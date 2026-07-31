from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from superagi.chat.sft import load_sft_records, tokenize_sft_messages
from superagi.chat.sft_training import (
    clear_sft_device_cache,
    evaluate_sft_loss,
    limit_sft_examples,
    parse_sft_source_weights,
    resolve_sft_pad_token_id,
    sample_sft_batch,
    should_log_sft_progress,
    source_summary,
    split_sft_examples,
)
from superagi.model.checkpoint import (
    load_checkpoint,
    retain_checkpoint_snapshot,
    save_checkpoint,
)
from superagi.training.train import (
    MetricSnapshot,
    TrainConfig,
    append_metrics_jsonl,
    learning_rate_for_step,
    train_accumulated_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune a checkpoint on User:/AGI: examples.",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--run-dir",
        default="",
        help="Directory for latest.pt, best.pt, final.pt, snapshots/, and metrics.jsonl.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Deprecated compatibility alias for the final checkpoint path.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Deprecated compatibility alias for the metrics JSONL path.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print train-loss progress every N steps without checkpointing; 0 logs on checkpoints only.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "auto", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--fused-adamw",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=("0", "1"),
        default="0",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Deterministically cap loaded SFT examples before train/validation split; 0 uses all examples.",
    )
    parser.add_argument(
        "--source-weights",
        default="",
        help="Comma-separated source=weight entries, e.g. anchor=4,wildchat=0.35",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive")
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be non-negative")
    if args.checkpoint_keep < 0:
        raise SystemExit("--checkpoint-keep must be non-negative")
    if args.log_interval < 0:
        raise SystemExit("--log-interval must be non-negative")
    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    if args.validation_batches <= 0:
        raise SystemExit("--validation-batches must be positive")
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")

    if bool(args.run_dir.strip()) == bool(args.out.strip()):
        raise SystemExit("supply exactly one of --run-dir or --out")
    if args.out.strip() and not args.metrics.strip():
        raise SystemExit("--metrics is required with deprecated --out")

    base_path = Path(args.base_checkpoint)
    data_paths = _parse_data_paths(args.data)
    if args.run_dir.strip():
        run_dir = Path(args.run_dir)
        final_alias_path: Path | None = None
        metrics_path = run_dir / "metrics.jsonl"
    else:
        final_alias_path = Path(args.out)
        run_dir = final_alias_path.parent
        metrics_path = Path(args.metrics)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    final_path = run_dir / "final.pt"
    snapshots_path = run_dir / "snapshots"
    try:
        source_weights = parse_sft_source_weights(args.source_weights)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = load_checkpoint(base_path, map_location="cpu")
    sft_pad_token_id = resolve_sft_pad_token_id(checkpoint.tokenizer)
    records = [
        record
        for data_path in data_paths
        for record in load_sft_records(
            data_path,
            default_source=data_path.stem,
        )
    ]
    tokenized_examples = [
        tokenize_sft_messages(
            record.messages,
            checkpoint.tokenizer,
            source=record.source,
        )
        for record in records
    ]
    examples = [
        example
        for example in tokenized_examples
        if len(example.input_ids) <= checkpoint.config.context_length
    ]
    skipped_examples = len(tokenized_examples) - len(examples)
    if not examples:
        raise SystemExit(
            "no SFT examples fit in the model context window; "
            "shorten examples or increase ctx_window"
        )
    if skipped_examples:
        print(
            f"Skipped {skipped_examples} SFT examples longer than "
            f"context_length={checkpoint.config.context_length}",
            flush=True,
        )
    total_examples_before_limit = len(examples)
    examples = limit_sft_examples(
        examples,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    if len(examples) < total_examples_before_limit:
        print(
            "SFT max examples: "
            f"using {len(examples)} of {total_examples_before_limit}",
            flush=True,
        )
    train_examples, validation_examples = split_sft_examples(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        "SFT examples: "
        f"train={len(train_examples)} validation={len(validation_examples)}",
        flush=True,
    )
    train_source_summary = source_summary(train_examples, source_weights)
    validation_source_summary = source_summary(validation_examples, source_weights)
    print(f"SFT train sources: {train_source_summary.counts_text}", flush=True)
    if validation_examples:
        print(
            f"SFT validation sources: {validation_source_summary.counts_text}",
            flush=True,
        )
    if source_weights:
        print(
            "SFT source weights: "
            f"{_format_source_weights(source_weights)}",
            flush=True,
        )
        print(
            "SFT weighted train sampling mass: "
            f"{train_source_summary.sampling_mass_text}",
            flush=True,
        )

    device = _resolve_device(args.device)
    checkpoint.model.config = replace(
        checkpoint.model.config,
        activation_checkpointing=args.activation_checkpointing == "1",
    )
    model = checkpoint.model.to(device)
    mixed_precision_dtype = _resolve_mixed_precision_dtype(args.mixed_precision, device)
    grad_scaler = _build_grad_scaler(mixed_precision_dtype, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **_adamw_kwargs(args.fused_adamw, device),
    )
    config = TrainConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        min_learning_rate=args.lr_min,
        warmup_steps=args.lr_warmup_steps,
        max_steps=args.steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    metrics: list[MetricSnapshot] = []
    best_validation_loss: float | None = None
    best_selection_loss: float | None = None
    start_time = time.perf_counter()
    for step_index in range(1, args.steps + 1):
        learning_rate = learning_rate_for_step(config, step_index)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        def _microbatches() -> list[tuple[torch.Tensor, torch.Tensor]]:
            return [
                sample_sft_batch(
                    train_examples,
                    batch_size=args.batch,
                    pad_token_id=sft_pad_token_id,
                    device=device,
                    generator=generator,
                    source_weights=source_weights,
                )
                for _ in range(args.grad_accum_steps)
            ]

        train_loss = train_accumulated_step(
            model=model,
            optimizer=optimizer,
            batches=_microbatches(),
            microbatch_count=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            mixed_precision_dtype=mixed_precision_dtype,
            grad_scaler=grad_scaler,
        )
        losses.append(train_loss)

        should_checkpoint = step_index == args.steps or (
            args.checkpoint_interval > 0
            and step_index % args.checkpoint_interval == 0
        )
        should_log = should_log_sft_progress(
            step=step_index,
            total_steps=args.steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
        )
        if should_checkpoint:
            validation_loss = (
                evaluate_sft_loss(
                    model,
                    validation_examples,
                    batch_size=args.batch,
                    pad_token_id=sft_pad_token_id,
                    device=device,
                    max_batches=args.validation_batches,
                    mixed_precision_dtype=mixed_precision_dtype,
                )
                if validation_examples
                else None
            )
            metric = MetricSnapshot(
                step=step_index,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - start_time,
            )
            metrics.append(metric)
            append_metrics_jsonl(metrics_path, [metric])
            metadata = _build_metadata(
                checkpoint=checkpoint,
                base_path=base_path,
                data_paths=data_paths,
                metrics_path=metrics_path,
                args=args,
                source_weights=source_weights,
                step=step_index,
                train_examples=len(train_examples),
                validation_examples=len(validation_examples),
                skipped_examples=skipped_examples,
            )
            save_checkpoint(
                latest_path,
                model=model,
                vocab=checkpoint.vocab,
                losses=losses,
                metrics=[asdict(item) for item in metrics],
                metadata=metadata,
            )
            retain_checkpoint_snapshot(
                latest_path,
                snapshots_path,
                step=step_index,
                keep=args.checkpoint_keep,
            )
            selection_loss = validation_loss if validation_loss is not None else train_loss
            if best_selection_loss is None or selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                best_validation_loss = validation_loss
                best_metadata = dict(metadata)
                best_metadata.update(
                    {
                        "best_validation_loss": best_validation_loss,
                        "best_validation_step": step_index,
                    }
                )
                save_checkpoint(
                    best_path,
                    model=model,
                    vocab=checkpoint.vocab,
                    losses=losses,
                    metrics=[asdict(item) for item in metrics],
                    metadata=best_metadata,
                )
            validation_text = (
                f"validation_loss={validation_loss:.6f} "
                if validation_loss is not None
                else "validation_loss=null "
            )
            print(
                "sft_step="
                f"{step_index} "
                f"train_loss={train_loss:.6f} "
                f"{validation_text}"
                f"learning_rate={learning_rate:.8f} "
                f"elapsed_seconds={metric.elapsed_seconds:.2f} "
                f"supervised_tokens_per_second={_sft_tokens_per_second(step_index, args, train_examples, metric.elapsed_seconds):.2f} "
                f"examples_per_second={(step_index * args.batch * args.grad_accum_steps) / metric.elapsed_seconds:.2f}",
                flush=True,
            )
            clear_sft_device_cache(device)
        elif should_log:
            elapsed_seconds = time.perf_counter() - start_time
            print(
                "sft_step="
                f"{step_index} "
                f"train_loss={train_loss:.6f} "
                f"learning_rate={learning_rate:.8f} "
                f"elapsed_seconds={elapsed_seconds:.2f}",
                flush=True,
            )

    if not best_path.exists():
        raise RuntimeError("SFT did not produce a best checkpoint")
    shutil.copy2(best_path, final_path)
    if final_alias_path is not None:
        shutil.copy2(best_path, final_alias_path)
    print(f"SFT run directory: {run_dir}")
    print(f"SFT checkpoint: {final_path}")
    print(f"SFT metrics: {metrics_path}")
    return 0


def _build_metadata(
    *,
    checkpoint: object,
    base_path: Path,
    data_paths: list[Path],
    metrics_path: Path,
    args: argparse.Namespace,
    source_weights: dict[str, float],
    step: int,
    train_examples: int,
    validation_examples: int,
    skipped_examples: int,
) -> dict[str, object]:
    metadata = dict(getattr(checkpoint, "metadata"))
    metadata.update(
        {
            "status": "sft",
            "base_checkpoint": str(base_path),
            "sft_data": ",".join(str(path) for path in data_paths),
            "sft_data_paths": [str(path) for path in data_paths],
            "sft_metrics_path": str(metrics_path),
            "sft_steps": step,
            "sft_batch_size": args.batch,
            "sft_grad_accum_steps": args.grad_accum_steps,
            "sft_learning_rate": args.lr,
            "sft_min_learning_rate": args.lr_min,
            "sft_warmup_steps": args.lr_warmup_steps,
            "sft_examples": train_examples + validation_examples,
            "sft_train_examples": train_examples,
            "sft_validation_examples": validation_examples,
            "sft_validation_fraction": args.validation_fraction,
            "sft_validation_batches": args.validation_batches,
            "sft_max_examples": args.max_examples,
            "sft_log_interval": args.log_interval,
            "sft_mixed_precision": args.mixed_precision,
            "sft_fused_adamw": args.fused_adamw,
            "sft_activation_checkpointing": args.activation_checkpointing == "1",
            "sft_checkpoint_keep": args.checkpoint_keep,
            "sft_source_weights": source_weights,
            "sft_skipped_examples": skipped_examples,
        }
    )
    return metadata


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _resolve_mixed_precision_dtype(
    mixed_precision: str,
    device: torch.device,
) -> torch.dtype | None:
    if mixed_precision == "none":
        return None
    if mixed_precision == "auto":
        return torch.float16 if device.type == "cuda" else None
    if mixed_precision == "float16":
        return torch.float16
    if mixed_precision == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported mixed precision setting: {mixed_precision}")


def _build_grad_scaler(
    mixed_precision_dtype: torch.dtype | None,
    device: torch.device,
) -> torch.amp.GradScaler | None:
    if mixed_precision_dtype != torch.float16 or device.type != "cuda":
        return None
    return torch.amp.GradScaler(device.type, enabled=True)


def _adamw_kwargs(mode: str, device: torch.device) -> dict[str, bool]:
    if mode == "off" or device.type != "cuda":
        return {}
    return {"fused": True}


def _sft_tokens_per_second(
    step: int,
    args: argparse.Namespace,
    examples: list[object] | tuple[object, ...],
    elapsed_seconds: float,
) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    # Example lengths vary, so this is deliberately a supervised-example proxy.
    mean_tokens = 0.0
    if examples:
        mean_tokens = sum(
            float(getattr(example, "supervised_token_count", 0))
            for example in examples
        ) / len(examples)
    return (
        step * args.batch * args.grad_accum_steps * mean_tokens / elapsed_seconds
    )


def _parse_data_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise SystemExit("--data must contain at least one JSONL path")
    return paths


def _format_source_weights(source_weights: dict[str, float]) -> str:
    return ", ".join(
        f"{source}={source_weights[source]:g}" for source in sorted(source_weights)
    )


if __name__ == "__main__":
    raise SystemExit(main())
