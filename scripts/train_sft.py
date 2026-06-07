from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from superagi.chat.sft import load_sft_records, tokenize_sft_messages
from superagi.chat.sft_training import (
    evaluate_sft_loss,
    parse_sft_source_weights,
    sample_sft_batch,
    source_summary,
    split_sft_examples,
)
from superagi.model.checkpoint import load_checkpoint, save_checkpoint
from superagi.training.train import (
    MetricSnapshot,
    TrainConfig,
    append_metrics_jsonl,
    learning_rate_for_step,
    train_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune a checkpoint on User:/AGI: examples.",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-batches", type=int, default=10)
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
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be non-negative")
    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    if args.validation_batches <= 0:
        raise SystemExit("--validation-batches must be positive")

    base_path = Path(args.base_checkpoint)
    data_paths = _parse_data_paths(args.data)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    try:
        source_weights = parse_sft_source_weights(args.source_weights)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = load_checkpoint(base_path, map_location="cpu")
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
    model = checkpoint.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    metrics: list[MetricSnapshot] = []
    start_time = time.perf_counter()
    for step_index in range(1, args.steps + 1):
        learning_rate = learning_rate_for_step(config, step_index)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        input_ids, target_ids = sample_sft_batch(
            train_examples,
            batch_size=args.batch,
            pad_token_id=0,
            device=device,
            generator=generator,
            source_weights=source_weights,
        )
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            target_ids=target_ids,
            grad_clip=args.grad_clip,
        )
        losses.append(train_loss)

        should_report = step_index == args.steps or (
            args.checkpoint_interval > 0
            and step_index % args.checkpoint_interval == 0
        )
        if should_report:
            validation_loss = (
                evaluate_sft_loss(
                    model,
                    validation_examples,
                    batch_size=args.batch,
                    pad_token_id=0,
                    device=device,
                    max_batches=args.validation_batches,
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
            save_checkpoint(
                out_path,
                model=model,
                vocab=checkpoint.vocab,
                losses=losses,
                metrics=[asdict(item) for item in metrics],
                metadata=_build_metadata(
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
                ),
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
                f"elapsed_seconds={metric.elapsed_seconds:.2f}",
                flush=True,
            )

    print(f"SFT checkpoint: {out_path}")
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
            "sft_learning_rate": args.lr,
            "sft_min_learning_rate": args.lr_min,
            "sft_warmup_steps": args.lr_warmup_steps,
            "sft_examples": train_examples + validation_examples,
            "sft_train_examples": train_examples,
            "sft_validation_examples": validation_examples,
            "sft_validation_fraction": args.validation_fraction,
            "sft_validation_batches": args.validation_batches,
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
