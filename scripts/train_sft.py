from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from superagi.chat.sft import IGNORE_INDEX, load_sft_records, tokenize_sft_messages
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
    LoadedCheckpoint,
    load_checkpoint,
    retain_checkpoint_snapshot,
    save_checkpoint,
)
from superagi.training.train import (
    TrainConfig,
    append_metrics_jsonl,
    learning_rate_for_step,
    train_accumulated_step,
)


TRAINER_STATE_FORMAT = "superagi-sft-trainer-state-v1"


@dataclass(frozen=True)
class SftMetricSnapshot:
    step: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    elapsed_seconds: float
    supervised_tokens_seen: int
    examples_seen: int
    supervised_tokens_per_second: float
    examples_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune a checkpoint on User:/AGI: examples.",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--run-dir",
        default="",
        help=(
            "Production directory for latest.pt, best.pt, final.pt, "
            "trainer-state.pt, snapshots/, and metrics.jsonl."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a validated, incomplete production --run-dir.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Deprecated bounded-local compatibility alias for the final checkpoint.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Deprecated bounded-local compatibility alias for metrics JSONL.",
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
        help="Persist and print train progress every N steps; 0 logs on checkpoints only.",
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
        help="Deterministically cap examples before splitting; 0 uses all examples.",
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
    _validate_args(args)

    production_mode = bool(args.run_dir.strip())
    base_path = Path(args.base_checkpoint)
    data_paths = _parse_data_paths(args.data)
    if production_mode:
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
    trainer_state_path = run_dir / "trainer-state.pt"
    snapshots_path = run_dir / "snapshots"

    _validate_artifact_paths(
        base_path=base_path,
        latest_path=latest_path,
        best_path=best_path,
        final_path=final_path,
        metrics_path=metrics_path,
        snapshots_path=snapshots_path,
        final_alias_path=final_alias_path,
    )
    _prepare_run_directory(
        run_dir,
        production_mode=production_mode,
        resume=args.resume,
        trainer_state_path=trainer_state_path,
        final_path=final_path,
    )
    if production_mode and args.validation_fraction <= 0:
        raise SystemExit(
            "production --run-dir requires a non-empty held-out validation split"
        )

    try:
        source_weights = parse_sft_source_weights(args.source_weights)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    _seed_global_torch_rng(args.seed)
    base_checkpoint = load_checkpoint(base_path, map_location="cpu")
    base_checkpoint_sha256 = _sha256_file(base_path)
    run_signature = _build_run_signature(
        args=args,
        base_path=base_path,
        base_checkpoint_sha256=base_checkpoint_sha256,
        data_paths=data_paths,
        source_weights=source_weights,
    )

    resume_state: dict[str, Any] | None = None
    model_checkpoint = base_checkpoint
    if args.resume:
        resume_state, model_checkpoint = _load_validated_resume(
            trainer_state_path=trainer_state_path,
            latest_path=latest_path,
            best_path=best_path,
            final_path=final_path,
            metrics_path=metrics_path,
            expected_run_signature=run_signature,
            expected_base_sha256=base_checkpoint_sha256,
        )

    sft_pad_token_id = resolve_sft_pad_token_id(base_checkpoint.tokenizer)
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
            base_checkpoint.tokenizer,
            source=record.source,
        )
        for record in records
    ]
    examples = [
        example
        for example in tokenized_examples
        if len(example.input_ids) <= base_checkpoint.config.context_length
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
            f"context_length={base_checkpoint.config.context_length}",
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
    _require_production_validation(
        production_mode=production_mode,
        validation_fraction=args.validation_fraction,
        validation_examples=validation_examples,
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
    model_checkpoint.model.config = replace(
        model_checkpoint.model.config,
        activation_checkpointing=args.activation_checkpointing == "1",
    )
    model = model_checkpoint.model.to(device)
    mixed_precision_dtype = _resolve_mixed_precision_dtype(args.mixed_precision, device)
    grad_scaler = _build_grad_scaler(mixed_precision_dtype, device)
    optimizer = _build_adamw_optimizer(
        list(model.parameters()),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        fused_mode=args.fused_adamw,
        device=device,
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
    metrics: list[SftMetricSnapshot] = []
    best_selection_loss: float | None = None
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    supervised_tokens_seen = 0
    examples_seen = 0
    elapsed_offset = 0.0
    first_step = 1
    if resume_state is not None:
        optimizer.load_state_dict(_required_mapping(resume_state, "optimizer_state"))
        scaler_state = resume_state.get("grad_scaler_state")
        if grad_scaler is None:
            if scaler_state is not None:
                raise SystemExit(
                    "trainer state requires a CUDA GradScaler, but this run does not"
                )
        elif scaler_state is not None:
            grad_scaler.load_state_dict(scaler_state)
        generator.set_state(_required_tensor(resume_state, "sampler_rng_state"))
        _restore_global_torch_rng(resume_state)
        losses = list(model_checkpoint.losses)
        metrics = [
            _metric_from_mapping(metric)
            for metric in model_checkpoint.metrics
        ]
        best_selection_loss = _optional_float(
            resume_state.get("best_validation_loss")
        )
        best_validation_loss = best_selection_loss
        best_validation_step = _optional_int(
            resume_state.get("best_validation_step")
        )
        supervised_tokens_seen = _required_int(
            resume_state,
            "supervised_tokens_seen",
        )
        examples_seen = _required_int(resume_state, "examples_seen")
        elapsed_offset = float(resume_state.get("elapsed_seconds", 0.0))
        first_step = _required_int(resume_state, "completed_step") + 1
        _trim_uncommitted_metrics(
            metrics_path,
            committed_metrics=model_checkpoint.metrics,
        )

    start_time = time.perf_counter()
    for step_index in range(first_step, args.steps + 1):
        learning_rate = learning_rate_for_step(config, step_index)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        batches, sampled_supervised_tokens, sampled_examples = (
            _sample_sft_microbatches(
                train_examples,
                batch_size=args.batch,
                grad_accum_steps=args.grad_accum_steps,
                pad_token_id=sft_pad_token_id,
                device=device,
                generator=generator,
                source_weights=source_weights,
            )
        )
        train_loss = train_accumulated_step(
            model=model,
            optimizer=optimizer,
            batches=batches,
            microbatch_count=len(batches),
            grad_clip=args.grad_clip,
            mixed_precision_dtype=mixed_precision_dtype,
            grad_scaler=grad_scaler,
        )
        losses.append(train_loss)
        supervised_tokens_seen += sampled_supervised_tokens
        examples_seen += sampled_examples

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
        if not should_checkpoint and not should_log:
            continue

        elapsed_seconds = elapsed_offset + (time.perf_counter() - start_time)
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
            if should_checkpoint and validation_examples
            else None
        )
        metric = SftMetricSnapshot(
            step=step_index,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            elapsed_seconds=elapsed_seconds,
            supervised_tokens_seen=supervised_tokens_seen,
            examples_seen=examples_seen,
            supervised_tokens_per_second=(
                supervised_tokens_seen / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
            examples_per_second=(
                examples_seen / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
        )
        metrics.append(metric)
        append_metrics_jsonl(metrics_path, [metric])

        if should_checkpoint:
            metadata = _build_metadata(
                checkpoint=base_checkpoint,
                base_path=base_path,
                data_paths=data_paths,
                metrics_path=metrics_path,
                args=args,
                source_weights=source_weights,
                step=step_index,
                train_examples=len(train_examples),
                validation_examples=len(validation_examples),
                skipped_examples=skipped_examples,
                supervised_tokens_seen=supervised_tokens_seen,
                examples_seen=examples_seen,
            )
            _save_checkpoint_atomic(
                latest_path,
                model=model,
                vocab=base_checkpoint.vocab,
                losses=losses,
                metrics=[asdict(item) for item in metrics],
                metadata=metadata,
            )
            selection_loss = (
                validation_loss
                if production_mode
                else (
                    validation_loss
                    if validation_loss is not None
                    else train_loss
                )
            )
            if selection_loss is None:
                raise RuntimeError(
                    "production best checkpoint selection requires validation loss"
                )
            if best_selection_loss is None or selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                best_validation_loss = validation_loss
                best_validation_step = step_index
                best_metadata = dict(metadata)
                best_metadata.update(
                    {
                        "best_validation_loss": best_validation_loss,
                        "best_validation_step": best_validation_step,
                    }
                )
                _save_checkpoint_atomic(
                    best_path,
                    model=model,
                    vocab=base_checkpoint.vocab,
                    losses=losses,
                    metrics=[asdict(item) for item in metrics],
                    metadata=best_metadata,
                )
            retain_checkpoint_snapshot(
                latest_path,
                snapshots_path,
                step=step_index,
                keep=args.checkpoint_keep,
            )
            if production_mode:
                _save_trainer_state(
                    trainer_state_path,
                    completed_step=step_index,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    generator=generator,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    supervised_tokens_seen=supervised_tokens_seen,
                    examples_seen=examples_seen,
                    elapsed_seconds=elapsed_seconds,
                    run_signature=run_signature,
                    base_checkpoint_sha256=base_checkpoint_sha256,
                    latest_path=latest_path,
                    best_path=best_path,
                    metrics_count=len(metrics),
                )
            clear_sft_device_cache(device)

        _print_progress(metric)

    if not best_path.exists():
        raise RuntimeError("SFT did not produce a best checkpoint")
    _atomic_copy(best_path, final_path)
    if final_alias_path is not None:
        _atomic_copy(best_path, final_alias_path)
    print(f"SFT run directory: {run_dir}")
    print(f"SFT checkpoint: {final_path}")
    print(f"SFT metrics: {metrics_path}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
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
    if args.resume and not args.run_dir.strip():
        raise SystemExit("--resume requires production --run-dir")


def _validate_artifact_paths(
    *,
    base_path: Path,
    latest_path: Path,
    best_path: Path,
    final_path: Path,
    metrics_path: Path,
    snapshots_path: Path,
    final_alias_path: Path | None,
) -> None:
    resolved_base = base_path.resolve(strict=False)
    resolved_snapshots = snapshots_path.resolve(strict=False)
    trainer_state_path = latest_path.parent / "trainer-state.pt"
    artifact_paths = {
        "latest checkpoint": latest_path,
        "best checkpoint": best_path,
        "final checkpoint": final_path,
        "metrics": metrics_path,
        "trainer state": trainer_state_path,
    }
    if final_alias_path is not None:
        artifact_paths["legacy output"] = final_alias_path
    resolved_artifacts = {
        name: path.resolve(strict=False)
        for name, path in artifact_paths.items()
    }
    if (
        resolved_base == resolved_snapshots
        or resolved_base.is_relative_to(resolved_snapshots)
        or resolved_base in resolved_artifacts.values()
    ):
        raise SystemExit(
            "base checkpoint must not collide with any SFT output artifact"
        )
    seen: dict[Path, str] = {}
    for name, path in resolved_artifacts.items():
        if path in seen:
            raise SystemExit(
                f"SFT output path collision: {name} and {seen[path]} resolve to {path}"
            )
        if path == resolved_snapshots or path.is_relative_to(resolved_snapshots):
            raise SystemExit(
                f"SFT output path collision: {name} is inside snapshots/"
            )
        seen[path] = name


def _prepare_run_directory(
    run_dir: Path,
    *,
    production_mode: bool,
    resume: bool,
    trainer_state_path: Path,
    final_path: Path,
) -> None:
    if not production_mode:
        return
    if resume:
        if not run_dir.is_dir() or not any(run_dir.iterdir()):
            raise SystemExit("cannot resume an empty or missing production run directory")
        if not trainer_state_path.is_file():
            raise SystemExit(
                "production resume requires a valid trainer state artifact"
            )
        if final_path.exists():
            raise SystemExit("production run is already complete; final.pt exists")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            "refusing to start a fresh production run in a non-empty run directory"
        )


def _require_production_validation(
    *,
    production_mode: bool,
    validation_fraction: float,
    validation_examples: Sequence[object],
) -> None:
    if production_mode and (
        validation_fraction <= 0 or not validation_examples
    ):
        raise SystemExit(
            "production --run-dir requires a non-empty held-out validation split"
        )


def _seed_global_torch_rng(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_global_torch_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(_required_tensor(state, "torch_rng_state"))
    cuda_rng_state = state.get("cuda_rng_state_all")
    if cuda_rng_state is not None:
        if not torch.cuda.is_available():
            raise SystemExit(
                "trainer state contains CUDA RNG state, but CUDA is unavailable"
            )
        if not isinstance(cuda_rng_state, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_rng_state
        ):
            raise SystemExit("trainer state has invalid CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_rng_state)


def _build_adamw_optimizer(
    parameters: Sequence[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    fused_mode: str,
    device: torch.device,
) -> torch.optim.Optimizer:
    parameter_list = list(parameters)
    if device.type != "cuda":
        if fused_mode == "on":
            raise ValueError("fused AdamW requires CUDA")
        return torch.optim.AdamW(
            parameter_list,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if fused_mode == "off":
        return torch.optim.AdamW(
            parameter_list,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    try:
        return torch.optim.AdamW(
            parameter_list,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
        )
    except (TypeError, RuntimeError, NotImplementedError):
        if fused_mode == "on":
            raise
        return torch.optim.AdamW(
            parameter_list,
            lr=learning_rate,
            weight_decay=weight_decay,
        )


def _sample_sft_microbatches(
    examples: Sequence[object],
    *,
    batch_size: int,
    grad_accum_steps: int,
    pad_token_id: int,
    device: torch.device,
    generator: torch.Generator,
    source_weights: Mapping[str, float],
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int]:
    batches = [
        sample_sft_batch(
            examples,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
            generator=generator,
            source_weights=source_weights,
        )
        for _ in range(grad_accum_steps)
    ]
    supervised_tokens = sum(
        int((target_ids != IGNORE_INDEX).sum().item())
        for _, target_ids in batches
    )
    sampled_examples = sum(int(target_ids.shape[0]) for _, target_ids in batches)
    return batches, supervised_tokens, sampled_examples


def _build_run_signature(
    *,
    args: argparse.Namespace,
    base_path: Path,
    base_checkpoint_sha256: str,
    data_paths: Sequence[Path],
    source_weights: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "base_checkpoint": str(base_path.resolve(strict=False)),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "data": [
            {
                "path": str(path.resolve(strict=False)),
                "sha256": _sha256_file(path),
            }
            for path in data_paths
        ],
        "steps": args.steps,
        "batch": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.lr,
        "min_learning_rate": args.lr_min,
        "warmup_steps": args.lr_warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_keep": args.checkpoint_keep,
        "log_interval": args.log_interval,
        "mixed_precision": args.mixed_precision,
        "fused_adamw": args.fused_adamw,
        "activation_checkpointing": args.activation_checkpointing,
        "validation_fraction": args.validation_fraction,
        "validation_batches": args.validation_batches,
        "max_examples": args.max_examples,
        "source_weights": dict(sorted(source_weights.items())),
        "seed": args.seed,
    }


def _load_validated_resume(
    *,
    trainer_state_path: Path,
    latest_path: Path,
    best_path: Path,
    final_path: Path,
    metrics_path: Path,
    expected_run_signature: Mapping[str, Any],
    expected_base_sha256: str,
) -> tuple[dict[str, Any], LoadedCheckpoint]:
    if final_path.exists():
        raise SystemExit("production run is already complete; final.pt exists")
    try:
        state = torch.load(trainer_state_path, map_location="cpu")
    except Exception as error:
        raise SystemExit(f"failed to load trainer state: {error}") from error
    if not isinstance(state, dict) or state.get("format") != TRAINER_STATE_FORMAT:
        raise SystemExit("trainer state has an unsupported or corrupt format")
    if state.get("run_signature") != dict(expected_run_signature):
        raise SystemExit("trainer state does not match the requested SFT run schedule")
    if state.get("base_checkpoint_sha256") != expected_base_sha256:
        raise SystemExit("trainer state base checkpoint hash does not match")
    if not latest_path.is_file() or not best_path.is_file():
        raise SystemExit("trainer state requires both latest.pt and best.pt")
    if state.get("latest_checkpoint_sha256") != _sha256_file(latest_path):
        raise SystemExit("latest.pt does not match trainer state; refusing corrupt reuse")
    if state.get("best_checkpoint_sha256") != _sha256_file(best_path):
        raise SystemExit("best.pt does not match trainer state; refusing corrupt reuse")
    try:
        latest = load_checkpoint(latest_path, map_location="cpu")
    except Exception as error:
        raise SystemExit(f"failed to load latest checkpoint: {error}") from error
    completed_step = _required_int(state, "completed_step")
    if completed_step <= 0:
        raise SystemExit("trainer state completed step must be positive")
    if latest.metadata.get("sft_steps") != completed_step:
        raise SystemExit("latest.pt step does not match trainer state")
    if len(latest.losses) != completed_step:
        raise SystemExit("latest.pt losses do not match trainer state step")
    expected_metrics_count = _required_int(state, "metrics_count")
    if expected_metrics_count != len(latest.metrics):
        raise SystemExit("latest.pt metrics do not match trainer state")
    persisted_metrics = _read_metrics_jsonl(metrics_path)
    if len(persisted_metrics) < expected_metrics_count:
        raise SystemExit("metrics.jsonl is incomplete for trainer state")
    if persisted_metrics[:expected_metrics_count] != latest.metrics:
        raise SystemExit("metrics.jsonl does not match trainer state")
    return state, latest


def _trim_uncommitted_metrics(
    path: Path,
    *,
    committed_metrics: Sequence[Mapping[str, Any]],
) -> None:
    persisted_metrics = _read_metrics_jsonl(path)
    if len(persisted_metrics) > len(committed_metrics):
        _write_metrics_jsonl_atomic(path, committed_metrics)


def _save_trainer_state(
    path: Path,
    *,
    completed_step: int,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler | None,
    generator: torch.Generator,
    best_validation_loss: float | None,
    best_validation_step: int | None,
    supervised_tokens_seen: int,
    examples_seen: int,
    elapsed_seconds: float,
    run_signature: Mapping[str, Any],
    base_checkpoint_sha256: str,
    latest_path: Path,
    best_path: Path,
    metrics_count: int,
) -> None:
    cuda_rng_state_all = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    _atomic_torch_save(
        {
            "format": TRAINER_STATE_FORMAT,
            "completed_step": completed_step,
            "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": (
                grad_scaler.state_dict() if grad_scaler is not None else None
            ),
            "sampler_rng_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": cuda_rng_state_all,
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "supervised_tokens_seen": supervised_tokens_seen,
            "examples_seen": examples_seen,
            "elapsed_seconds": elapsed_seconds,
            "run_signature": dict(run_signature),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "latest_checkpoint_sha256": _sha256_file(latest_path),
            "best_checkpoint_sha256": _sha256_file(best_path),
            "metrics_count": metrics_count,
        },
        path,
    )


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
    supervised_tokens_seen: int,
    examples_seen: int,
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
            "sft_supervised_tokens_seen": supervised_tokens_seen,
            "sft_sampled_examples_seen": examples_seen,
        }
    )
    return metadata


def _save_checkpoint_atomic(
    path: Path,
    *,
    model: torch.nn.Module,
    vocab: dict[str, Any],
    losses: Sequence[float],
    metrics: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        save_checkpoint(
            temporary_path,
            model=model,
            vocab=vocab,
            losses=losses,
            metrics=metrics,
            metadata=metadata,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(path)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(destination)
    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp")


def _read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit("trainer state requires metrics.jsonl")
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to read metrics.jsonl: {error}") from error


def _write_metrics_jsonl_atomic(
    path: Path,
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for metric in metrics:
                file.write(json.dumps(dict(metric), sort_keys=True))
                file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _metric_from_mapping(payload: Mapping[str, Any]) -> SftMetricSnapshot:
    field_names = {field.name for field in fields(SftMetricSnapshot)}
    try:
        return SftMetricSnapshot(
            **{name: payload[name] for name in field_names}
        )
    except KeyError as error:
        raise SystemExit(
            f"latest.pt contains an incomplete SFT metric: {error}"
        ) from error


def _print_progress(metric: SftMetricSnapshot) -> None:
    validation_text = (
        f"validation_loss={metric.validation_loss:.6f} "
        if metric.validation_loss is not None
        else "validation_loss=null "
    )
    print(
        "sft_step="
        f"{metric.step} "
        f"train_loss={metric.train_loss:.6f} "
        f"{validation_text}"
        f"learning_rate={metric.learning_rate:.8f} "
        f"elapsed_seconds={metric.elapsed_seconds:.2f} "
        f"supervised_tokens_per_second={metric.supervised_tokens_per_second:.2f} "
        f"examples_per_second={metric.examples_per_second:.2f}",
        flush=True,
    )


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


def _parse_data_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise SystemExit("--data must contain at least one JSONL path")
    return paths


def _format_source_weights(source_weights: dict[str, float]) -> str:
    return ", ".join(
        f"{source}={source_weights[source]:g}" for source in sorted(source_weights)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"trainer state field {key!r} must be a mapping")
    return value


def _required_tensor(
    payload: Mapping[str, Any],
    key: str,
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SystemExit(f"trainer state field {key!r} must be a tensor")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise SystemExit(f"trainer state field {key!r} must be an integer")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise SystemExit("trainer state best validation step must be an integer")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (float, int)):
        raise SystemExit("trainer state best validation loss must be numeric")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
