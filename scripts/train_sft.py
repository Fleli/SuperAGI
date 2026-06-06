from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from superagi.chat.sft import load_sft_jsonl, tokenize_sft_messages
from superagi.chat.sft_training import sample_sft_batch
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be non-negative")

    base_path = Path(args.base_checkpoint)
    data_path = Path(args.data)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)

    checkpoint = load_checkpoint(base_path, map_location="cpu")
    conversations = load_sft_jsonl(data_path)
    tokenized_examples = [
        tokenize_sft_messages(messages, checkpoint.tokenizer)
        for messages in conversations
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
            examples,
            batch_size=args.batch,
            pad_token_id=0,
            device=device,
            generator=generator,
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
            metric = MetricSnapshot(
                step=step_index,
                train_loss=train_loss,
                validation_loss=None,
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
                    data_path=data_path,
                    metrics_path=metrics_path,
                    args=args,
                    step=step_index,
                    examples=len(examples),
                    skipped_examples=skipped_examples,
                ),
            )
            print(
                "sft_step="
                f"{step_index} "
                f"train_loss={train_loss:.6f} "
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
    data_path: Path,
    metrics_path: Path,
    args: argparse.Namespace,
    step: int,
    examples: int,
    skipped_examples: int,
) -> dict[str, object]:
    metadata = dict(getattr(checkpoint, "metadata"))
    metadata.update(
        {
            "status": "sft",
            "base_checkpoint": str(base_path),
            "sft_data": str(data_path),
            "sft_metrics_path": str(metrics_path),
            "sft_steps": step,
            "sft_batch_size": args.batch,
            "sft_learning_rate": args.lr,
            "sft_min_learning_rate": args.lr_min,
            "sft_warmup_steps": args.lr_warmup_steps,
            "sft_examples": examples,
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


if __name__ == "__main__":
    raise SystemExit(main())
