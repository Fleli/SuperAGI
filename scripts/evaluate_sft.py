#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from superagi.chat.sft_evaluation import (  # noqa: E402
    EvaluationGates,
    evaluate_responses,
    generate_evaluation_response,
    load_evaluation_prompts,
    write_evaluation_artifacts,
)
from superagi.model.checkpoint import load_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic behavioral gates against an SFT checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint to evaluate.")
    parser.add_argument(
        "--prompts",
        default="data/sft/eval_prompts.jsonl",
        help="Tracked held-out evaluation prompt suite.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="JSONL result path; defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="JSON summary path; defaults beside the checkpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-eos-termination-rate", type=float, default=0.90)
    parser.add_argument("--min-nonempty-response-rate", type=float, default=0.95)
    parser.add_argument("--max-repetition-failure-rate", type=float, default=0.05)
    parser.add_argument("--min-topic-reset-pass-rate", type=float, default=0.80)
    return parser


def resolve_output_paths(
    *,
    checkpoint_path: Path,
    results_path: str | Path | None,
    summary_path: str | Path | None,
) -> tuple[Path, Path]:
    output_dir = checkpoint_path.parent
    return (
        Path(results_path) if results_path else output_dir / "evaluation.jsonl",
        Path(summary_path)
        if summary_path
        else output_dir / "evaluation.summary.json",
    )


def run_evaluation(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing evaluation checkpoint: {checkpoint_path}")
    prompts = load_evaluation_prompts(args.prompts)
    device = _resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint.model.to(device)
    checkpoint.model.eval()

    outcomes = []
    for index, prompt in enumerate(prompts):
        _seed_generation(args.seed + index)
        outcomes.append(
            generate_evaluation_response(
                checkpoint=checkpoint,
                prompt=prompt,
                temperature=args.temperature,
                top_k=args.top_k or None,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                device=device,
            )
        )

    report = evaluate_responses(
        prompts,
        outcomes,
        gates=EvaluationGates(
            min_eos_termination_rate=args.min_eos_termination_rate,
            min_nonempty_response_rate=args.min_nonempty_response_rate,
            max_repetition_failure_rate=args.max_repetition_failure_rate,
            min_topic_reset_pass_rate=args.min_topic_reset_pass_rate,
        ),
    )
    results_path, summary_path = resolve_output_paths(
        checkpoint_path=checkpoint_path,
        results_path=args.results,
        summary_path=args.summary,
    )
    write_evaluation_artifacts(
        report,
        results_path=results_path,
        summary_path=summary_path,
    )
    summary = report.summary_mapping()
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print(f"Evaluation results: {results_path}")
    print(f"Evaluation summary: {summary_path}")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")
    if args.repetition_window < 0:
        parser.error("--repetition-window must be non-negative")
    try:
        return run_evaluation(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_generation(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
