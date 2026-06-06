#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from superagi.model.checkpoint import generate_from_checkpoint  # noqa: E402
from superagi.chat.formatting import format_user_prompt  # noqa: E402


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: 1, 0, true, false, yes, no")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text from a SuperAGI checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/latest.pt",
        help="Path to a portable model checkpoint.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to continue.")
    parser.add_argument(
        "--chat",
        type=_parse_bool,
        default=False,
        help="Format prompt as a User:/AGI: chat turn before generation.",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Limit sampling to the k most likely next tokens; 0 disables it.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty applied to recently used token logits; 1.0 disables it.",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=128,
        help="Number of recent tokens to penalize; 0 disables the window.",
    )
    parser.add_argument(
        "--stream",
        type=_parse_bool,
        default=True,
        help="Print generated token text as it is sampled; use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to run on: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")
    if args.repetition_window < 0:
        parser.error("--repetition-window must be non-negative")

    prompt = format_user_prompt(args.prompt).text if args.chat else args.prompt

    if args.stream:
        print(prompt, end="", flush=True)
        generate_from_checkpoint(
            args.checkpoint,
            prompt=prompt,
            max_new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            on_text=lambda text: print(text, end="", flush=True),
            device=args.device,
            seed=args.seed,
        )
        print()
    else:
        generated = generate_from_checkpoint(
            args.checkpoint,
            prompt=prompt,
            max_new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            device=args.device,
            seed=args.seed,
        )
        print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
