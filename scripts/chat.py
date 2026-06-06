#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from superagi.chat.formatting import ChatMessage  # noqa: E402
from superagi.chat.session import generate_chat_reply  # noqa: E402
from superagi.model.checkpoint import load_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start an interactive SuperAGI chat.")
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/latest.pt",
        help="Path to a model checkpoint.",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate for each AGI turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Limit sampling to the k most likely next tokens; 0 disables it.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty applied to recently used token logits; 1.0 disables it.",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=128,
        help="Number of recent tokens to penalize; 0 disables the window.",
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
    if args.new_tokens <= 0:
        parser.error("--new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")
    if args.repetition_window < 0:
        parser.error("--repetition-window must be non-negative")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run `make sft-train ...` first, or pass an existing checkpoint such as "
            "`CHAT_CHECKPOINT=./best-svl-current.pt`.",
            file=sys.stderr,
        )
        return 1

    device = _resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint.model.to(device)
    messages: list[ChatMessage] = []

    print("SuperAGI chat. Type /exit or /quit to stop.")
    while True:
        try:
            user_text = input("User: ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break

        messages.append(ChatMessage(role="user", content=user_text))
        print("AGI: ", end="", flush=True)
        reply = generate_chat_reply(
            checkpoint=checkpoint,
            messages=messages,
            max_new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            device=device,
            on_text=lambda text: print(text, end="", flush=True),
        )
        print()
        messages.append(ChatMessage(role="agi", content=reply or "..."))

    return 0


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
