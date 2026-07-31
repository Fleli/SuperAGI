from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from superagi.chat.sft_audit import (  # noqa: E402
    AuditConfig,
    AuditReport,
    audit_sft_corpus,
)
from superagi.chat.sft_training import parse_sft_source_weights  # noqa: E402
from superagi.model.checkpoint import load_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SuperAGI SFT JSONL corpora")
    parser.add_argument("--data", required=True, help="Comma-separated SFT JSONL paths")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint for token/context checks")
    parser.add_argument("--source-weights", default="", help="Comma-separated source=weight entries")
    parser.add_argument("--mode", choices=("curated", "mixed", "style"), default="curated")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    parser.add_argument("--curated-sampling-mass-min", type=float, default=0.15)
    parser.add_argument("--curated-sampling-mass-max", type=float, default=0.25)
    parser.add_argument(
        "--curated-source-families",
        default="curated_core,curated",
        help="Comma-separated source families counted as curated sampling mass",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_audit(args)


def run_audit(args: argparse.Namespace) -> int:
    paths = tuple(Path(value.strip()) for value in args.data.split(",") if value.strip())
    if not paths:
        raise SystemExit("--data must contain at least one JSONL path")
    try:
        source_weights = parse_sft_source_weights(args.source_weights)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    curated_source_families = tuple(
        value.strip()
        for value in getattr(
            args,
            "curated_source_families",
            "curated_core,curated",
        ).split(",")
        if value.strip()
    )
    if not curated_source_families:
        raise SystemExit("--curated-source-families must contain at least one family")

    tokenizer = None
    context_length = None
    if args.checkpoint.strip():
        checkpoint = load_checkpoint(args.checkpoint.strip(), map_location="cpu")
        tokenizer = checkpoint.tokenizer
        context_length = checkpoint.config.context_length

    report = audit_sft_corpus(
        paths,
        mode=args.mode,
        tokenizer=tokenizer,
        context_length=context_length,
        source_weights=source_weights,
        config=AuditConfig(
            curated_sampling_mass_min=getattr(
                args,
                "curated_sampling_mass_min",
                0.15,
            ),
            curated_sampling_mass_max=getattr(
                args,
                "curated_sampling_mass_max",
                0.25,
            ),
            curated_source_families=curated_source_families,
        ),
    )
    _print_summary(report)
    if args.report.strip():
        report_path = Path(args.report.strip())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.to_json(), encoding="utf-8")
        print(f"SFT audit report: {report_path}", flush=True)
    return 0 if report.ok else 1


def _print_summary(report: AuditReport) -> None:
    print(f"SFT audit: {'PASS' if report.ok else 'FAIL'}", flush=True)
    print(f"Sources: {_format_mapping(report.source_counts)}", flush=True)
    print(f"Turns: {_format_mapping(report.turn_counts)}", flush=True)
    print(f"Word quantiles: {_format_mapping(report.word_quantiles)}", flush=True)
    print(
        "Token quantiles: "
        + (
            _format_mapping(report.token_quantiles)
            if report.token_quantiles is not None
            else "unavailable (supply --checkpoint)"
        ),
        flush=True,
    )
    print(
        f"Response-length quantiles: {_format_mapping(report.response_length_quantiles)}",
        flush=True,
    )
    print(f"Top openings: {_format_top(report.repeated_openings)}", flush=True)
    print(f"Repeated n-grams: {_format_top(report.repeated_ngrams)}", flush=True)
    print(f"Coverage: {_format_mapping(report.coverage_categories)}", flush=True)
    print(f"Identity share: {report.identity_share:.1%}", flush=True)
    if report.curated_sampling_mass is not None:
        print(f"Curated sampling mass: {report.curated_sampling_mass:.1%}", flush=True)
    for finding in report.findings:
        print(f"{finding.severity.upper()} [{finding.code}]: {finding.message}", flush=True)
        for example in finding.examples:
            print(f"  - {example}", flush=True)


def _format_mapping(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "none"


def _format_top(values: dict[str, int]) -> str:
    return ", ".join(f"{key!r}={value}" for key, value in list(values.items())[:5]) or "none"


if __name__ == "__main__":
    raise SystemExit(main())
