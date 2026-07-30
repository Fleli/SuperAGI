from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset

from superagi.chat.sft_public_import import (
    ImportFilterConfig,
    ImportedSftExample,
    ImportResult,
    PublicSftImporter,
    convert_dolly_row,
    convert_no_robots_row,
    convert_ultrachat_row,
    convert_wildchat_row,
    iter_openassistant_conversations,
    seeded_source_sample,
)
from superagi.model.checkpoint import load_checkpoint


DEFAULT_SOURCES = ("no_robots", "dolly", "openassistant", "ultrachat")

SOURCE_DATASETS = {
    "no_robots": ("HuggingFaceH4/no_robots", "train"),
    "dolly": ("databricks/databricks-dolly-15k", "train"),
    "openassistant": ("OpenAssistant/oasst1", "train"),
    "wildchat": ("allenai/WildChat", "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public instruction/chat datasets and filter them into SuperAGI SFT JSONL.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="data/sft/imported/public-mixed.jsonl")
    parser.add_argument(
        "--metadata",
        default="data/sft/imported/public-mixed.metadata.json",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source names: no_robots,dolly,openassistant,wildchat,ultrachat",
    )
    parser.add_argument("--max-rows-per-source", type=int, default=50000)
    parser.add_argument("--max-examples-per-source", type=int, default=5000)
    parser.add_argument("--max-context-tokens", type=int, default=900)
    parser.add_argument("--max-messages", type=int, default=8)
    parser.add_argument("--max-agi-chars", type=int, default=1200)
    parser.add_argument("--min-agi-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    return run_import(parse_args())


def run_import(args: argparse.Namespace) -> int:
    sources = _parse_sources(args.sources)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    importer = PublicSftImporter(
        tokenizer=checkpoint.tokenizer,
        filter_config=ImportFilterConfig(
            max_context_tokens=args.max_context_tokens,
            min_agi_chars=args.min_agi_chars,
            max_agi_chars=args.max_agi_chars,
            max_messages=args.max_messages,
        ),
    )

    candidates: list[tuple[str, object]] = []
    candidate_counts: Counter[str] = Counter()
    for source in sources:
        print(f"Importing source: {source}", flush=True)
        source_candidates = list(
            _iter_source_candidates(
                source=source,
                max_rows=args.max_rows_per_source,
                max_messages=args.max_messages,
            )
        )
        candidates.extend(source_candidates)
        candidate_counts[source] = len(source_candidates)

    imported = importer.import_conversations(candidates)
    accepted_by_source: dict[str, list[ImportedSftExample]] = defaultdict(list)
    for example in imported.examples:
        accepted_by_source[_source_family(example.source)].append(example)

    all_examples = []
    accepted_before_limit_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for source in sources:
        accepted_examples = accepted_by_source[source]
        accepted_before_limit_counts[source] = len(accepted_examples)
        selected_examples = seeded_source_sample(
            accepted_examples,
            limit=args.max_examples_per_source,
            seed=args.seed,
            source=source,
        )
        selected_counts[source] = len(selected_examples)
        all_examples.extend(selected_examples)
        print(
            f"Selected {len(selected_examples)} / {len(accepted_examples)} accepted "
            f"from {candidate_counts[source]} candidates",
            flush=True,
        )

    result = ImportResult(examples=tuple(all_examples), stats=imported.stats)
    importer.write_import(
        result,
        out_path=Path(args.out),
        metadata_path=Path(args.metadata),
    )
    _append_source_metadata(
        Path(args.metadata),
        seed=args.seed,
        sources=sources,
        candidate_counts=candidate_counts,
        accepted_before_limit_counts=accepted_before_limit_counts,
        selected_counts=selected_counts,
        result=result,
    )
    print(f"Wrote {len(all_examples)} public SFT examples to {args.out}")
    print(f"Metadata: {args.metadata}")
    return 0


def _iter_source_candidates(
    *,
    source: str,
    max_rows: int,
    max_messages: int,
) -> Iterable[tuple[str, object]]:
    dataset_name, split = SOURCE_DATASETS[source]
    if source == "openassistant":
        rows = list(_take_rows(load_dataset(dataset_name, split=split, streaming=True), max_rows))
        for index, messages in enumerate(
            iter_openassistant_conversations(rows, max_messages=max_messages),
            start=1,
        ):
            yield f"{source}:{index}", messages
        return

    rows = _take_rows(load_dataset(dataset_name, split=split, streaming=True), max_rows)
    for index, row in enumerate(rows, start=1):
        if source == "no_robots":
            messages = convert_no_robots_row(row)
        elif source == "dolly":
            messages = convert_dolly_row(row)
        elif source == "wildchat":
            messages = convert_wildchat_row(row, max_turns=max_messages)
        elif source == "ultrachat":
            messages = convert_ultrachat_row(row)
        else:
            raise ValueError(f"unsupported source: {source}")
        yield f"{source}:{index}", messages


def _take_rows(dataset: Iterable[dict[str, Any]], max_rows: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0:
        raise ValueError("--max-rows-per-source must be positive")
    for index, row in enumerate(dataset):
        if index >= max_rows:
            break
        yield row


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(source.strip() for source in value.split(",") if source.strip())
    if not sources:
        raise ValueError("--sources must contain at least one source")
    unknown_sources = sorted(set(sources) - set(SOURCE_DATASETS))
    if unknown_sources:
        raise ValueError(f"unknown SFT sources: {', '.join(unknown_sources)}")
    return sources


def _append_source_metadata(
    metadata_path: Path,
    *,
    seed: int,
    sources: tuple[str, ...],
    candidate_counts: Mapping[str, int],
    accepted_before_limit_counts: Mapping[str, int],
    selected_counts: Mapping[str, int],
    result: ImportResult,
) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_summaries = {
        source: {
            "candidates": candidate_counts[source],
            "accepted_before_source_limit": accepted_before_limit_counts[source],
            "selected": selected_counts[source],
        }
        for source in sources
    }
    rejection_reasons = dict(sorted(result.stats.rejected_by_reason.items()))
    payload["seed"] = seed
    payload["sources"] = source_summaries
    payload["selected_source_counts"] = dict(sorted(selected_counts.items()))
    payload["accepted_before_limit_source_counts"] = dict(
        sorted(accepted_before_limit_counts.items())
    )
    payload["rejection_reasons"] = rejection_reasons
    payload["exact_duplicate_count"] = rejection_reasons.get("duplicate_answer", 0)
    payload["near_duplicate_count"] = rejection_reasons.get("near_duplicate", 0)
    payload["licenses_note"] = (
        "Review each upstream dataset license before using imported SFT data outside "
        "local learning experiments."
    )
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_family(source: str) -> str:
    return source.split(":", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
