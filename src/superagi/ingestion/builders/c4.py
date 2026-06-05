from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from superagi.ingestion.builders.common import (
    BuildResult,
    normalize_document_text,
    write_build_metadata,
    write_raw_document,
)


def build_c4_corpus(
    raw_root: Path | str = Path("data/raw"),
    max_documents: int = 100,
    min_chars: int = 200,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
    examples: Iterable[dict[str, Any]] | None = None,
) -> BuildResult:
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")

    output_dir = Path(raw_root) / "c4"
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = examples if examples is not None else _load_c4_stream(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
    )

    document_paths = []
    documents_metadata = []
    for example in stream:
        text = normalize_document_text(str(example.get("text", "")))
        if len(text) < min_chars:
            continue

        document_number = len(document_paths) + 1
        document_path = write_raw_document(
            output_dir=output_dir,
            filename_stem=f"document-{document_number:06d}",
            text=text,
        )
        document_paths.append(document_path)
        documents_metadata.append(
            {
                "path": str(document_path),
                "url": example.get("url"),
                "timestamp": example.get("timestamp"),
                "chars": len(text),
            }
        )
        if len(document_paths) >= max_documents:
            break

    metadata_path = write_build_metadata(
        output_dir=output_dir,
        metadata={
            "source": "c4",
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "max_documents": max_documents,
            "min_chars": min_chars,
            "documents": documents_metadata,
        },
    )
    return BuildResult(
        source="c4",
        output_dir=output_dir,
        documents_written=len(document_paths),
        document_paths=tuple(document_paths),
        metadata_path=metadata_path,
    )


def _load_c4_stream(
    dataset_name: str,
    dataset_config: str,
    split: str,
):
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )
