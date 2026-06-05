from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildResult:
    source: str
    output_dir: Path
    documents_written: int
    document_paths: tuple[Path, ...]
    metadata_path: Path


def slugify(value: str, max_length: int = 80) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "document"
    return value[:max_length].strip("-") or "document"


def normalize_document_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    normalized_lines = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip()


def write_raw_document(output_dir: Path | str, filename_stem: str, text: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    document_path = output_path / f"{slugify(filename_stem)}.txt"
    document_path.write_text(f"{normalize_document_text(text)}\n", encoding="utf-8")
    return document_path


def write_build_metadata(
    output_dir: Path | str,
    metadata: dict[str, Any],
    filename: str = "metadata.json",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / filename
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path
