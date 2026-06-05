from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from superagi.ingestion.tokenizer import CharTokenizer


DEFAULT_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class RawCorpus:
    text: str
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TokenizedCorpus:
    token_ids: list[int]
    tokenizer: CharTokenizer
    source_paths: tuple[Path, ...]


def read_raw_corpus(
    raw_dir: Path | str,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    separator: str = "\n",
) -> RawCorpus:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw corpus directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"raw corpus path is not a directory: {raw_path}")

    allowed_suffixes = tuple(suffix.lower() for suffix in suffixes)
    source_paths = tuple(
        path
        for path in sorted(raw_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    if not source_paths:
        raise ValueError(f"no text files found in {raw_path}")

    texts = [path.read_text(encoding="utf-8") for path in source_paths]
    text = separator.join(texts)
    if not text:
        raise ValueError("raw corpus is empty")
    return RawCorpus(text=text, source_paths=source_paths)


def tokenize_raw_corpus(raw_corpus: RawCorpus) -> TokenizedCorpus:
    tokenizer = CharTokenizer.from_text(raw_corpus.text)
    token_ids = tokenizer.encode(raw_corpus.text)
    return TokenizedCorpus(
        token_ids=token_ids,
        tokenizer=tokenizer,
        source_paths=raw_corpus.source_paths,
    )


def ingest_raw_corpus(
    raw_dir: Path | str,
    processed_dir: Path | str,
    artifact_name: str = "train",
) -> TokenizedCorpus:
    raw_corpus = read_raw_corpus(raw_dir)
    tokenized = tokenize_raw_corpus(raw_corpus)
    save_tokenized_corpus(tokenized, processed_dir, artifact_name)
    return tokenized


def save_tokenized_corpus(
    tokenized: TokenizedCorpus,
    processed_dir: Path | str,
    artifact_name: str,
) -> None:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    token_path = processed_path / f"{artifact_name}_tokens.pt"
    vocab_path = processed_path / f"{artifact_name}_vocab.json"
    torch.save(torch.tensor(tokenized.token_ids, dtype=torch.long), token_path)
    vocab_payload = {
        "id_to_char": list(tokenized.tokenizer.id_to_char),
        "source_paths": [str(path) for path in tokenized.source_paths],
    }
    vocab_path.write_text(
        json.dumps(vocab_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
