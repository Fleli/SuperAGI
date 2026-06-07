from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from superagi.ingestion.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    BpeTokenizer,
    CharTokenizer,
    TokenizerLike,
)


DEFAULT_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class RawCorpus:
    text: str
    documents: tuple[str, ...]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TokenizedCorpus:
    token_ids: list[int]
    tokenizer: TokenizerLike
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
    return RawCorpus(text=text, documents=tuple(texts), source_paths=source_paths)


def tokenize_raw_corpus(
    raw_corpus: RawCorpus,
    *,
    tokenizer_type: str = "bpe",
    bpe_vocab_size: int = 8000,
    bpe_min_frequency: int = 2,
) -> TokenizedCorpus:
    if tokenizer_type == "bpe":
        text = format_pretraining_documents(raw_corpus.documents)
        tokenizer = BpeTokenizer.from_text(
            text,
            vocab_size=bpe_vocab_size,
            min_frequency=bpe_min_frequency,
        )
    elif tokenizer_type == "char":
        text = raw_corpus.text
        tokenizer = CharTokenizer.from_text(raw_corpus.text)
    else:
        raise ValueError(f"unknown tokenizer_type: {tokenizer_type!r}")

    token_ids = tokenizer.encode(text)
    return TokenizedCorpus(
        token_ids=token_ids,
        tokenizer=tokenizer,
        source_paths=raw_corpus.source_paths,
    )


def format_pretraining_documents(documents: Iterable[str]) -> str:
    formatted_documents = []
    for document in documents:
        normalized = document.strip()
        if normalized:
            formatted_documents.append(f"{BOS_TOKEN}{normalized}{EOS_TOKEN}")
    if not formatted_documents:
        raise ValueError("raw corpus is empty")
    return "\n".join(formatted_documents)


def ingest_raw_corpus(
    raw_dir: Path | str,
    processed_dir: Path | str,
    artifact_name: str = "train",
    validation_fraction: float = 0.0,
    validation_artifact_name: str = "val",
    tokenizer_type: str = "bpe",
    bpe_vocab_size: int = 8000,
    bpe_min_frequency: int = 2,
) -> TokenizedCorpus:
    raw_corpus = read_raw_corpus(raw_dir)
    tokenized = tokenize_raw_corpus(
        raw_corpus,
        tokenizer_type=tokenizer_type,
        bpe_vocab_size=bpe_vocab_size,
        bpe_min_frequency=bpe_min_frequency,
    )
    save_tokenized_corpus(
        tokenized,
        processed_dir,
        artifact_name,
        validation_fraction=validation_fraction,
        validation_artifact_name=validation_artifact_name,
    )
    return tokenized


def save_tokenized_corpus(
    tokenized: TokenizedCorpus,
    processed_dir: Path | str,
    artifact_name: str,
    validation_fraction: float = 0.0,
    validation_artifact_name: str = "val",
) -> None:
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0)")

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    train_token_ids, validation_token_ids = _split_train_validation(
        tokenized.token_ids,
        validation_fraction,
    )
    token_path = processed_path / f"{artifact_name}_tokens.pt"
    vocab_path = processed_path / f"{artifact_name}_vocab.json"
    torch.save(torch.tensor(train_token_ids, dtype=torch.long), token_path)
    if validation_token_ids:
        validation_token_path = processed_path / f"{validation_artifact_name}_tokens.pt"
        torch.save(
            torch.tensor(validation_token_ids, dtype=torch.long),
            validation_token_path,
        )

    vocab_payload = {
        **tokenized.tokenizer.to_payload(),
        "source_paths": [str(path) for path in tokenized.source_paths],
        "train_tokens": len(train_token_ids),
        "validation_tokens": len(validation_token_ids),
    }
    vocab_path.write_text(
        json.dumps(vocab_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _split_train_validation(
    token_ids: list[int],
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    if validation_fraction == 0 or len(token_ids) < 2:
        return list(token_ids), []

    validation_count = int(len(token_ids) * validation_fraction)
    validation_count = max(1, min(validation_count, len(token_ids) - 1))
    split_index = len(token_ids) - validation_count
    return list(token_ids[:split_index]), list(token_ids[split_index:])
