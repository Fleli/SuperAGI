from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from superagi.ingestion.tokenizer import BOS_TOKEN, EOS_TOKEN, BpeTokenizer


@dataclass(frozen=True)
class TokenShardBuildResult:
    tokenizer: BpeTokenizer
    processed_dir: Path
    vocab_path: Path
    manifest_path: Path
    validation_token_path: Path
    train_shard_paths: tuple[Path, ...]
    train_tokens: int
    validation_tokens: int
    documents_tokenized: int
    target_train_tokens: int | None = None


def normalize_stream_document_text(text: str) -> str:
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


def build_c4_token_shards(
    processed_dir: Path | str = Path("data/processed"),
    *,
    max_documents: int = 1000,
    tokenizer_sample_documents: int = 500,
    shard_token_count: int = 1_000_000,
    validation_token_count: int = 100_000,
    target_train_tokens: int | None = None,
    min_chars: int = 500,
    bpe_vocab_size: int = 8000,
    bpe_min_frequency: int = 2,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
) -> TokenShardBuildResult:
    def examples_factory() -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        return load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
        )

    return build_token_shards_from_stream(
        examples_factory=examples_factory,
        processed_dir=processed_dir,
        max_documents=max_documents,
        tokenizer_sample_documents=tokenizer_sample_documents,
        shard_token_count=shard_token_count,
        validation_token_count=validation_token_count,
        target_train_tokens=target_train_tokens,
        min_chars=min_chars,
        bpe_vocab_size=bpe_vocab_size,
        bpe_min_frequency=bpe_min_frequency,
        metadata={
            "source": "c4",
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
        },
    )


def build_token_shards_from_stream(
    *,
    examples_factory: Callable[[], Iterable[dict[str, Any]]],
    processed_dir: Path | str,
    max_documents: int,
    tokenizer_sample_documents: int,
    shard_token_count: int,
    validation_token_count: int,
    target_train_tokens: int | None = None,
    min_chars: int = 0,
    bpe_vocab_size: int = 8000,
    bpe_min_frequency: int = 2,
    metadata: dict[str, Any] | None = None,
    shard_published_callback: Callable[[Path], None] | None = None,
) -> TokenShardBuildResult:
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if tokenizer_sample_documents <= 0:
        raise ValueError("tokenizer_sample_documents must be positive")
    if shard_token_count <= 0:
        raise ValueError("shard_token_count must be positive")
    if validation_token_count < 0:
        raise ValueError("validation_token_count must be non-negative")
    if target_train_tokens is not None and target_train_tokens <= 0:
        raise ValueError("target_train_tokens must be positive when set")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    shard_dir = processed_path / "train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    validation_token_path = processed_path / "val_tokens.pt"

    tokenizer = BpeTokenizer.from_texts(
        _iter_normalized_texts(
            examples_factory(),
            max_documents=tokenizer_sample_documents,
            min_chars=min_chars,
        ),
        vocab_size=bpe_vocab_size,
        min_frequency=bpe_min_frequency,
    )

    validation_buffer: list[int] = []
    train_buffer: list[int] = []
    shard_paths: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    train_tokens = 0
    documents_tokenized = 0
    separator_ids = tokenizer.encode("\n")
    validation_tokens_published = validation_token_count == 0
    vocab_path = processed_path / "train_vocab.json"
    manifest_path = shard_dir / "manifest.json"
    _write_vocab_payload(
        vocab_path,
        tokenizer=tokenizer,
        train_tokens=0,
        validation_tokens=0,
        target_train_tokens=target_train_tokens,
        shard_dir=shard_dir,
        documents_tokenized=0,
        metadata=metadata,
    )

    target_reached = False
    for text in _iter_normalized_texts(
        examples_factory(),
        max_documents=max_documents,
        min_chars=min_chars,
    ):
        document_ids = (
            tokenizer.encode(f"{BOS_TOKEN}{text}{EOS_TOKEN}") + separator_ids
        )
        documents_tokenized += 1

        if len(validation_buffer) < validation_token_count:
            needed = validation_token_count - len(validation_buffer)
            validation_buffer.extend(document_ids[:needed])
            document_ids = document_ids[needed:]
            if len(validation_buffer) >= validation_token_count:
                _write_token_tensor_atomic(validation_token_path, validation_buffer)
                validation_tokens_published = True

        train_buffer.extend(document_ids)
        while len(train_buffer) >= shard_token_count:
            shard_ids = train_buffer[:shard_token_count]
            del train_buffer[:shard_token_count]
            shard_path = _write_train_shard(shard_dir, len(shard_paths) + 1, shard_ids)
            shard_paths.append(shard_path)
            shard_records.append(
                {
                    "path": shard_path.name,
                    "tokens": len(shard_ids),
                }
            )
            train_tokens += len(shard_ids)
            _write_manifest_payload(
                manifest_path,
                train_tokens=train_tokens,
                validation_tokens=len(validation_buffer),
                target_train_tokens=target_train_tokens,
                shard_token_count=shard_token_count,
                documents_tokenized=documents_tokenized,
                shard_records=shard_records,
            )
            if shard_published_callback is not None:
                shard_published_callback(manifest_path)
            if target_train_tokens is not None and train_tokens >= target_train_tokens:
                target_reached = True
                train_buffer.clear()
                break

        if target_reached:
            break

    if train_buffer:
        shard_path = _write_train_shard(shard_dir, len(shard_paths) + 1, train_buffer)
        shard_paths.append(shard_path)
        shard_records.append(
            {
                "path": shard_path.name,
                "tokens": len(train_buffer),
            }
        )
        train_tokens += len(train_buffer)
        _write_manifest_payload(
            manifest_path,
            train_tokens=train_tokens,
            validation_tokens=len(validation_buffer),
            target_train_tokens=target_train_tokens,
            shard_token_count=shard_token_count,
            documents_tokenized=documents_tokenized,
            shard_records=shard_records,
        )
        if shard_published_callback is not None:
            shard_published_callback(manifest_path)

    if not shard_paths:
        raise ValueError("stream did not produce any training tokens")

    if not validation_tokens_published:
        _write_token_tensor_atomic(validation_token_path, validation_buffer)

    _write_vocab_payload(
        vocab_path,
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=len(validation_buffer),
        target_train_tokens=target_train_tokens,
        shard_dir=shard_dir,
        documents_tokenized=documents_tokenized,
        metadata=metadata,
    )
    _write_manifest_payload(
        manifest_path,
        train_tokens=train_tokens,
        validation_tokens=len(validation_buffer),
        target_train_tokens=target_train_tokens,
        shard_token_count=shard_token_count,
        documents_tokenized=documents_tokenized,
        shard_records=shard_records,
    )

    return TokenShardBuildResult(
        tokenizer=tokenizer,
        processed_dir=processed_path,
        vocab_path=vocab_path,
        manifest_path=manifest_path,
        validation_token_path=validation_token_path,
        train_shard_paths=tuple(shard_paths),
        train_tokens=train_tokens,
        validation_tokens=len(validation_buffer),
        documents_tokenized=documents_tokenized,
        target_train_tokens=target_train_tokens,
    )


def _iter_normalized_texts(
    examples: Iterable[dict[str, Any]],
    *,
    max_documents: int,
    min_chars: int,
) -> Iterable[str]:
    yielded = 0
    for example in examples:
        text = normalize_stream_document_text(str(example.get("text", "")))
        if len(text) < min_chars:
            continue
        yield text
        yielded += 1
        if yielded >= max_documents:
            break


def _write_train_shard(
    shard_dir: Path,
    shard_number: int,
    token_ids: list[int],
) -> Path:
    shard_path = shard_dir / f"train-{shard_number:06d}.pt"
    _write_token_tensor_atomic(shard_path, token_ids)
    return shard_path


def _write_token_tensor_atomic(path: Path, token_ids: list[int]) -> None:
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save(torch.tensor(token_ids, dtype=torch.long), temporary_path)
    temporary_path.replace(path)


def _write_vocab_payload(
    vocab_path: Path,
    *,
    tokenizer: BpeTokenizer,
    train_tokens: int,
    validation_tokens: int,
    target_train_tokens: int | None,
    shard_dir: Path,
    documents_tokenized: int,
    metadata: dict[str, Any] | None,
) -> None:
    vocab_payload = {
        **tokenizer.to_payload(),
        "source": "stream",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "target_train_tokens": target_train_tokens,
        "train_shard_manifest": str(shard_dir / "manifest.json"),
        "documents_tokenized": documents_tokenized,
    }
    if metadata:
        vocab_payload.update(metadata)
    _write_json_atomic(vocab_path, vocab_payload)


def _write_manifest_payload(
    manifest_path: Path,
    *,
    train_tokens: int,
    validation_tokens: int,
    target_train_tokens: int | None,
    shard_token_count: int,
    documents_tokenized: int,
    shard_records: list[dict[str, Any]],
) -> None:
    manifest_payload = {
        "format": "superagi-token-shards-v1",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "target_train_tokens": target_train_tokens,
        "shard_token_count": shard_token_count,
        "documents_tokenized": documents_tokenized,
        "shards": list(shard_records),
    }
    _write_json_atomic(manifest_path, manifest_payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
