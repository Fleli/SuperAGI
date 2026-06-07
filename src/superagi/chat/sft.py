from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from superagi.chat.formatting import ChatMessage, TextSpan, format_chat_messages
from superagi.ingestion.tokenizer import TokenizerLike


IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizedSftExample:
    text: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    supervised_token_count: int
    source: str = "manual"


@dataclass(frozen=True)
class SftConversation:
    messages: tuple[ChatMessage, ...]
    source: str


def tokenize_sft_messages(
    messages: Sequence[ChatMessage | Mapping[str, str]],
    tokenizer: TokenizerLike,
    *,
    source: str = "manual",
) -> TokenizedSftExample:
    formatted = format_chat_messages(messages)
    encoding = tokenizer.encode_with_offsets(formatted.text)
    if len(encoding.ids) < 2:
        raise ValueError("formatted SFT conversation must contain at least two tokens")

    labels = [
        token_id
        if _token_overlaps_any_span(offset, formatted.agi_spans)
        else IGNORE_INDEX
        for token_id, offset in zip(encoding.ids, encoding.offsets, strict=True)
    ]
    input_ids = encoding.ids[:-1]
    target_ids = tuple(labels[1:])
    supervised_token_count = sum(
        1 for token_id in target_ids if token_id != IGNORE_INDEX
    )
    if supervised_token_count == 0:
        raise ValueError("SFT conversation has no AGI response tokens")

    return TokenizedSftExample(
        text=formatted.text,
        input_ids=input_ids,
        target_ids=target_ids,
        supervised_token_count=supervised_token_count,
        source=source,
    )


def load_sft_jsonl(path: Path | str) -> list[tuple[ChatMessage, ...]]:
    return [record.messages for record in load_sft_records(path)]


def load_sft_records(
    path: Path | str,
    *,
    default_source: str = "manual",
) -> list[SftConversation]:
    sft_path = Path(path)
    conversations: list[SftConversation] = []
    for line_number, line in enumerate(
        sft_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"SFT line {line_number} must contain messages")
        source = payload.get("source", default_source)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"SFT line {line_number} source must be a non-empty string")
        conversations.append(
            SftConversation(
                messages=tuple(
                    _coerce_chat_message(message, line_number) for message in messages
                ),
                source=source.strip(),
            )
        )
    if not conversations:
        raise ValueError(f"no SFT examples found in {sft_path}")
    return conversations


def _coerce_chat_message(
    message: object,
    line_number: int,
) -> ChatMessage:
    if not isinstance(message, dict):
        raise ValueError(f"SFT line {line_number} messages must be objects")
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "agi"}:
        raise ValueError(f"SFT line {line_number} has unknown role: {role!r}")
    if not isinstance(content, str):
        raise ValueError(f"SFT line {line_number} content must be a string")
    return ChatMessage(role=role, content=content)


def _token_overlaps_any_span(
    offset: tuple[int, int],
    spans: Sequence[TextSpan],
) -> bool:
    start, end = offset
    if end <= start:
        return False
    return any(start < span.end and end > span.start for span in spans)
