from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


ChatRole = Literal["user", "agi"]
ROLE_LABELS: dict[ChatRole, str] = {
    "user": "User",
    "agi": "AGI",
}


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class ChatFormat:
    text: str
    agi_spans: tuple[TextSpan, ...]


def format_user_prompt(content: str) -> ChatFormat:
    return format_chat_messages(
        [ChatMessage(role="user", content=content)],
        add_generation_prompt=True,
    )


def format_chat_messages(
    messages: Sequence[ChatMessage | Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> ChatFormat:
    if not messages:
        raise ValueError("at least one chat message is required")

    text_parts: list[str] = []
    agi_spans: list[TextSpan] = []
    current_offset = 0
    for raw_message in messages:
        message = _coerce_message(raw_message)
        label = ROLE_LABELS[message.role]
        prefix = f"{label}: "
        content = _normalize_content(message.content)
        line = f"{prefix}{content}\n"

        if message.role == "agi":
            span_start = current_offset + len(prefix)
            span_end = span_start + len(content)
            agi_spans.append(TextSpan(start=span_start, end=span_end))

        text_parts.append(line)
        current_offset += len(line)

    if add_generation_prompt:
        text_parts.append("AGI:")

    return ChatFormat(
        text="".join(text_parts),
        agi_spans=tuple(agi_spans),
    )


def _coerce_message(message: ChatMessage | Mapping[str, str]) -> ChatMessage:
    if isinstance(message, ChatMessage):
        role = message.role
        content = message.content
    else:
        role = message.get("role")
        content = message.get("content")

    if role not in ROLE_LABELS:
        raise ValueError(f"unknown chat role: {role!r}")
    if not isinstance(content, str):
        raise ValueError("chat message content must be a string")
    return ChatMessage(role=role, content=content)


def _normalize_content(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("chat message content must not be empty")
    return normalized
