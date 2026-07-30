from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence

from superagi.chat.formatting import ChatMessage


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def canonical_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(unicodedata.normalize("NFKC", value).lower()))


def validate_role_sequence(messages: Sequence[ChatMessage]) -> None:
    if not messages:
        raise ValueError("SFT conversation must contain messages")
    index = 1 if messages[0].role == "system" else 0
    if index == len(messages):
        raise ValueError("SFT conversation must contain a user/agi exchange")
    expected = "user"
    for message in messages[index:]:
        if message.role != expected:
            raise ValueError(f"SFT role order expected {expected!r}, got {message.role!r}")
        expected = "agi" if expected == "user" else "user"
    if messages[-1].role != "agi":
        raise ValueError("SFT conversation must end with an agi response")


def conversation_fingerprint(messages: Sequence[ChatMessage]) -> str:
    return _hash_canonical_pairs(
        (message.role, canonical_text(message.content)) for message in messages
    )


def conversation_group_key(messages: Sequence[ChatMessage]) -> str:
    return _hash_canonical_pairs(
        (message.role, canonical_text(message.content))
        for message in messages
        if message.role in {"user", "agi"}
    )


def _hash_canonical_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    serialized = json.dumps(
        list(pairs),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
