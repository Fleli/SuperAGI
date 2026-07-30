from __future__ import annotations

import json
import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from superagi.chat.formatting import ChatMessage
from superagi.chat.sft import tokenize_sft_messages
from superagi.chat.sft_quality import canonical_text, validate_role_sequence
from superagi.ingestion.tokenizer import TokenizerLike


ROLE_MAP = {
    "user": "user",
    "prompter": "user",
    "assistant": "agi",
    "agi": "agi",
    "system": "system",
}

DISALLOWED_PATTERNS = (
    re.compile(r"\[[a-z_]+-\d+", re.IGNORECASE),
    re.compile(r"\bSTRIPTIONS\b", re.IGNORECASE),
    re.compile(r"\bemail-magic\b", re.IGNORECASE),
)

_LEAKED_CONTROL_TOKEN_RE = re.compile(
    r"<(?:pad|bos|eos|user|agi|system)>",
    re.IGNORECASE,
)
_FALSE_CAPABILITY_OR_IDENTITY_PATTERNS = (
    re.compile(
        r"\bi\s+(?:am|['\u2019]m)\s+(?:a\s+|an\s+)?(?:licensed|certified|registered|qualified)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi\s+live\s+in\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:have\s+)?browsed\s+(?:the\s+)?web\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+have\s+worked\s+(?:here|there|at\s+\S+)\s+for\s+\d+\s+years?\b",
        re.IGNORECASE,
    ),
)
_GENERIC_REFUSAL_RE = re.compile(
    r"\bas\s+an\s+ai(?:\s+language\s+model)?\b.*\b(?:cannot|can['\u2019]t|unable|not\s+able)\b",
    re.IGNORECASE,
)
_NON_HARMLESS_PROMPT_RE = re.compile(
    r"\b(?:kill|murder|hurt|harm|suicide|self-harm|bomb|weapon|explosive|malware|ransomware|phishing|fraud|steal)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImportFilterConfig:
    max_context_tokens: int = 900
    min_agi_chars: int = 20
    max_agi_chars: int = 1200
    max_user_chars: int = 4000
    max_messages: int = 8
    max_repeated_five_grams: int = 3
    near_duplicate_threshold: float = 0.88


@dataclass(frozen=True)
class ImportedSftExample:
    source: str
    messages: tuple[ChatMessage, ...]
    token_count: int
    supervised_token_count: int

    def to_json_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ],
        }


@dataclass
class ImportStats:
    seen: int = 0
    accepted: int = 0
    rejected_by_reason: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.rejected_by_reason[reason] += 1

    def to_json_payload(self) -> dict[str, object]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
        }


@dataclass(frozen=True)
class ImportResult:
    examples: tuple[ImportedSftExample, ...]
    stats: ImportStats


class PublicSftImporter:
    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        filter_config: ImportFilterConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.filter_config = filter_config or ImportFilterConfig()

    def import_conversations(
        self,
        conversations: Iterable[tuple[str, Sequence[ChatMessage | Mapping[str, str]]]],
    ) -> ImportResult:
        stats = ImportStats()
        examples: list[ImportedSftExample] = []
        seen_answers: set[str] = set()
        seen_answer_buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)

        for source, raw_messages in conversations:
            stats.seen += 1
            messages = _coerce_messages(raw_messages)
            rejection_reason = self._rejection_reason(
                messages,
                seen_answers,
                seen_answer_buckets,
            )
            if rejection_reason is not None:
                stats.reject(rejection_reason)
                continue

            tokenized = tokenize_sft_messages(messages, self.tokenizer)
            examples.append(
                ImportedSftExample(
                    source=source,
                    messages=messages,
                    token_count=len(tokenized.input_ids),
                    supervised_token_count=tokenized.supervised_token_count,
                )
            )
            for answer in _normalized_agi_answers(messages):
                seen_answers.add(answer)
                seen_answer_buckets[_answer_prefix_bucket(answer)].append(answer)
            stats.accepted += 1

        return ImportResult(examples=tuple(examples), stats=stats)

    def write_import(
        self,
        imported: ImportResult,
        *,
        out_path: Path,
        metadata_path: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for example in imported.examples:
                handle.write(
                    json.dumps(example.to_json_payload(), ensure_ascii=False) + "\n"
                )

        metadata = {
            "filter_config": asdict(self.filter_config),
            "stats": imported.stats.to_json_payload(),
            "token_counts": {
                "min": min((example.token_count for example in imported.examples), default=0),
                "max": max((example.token_count for example in imported.examples), default=0),
                "total": sum(example.token_count for example in imported.examples),
            },
            "supervised_token_counts": {
                "min": min(
                    (example.supervised_token_count for example in imported.examples),
                    default=0,
                ),
                "max": max(
                    (example.supervised_token_count for example in imported.examples),
                    default=0,
                ),
                "total": sum(
                    example.supervised_token_count for example in imported.examples
                ),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _rejection_reason(
        self,
        messages: tuple[ChatMessage, ...],
        seen_answers: set[str],
        seen_answer_buckets: Mapping[tuple[str, ...], Sequence[str]],
    ) -> str | None:
        config = self.filter_config
        if not messages:
            return "empty"
        try:
            validate_role_sequence(messages)
        except ValueError:
            return "role_sequence"
        if len(messages) > config.max_messages:
            return "too_many_messages"
        if any(not message.content.strip() for message in messages):
            return "empty_content"
        if any(
            message.role == "user" and len(message.content) > config.max_user_chars
            for message in messages
        ):
            return "user_too_long"
        agi_answers = [message.content for message in messages if message.role == "agi"]
        if any(len(answer) < config.min_agi_chars for answer in agi_answers):
            return "answer_too_short"
        if any(len(answer) > config.max_agi_chars for answer in agi_answers):
            return "answer_too_long"
        joined_text = "\n".join(message.content for message in messages)
        if _contains_artifact(joined_text):
            return "artifact"
        if any(_claims_false_capability_or_identity(answer) for answer in agi_answers):
            return "false_capability_or_identity"
        if _is_generic_refusal_for_harmless_prompt(messages):
            return "generic_refusal"
        if any(_has_repeated_five_grams(answer, config) for answer in agi_answers):
            return "repeated_phrase"
        normalized_answers = _normalized_agi_answers(messages)
        if any(answer in seen_answers for answer in normalized_answers):
            return "duplicate_answer"
        if any(
            _is_near_duplicate(
                answer,
                seen_answer_buckets.get(_answer_prefix_bucket(answer), ()),
                threshold=config.near_duplicate_threshold,
            )
            for answer in normalized_answers
        ):
            return "near_duplicate"

        try:
            tokenized = tokenize_sft_messages(messages, self.tokenizer)
        except ValueError:
            return "tokenization_error"
        if len(tokenized.input_ids) > config.max_context_tokens:
            return "too_long"
        return None


def convert_no_robots_row(row: Mapping[str, Any]) -> tuple[ChatMessage, ...]:
    return _convert_messages(row.get("messages", ()))


def convert_dolly_row(row: Mapping[str, Any]) -> tuple[ChatMessage, ...]:
    instruction = _clean_text(row.get("instruction"))
    response = _clean_text(row.get("response"))
    context = _clean_text(row.get("context"))
    if not instruction or not response:
        return ()
    user_content = (
        f"Context:\n{context}\n\nInstruction:\n{instruction}"
        if context
        else instruction
    )
    return (
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="agi", content=response),
    )


def convert_ultrachat_row(row: Mapping[str, Any]) -> tuple[ChatMessage, ...]:
    return _convert_messages(row.get("messages", ()))


def convert_wildchat_row(
    row: Mapping[str, Any],
    *,
    max_turns: int = 8,
) -> tuple[ChatMessage, ...]:
    if row.get("toxic") is True or row.get("redacted") is True:
        return ()
    language = str(row.get("language", "")).lower()
    if language and language not in {"english", "en"}:
        return ()

    raw_messages = row.get("conversation", ())
    if not isinstance(raw_messages, list):
        return ()
    for message in raw_messages:
        if not isinstance(message, Mapping):
            return ()
        if message.get("toxic") is True or message.get("redacted") is True:
            return ()
        message_language = str(message.get("language", "")).lower()
        if message_language and message_language not in {"english", "en"}:
            return ()
    return _convert_messages(raw_messages[:max_turns])


def iter_openassistant_conversations(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_messages: int = 8,
) -> Iterable[tuple[ChatMessage, ...]]:
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    assistant_ids: list[str] = []
    for row in rows:
        message_id = row.get("message_id")
        if not isinstance(message_id, str):
            continue
        rows_by_id[message_id] = row
        if row.get("role") == "assistant":
            assistant_ids.append(message_id)

    for assistant_id in assistant_ids:
        path = _openassistant_path(assistant_id, rows_by_id)
        if path is None or len(path) > max_messages:
            continue
        messages = _convert_messages(
            [
                {"role": row.get("role"), "content": row.get("text")}
                for row in path
                if _openassistant_row_is_usable(row)
            ]
        )
        if messages and messages[-1].role == "agi":
            yield messages


def _openassistant_path(
    assistant_id: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...] | None:
    path: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    current_id: str | None = assistant_id
    while current_id is not None:
        if current_id in seen:
            return None
        seen.add(current_id)
        row = rows_by_id.get(current_id)
        if row is None or not _openassistant_row_is_usable(row):
            return None
        path.append(row)
        parent_id = row.get("parent_id")
        current_id = parent_id if isinstance(parent_id, str) else None
    path.reverse()
    if not path or path[0].get("role") != "prompter":
        return None
    return tuple(path)


def _openassistant_row_is_usable(row: Mapping[str, Any]) -> bool:
    return (
        row.get("lang") == "en"
        and row.get("deleted") is not True
        and row.get("review_result") is True
        and row.get("role") in {"prompter", "assistant"}
    )


def _convert_messages(raw_messages: object) -> tuple[ChatMessage, ...]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str):
        return ()
    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            return ()
        raw_role = raw_message.get("role")
        role = ROLE_MAP.get(str(raw_role))
        content = _clean_text(raw_message.get("content"))
        if role is None or not content:
            return ()
        messages.append(ChatMessage(role=role, content=content))
    return tuple(messages)


def _coerce_messages(
    raw_messages: Sequence[ChatMessage | Mapping[str, str]],
) -> tuple[ChatMessage, ...]:
    messages: list[ChatMessage] = []
    for message in raw_messages:
        if isinstance(message, ChatMessage):
            messages.append(
                ChatMessage(role=message.role, content=_normalize_whitespace(message.content))
            )
        else:
            role = ROLE_MAP.get(str(message.get("role")))
            content = _clean_text(message.get("content"))
            if role is None or not content:
                return ()
            messages.append(ChatMessage(role=role, content=content))
    return tuple(messages)


def _clean_text(value: object) -> str:
    return _normalize_whitespace(value if isinstance(value, str) else "")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalized_agi_answers(messages: Sequence[ChatMessage]) -> tuple[str, ...]:
    return tuple(
        canonical_text(message.content)
        for message in messages
        if message.role == "agi"
    )


def seeded_source_sample(
    examples: Sequence[ImportedSftExample],
    *,
    limit: int,
    seed: int,
    source: str,
) -> tuple[ImportedSftExample, ...]:
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    if limit >= len(examples):
        return tuple(examples)
    source_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()[:8],
        "big",
    )
    indices = random.Random(source_seed).sample(range(len(examples)), limit)
    return tuple(examples[index] for index in sorted(indices))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(canonical_text(left).split())
    right_tokens = set(canonical_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _answer_prefix_bucket(answer: str) -> tuple[str, ...]:
    return tuple(answer.split()[:4])


def _is_near_duplicate(
    answer: str,
    accepted_answers: Sequence[str],
    *,
    threshold: float,
) -> bool:
    return any(token_jaccard(answer, accepted) >= threshold for accepted in accepted_answers)


def _contains_artifact(value: str) -> bool:
    return (
        "\ufffd" in value
        or bool(_LEAKED_CONTROL_TOKEN_RE.search(value))
        or any(pattern.search(value) for pattern in DISALLOWED_PATTERNS)
    )


def _claims_false_capability_or_identity(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in _FALSE_CAPABILITY_OR_IDENTITY_PATTERNS)


def _is_generic_refusal_for_harmless_prompt(messages: Sequence[ChatMessage]) -> bool:
    user_text = "\n".join(message.content for message in messages if message.role == "user")
    if _NON_HARMLESS_PROMPT_RE.search(user_text):
        return False
    return any(
        _GENERIC_REFUSAL_RE.search(message.content)
        for message in messages
        if message.role == "agi"
    )


def _has_repeated_five_grams(
    text: str,
    config: ImportFilterConfig,
) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 5:
        return False
    counts = defaultdict(int)
    for index in range(len(words) - 4):
        counts[tuple(words[index : index + 5])] += 1
    return any(count > config.max_repeated_five_grams for count in counts.values())
