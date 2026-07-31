from __future__ import annotations

import json
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from superagi.chat.formatting import ChatMessage
from superagi.chat.sft import TokenizedSftExample, tokenize_sft_messages
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
_JACCARD_ROUNDING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ImportFilterConfig:
    max_context_tokens: int = 900
    min_agi_chars: int = 20
    max_agi_chars: int = 1200
    max_agi_tokens: int = 512
    max_user_chars: int = 4000
    max_messages: int = 8
    max_repeated_five_grams: int = 3
    cross_example_ngram_size: int = 5
    max_cross_example_ngram_count: int = 3
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


@dataclass(frozen=True)
class _IndexedAnswer:
    token_set: frozenset[str]
    token_count: int


@dataclass(frozen=True)
class _PreparedConversation:
    source: str
    messages: tuple[ChatMessage, ...]
    pre_dedupe_rejection_reason: str | None
    normalized_answers: tuple[str, ...]


class _JaccardCandidateIndex:
    """Exact, threshold-preserving candidate retrieval for accepted answers.

    The corpus-wide rarest-first token order is frozen before insertion. For
    sets with Jaccard similarity at least ``threshold``, their threshold
    prefixes must overlap under any shared fixed order. Freezing this order for
    the complete import keeps the standard prefix-filter proof valid while
    pushing common boilerplate tokens out of prefixes. The index stores only
    accepted-answer prefixes, partitioned by token-set length.
    """

    def __init__(self, threshold: float, token_order: Mapping[str, int]) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        self.threshold = threshold
        self._token_order = token_order
        self._entries: list[_IndexedAnswer] = []
        self._postings: dict[str, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.comparison_count = 0
        self.retrieval_work = 0

    def has_near_duplicate(self, answer: str) -> bool:
        token_set = frozenset(answer.split())
        if not token_set:
            return False

        query_length = len(token_set)
        min_length, max_length = _jaccard_length_bounds(
            query_length,
            self.threshold,
        )
        candidate_ids: set[int] = set()
        for token in _jaccard_prefix(
            token_set,
            self.threshold,
            token_order=self._token_order,
        ):
            for length, posting in self._postings.get(token, {}).items():
                if min_length <= length <= max_length:
                    self.retrieval_work += len(posting)
                    candidate_ids.update(posting)

        for candidate_id in sorted(candidate_ids):
            candidate = self._entries[candidate_id]
            intersection_size = len(token_set & candidate.token_set)
            if intersection_size < _minimum_jaccard_intersection(
                query_length,
                candidate.token_count,
                self.threshold,
            ):
                continue
            self.comparison_count += 1
            union_size = query_length + candidate.token_count - intersection_size
            if intersection_size / union_size >= self.threshold:
                return True
        return False

    def add(self, answer: str) -> None:
        token_set = frozenset(answer.split())
        if not token_set:
            return
        entry_id = len(self._entries)
        token_count = len(token_set)
        self._entries.append(_IndexedAnswer(token_set, token_count))
        for token in _jaccard_prefix(
            token_set,
            self.threshold,
            token_order=self._token_order,
        ):
            self._postings[token][token_count].add(entry_id)


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
        *,
        ngram_reference_conversations: Iterable[
            Sequence[ChatMessage | Mapping[str, str]]
        ] = (),
    ) -> ImportResult:
        stats = ImportStats()
        examples: list[ImportedSftExample] = []
        seen_answers: set[str] = set()
        response_ngram_counts: Counter[str] = Counter()
        for raw_messages in ngram_reference_conversations:
            response_ngram_counts.update(
                _response_ngram_counts(
                    _coerce_messages(raw_messages),
                    self.filter_config.cross_example_ngram_size,
                )
            )
        prepared_conversations = self._prepare_conversations(conversations)
        near_duplicate_index = _JaccardCandidateIndex(
            self.filter_config.near_duplicate_threshold,
            _rarest_first_token_order(prepared_conversations),
        )
        self._near_duplicate_comparisons = 0
        self._near_duplicate_retrieval_work = 0

        for prepared in prepared_conversations:
            stats.seen += 1
            rejection_reason = prepared.pre_dedupe_rejection_reason
            if rejection_reason is not None:
                stats.reject(rejection_reason)
                continue

            if any(answer in seen_answers for answer in prepared.normalized_answers):
                stats.reject("duplicate_answer")
                continue
            if any(
                near_duplicate_index.has_near_duplicate(answer)
                for answer in prepared.normalized_answers
            ):
                stats.reject("near_duplicate")
                continue

            rejection_reason, tokenized = self._post_dedupe_filter(
                prepared.messages
            )
            if rejection_reason is not None:
                stats.reject(rejection_reason)
                continue
            assert tokenized is not None
            candidate_ngram_counts = _response_ngram_counts(
                prepared.messages,
                self.filter_config.cross_example_ngram_size,
            )
            if any(
                response_ngram_counts[ngram] + count
                > self.filter_config.max_cross_example_ngram_count
                for ngram, count in candidate_ngram_counts.items()
            ):
                stats.reject("cross_example_repetition")
                continue
            examples.append(
                ImportedSftExample(
                    source=prepared.source,
                    messages=prepared.messages,
                    token_count=len(tokenized.input_ids),
                    supervised_token_count=tokenized.supervised_token_count,
                )
            )
            for answer in prepared.normalized_answers:
                seen_answers.add(answer)
                near_duplicate_index.add(answer)
            response_ngram_counts.update(candidate_ngram_counts)
            stats.accepted += 1

        self._near_duplicate_comparisons = near_duplicate_index.comparison_count
        self._near_duplicate_retrieval_work = near_duplicate_index.retrieval_work
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

    def _prepare_conversations(
        self,
        conversations: Iterable[tuple[str, Sequence[ChatMessage | Mapping[str, str]]]],
    ) -> tuple[_PreparedConversation, ...]:
        prepared: list[_PreparedConversation] = []
        for source, raw_messages in conversations:
            messages = _coerce_messages(raw_messages)
            rejection_reason = self._pre_dedupe_rejection_reason(messages)
            prepared.append(
                _PreparedConversation(
                    source=source,
                    messages=messages,
                    pre_dedupe_rejection_reason=rejection_reason,
                    normalized_answers=(
                        _normalized_agi_answers(messages)
                        if rejection_reason is None
                        else ()
                    ),
                )
            )
        return tuple(prepared)

    def _pre_dedupe_rejection_reason(
        self,
        messages: tuple[ChatMessage, ...],
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
        return None

    def _post_dedupe_filter(
        self,
        messages: tuple[ChatMessage, ...],
    ) -> tuple[str | None, TokenizedSftExample | None]:
        for message in messages:
            if message.role != "agi":
                continue
            if (
                len(self.tokenizer.encode_with_offsets(message.content).ids)
                > self.filter_config.max_agi_tokens
            ):
                return "answer_token_limit", None
        try:
            tokenized = tokenize_sft_messages(messages, self.tokenizer)
        except ValueError:
            return "tokenization_error", None
        if len(tokenized.input_ids) > self.filter_config.max_context_tokens:
            return "too_long", None
        return None, tokenized


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


def _response_ngram_counts(
    messages: Sequence[ChatMessage],
    ngram_size: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for message in messages:
        if message.role != "agi":
            continue
        words = canonical_text(message.content).split()
        response_ngrams = {
            " ".join(words[index : index + ngram_size])
            for index in range(len(words) - ngram_size + 1)
        }
        counts.update(response_ngrams)
    return counts


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


def _jaccard_length_bounds(length: int, threshold: float) -> tuple[int, int]:
    # Expand bounds by a tiny tolerance: rounding may retrieve an extra
    # candidate, but cannot discard one that meets the exact threshold.
    return (
        math.ceil((threshold * length) - _JACCARD_ROUNDING_TOLERANCE),
        math.floor((length / threshold) + _JACCARD_ROUNDING_TOLERANCE),
    )


def _jaccard_prefix(
    token_set: frozenset[str],
    threshold: float,
    *,
    token_order: Mapping[str, int],
) -> tuple[str, ...]:
    fallback_rank = len(token_order)
    ordered_tokens = tuple(
        sorted(
            token_set,
            key=lambda token: (token_order.get(token, fallback_rank), token),
        )
    )
    prefix_length = (
        len(ordered_tokens)
        - math.ceil(
            (threshold * len(ordered_tokens)) - _JACCARD_ROUNDING_TOLERANCE
        )
        + 1
    )
    return ordered_tokens[:prefix_length]


def _rarest_first_token_order(
    conversations: Sequence[_PreparedConversation],
) -> dict[str, int]:
    document_frequency: Counter[str] = Counter()
    for conversation in conversations:
        if conversation.pre_dedupe_rejection_reason is not None:
            continue
        for answer in conversation.normalized_answers:
            document_frequency.update(frozenset(answer.split()))
    return {
        token: rank
        for rank, (token, _) in enumerate(
            sorted(document_frequency.items(), key=lambda item: (item[1], item[0]))
        )
    }


def _minimum_jaccard_intersection(
    left_length: int,
    right_length: int,
    threshold: float,
) -> int:
    return math.ceil(
        ((threshold * (left_length + right_length)) / (1.0 + threshold))
        - _JACCARD_ROUNDING_TOLERANCE
    )


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
