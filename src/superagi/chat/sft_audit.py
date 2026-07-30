from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from superagi.chat.formatting import ChatMessage
from superagi.chat.sft import SftConversation, load_sft_records, tokenize_sft_messages
from superagi.chat.sft_public_import import token_jaccard
from superagi.chat.sft_quality import (
    canonical_text,
    conversation_fingerprint,
    validate_role_sequence,
)
from superagi.ingestion.tokenizer import (
    AGI_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    TokenizerLike,
)


AuditMode = Literal["curated", "mixed", "style"]
AuditSeverity = Literal["error", "warning"]

_LEAKED_CONTROL_TOKEN_RE = re.compile(r"<(?:user|agi|system|bos)>", re.IGNORECASE)
_SYNTHETIC_TAG_RE = re.compile(r"\[[a-z_-]+-\d+", re.IGNORECASE)
_IDENTITY_OR_LIMITATION_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:am|['\u2019]m)\s+(?:an?\s+)?(?:ai|language\s+model|superagi|"
    r"small\s+experimental\s+model)|"
    r"as\s+an\s+ai|"
    r"i\s+(?:cannot|can['\u2019]t|do\s+not|don't)\s+(?:access|browse|know|provide)|"
    r"(?:limited\s+training|limited\s+resources|limited\s+capabilities)"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditConfig:
    near_duplicate_threshold: float = 0.88
    max_duplicate_conversations: int = 0
    max_duplicate_agi_answers: int = 0
    max_near_duplicate_answers: int = 0
    opening_token_count: int = 3
    opening_frequency_limit: float = 0.08
    opening_frequency_min_responses: int = 100
    ngram_size: int = 5
    max_repeated_ngram_count: int = 3
    identity_share_limit: float = 0.03
    curated_sampling_mass_min: float = 0.15
    curated_sampling_mass_max: float = 0.25
    max_context_tokens: int | None = None
    curated_source_families: tuple[str, ...] = ("curated_core", "curated")
    example_limit: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if min(
            self.max_duplicate_conversations,
            self.max_duplicate_agi_answers,
            self.max_near_duplicate_answers,
        ) < 0:
            raise ValueError("duplicate limits must be non-negative")
        if self.opening_token_count <= 0:
            raise ValueError("opening_token_count must be positive")
        if not 0.0 <= self.opening_frequency_limit <= 1.0:
            raise ValueError("opening_frequency_limit must be in [0, 1]")
        if self.opening_frequency_min_responses < 0:
            raise ValueError("opening_frequency_min_responses must be non-negative")
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if self.max_repeated_ngram_count < 0:
            raise ValueError("max_repeated_ngram_count must be non-negative")
        if not 0.0 <= self.identity_share_limit <= 1.0:
            raise ValueError("identity_share_limit must be in [0, 1]")
        if not 0.0 <= self.curated_sampling_mass_min <= self.curated_sampling_mass_max <= 1.0:
            raise ValueError("curated sampling mass limits must be in [0, 1]")
        if self.example_limit <= 0:
            raise ValueError("example_limit must be positive")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive when supplied")


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: AuditSeverity
    message: str
    examples: tuple[str, ...] = ()

    def to_json_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class AuditReport:
    mode: AuditMode
    config: AuditConfig
    conversation_count: int
    response_count: int
    source_counts: dict[str, int]
    turn_counts: dict[str, int]
    token_quantiles: dict[str, int]
    response_length_quantiles: dict[str, int]
    repeated_openings: dict[str, int]
    repeated_ngrams: dict[str, int]
    identity_share: float
    curated_sampling_mass: float | None
    findings: tuple[AuditFinding, ...]

    @property
    def ok(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def has_error(self, code: str) -> bool:
        return any(
            finding.code == code and finding.severity == "error"
            for finding in self.findings
        )

    def to_json_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "config": asdict(self.config),
            "conversation_count": self.conversation_count,
            "response_count": self.response_count,
            "source_counts": self.source_counts,
            "turn_counts": self.turn_counts,
            "token_quantiles": self.token_quantiles,
            "response_length_quantiles": self.response_length_quantiles,
            "repeated_openings": self.repeated_openings,
            "repeated_ngrams": self.repeated_ngrams,
            "identity_share": self.identity_share,
            "curated_sampling_mass": self.curated_sampling_mass,
            "findings": [finding.to_json_payload() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def audit_sft_corpus(
    paths: Sequence[Path | str],
    *,
    mode: AuditMode,
    tokenizer: TokenizerLike | object | None = None,
    context_length: int | None = None,
    source_weights: Mapping[str, float] | None = None,
    config: AuditConfig | None = None,
) -> AuditReport:
    if mode not in {"curated", "mixed", "style"}:
        raise ValueError("mode must be one of curated, mixed, or style")
    if context_length is not None and context_length <= 0:
        raise ValueError("context_length must be positive when supplied")

    audit_config = config or AuditConfig()
    findings: list[AuditFinding] = []
    records = _load_records(paths, findings, audit_config)
    if tokenizer is not None:
        _validate_special_token_ids(tokenizer, findings)

    source_counts = Counter(_source_family(record.source) for record in records)
    turn_counts = Counter(str(sum(message.role == "agi" for message in record.messages)) for record in records)
    responses = [
        (record.source, message.content)
        for record in records
        for message in record.messages
        if message.role == "agi"
    ]
    _append_duplicate_findings(records, responses, mode, audit_config, findings)
    _append_content_findings(records, audit_config, findings)

    effective_context_length = _effective_context_length(
        context_length,
        audit_config.max_context_tokens,
    )
    token_counts = _token_counts(
        records,
        tokenizer,
        effective_context_length,
        findings,
        audit_config,
    )
    response_lengths = [len(canonical_text(answer).split()) for _, answer in responses]
    repeated_openings = _repeated_openings(responses, audit_config)
    repeated_ngrams = _repeated_ngrams(responses, audit_config)
    _append_repetition_findings(
        responses,
        repeated_openings,
        repeated_ngrams,
        audit_config,
        findings,
    )

    identity_share = _identity_share(responses)
    if mode == "curated" and identity_share > audit_config.identity_share_limit:
        findings.append(
            AuditFinding(
                code="identity_share",
                severity="error",
                message=(
                    "identity/capability/limitation responses account for "
                    f"{identity_share:.1%}, above the {audit_config.identity_share_limit:.1%} curated limit"
                ),
                examples=_examples_matching(responses, _IDENTITY_OR_LIMITATION_RE, audit_config),
            )
        )

    curated_sampling_mass = None
    if mode == "mixed":
        curated_sampling_mass = _curated_sampling_mass(
            records,
            source_weights or {},
            audit_config,
        )
        if not (
            audit_config.curated_sampling_mass_min
            <= curated_sampling_mass
            <= audit_config.curated_sampling_mass_max
        ):
            findings.append(
                AuditFinding(
                    code="curated_sampling_mass",
                    severity="error",
                    message=(
                        "effective curated source sampling mass is "
                        f"{curated_sampling_mass:.1%}; expected "
                        f"{audit_config.curated_sampling_mass_min:.0%}-"
                        f"{audit_config.curated_sampling_mass_max:.0%}"
                    ),
                )
            )

    return AuditReport(
        mode=mode,
        config=audit_config,
        conversation_count=len(records),
        response_count=len(responses),
        source_counts=dict(sorted(source_counts.items())),
        turn_counts=dict(sorted(turn_counts.items(), key=lambda item: int(item[0]))),
        token_quantiles=_nearest_rank_quantiles(token_counts),
        response_length_quantiles=_nearest_rank_quantiles(response_lengths),
        repeated_openings=repeated_openings,
        repeated_ngrams=repeated_ngrams,
        identity_share=identity_share,
        curated_sampling_mass=curated_sampling_mass,
        findings=tuple(findings),
    )


def _load_records(
    paths: Sequence[Path | str],
    findings: list[AuditFinding],
    config: AuditConfig,
) -> list[SftConversation]:
    records: list[SftConversation] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            records.extend(load_sft_records(path, default_source=path.stem))
            continue
        except (OSError, ValueError, json.JSONDecodeError) as error:
            findings.append(
                AuditFinding(
                    code=_loading_error_code(error),
                    severity="error",
                    message=f"failed to load {path}: {error}",
                )
            )
        records.extend(_recover_valid_records(path, findings, config))
    return records


def _recover_valid_records(
    path: Path,
    findings: list[AuditFinding],
    config: AuditConfig,
) -> list[SftConversation]:
    recovered: list[SftConversation] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return recovered
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            messages_payload = payload["messages"]
            if not isinstance(messages_payload, list):
                raise ValueError("messages must be a list")
            source = payload.get("source", path.stem)
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source must be a non-empty string")
            messages = tuple(
                ChatMessage(role=item["role"], content=item["content"])
                for item in messages_payload
                if isinstance(item, dict)
            )
            if len(messages) != len(messages_payload):
                raise ValueError("messages must be objects")
            if any(
                message.role not in {"system", "user", "agi"}
                or not isinstance(message.content, str)
                for message in messages
            ):
                raise ValueError("message role/content is invalid")
            validate_role_sequence(messages)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(
                AuditFinding(
                    code=_loading_error_code(error),
                    severity="error",
                    message=f"invalid SFT record {path}:{line_number}: {error}",
                )
            )
            continue
        recovered.append(SftConversation(messages=messages, source=source.strip()))
    return recovered


def _loading_error_code(error: BaseException) -> str:
    message = str(error)
    if "role" in message or "end with an agi" in message or "user/agi" in message:
        return "invalid_role_sequence"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    return "invalid_sft_record"


def _validate_special_token_ids(tokenizer: object, findings: list[AuditFinding]) -> None:
    special_token_id = getattr(tokenizer, "special_token_id", None)
    if not callable(special_token_id):
        findings.append(
            AuditFinding(
                code="missing_special_token_ids",
                severity="error",
                message="SFT audit tokenizer must resolve named special token IDs",
            )
        )
        return
    try:
        for token in (PAD_TOKEN, BOS_TOKEN, USER_TOKEN, AGI_TOKEN, SYSTEM_TOKEN):
            special_token_id(token)
    except (KeyError, ValueError) as error:
        findings.append(
            AuditFinding(
                code="missing_special_token_ids",
                severity="error",
                message=f"SFT audit tokenizer lacks a required special token: {error}",
            )
        )


def _append_duplicate_findings(
    records: Sequence[SftConversation],
    responses: Sequence[tuple[str, str]],
    mode: AuditMode,
    config: AuditConfig,
    findings: list[AuditFinding],
) -> None:
    fingerprints: dict[str, list[str]] = defaultdict(list)
    for record in records:
        fingerprints[conversation_fingerprint(record.messages)].append(record.source)
    duplicate_conversations = [
        f"{sources[0]} (repeated {len(sources)} times)"
        for sources in fingerprints.values()
        if len(sources) > 1
    ]
    if len(duplicate_conversations) > config.max_duplicate_conversations:
        findings.append(
            AuditFinding(
                code="duplicate_conversation",
                severity="error",
                message=(
                    f"found {len(duplicate_conversations)} exact duplicate conversations; "
                    f"limit is {config.max_duplicate_conversations}"
                ),
                examples=tuple(duplicate_conversations[: config.example_limit]),
            )
        )

    canonical_answers: dict[str, list[str]] = defaultdict(list)
    for source, answer in responses:
        canonical_answers[canonical_text(answer)].append(source)
    duplicate_answers = [
        f"{sources[0]} (repeated {len(sources)} times)"
        for sources in canonical_answers.values()
        if len(sources) > 1
    ]
    if mode == "curated" and len(duplicate_answers) > config.max_duplicate_agi_answers:
        findings.append(
            AuditFinding(
                code="duplicate_agi_answer",
                severity="error",
                message=(
                    f"found {len(duplicate_answers)} duplicate canonical AGI answers; "
                    f"limit is {config.max_duplicate_agi_answers}"
                ),
                examples=tuple(duplicate_answers[: config.example_limit]),
            )
        )

    near_duplicates = _near_duplicate_pairs(responses, config.near_duplicate_threshold)
    if len(near_duplicates) > config.max_near_duplicate_answers:
        findings.append(
            AuditFinding(
                code="near_duplicate_agi_answer",
                severity="error",
                message=(
                    f"found {len(near_duplicates)} AGI answer pairs with canonical "
                    f"Jaccard similarity >= {config.near_duplicate_threshold:.2f}; "
                    f"limit is {config.max_near_duplicate_answers}"
                ),
                examples=tuple(near_duplicates[: config.example_limit]),
            )
        )

    prompt_answer_pairs = [
        (
            record.source,
            " ".join(
                f"{message.role} {canonical_text(message.content)}"
                for message in record.messages
                if message.role in {"user", "agi"}
            ),
        )
        for record in records
    ]
    near_duplicate_pairs = _near_duplicate_pairs_from_texts(
        prompt_answer_pairs,
        config.near_duplicate_threshold,
    )
    if len(near_duplicate_pairs) > config.max_near_duplicate_answers:
        findings.append(
            AuditFinding(
                code="near_duplicate_prompt_answer_pair",
                severity="error",
                message=(
                    f"found {len(near_duplicate_pairs)} prompt/answer pairs with "
                    f"canonical Jaccard similarity >= {config.near_duplicate_threshold:.2f}; "
                    f"limit is {config.max_near_duplicate_answers}"
                ),
                examples=tuple(near_duplicate_pairs[: config.example_limit]),
            )
        )


def _append_content_findings(
    records: Sequence[SftConversation],
    config: AuditConfig,
    findings: list[AuditFinding],
) -> None:
    content_by_problem: dict[str, list[str]] = {
        "empty_agi_response": [],
        "leaked_control_token": [],
        "synthetic_tag": [],
        "replacement_character": [],
    }
    for record in records:
        for message in record.messages:
            example = f"{record.source}: {message.content[:160]}"
            if message.role == "agi" and not message.content.strip():
                content_by_problem["empty_agi_response"].append(example)
            if _LEAKED_CONTROL_TOKEN_RE.search(message.content):
                content_by_problem["leaked_control_token"].append(example)
            if _SYNTHETIC_TAG_RE.search(message.content):
                content_by_problem["synthetic_tag"].append(example)
            if "\ufffd" in message.content:
                content_by_problem["replacement_character"].append(example)
    for code, examples in content_by_problem.items():
        if examples:
            findings.append(
                AuditFinding(
                    code=code,
                    severity="error",
                    message=f"found {len(examples)} {code.replace('_', ' ')} occurrences",
                    examples=tuple(examples[: config.example_limit]),
                )
            )


def _token_counts(
    records: Sequence[SftConversation],
    tokenizer: TokenizerLike | object | None,
    context_length: int | None,
    findings: list[AuditFinding],
    config: AuditConfig,
) -> list[int]:
    if tokenizer is None:
        return [
            len(canonical_text(" ".join(message.content for message in record.messages)).split())
            for record in records
        ]

    counts: list[int] = []
    for record in records:
        try:
            tokenized = tokenize_sft_messages(record.messages, tokenizer, source=record.source)
        except ValueError as error:
            code = "zero_supervised_labels" if "no AGI response tokens" in str(error) else "tokenization_error"
            findings.append(
                AuditFinding(
                    code=code,
                    severity="error",
                    message=f"{record.source}: {error}",
                )
            )
            continue
        counts.append(len(tokenized.input_ids))
        if tokenized.supervised_token_count == 0:
            findings.append(
                AuditFinding(
                    code="zero_supervised_labels",
                    severity="error",
                    message=f"{record.source} has zero supervised AGI tokens",
                )
            )
        if context_length is not None and len(tokenized.input_ids) > context_length:
            findings.append(
                AuditFinding(
                    code="context_overflow",
                    severity="error",
                    message=(
                        f"{record.source} uses {len(tokenized.input_ids)} tokens, "
                        f"above context_length={context_length}"
                    ),
                )
            )
    return counts


def _effective_context_length(
    checkpoint_context_length: int | None,
    configured_context_limit: int | None,
) -> int | None:
    limits = [
        limit
        for limit in (checkpoint_context_length, configured_context_limit)
        if limit is not None
    ]
    return min(limits) if limits else None


def _repeated_openings(
    responses: Sequence[tuple[str, str]],
    config: AuditConfig,
) -> dict[str, int]:
    openings = Counter(
        " ".join(canonical_text(answer).split()[: config.opening_token_count])
        for _, answer in responses
    )
    return {
        opening: count
        for opening, count in sorted(openings.items(), key=lambda item: (-item[1], item[0]))
        if opening
    }


def _repeated_ngrams(
    responses: Sequence[tuple[str, str]],
    config: AuditConfig,
) -> dict[str, int]:
    ngrams: Counter[str] = Counter()
    for _, answer in responses:
        words = canonical_text(answer).split()
        ngrams.update(
            set(
                " ".join(words[index : index + config.ngram_size])
                for index in range(len(words) - config.ngram_size + 1)
            )
        )
    return {
        ngram: count
        for ngram, count in sorted(ngrams.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    }


def _append_repetition_findings(
    responses: Sequence[tuple[str, str]],
    repeated_openings: Mapping[str, int],
    repeated_ngrams: Mapping[str, int],
    config: AuditConfig,
    findings: list[AuditFinding],
) -> None:
    if len(responses) >= config.opening_frequency_min_responses:
        excessive_openings = {
            opening: count
            for opening, count in repeated_openings.items()
            if count / len(responses) > config.opening_frequency_limit
        }
        if excessive_openings:
            findings.append(
                AuditFinding(
                    code="opening_frequency",
                    severity="error",
                    message=(
                        "response opening frequency exceeds "
                        f"{config.opening_frequency_limit:.0%}"
                    ),
                    examples=tuple(
                        f"{opening!r}: {count}/{len(responses)}"
                        for opening, count in list(excessive_openings.items())[: config.example_limit]
                    ),
                )
            )
    excessive_ngrams = {
        ngram: count
        for ngram, count in repeated_ngrams.items()
        if count > config.max_repeated_ngram_count
    }
    if excessive_ngrams:
        findings.append(
            AuditFinding(
                code="repeated_ngram",
                severity="error",
                message=(
                    f"{config.ngram_size}-gram reuse exceeds "
                    f"{config.max_repeated_ngram_count} responses"
                ),
                examples=tuple(
                    f"{ngram!r}: {count}"
                    for ngram, count in list(excessive_ngrams.items())[: config.example_limit]
                ),
            )
        )


def _identity_share(responses: Sequence[tuple[str, str]]) -> float:
    if not responses:
        return 0.0
    return sum(
        bool(_IDENTITY_OR_LIMITATION_RE.search(answer))
        for _, answer in responses
    ) / len(responses)


def _examples_matching(
    responses: Sequence[tuple[str, str]],
    pattern: re.Pattern[str],
    config: AuditConfig,
) -> tuple[str, ...]:
    return tuple(
        f"{source}: {answer[:160]}"
        for source, answer in responses
        if pattern.search(answer)
    )[: config.example_limit]


def _curated_sampling_mass(
    records: Sequence[SftConversation],
    source_weights: Mapping[str, float],
    config: AuditConfig,
) -> float:
    mass_by_family: Counter[str] = Counter()
    for record in records:
        family = _source_family(record.source)
        mass_by_family[family] += _weight_for_source(record.source, source_weights)
    total_mass = sum(mass_by_family.values())
    if total_mass <= 0:
        return 0.0
    curated_mass = sum(
        mass_by_family[family]
        for family in config.curated_source_families
    )
    return curated_mass / total_mass


def _weight_for_source(source: str, source_weights: Mapping[str, float]) -> float:
    family = _source_family(source)
    if source in source_weights:
        return source_weights[source]
    if family in source_weights:
        return source_weights[family]
    return source_weights.get("default", 1.0)


def _source_family(source: str) -> str:
    return source.split(":", maxsplit=1)[0]


def _nearest_rank_quantiles(values: Iterable[int]) -> dict[str, int]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        f"p{int(percentile * 100)}": sorted_values[
            max(0, math.ceil(percentile * len(sorted_values)) - 1)
        ]
        for percentile in (0.50, 0.90, 0.95, 0.99)
    }


def _near_duplicate_pairs(
    responses: Sequence[tuple[str, str]],
    threshold: float,
) -> list[str]:
    return _near_duplicate_pairs_from_texts(responses, threshold)


def _near_duplicate_pairs_from_texts(
    values: Sequence[tuple[str, str]],
    threshold: float,
) -> list[str]:
    normalized = [
        (source, text, frozenset(canonical_text(text).split()))
        for source, text in values
    ]
    document_frequency: Counter[str] = Counter()
    for _, _, token_set in normalized:
        document_frequency.update(token_set)
    token_order = {
        token: rank
        for rank, (token, _) in enumerate(
            sorted(document_frequency.items(), key=lambda item: (item[1], item[0]))
        )
    }
    index = _NearDuplicateIndex(threshold, token_order)
    duplicates: list[str] = []
    for source, _, token_set in normalized:
        match = index.first_match(token_set)
        if match is not None:
            duplicates.append(f"{match} ~= {source}")
        index.add(source, token_set)
    return duplicates


class _NearDuplicateIndex:
    def __init__(self, threshold: float, token_order: Mapping[str, int]) -> None:
        self.threshold = threshold
        self.token_order = token_order
        self.entries: list[tuple[str, frozenset[str]]] = []
        self.postings: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

    def first_match(self, token_set: frozenset[str]) -> str | None:
        if not token_set:
            return None
        lower, upper = _jaccard_length_bounds(len(token_set), self.threshold)
        candidates: set[int] = set()
        for token in _jaccard_prefix(token_set, self.threshold, self.token_order):
            for length, posting in self.postings.get(token, {}).items():
                if lower <= length <= upper:
                    candidates.update(posting)
        for index in sorted(candidates):
            source, candidate = self.entries[index]
            if token_jaccard(" ".join(token_set), " ".join(candidate)) >= self.threshold:
                return source
        return None

    def add(self, source: str, token_set: frozenset[str]) -> None:
        if not token_set:
            return
        index = len(self.entries)
        self.entries.append((source, token_set))
        for token in _jaccard_prefix(token_set, self.threshold, self.token_order):
            self.postings[token][len(token_set)].add(index)


def _jaccard_length_bounds(length: int, threshold: float) -> tuple[int, int]:
    return math.ceil(threshold * length), math.floor(length / threshold)


def _jaccard_prefix(
    token_set: frozenset[str],
    threshold: float,
    token_order: Mapping[str, int],
) -> tuple[str, ...]:
    ordered = tuple(sorted(token_set, key=lambda token: (token_order[token], token)))
    prefix_length = len(ordered) - math.ceil(threshold * len(ordered)) + 1
    return ordered[:prefix_length]
