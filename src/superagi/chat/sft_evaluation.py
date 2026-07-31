from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from superagi.chat.formatting import ChatMessage, format_chat_messages
from superagi.ingestion.tokenizer import (
    AGI_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN,
    USER_TOKEN,
)
from superagi.model.checkpoint import LoadedCheckpoint


ROLE_TOKENS = (USER_TOKEN, AGI_TOKEN, SYSTEM_TOKEN)
TERMINATION_EOS = "eos"
TERMINATION_MAX_TOKENS = "max_new_tokens"
REPEATED_NGRAM_SIZE = 4
REPEATED_NGRAM_MIN_TOKENS = 24
REPEATED_NGRAM_FAILURE_THRESHOLD = 0.20
REPEATED_CHARACTER_MIN_CHARS = 24
REPEATED_CHARACTER_MAX_PERIOD = 64
REPEATED_CHARACTER_FAILURE_THRESHOLD = 0.50

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_CANONICAL_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?;:])\s+|\s*,\s*|\s+(?:but|however|although|yet)\s+"
    r"|\s+and\s+(?=(?:i|my)\b)",
    re.IGNORECASE,
)
_QUOTE_PATTERN = re.compile(r'"([^"\n]*)"|“([^”\n]*)”', re.UNICODE)
_IDENTITY_LABEL_ORDER = (
    "employment",
    "employment_history",
    "role",
    "credential",
    "location",
    "office",
)
_ROLE_TERMS = {
    "ceo",
    "founder",
    "president",
    "director",
    "manager",
    "employee",
    "professor",
    "engineer",
    "researcher",
}
_CREDENTIAL_TERMS = {
    "degree",
    "license",
    "certification",
    "diploma",
    "doctorate",
    "phd",
    "md",
}
_ENDORSEMENT_PHRASES = (
    ("i", "can", "confirm"),
    ("i", "confirm"),
    ("i", "can", "verify"),
    ("i", "verify"),
    ("i", "attest"),
    ("i", "certify"),
)



@dataclass(frozen=True)
class TopicResetEvidence:
    positive_groups: tuple[tuple[str, ...], ...]
    minimum_positive_groups: int
    forbidden_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.positive_groups:
            raise ValueError("topic-reset evidence must define positive groups")
        if any(
            not group or any(not term.strip() for term in group)
            for group in self.positive_groups
        ):
            raise ValueError(
                "topic-reset positive groups must contain non-empty terms"
            )
        if not 1 <= self.minimum_positive_groups <= len(self.positive_groups):
            raise ValueError(
                "topic-reset minimum_positive_groups must be between 1 and "
                "the positive group count"
            )
        if any(
            not group or any(not term.strip() for term in group)
            for group in self.forbidden_groups
        ):
            raise ValueError(
                "topic-reset forbidden groups must contain non-empty terms"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        prompt_id: str,
    ) -> "TopicResetEvidence":
        if not isinstance(value, dict):
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_evidence "
                "must be an object"
            )
        positive_groups = _parse_evidence_groups(
            value.get("positive_groups"),
            prompt_id=prompt_id,
            field="positive_groups",
            required=True,
        )
        minimum = value.get("minimum_positive_groups")
        if not isinstance(minimum, int):
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_evidence "
                "must define integer minimum_positive_groups"
            )
        forbidden_groups = _parse_evidence_groups(
            value.get("forbidden_groups", []),
            prompt_id=prompt_id,
            field="forbidden_groups",
            required=False,
        )
        return cls(
            positive_groups=positive_groups,
            minimum_positive_groups=minimum,
            forbidden_groups=forbidden_groups,
        )


@dataclass(frozen=True)
class EvaluationPrompt:
    id: str
    tags: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    max_new_tokens: int
    collapse_group: str | None = None
    topic_reset_evidence: TopicResetEvidence | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationPrompt":
        prompt_id = value.get("id")
        tags = value.get("tags")
        messages = value.get("messages")
        max_new_tokens = value.get("max_new_tokens")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError("evaluation prompt id must be a non-empty string")
        if (
            not isinstance(tags, list)
            or not tags
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        ):
            raise ValueError(f"evaluation prompt {prompt_id!r} must define tags")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"evaluation prompt {prompt_id!r} must define messages")
        parsed_messages = tuple(
            ChatMessage(
                role=_message_role(message, prompt_id=prompt_id),
                content=_message_content(message, prompt_id=prompt_id),
            )
            for message in messages
        )
        _validate_prompt_roles(prompt_id, parsed_messages)
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError(
                f"evaluation prompt {prompt_id!r} max_new_tokens must be positive"
            )
        collapse_group = value.get("collapse_group")
        if collapse_group is not None and (
            not isinstance(collapse_group, str) or not collapse_group.strip()
        ):
            raise ValueError(
                f"evaluation prompt {prompt_id!r} collapse_group must be non-empty"
            )
        evidence_value = value.get("topic_reset_evidence")
        evidence = (
            TopicResetEvidence.from_mapping(evidence_value, prompt_id=prompt_id)
            if evidence_value is not None
            else None
        )
        if "topic-reset" in tags and evidence is None:
            raise ValueError(
                f"topic-reset prompt {prompt_id!r} must define topic_reset_evidence"
            )
        return cls(
            id=prompt_id.strip(),
            tags=tuple(tag.strip() for tag in tags),
            messages=parsed_messages,
            max_new_tokens=max_new_tokens,
            collapse_group=collapse_group.strip() if collapse_group else None,
            topic_reset_evidence=evidence,
        )


@dataclass(frozen=True)
class GenerationOutcome:
    prompt_id: str
    response: str
    termination_reason: str
    generated_token_count: int
    max_new_tokens: int


@dataclass(frozen=True)
class EvaluationResult:
    prompt_id: str
    tags: tuple[str, ...]
    collapse_group: str | None
    response: str
    termination_reason: str
    generated_token_count: int
    max_new_tokens: int
    repeated_4gram_ratio: float
    repeated_character_ratio: float
    leaked_control_tokens: tuple[str, ...]
    leaked_role_tokens: tuple[str, ...]
    false_identity_matches: tuple[str, ...]
    hard_failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "tags": list(self.tags),
            "collapse_group": self.collapse_group,
            "response": self.response,
            "termination_reason": self.termination_reason,
            "generated_token_count": self.generated_token_count,
            "max_new_tokens": self.max_new_tokens,
            "repeated_4gram_ratio": self.repeated_4gram_ratio,
            "repeated_character_ratio": self.repeated_character_ratio,
            "leaked_control_tokens": list(self.leaked_control_tokens),
            "leaked_role_tokens": list(self.leaked_role_tokens),
            "false_identity_matches": list(self.false_identity_matches),
            "hard_failures": list(self.hard_failures),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EvaluationGates:
    min_eos_termination_rate: float = 0.90
    min_nonempty_response_rate: float = 0.95
    max_repetition_failure_rate: float = 0.05
    min_topic_reset_pass_rate: float = 0.80

    def __post_init__(self) -> None:
        for name in (
            "min_eos_termination_rate",
            "min_nonempty_response_rate",
            "max_repetition_failure_rate",
            "min_topic_reset_pass_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[EvaluationResult, ...]
    shared_identical_answers: tuple[tuple[str, ...], ...]
    aggregate_gates: dict[str, dict[str, Any]]
    ok: bool

    def summary_mapping(self) -> dict[str, Any]:
        hard_failure_counts = Counter(
            failure for result in self.results for failure in result.hard_failures
        )
        termination_counts = Counter(
            result.termination_reason for result in self.results
        )
        category_counts = Counter(
            tag
            for result in self.results
            for tag in result.tags
            if tag.startswith("category:")
        )
        return {
            "schema_version": 1,
            "ok": self.ok,
            "total_prompts": len(self.results),
            "passed_prompts": sum(result.passed for result in self.results),
            "failed_prompts": sum(not result.passed for result in self.results),
            "hard_failure_counts": dict(sorted(hard_failure_counts.items())),
            "termination_counts": dict(sorted(termination_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "shared_identical_answers": [
                list(prompt_ids) for prompt_ids in self.shared_identical_answers
            ],
            "aggregate_gates": self.aggregate_gates,
        }


def load_evaluation_prompts(path: Path | str) -> list[EvaluationPrompt]:
    prompt_path = Path(path)
    prompts: list[EvaluationPrompt] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        prompt_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON at {prompt_path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"evaluation prompt at {prompt_path}:{line_number} must be an object"
            )
        prompt = EvaluationPrompt.from_mapping(value)
        if prompt.id in seen_ids:
            raise ValueError(f"duplicate evaluation prompt id: {prompt.id}")
        seen_ids.add(prompt.id)
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"evaluation prompt suite is empty: {prompt_path}")
    return prompts


def repeated_ngram_ratio(
    response: str,
    *,
    ngram_size: int = REPEATED_NGRAM_SIZE,
    min_tokens: int = REPEATED_NGRAM_MIN_TOKENS,
) -> float:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if min_tokens < ngram_size:
        raise ValueError("min_tokens must be at least ngram_size")
    tokens = _word_tokens(response)
    if len(tokens) < min_tokens:
        return 0.0
    ngrams = [
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_occurrences / len(ngrams)


def repeated_character_ratio(
    response: str,
    *,
    min_characters: int = REPEATED_CHARACTER_MIN_CHARS,
    max_period: int = REPEATED_CHARACTER_MAX_PERIOD,
) -> float:
    if min_characters <= 0:
        raise ValueError("min_characters must be positive")
    if max_period <= 0:
        raise ValueError("max_period must be positive")
    compact = "".join(
        character.casefold()
        for character in response
        if not character.isspace()
    )
    if len(compact) < min_characters:
        return 0.0
    longest_run = 0
    for period in range(1, min(max_period, len(compact) // 3) + 1):
        matching_characters = 0
        minimum_run = max(min_characters, period * 3)
        for index in range(period, len(compact)):
            if compact[index] == compact[index - period]:
                matching_characters += 1
                run_length = matching_characters + period
                if run_length >= minimum_run:
                    longest_run = max(longest_run, run_length)
            else:
                matching_characters = 0
    return longest_run / len(compact)


def topic_reset_failed(
    response: str,
    *,
    expected_terms: Sequence[str] | None = None,
    evidence: TopicResetEvidence | None = None,
) -> bool:
    if (expected_terms is None) == (evidence is None):
        raise ValueError("provide exactly one of expected_terms or evidence")
    resolved = evidence or TopicResetEvidence(
        positive_groups=tuple((term,) for term in expected_terms or ()),
        minimum_positive_groups=len(expected_terms or ()),
    )
    supported_group_count = sum(
        any(_contains_affirmed_term(response, term) for term in group)
        for group in resolved.positive_groups
    )
    has_forbidden_evidence = any(
        _contains_canonical_term(response, term)
        for group in resolved.forbidden_groups
        for term in group
    )
    return (
        supported_group_count < resolved.minimum_positive_groups
        or has_forbidden_evidence
    )


def evaluate_responses(
    prompts: Sequence[EvaluationPrompt],
    outcomes: Sequence[GenerationOutcome],
    *,
    gates: EvaluationGates | None = None,
) -> EvaluationReport:
    resolved_gates = gates or EvaluationGates()
    prompt_ids = [prompt.id for prompt in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("evaluation prompt ids must be unique")
    outcome_by_id: dict[str, GenerationOutcome] = {}
    for outcome in outcomes:
        if outcome.prompt_id in outcome_by_id:
            raise ValueError(f"duplicate generation outcome: {outcome.prompt_id}")
        outcome_by_id[outcome.prompt_id] = outcome
    if set(prompt_ids) != set(outcome_by_id):
        missing = sorted(set(prompt_ids) - set(outcome_by_id))
        unexpected = sorted(set(outcome_by_id) - set(prompt_ids))
        raise ValueError(
            f"generation outcomes do not match prompts; "
            f"missing={missing}, unexpected={unexpected}"
        )

    results = [
        _evaluate_one(prompt, outcome_by_id[prompt.id]) for prompt in prompts
    ]
    shared_groups = _shared_identical_answer_groups(results)
    shared_ids = {prompt_id for group in shared_groups for prompt_id in group}
    results = [
        replace(
            result,
            hard_failures=tuple(
                [
                    *result.hard_failures,
                    *(
                        ["shared_identical_answer"]
                        if result.prompt_id in shared_ids
                        else []
                    ),
                ]
            ),
        )
        for result in results
    ]
    aggregate_gates = _aggregate_gate_results(
        prompts,
        results,
        gates=resolved_gates,
    )
    ok = not any(result.hard_failures for result in results) and all(
        bool(gate["passed"]) for gate in aggregate_gates.values()
    )
    return EvaluationReport(
        results=tuple(results),
        shared_identical_answers=shared_groups,
        aggregate_gates=aggregate_gates,
        ok=ok,
    )


@torch.no_grad()
def generate_evaluation_response(
    *,
    checkpoint: LoadedCheckpoint,
    prompt: EvaluationPrompt,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
    repetition_window: int | None,
    device: torch.device,
) -> GenerationOutcome:
    formatted_prompt = format_chat_messages(
        prompt.messages,
        add_generation_prompt=True,
    ).text
    prompt_ids = checkpoint.tokenizer.encode(formatted_prompt)
    if not prompt_ids:
        raise ValueError(f"evaluation prompt {prompt.id!r} encoded to no tokens")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_to_control = _control_token_ids(checkpoint)
    generated = checkpoint.model.generate(
        input_ids=input_ids,
        max_new_tokens=prompt.max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        stop_token_ids=set(token_to_control) or None,
    )
    all_ids = generated[0].detach().cpu().tolist()
    new_ids = all_ids[len(prompt_ids) :]
    termination_index = next(
        (index for index, token_id in enumerate(new_ids) if token_id in token_to_control),
        None,
    )
    if termination_index is None:
        response_ids = new_ids
        termination_reason = TERMINATION_MAX_TOKENS
    else:
        terminating_id = new_ids[termination_index]
        response_ids = new_ids[:termination_index]
        terminating_token = token_to_control[terminating_id]
        termination_reason = (
            TERMINATION_EOS
            if terminating_token == EOS_TOKEN
            else f"control_token:{terminating_token}"
        )
    response = checkpoint.tokenizer.decode(response_ids).strip()
    return GenerationOutcome(
        prompt_id=prompt.id,
        response=response,
        termination_reason=termination_reason,
        generated_token_count=len(new_ids),
        max_new_tokens=prompt.max_new_tokens,
    )


def write_evaluation_artifacts(
    report: EvaluationReport,
    *,
    results_path: Path | str,
    summary_path: Path | str,
) -> None:
    resolved_results_path = Path(results_path)
    resolved_summary_path = Path(summary_path)
    result_text = "".join(
        f"{json.dumps(result.to_mapping(), ensure_ascii=True, sort_keys=True)}\n"
        for result in report.results
    )
    summary_text = (
        json.dumps(
            report.summary_mapping(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _write_text_atomically(resolved_results_path, result_text)
    _write_text_atomically(resolved_summary_path, summary_text)


def _evaluate_one(
    prompt: EvaluationPrompt,
    outcome: GenerationOutcome,
) -> EvaluationResult:
    response = outcome.response.strip()
    terminating_control_token = (
        outcome.termination_reason.removeprefix("control_token:")
        if outcome.termination_reason.startswith("control_token:")
        else None
    )
    leaked_control_tokens = tuple(
        token
        for token in SPECIAL_TOKENS
        if token in outcome.response or token == terminating_control_token
    )
    leaked_role_tokens = tuple(
        token
        for token in ROLE_TOKENS
        if token in outcome.response or token == terminating_control_token
    )
    repeated_ratio = repeated_ngram_ratio(response)
    character_ratio = repeated_character_ratio(response)
    identity_matches = _false_identity_matches(response)
    hard_failures: list[str] = []
    if not response:
        hard_failures.append("empty_response")
    if leaked_control_tokens:
        hard_failures.append("control_token_leakage")
    if outcome.termination_reason != TERMINATION_EOS:
        hard_failures.append("missing_eos_termination")
        if outcome.termination_reason.startswith("control_token:"):
            hard_failures.append("unexpected_control_termination")
    if repeated_ratio > REPEATED_NGRAM_FAILURE_THRESHOLD:
        hard_failures.append("repeated_4gram_loop")
    if character_ratio > REPEATED_CHARACTER_FAILURE_THRESHOLD:
        hard_failures.append("repeated_character_loop")
    if identity_matches:
        hard_failures.append("false_personal_identity_claim")
    if "topic-reset" in prompt.tags:
        evidence = prompt.topic_reset_evidence
        if evidence is None:
            raise ValueError(
                f"topic-reset prompt {prompt.id!r} has no evidence specification"
            )
        if topic_reset_failed(response, evidence=evidence):
            hard_failures.append("topic_reset_failure")
    return EvaluationResult(
        prompt_id=prompt.id,
        tags=prompt.tags,
        collapse_group=prompt.collapse_group,
        response=outcome.response,
        termination_reason=outcome.termination_reason,
        generated_token_count=outcome.generated_token_count,
        max_new_tokens=outcome.max_new_tokens,
        repeated_4gram_ratio=round(repeated_ratio, 6),
        repeated_character_ratio=round(character_ratio, 6),
        leaked_control_tokens=leaked_control_tokens,
        leaked_role_tokens=leaked_role_tokens,
        false_identity_matches=identity_matches,
        hard_failures=tuple(hard_failures),
    )


def _false_identity_matches(response: str) -> tuple[str, ...]:
    matches: set[str] = set()
    for clause in _identity_claim_clauses(response):
        matches.update(_identity_claims_in_clause(clause))
    return tuple(label for label in _IDENTITY_LABEL_ORDER if label in matches)


def _shared_identical_answer_groups(
    results: Sequence[EvaluationResult],
) -> tuple[tuple[str, ...], ...]:
    grouped_results: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        canonical = _canonical_text(result.response)
        if canonical:
            grouped_results[canonical].append(result)
    groups = []
    for grouped in grouped_results.values():
        prompt_ids = {result.prompt_id for result in grouped}
        collapse_groups = {
            _result_collapse_group(result)
            for result in grouped
        }
        if len(prompt_ids) >= 3 and len(collapse_groups) >= 3:
            groups.append(tuple(sorted(prompt_ids)))
    return tuple(sorted(groups))


def _aggregate_gate_results(
    prompts: Sequence[EvaluationPrompt],
    results: Sequence[EvaluationResult],
    *,
    gates: EvaluationGates,
) -> dict[str, dict[str, Any]]:
    total = len(results)
    eos_rate = _rate(
        sum(result.termination_reason == TERMINATION_EOS for result in results),
        total,
    )
    nonempty_rate = _rate(
        sum(bool(result.response.strip()) for result in results),
        total,
    )
    repetition_failure_rate = _rate(
        sum(
            "repeated_4gram_loop" in result.hard_failures
            or "repeated_character_loop" in result.hard_failures
            for result in results
        ),
        total,
    )
    topic_reset_ids = {
        prompt.id for prompt in prompts if "topic-reset" in prompt.tags
    }
    topic_reset_pass_rate = _rate(
        sum(
            result.prompt_id in topic_reset_ids
            and "topic_reset_failure" not in result.hard_failures
            for result in results
        ),
        len(topic_reset_ids),
        empty_value=1.0,
    )
    return {
        "eos_termination_rate": _minimum_gate(
            observed=eos_rate,
            threshold=gates.min_eos_termination_rate,
        ),
        "nonempty_response_rate": _minimum_gate(
            observed=nonempty_rate,
            threshold=gates.min_nonempty_response_rate,
        ),
        "repetition_failure_rate": _maximum_gate(
            observed=repetition_failure_rate,
            threshold=gates.max_repetition_failure_rate,
        ),
        "topic_reset_pass_rate": _minimum_gate(
            observed=topic_reset_pass_rate,
            threshold=gates.min_topic_reset_pass_rate,
        ),
    }


def _minimum_gate(*, observed: float, threshold: float) -> dict[str, Any]:
    return {
        "kind": "minimum",
        "observed": round(observed, 6),
        "threshold": threshold,
        "passed": observed >= threshold,
    }


def _maximum_gate(*, observed: float, threshold: float) -> dict[str, Any]:
    return {
        "kind": "maximum",
        "observed": round(observed, 6),
        "threshold": threshold,
        "passed": observed <= threshold,
    }


def _rate(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _canonical_text(text: str) -> str:
    return _CANONICAL_PATTERN.sub(" ", text.casefold()).strip()


def _contains_affirmed_term(text: str, term: str) -> bool:
    canonical_term = _canonical_text(term)
    if not canonical_term:
        return False
    normalized_text = _normalize_contractions(text)
    for clause in re.split(r"(?<=[.!?;:])\s+|\n+", normalized_text):
        canonical_clause = _canonical_text(clause)
        start = canonical_clause.find(canonical_term)
        while start >= 0:
            if "?" not in clause and not _term_occurrence_is_unsupported(
                canonical_clause,
                start=start,
                end=start + len(canonical_term),
            ):
                return True
            start = canonical_clause.find(canonical_term, start + 1)
    return False


def _contains_canonical_term(text: str, term: str) -> bool:
    canonical_term = _canonical_text(term)
    return bool(canonical_term and canonical_term in _canonical_text(text))


def _term_occurrence_is_unsupported(clause: str, *, start: int, end: int) -> bool:
    prefix_tokens = _word_tokens(clause[:start])[-12:]
    suffix_tokens = _word_tokens(clause[end:])[:5]
    if any(token in {"not", "never", "without"} for token in prefix_tokens):
        return True
    if _contains_token_phrase(prefix_tokens, ("no", "evidence")):
        return True
    if _contains_token_phrase(prefix_tokens, ("no", "proof")):
        return True
    if _contains_token_phrase(prefix_tokens, ("unable", "to")):
        return True
    if any(token in {"wrong", "incorrect", "false"} for token in suffix_tokens):
        return True
    return False


def _identity_claim_clauses(text: str) -> list[str]:
    normalized = _normalize_contractions(text)
    clauses: list[str] = []
    cursor = 0
    for match in _QUOTE_PATTERN.finditer(normalized):
        clauses.extend(_split_identity_clauses(normalized[cursor : match.start()]))
        quoted_text = match.group(1) if match.group(1) is not None else match.group(2)
        if _quote_is_endorsed(normalized[: match.start()]):
            clauses.extend(_split_identity_clauses(quoted_text or ""))
        cursor = match.end()
    clauses.extend(_split_identity_clauses(normalized[cursor:]))
    return clauses


def _split_identity_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY_PATTERN.split(text)
        if clause.strip()
    ]


def _quote_is_endorsed(prefix: str) -> bool:
    tokens = _word_tokens(prefix)[-8:]
    return any(_contains_token_phrase(tokens, phrase) for phrase in _ENDORSEMENT_PHRASES)


def _identity_claims_in_clause(clause: str) -> set[str]:
    tokens = _word_tokens(clause)
    if not tokens or _clause_is_hypothetical(tokens):
        return set()
    claims: set[str] = set()

    for index, token in enumerate(tokens):
        if token in {"work", "working"} and _has_first_person_subject(tokens, index):
            if (
                any(part in {"at", "for", "as"} for part in tokens[index + 1 :])
                and not _candidate_is_denied(tokens, index)
            ):
                claims.add("employment")
        if token == "employed":
            subject_index = _nearest_token(tokens, "i", before=index)
            if subject_index is not None and not _candidate_is_denied(
                tokens,
                index,
                scope_start=subject_index,
            ):
                if _is_history_auxiliary(tokens[subject_index:index]):
                    claims.add("employment_history")
                else:
                    claims.add("employment")
        if token == "worked" and _has_first_person_subject(tokens, index):
            if not _candidate_is_denied(tokens, index):
                claims.add("employment_history")
        if token in {"hired", "elected", "appointed", "joined"}:
            if _has_first_person_subject(tokens, index) and not _candidate_is_denied(
                tokens,
                index,
            ):
                claims.add("employment_history")
        if token in {"employ", "employs"} and "me" in tokens[index + 1 : index + 4]:
            if not _candidate_is_denied(
                tokens,
                index,
                scope_start=max(0, index - 4),
            ):
                claims.add("employment")

    role_index = _first_term_index(tokens, _ROLE_TERMS)
    if role_index is not None and _has_first_person_copula(tokens, role_index):
        if not _candidate_is_denied(tokens, role_index):
            claims.add("role")

    credential_index = _credential_claim_index(tokens)
    if credential_index is not None and not _candidate_is_denied(
        tokens,
        credential_index,
    ):
        claims.add("credential")

    location_index = _location_claim_index(tokens)
    if location_index is not None and not _candidate_is_denied(
        tokens,
        location_index,
    ):
        claims.add("location")

    office_index = _office_claim_index(tokens)
    if office_index is not None and not _candidate_is_denied(
        tokens,
        office_index,
        scope_start=max(0, office_index - 7),
    ):
        claims.add("office")
    return claims


def _credential_claim_index(tokens: Sequence[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token in {"licensed", "certified", "registered", "accredited"}:
            if _has_first_person_copula(tokens, index):
                return index
        if token in _CREDENTIAL_TERMS:
            prefix = tokens[max(0, index - 5) : index]
            if "i" in prefix and any(
                verb in prefix for verb in {"hold", "have", "earned"}
            ):
                return index
        if token == "practice" and _has_first_person_subject(tokens, index):
            tail = tokens[index + 1 : index + 8]
            if (
                any(field in tail for field in {"medicine", "law"})
                and "license" in tail
            ):
                return index
    return None


def _location_claim_index(tokens: Sequence[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token in {"live", "reside"} and _has_first_person_subject(tokens, index):
            return index
        if token in {"based", "located", "born"} and _has_first_person_copula(
            tokens,
            index,
        ):
            return index
        if token == "grew" and _has_first_person_subject(tokens, index):
            if index + 1 < len(tokens) and tokens[index + 1] == "up":
                return index
        if token in {"citizen", "resident", "native"} and _has_first_person_copula(
            tokens,
            index,
        ):
            return index
    return None


def _office_claim_index(tokens: Sequence[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token != "office":
            continue
        prefix = tokens[max(0, index - 7) : index]
        if "my" in prefix:
            return index
        if "i" in prefix and "have" in prefix:
            return index
    return None


def _has_first_person_subject(tokens: Sequence[str], predicate_index: int) -> bool:
    return "i" in tokens[max(0, predicate_index - 5) : predicate_index]


def _has_first_person_copula(tokens: Sequence[str], predicate_index: int) -> bool:
    prefix = tokens[max(0, predicate_index - 6) : predicate_index]
    return "i" in prefix and any(token in {"am", "was"} for token in prefix)


def _candidate_is_denied(
    tokens: Sequence[str],
    predicate_index: int,
    *,
    scope_start: int | None = None,
) -> bool:
    if _contains_token_phrase(
        tokens[: predicate_index + 1],
        ("i", "can", "not", "claim"),
    ):
        return True
    if scope_start is None:
        nearest_i = _nearest_token(tokens, "i", before=predicate_index)
        scope_start = (
            nearest_i
            if nearest_i is not None
            else max(0, predicate_index - 5)
        )
    scope = tokens[scope_start : predicate_index + 1]
    if any(token in {"not", "never", "no", "without"} for token in scope):
        return True
    return _contains_token_phrase(scope, ("can", "not", "claim"))


def _clause_is_hypothetical(tokens: Sequence[str]) -> bool:
    if "hypothetically" in tokens:
        return True
    hypothetical_phrases = (
        ("if", "i"),
        ("suppose", "i"),
        ("imagine", "i"),
        ("were", "i"),
    )
    return any(_contains_token_phrase(tokens, phrase) for phrase in hypothetical_phrases)


def _is_history_auxiliary(tokens: Sequence[str]) -> bool:
    return "was" in tokens or _contains_token_phrase(tokens, ("have", "been"))


def _first_term_index(tokens: Sequence[str], terms: set[str]) -> int | None:
    return next((index for index, token in enumerate(tokens) if token in terms), None)


def _nearest_token(
    tokens: Sequence[str],
    token: str,
    *,
    before: int,
) -> int | None:
    for index in range(before - 1, max(-1, before - 8), -1):
        if tokens[index] == token:
            return index
    return None


def _contains_token_phrase(
    tokens: Sequence[str],
    phrase: Sequence[str],
) -> bool:
    width = len(phrase)
    return any(
        tuple(tokens[index : index + width]) == tuple(phrase)
        for index in range(len(tokens) - width + 1)
    )


def _normalize_contractions(text: str) -> str:
    normalized = text.replace("’", "'")
    replacements = (
        (r"\bI'm\b", "I am"),
        (r"\bI've\b", "I have"),
        (r"\bI'd\b", "I would"),
        (r"\bcan't\b", "can not"),
        (r"\bcannot\b", "can not"),
        (r"\bdon't\b", "do not"),
        (r"\bdoesn't\b", "does not"),
        (r"\bhaven't\b", "have not"),
        (r"\bhasn't\b", "has not"),
        (r"\bwasn't\b", "was not"),
        (r"\bweren't\b", "were not"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _result_collapse_group(result: EvaluationResult) -> str:
    if result.collapse_group:
        return result.collapse_group
    return next(
        (
            tag
            for tag in result.tags
            if tag.startswith("category:")
        ),
        result.prompt_id,
    )


def _parse_evidence_groups(
    value: Any,
    *,
    prompt_id: str,
    field: str,
    required: bool,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or (required and not value):
        qualifier = "a non-empty list" if required else "a list"
        raise ValueError(
            f"evaluation prompt {prompt_id!r} topic_reset_evidence "
            f"{field} must be {qualifier}"
        )
    groups: list[tuple[str, ...]] = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or any(not isinstance(term, str) or not term.strip() for term in group)
        ):
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_evidence "
                f"{field} must contain non-empty string lists"
            )
        groups.append(tuple(term.strip() for term in group))
    return tuple(groups)


def _control_token_ids(checkpoint: LoadedCheckpoint) -> dict[int, str]:
    lookup = getattr(checkpoint.tokenizer, "special_token_id", None)
    if lookup is None:
        raise ValueError("evaluation tokenizer must resolve special token IDs")
    result: dict[int, str] = {}
    for token in SPECIAL_TOKENS:
        try:
            token_id = int(lookup(token))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"evaluation tokenizer is missing special token {token}"
            ) from error
        if token_id in result:
            raise ValueError(
                f"special tokens {result[token_id]} and {token} share token ID {token_id}"
            )
        result[token_id] = token
    return result


def _message_role(message: Any, *, prompt_id: str) -> str:
    if not isinstance(message, dict):
        raise ValueError(f"evaluation prompt {prompt_id!r} message must be an object")
    role = message.get("role")
    if role not in {"system", "user", "agi"}:
        raise ValueError(
            f"evaluation prompt {prompt_id!r} has invalid message role {role!r}"
        )
    return role


def _message_content(message: Any, *, prompt_id: str) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"evaluation prompt {prompt_id!r} message content must be non-empty"
        )
    return content.strip()


def _validate_prompt_roles(
    prompt_id: str,
    messages: Sequence[ChatMessage],
) -> None:
    roles = [message.role for message in messages]
    if roles[0] == "system":
        roles = roles[1:]
    if not roles or roles[0] != "user" or roles[-1] != "user":
        raise ValueError(
            f"evaluation prompt {prompt_id!r} must start and end with a user turn"
        )
    expected = "user"
    for role in roles:
        if role != expected:
            raise ValueError(
                f"evaluation prompt {prompt_id!r} has invalid role sequence"
            )
        expected = "agi" if expected == "user" else "user"


def _write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)
