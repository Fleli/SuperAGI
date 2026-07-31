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
    r"(?<=[.!?;:])\s+|\s+(?:but|however|although|yet)\s+"
    r"|\s+and\s+(?=(?:i|my)\b)",
    re.IGNORECASE,
)
_REALITY_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"\s*,?\s*(?:while|then|though|whereas)\s+"
    r"(?=(?:in\s+reality\b|i\s+(?:actually\s+)?"
    r"(?:work|live|reside|lead|hold|serve|have)\b))",
    re.IGNORECASE,
)
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
_ROLE_PHRASES = (
    ("chief", "executive"),
    ("chief", "executive", "officer"),
)
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
    ("i", "can", "state"),
    ("i", "state"),
    ("i", "attest"),
    ("i", "certify"),
)
_TOPIC_RESET_VALIDATORS = frozenset(
    {
        "accepted_translation",
        "causal_explanation",
        "concept_response",
        "grammar_correction",
        "invitation",
        "numeric_result",
        "ordered_actions",
        "poem",
        "structured_fields",
    }
)
_TASK_ACTION_TERMS = frozenset(
    {
        "answer",
        "calculate",
        "compose",
        "draft",
        "explain",
        "give",
        "help",
        "list",
        "provide",
        "solve",
        "write",
    }
)



@dataclass(frozen=True)
class TopicResetContract:
    validator: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.validator not in _TOPIC_RESET_VALIDATORS:
            raise ValueError(
                f"unsupported topic-reset validator {self.validator!r}"
            )
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValueError("topic-reset contract payload must be non-empty")
        _validate_topic_reset_payload(
            self.validator,
            self.payload,
            prompt_id="<direct>",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        prompt_id: str,
    ) -> "TopicResetContract":
        if not isinstance(value, dict):
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_contract "
                "must be an object"
            )
        validator = value.get("type")
        if not isinstance(validator, str) or validator not in _TOPIC_RESET_VALIDATORS:
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_contract "
                "must define a supported type"
            )
        payload = value.get("payload")
        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                f"evaluation prompt {prompt_id!r} topic_reset_contract "
                "must define a non-empty payload"
            )
        _validate_topic_reset_payload(
            validator,
            payload,
            prompt_id=prompt_id,
        )
        return cls(validator=validator, payload=dict(payload))


@dataclass(frozen=True)
class EvaluationPrompt:
    id: str
    tags: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    collapse_group: str
    max_new_tokens: int
    topic_reset_contract: TopicResetContract | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.collapse_group, str)
            or not self.collapse_group.strip()
        ):
            raise ValueError("collapse_group must be a non-empty string")
        object.__setattr__(self, "collapse_group", self.collapse_group.strip())

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
        if not isinstance(collapse_group, str) or not collapse_group.strip():
            raise ValueError(
                f"evaluation prompt {prompt_id!r} must define non-empty "
                "collapse_group"
            )
        contract_value = value.get("topic_reset_contract")
        contract = (
            TopicResetContract.from_mapping(contract_value, prompt_id=prompt_id)
            if contract_value is not None
            else None
        )
        if "topic-reset" in tags and contract is None:
            raise ValueError(
                f"topic-reset prompt {prompt_id!r} must define topic_reset_contract"
            )
        return cls(
            id=prompt_id.strip(),
            tags=tuple(tag.strip() for tag in tags),
            messages=parsed_messages,
            max_new_tokens=max_new_tokens,
            collapse_group=collapse_group.strip(),
            topic_reset_contract=contract,
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
    collapse_group: str
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
    contract: TopicResetContract,
    request_text: str | None = None,
) -> bool:
    if _contains_task_refusal(response):
        return True
    stale_terms = _string_sequence(contract.payload.get("stale_terms", ()))
    if any(_contains_canonical_term(response, term) for term in stale_terms):
        return True
    if request_text is not None and _response_echoes_request(
        response,
        request_text,
    ):
        return True
    return not _validate_topic_reset_response(
        response,
        validator=contract.validator,
        payload=contract.payload,
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
        contract = prompt.topic_reset_contract
        if contract is None:
            raise ValueError(
                f"topic-reset prompt {prompt.id!r} has no validator contract"
            )
        if topic_reset_failed(
            response,
            contract=contract,
            request_text=prompt.messages[-1].content,
        ):
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
        collapse_groups = {result.collapse_group for result in grouped}
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


def _contains_canonical_term(text: str, term: str) -> bool:
    canonical_term = _canonical_text(term)
    return bool(canonical_term and canonical_term in _canonical_text(text))


def _validate_topic_reset_response(
    response: str,
    *,
    validator: str,
    payload: Mapping[str, Any],
) -> bool:
    validators = {
        "accepted_translation": _validate_translation_response,
        "causal_explanation": _validate_causal_response,
        "concept_response": _validate_concept_response,
        "grammar_correction": _validate_grammar_response,
        "invitation": _validate_invitation_response,
        "numeric_result": _validate_numeric_response,
        "ordered_actions": _validate_ordered_actions_response,
        "poem": _validate_poem_response,
        "structured_fields": _validate_structured_fields_response,
    }
    return validators[validator](response, payload)


def _validate_numeric_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    return any(
        _asserted_literal(response, result)
        for result in _string_sequence(payload["accepted_results"])
    )


def _validate_translation_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    canonical = _canonical_text(response)
    return any(
        _canonical_text(source) in canonical
        and _canonical_text(meaning) in canonical
        for source, meaning in _pair_sequence(payload["accepted_pairs"])
    )


def _validate_invitation_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    return (
        _contains_any_term(response, payload["event_terms"])
        and _contains_any_term(response, payload["date_terms"])
        and _contains_any_term(response, payload["intent_terms"])
        and not _contains_meta_response(response)
    )


def _validate_structured_fields_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    fields = _group_sequence(payload["fields"])
    if not all(_contains_any_term(response, field) for field in fields):
        return False
    separators = (
        response.count("|")
        + response.count(",")
        + response.count(";")
        + response.count("\n")
    )
    return separators >= len(fields) - 1 and not _contains_meta_response(response)


def _validate_concept_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    groups = _group_sequence(payload["concept_groups"])
    minimum = int(payload.get("minimum_groups", len(groups)))
    if sum(_contains_any_term(response, group) for group in groups) < minimum:
        return False
    sentence_count = _sentence_count(response)
    minimum_sentences = int(payload.get("minimum_sentences", 1))
    maximum_sentences = int(payload.get("maximum_sentences", 10_000))
    return (
        minimum_sentences <= sentence_count <= maximum_sentences
        and not _contains_meta_response(response)
    )


def _validate_grammar_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    corrected = any(
        sentence.casefold() in response.casefold()
        for sentence in _string_sequence(payload["accepted_sentences"])
    )
    explained = _contains_any_term(response, payload["explanation_terms"])
    return corrected and explained and not _contains_meta_response(response)


def _validate_causal_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    if not _contains_any_term(response, payload["subject_terms"]):
        return False
    supported = sum(
        _supports_causal_relation(response, relation)
        for relation in _mapping_sequence(payload["relations"])
    )
    return supported >= int(payload["minimum_relations"])


def _validate_ordered_actions_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    groups = _group_sequence(payload["action_groups"])
    supported = sum(_contains_any_term(response, group) for group in groups)
    return (
        supported >= int(payload["minimum_actions"])
        and not _contains_meta_response(response)
    )


def _validate_poem_response(
    response: str,
    payload: Mapping[str, Any],
) -> bool:
    lines = [line for line in response.splitlines() if line.strip()]
    groups = _group_sequence(payload["concept_groups"])
    return (
        len(lines) >= int(payload["minimum_lines"])
        and all(_contains_any_term(response, group) for group in groups)
        and not _contains_meta_response(response)
    )


def _supports_causal_relation(
    response: str,
    relation: Mapping[str, Any],
) -> bool:
    for clause in _causal_clauses(response):
        if (
            _contains_any_term(clause, relation["cause_terms"])
            and _contains_any_term(clause, relation["effect_terms"])
            and not _has_negated_causal_relation(clause)
        ):
            return True
    return False


def _causal_clauses(response: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"(?<=[.!?;])\s+|\n+", response)
        if clause.strip()
    ]


def _has_negated_causal_relation(clause: str) -> bool:
    tokens = _word_tokens(_normalize_contractions(clause))
    negated_relations = (
        ("does", "not", "cause"),
        ("do", "not", "cause"),
        ("did", "not", "cause"),
        ("is", "not", "caused"),
        ("are", "not", "caused"),
        ("not", "due", "to"),
        ("not", "because"),
        ("does", "not", "lead"),
        ("does", "not", "create"),
        ("does", "not", "produce"),
        ("does", "not", "enter"),
        ("does", "not", "let"),
        ("can", "not", "cause"),
        ("can", "not", "be", "causing"),
        ("is", "not", "responsible"),
        ("no", "connection"),
        ("no", "causal", "link"),
    )
    return (
        "unrelated" in tokens
        or any(
            _contains_token_phrase(tokens, phrase)
            for phrase in negated_relations
        )
    )


def _asserted_literal(response: str, literal: str) -> bool:
    pattern = re.compile(
        rf"(?<!\d){re.escape(literal)}(?!\d)",
        re.IGNORECASE,
    )
    normalized = _normalize_contractions(response)
    for clause in _causal_clauses(normalized):
        for match in pattern.finditer(clause):
            prefix = _word_tokens(clause[: match.start()])[-3:]
            suffix = _word_tokens(clause[match.end() :])[:5]
            if any(token in {"not", "never", "no"} for token in prefix):
                continue
            if any(
                token in {"wrong", "incorrect", "false"}
                for token in suffix
            ):
                continue
            return True
    return False


def _contains_any_term(text: str, terms: Any) -> bool:
    return any(
        _contains_canonical_term(text, term)
        for term in _string_sequence(terms)
    )


def _contains_meta_response(response: str) -> bool:
    canonical = _canonical_text(response)
    return any(
        marker in canonical
        for marker in (
            "the prompt",
            "the request",
            "you asked",
            "you requested",
            "i was asked",
            "this is an invitation about",
        )
    )


def _sentence_count(response: str) -> int:
    return len(
        [
            sentence
            for sentence in re.split(r"(?<=[.!?])(?:\s+|$)|\n+", response)
            if sentence.strip()
        ]
    )


def _identity_claim_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    cursor = 0
    for start, end, quoted_text in _quoted_spans(text):
        clauses.extend(
            _split_identity_clauses(
                _normalize_contractions(text[cursor:start])
            )
        )
        if _quote_is_endorsed(text[:start], text[end:]):
            clauses.extend(
                _split_identity_clauses(
                    _normalize_contractions(quoted_text)
                )
            )
        cursor = end
    clauses.extend(
        _split_identity_clauses(
            _normalize_contractions(text[cursor:])
        )
    )
    return clauses


def _quoted_spans(text: str) -> list[tuple[int, int, str]]:
    quote_pairs = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        opener = text[index]
        closer = quote_pairs.get(opener)
        if closer is None or _is_internal_apostrophe(text, index):
            index += 1
            continue
        closing_index = index + 1
        while closing_index < len(text):
            if (
                text[closing_index] == closer
                and not _is_internal_apostrophe(text, closing_index)
            ):
                spans.append(
                    (index, closing_index + 1, text[index + 1 : closing_index])
                )
                index = closing_index + 1
                break
            closing_index += 1
        else:
            index += 1
    return spans


def _is_internal_apostrophe(text: str, index: int) -> bool:
    if text[index] not in {"'", "’"}:
        return False
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
    )


def _split_identity_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for clause in _CLAUSE_BOUNDARY_PATTERN.split(text):
        clauses.extend(
            segment.strip()
            for segment in _REALITY_CLAUSE_BOUNDARY_PATTERN.split(clause)
            if segment.strip()
        )
    return clauses


def _quote_is_endorsed(prefix: str, suffix: str) -> bool:
    prefix_tokens = _word_tokens(_normalize_contractions(prefix))[-12:]
    suffix_tokens = _word_tokens(_normalize_contractions(suffix))[:8]
    denial_phrases = (
        ("is", "false"),
        ("is", "not", "true"),
        ("is", "fictional"),
        ("is", "hypothetical"),
    )
    if any(
        _contains_token_phrase(prefix_tokens, phrase)
        or _contains_token_phrase(suffix_tokens, phrase)
        for phrase in denial_phrases
    ):
        return False
    return any(
        _contains_token_phrase(prefix_tokens, phrase)
        for phrase in _ENDORSEMENT_PHRASES
    )


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
                if _contains_token_phrase(
                    tokens[max(0, index - 4) : index + 1],
                    ("used", "to", "work"),
                ):
                    claims.add("employment_history")
                else:
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

    role_index = _role_claim_index(tokens)
    if role_index is not None:
        role_prefix = tokens[max(0, role_index - 7) : role_index]
        role_suffix = tokens[role_index + 1 : role_index + 3]
        has_service_claim = (
            "i" in role_prefix
            and any(token in {"serve", "served"} for token in role_prefix)
            and "as" in role_prefix
        )
        has_possessive_role_claim = (
            _contains_token_phrase(role_prefix, ("my", "role", "is"))
            or tuple(role_prefix[-3:]) == ("my", "job", "is")
            or (
                bool(role_prefix)
                and role_prefix[-1] == "my"
                and bool(role_suffix)
                and role_suffix[0] == "role"
            )
            or (
                bool(tokens)
                and tokens[0] == "as"
                and role_index <= 4
            )
        )
        has_held_role_claim = (
            "i" in role_prefix
            and any(token in {"hold", "held"} for token in role_prefix)
            and bool(role_suffix)
            and role_suffix[0] == "role"
        )
        has_appointed_role_claim = (
            "me" in role_prefix
            and any(
                token in {"appointed", "elected", "hired"}
                for token in role_prefix
            )
        )
        has_leadership_claim = (
            "i" in role_prefix
            and any(token in {"lead", "led"} for token in role_prefix)
            and "as" in role_prefix
        )
        if (
            (
                _has_first_person_copula(tokens, role_index)
                or has_service_claim
                or has_possessive_role_claim
                or has_held_role_claim
                or has_appointed_role_claim
                or has_leadership_claim
            )
            and not _candidate_is_denied(tokens, role_index)
        ):
            claims.add("role")
            if (
                "was" in role_prefix
                or "served" in role_prefix
                or _contains_token_phrase(role_prefix, ("used", "to"))
            ):
                claims.add("employment_history")

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
        scope_end=min(len(tokens), office_index + 5),
    ):
        claims.add("office")
    return claims


def _role_claim_index(tokens: Sequence[str]) -> int | None:
    direct_index = _first_term_index(tokens, _ROLE_TERMS)
    phrase_indices = [
        index
        for phrase in _ROLE_PHRASES
        for index in range(len(tokens) - len(phrase) + 1)
        if tuple(tokens[index : index + len(phrase)]) == phrase
    ]
    candidates = [
        index
        for index in (direct_index, *phrase_indices)
        if index is not None
    ]
    return min(candidates) if candidates else None


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
    scope_end: int | None = None,
) -> bool:
    prefix = tokens[: predicate_index + 1]
    if _contains_token_phrase(
        prefix,
        ("i", "can", "not", "claim"),
    ):
        return True
    if (
        _contains_token_phrase(prefix, ("it", "is", "not", "true", "that"))
        or _contains_token_phrase(prefix, ("that", "is", "not", "true"))
    ):
        return True
    if scope_start is None:
        nearest_i = _nearest_token(tokens, "i", before=predicate_index)
        scope_start = (
            nearest_i
            if nearest_i is not None
            else max(0, predicate_index - 5)
        )
    scope = tokens[scope_start : scope_end or predicate_index + 1]
    if any(token in {"not", "never", "no", "without"} for token in scope):
        return True
    return _contains_token_phrase(scope, ("can", "not", "claim"))


def _clause_is_hypothetical(tokens: Sequence[str]) -> bool:
    if any(token in {"hypothetical", "hypothetically"} for token in tokens):
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
        (r"\bwon't\b", "will not"),
        (r"\bisn't\b", "is not"),
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


def _validate_topic_reset_payload(
    validator: str,
    payload: Mapping[str, Any],
    *,
    prompt_id: str,
) -> None:
    required_fields = {"stale_terms"} | {
        "accepted_translation": {"accepted_pairs"},
        "causal_explanation": {
            "subject_terms",
            "relations",
            "minimum_relations",
        },
        "concept_response": {"concept_groups"},
        "grammar_correction": {"accepted_sentences", "explanation_terms"},
        "invitation": {"event_terms", "date_terms", "intent_terms"},
        "numeric_result": {"accepted_results"},
        "ordered_actions": {"action_groups", "minimum_actions"},
        "poem": {"concept_groups", "minimum_lines"},
        "structured_fields": {"fields"},
    }[validator]
    missing = sorted(required_fields - payload.keys())
    if missing:
        raise ValueError(
            f"evaluation prompt {prompt_id!r} topic_reset_contract "
            f"{validator} payload is missing {missing}"
        )

    _string_sequence(payload["stale_terms"])
    if validator == "accepted_translation":
        _pair_sequence(payload["accepted_pairs"])
    elif validator == "causal_explanation":
        _string_sequence(payload["subject_terms"])
        relations = _mapping_sequence(payload["relations"])
        for relation in relations:
            _string_sequence(relation.get("cause_terms"))
            _string_sequence(relation.get("effect_terms"))
        minimum = payload["minimum_relations"]
        if (
            not isinstance(minimum, int)
            or not 1 <= minimum <= len(relations)
        ):
            raise ValueError("minimum_relations is outside relation count")
    elif validator == "concept_response":
        groups = _group_sequence(payload["concept_groups"])
        minimum = payload.get("minimum_groups", len(groups))
        if not isinstance(minimum, int) or not 1 <= minimum <= len(groups):
            raise ValueError("minimum_groups is outside concept group count")
        _validate_optional_count(payload, "minimum_sentences", minimum=1)
        _validate_optional_count(payload, "maximum_sentences", minimum=1)
    elif validator == "grammar_correction":
        _string_sequence(payload["accepted_sentences"])
        _string_sequence(payload["explanation_terms"])
    elif validator == "invitation":
        _string_sequence(payload["event_terms"])
        _string_sequence(payload["date_terms"])
        _string_sequence(payload["intent_terms"])
    elif validator == "numeric_result":
        _string_sequence(payload["accepted_results"])
    elif validator == "ordered_actions":
        groups = _group_sequence(payload["action_groups"])
        minimum = payload["minimum_actions"]
        if not isinstance(minimum, int) or not 1 <= minimum <= len(groups):
            raise ValueError("minimum_actions is outside action group count")
    elif validator == "poem":
        _group_sequence(payload["concept_groups"])
        minimum_lines = payload["minimum_lines"]
        if not isinstance(minimum_lines, int) or minimum_lines < 1:
            raise ValueError("minimum_lines must be a positive integer")
    elif validator == "structured_fields":
        _group_sequence(payload["fields"])


def _validate_optional_count(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
) -> None:
    value = payload.get(field)
    if value is not None and (not isinstance(value, int) or value < minimum):
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _string_sequence(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError("expected a non-empty sequence of strings")
    return tuple(item.strip() for item in value)


def _group_sequence(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("expected a non-empty sequence of term groups")
    return tuple(_string_sequence(group) for group in value)


def _pair_sequence(value: Any) -> tuple[tuple[str, str], ...]:
    groups = _group_sequence(value)
    if any(len(group) != 2 for group in groups):
        raise ValueError("translation pairs must contain exactly two strings")
    return tuple((group[0], group[1]) for group in groups)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise ValueError("expected a non-empty sequence of mappings")
    return tuple(value)


def _contains_task_refusal(response: str) -> bool:
    task_objects = {"assignment", "instruction", "question", "request", "task"}
    refusal_phrases = (
        ("can", "not"),
        ("will", "not"),
        ("unable", "to"),
        ("decline", "to"),
        ("refuse", "to"),
    )
    for clause in _causal_clauses(_normalize_contractions(response)):
        tokens = _word_tokens(clause)
        for index in range(len(tokens)):
            if tokens[index] in {"decline", "refuse"}:
                refusal_tail = tokens[index + 1 : index + 6]
                if (
                    any(action in _TASK_ACTION_TERMS for action in refusal_tail)
                    or any(item in task_objects for item in refusal_tail)
                ):
                    return True
            for refusal in refusal_phrases:
                if tuple(tokens[index : index + len(refusal)]) != refusal:
                    continue
                refusal_tail = tokens[
                    index + len(refusal) : index + len(refusal) + 8
                ]
                if any(action in _TASK_ACTION_TERMS for action in refusal_tail):
                    return True
    return False


def _response_echoes_request(
    response: str,
    request_text: str,
) -> bool:
    canonical_response = _canonical_text(response)
    canonical_request = _canonical_text(request_text)
    if not canonical_response:
        return False
    if (
        canonical_response == canonical_request
        or canonical_request in canonical_response
    ):
        return True

    response_tokens = set(_word_tokens(response))
    request_tokens = set(_word_tokens(request_text))
    overlap = response_tokens & request_tokens
    meta_phrases = (
        "the prompt",
        "the request",
        "you asked",
        "you requested",
        "i was asked",
    )
    if (
        any(phrase in response.casefold() for phrase in meta_phrases)
        and len(overlap) >= 2
    ):
        return True
    return False


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
