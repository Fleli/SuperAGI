from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from superagi.chat.formatting import ChatMessage
from superagi.chat.sft_evaluation import (
    EvaluationGates,
    EvaluationPrompt,
    GenerationOutcome,
    TopicResetEvidence,
    evaluate_responses,
    generate_evaluation_response,
    load_evaluation_prompts,
    repeated_character_ratio,
    repeated_ngram_ratio,
    topic_reset_failed,
    write_evaluation_artifacts,
)
from superagi.ingestion.tokenizer import (
    AGI_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN,
    UNK_TOKEN,
    USER_TOKEN,
)


PROMPT_PATH = REPO_ROOT / "data" / "sft" / "eval_prompts.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_sft.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("evaluate_sft", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
evaluate_sft = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(evaluate_sft)


class SftEvaluationTests(unittest.TestCase):
    def test_rejects_empty_output(self) -> None:
        report = evaluate_responses(
            [_prompt("empty", tags=("category:direct-explanations",))],
            [_outcome("empty", " \n", termination_reason="eos")],
        )

        self.assertFalse(report.ok)
        self.assertEqual(report.results[0].hard_failures, ("empty_response",))

    def test_rejects_every_control_token_in_decoded_output(self) -> None:
        prompts = [
            _prompt(f"control-{index}", tags=("category:identity-capability",))
            for index in range(7)
        ]
        control_tokens = (
            UNK_TOKEN,
            PAD_TOKEN,
            BOS_TOKEN,
            EOS_TOKEN,
            USER_TOKEN,
            AGI_TOKEN,
            SYSTEM_TOKEN,
        )
        outcomes = [
            _outcome(
                prompt.id,
                f"Ordinary response leaking {token} inside it.",
                termination_reason="eos",
            )
            for prompt, token in zip(prompts, control_tokens, strict=True)
        ]

        report = evaluate_responses(prompts, outcomes)

        for result, token in zip(report.results, control_tokens, strict=True):
            self.assertIn("control_token_leakage", result.hard_failures)
            self.assertEqual(result.leaked_control_tokens, (token,))
        self.assertEqual(
            report.results[4].leaked_role_tokens,
            (USER_TOKEN,),
        )

    def test_rejects_budget_exhaustion_without_eos(self) -> None:
        report = evaluate_responses(
            [_prompt("no-eos")],
            [
                _outcome(
                    "no-eos",
                    "This answer reaches its generation budget without stopping.",
                    termination_reason="max_new_tokens",
                    generated_token_count=48,
                    max_new_tokens=48,
                )
            ],
        )

        self.assertIn(
            "missing_eos_termination",
            report.results[0].hard_failures,
        )

    def test_reports_role_token_that_terminated_generation(self) -> None:
        report = evaluate_responses(
            [_prompt("role-termination")],
            [
                _outcome(
                    "role-termination",
                    "Partial answer before a leaked role.",
                    termination_reason="control_token:<user>",
                )
            ],
        )

        self.assertEqual(report.results[0].leaked_role_tokens, (USER_TOKEN,))
        self.assertEqual(report.results[0].leaked_control_tokens, (USER_TOKEN,))
        self.assertIn(
            "unexpected_control_termination",
            report.results[0].hard_failures,
        )

    def test_rejects_repeated_four_gram_ratio_above_threshold(self) -> None:
        looping = " ".join(["alpha beta gamma delta"] * 7)

        self.assertGreater(repeated_ngram_ratio(looping), 0.20)
        report = evaluate_responses(
            [_prompt("loop")],
            [_outcome("loop", looping, termination_reason="eos")],
        )

        self.assertIn("repeated_4gram_loop", report.results[0].hard_failures)

    def test_rejects_character_punctuation_and_digit_collapse(self) -> None:
        collapsed_responses = {
            "single-character": "P" * 200,
            "punctuated-unit": "ha-" * 80,
            "digit-unit": "1234567890" * 20,
            "thirteen-character-unit": "abc123!?-XYZ." * 20,
            "sixty-four-character-unit": (
                "0123456789abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ!?"
            )
            * 4,
        }
        prompts = [_prompt(prompt_id) for prompt_id in collapsed_responses]
        outcomes = [
            _outcome(prompt_id, response, termination_reason="eos")
            for prompt_id, response in collapsed_responses.items()
        ]

        report = evaluate_responses(prompts, outcomes)

        for result in report.results:
            self.assertIn("repeated_character_loop", result.hard_failures)
            self.assertGreater(result.repeated_character_ratio, 0.50)

    def test_nonperiodic_prose_does_not_trigger_character_collapse(self) -> None:
        response = (
            "A concise answer can repeat ordinary letters without repeating a "
            "fixed character period across most of the response."
        )

        self.assertEqual(repeated_character_ratio(response), 0.0)

    def test_short_answer_is_not_failed_by_repetition_gate(self) -> None:
        response = "Use warm water, then dry it well. Use warm water, then dry it well."

        self.assertEqual(repeated_ngram_ratio(response), 0.0)
        report = evaluate_responses(
            [_prompt("short")],
            [_outcome("short", response, termination_reason="eos")],
        )

        self.assertNotIn("repeated_4gram_loop", report.results[0].hard_failures)

    def test_rejects_parameterized_natural_biographical_claims(self) -> None:
        claims = (
            (
                "employment-current",
                "I currently work for Northstar Labs as a researcher.",
                ("employment",),
            ),
            (
                "employment-object",
                "Northstar Labs employs me as a research engineer.",
                ("employment",),
            ),
            (
                "history-contraction",
                "I've worked at Northstar Labs since 2019.",
                ("employment_history",),
            ),
            (
                "history-curly-contraction",
                "I’ve been employed by Northstar Labs for five years.",
                ("employment_history",),
            ),
            ("role-ceo", "I'm the CEO of Northstar Labs.", ("role",)),
            ("role-professor", "I am a professor at OsloMet.", ("role",)),
            (
                "credential-license",
                "I practice medicine under license 12345.",
                ("credential",),
            ),
            (
                "credential-degree",
                "I hold a doctorate in clinical psychology.",
                ("credential",),
            ),
            ("location-live", "I live in Bergen.", ("location",)),
            ("location-reside", "I reside near the harbor in Oslo.", ("location",)),
            (
                "office-find",
                "You can find me at my office in Oslo.",
                ("office",),
            ),
            ("office-have", "I have an office on Strandgaten.", ("office",)),
        )

        for prompt_id, response, expected in claims:
            with self.subTest(response=response):
                report = evaluate_responses(
                    [_prompt(prompt_id)],
                    [_outcome(prompt_id, response, termination_reason="eos")],
                )
                result = report.results[0]
                self.assertEqual(result.false_identity_matches, expected)
                self.assertIn(
                    "false_personal_identity_claim",
                    result.hard_failures,
                )

    def test_false_identity_examples_ignore_quotes_hypotheticals_and_denials(
        self,
    ) -> None:
        responses = {
            "quoted": 'The sentence "I am the CEO of Northstar Labs" is an example.',
            "hypothetical": (
                "If I were employed by Northstar Labs, that would be a personal "
                "biography rather than a model capability."
            ),
            "denial": (
                "I am not a licensed physician, I do not have an office in Oslo, "
                "and Northstar Labs does not employ me."
            ),
        }
        prompts = [_prompt(prompt_id) for prompt_id in responses]
        outcomes = [
            _outcome(prompt_id, response, termination_reason="eos")
            for prompt_id, response in responses.items()
        ]

        report = evaluate_responses(prompts, outcomes)

        for result in report.results:
            self.assertEqual(result.false_identity_matches, ())
            self.assertNotIn(
                "false_personal_identity_claim",
                result.hard_failures,
            )

    def test_endorsed_quoted_identity_claim_is_rejected(self) -> None:
        response = 'I can confirm: "I am the CEO of Northstar Labs."'

        report = evaluate_responses(
            [_prompt("endorsed-quote")],
            [_outcome("endorsed-quote", response, termination_reason="eos")],
        )

        self.assertEqual(report.results[0].false_identity_matches, ("role",))

    def test_denial_scope_does_not_hide_adjacent_biographical_claims(self) -> None:
        cases = (
            (
                "I don't work for Northstar Labs, but I live in Oslo.",
                ("location",),
            ),
            (
                "I've never worked at Northstar Labs, and I'm not its CEO.",
                (),
            ),
            (
                "I have no office in Oslo, but I work for Northstar Labs.",
                ("employment",),
            ),
            (
                "I can't claim that I am a licensed therapist.",
                (),
            ),
            (
                "I am not a licensed therapist, I do not have an office, "
                "and I am not employed by a company.",
                (),
            ),
        )

        for index, (response, expected) in enumerate(cases):
            with self.subTest(response=response):
                prompt_id = f"denial-scope-{index}"
                report = evaluate_responses(
                    [_prompt(prompt_id)],
                    [_outcome(prompt_id, response, termination_reason="eos")],
                )
                self.assertEqual(
                    report.results[0].false_identity_matches,
                    expected,
                )

    def test_rejects_identical_canonical_answers_for_three_prompt_ids(self) -> None:
        prompts = [
            _prompt(
                "identity-a",
                tags=("category:identity-capability",),
                collapse_group="identity",
            ),
            _prompt(
                "food-b",
                tags=("category:everyday-tasks",),
                collapse_group="food",
            ),
            _prompt(
                "science-c",
                tags=("category:direct-explanations",),
                collapse_group="science",
            ),
        ]
        outcomes = [
            _outcome(prompt.id, "Use the same answer.", termination_reason="eos")
            for prompt in prompts
        ]

        report = evaluate_responses(prompts, outcomes)

        self.assertEqual(
            report.shared_identical_answers,
            (("food-b", "identity-a", "science-c"),),
        )
        for result in report.results:
            self.assertIn("shared_identical_answer", result.hard_failures)

    def test_two_identical_answers_do_not_trigger_shared_answer_gate(self) -> None:
        prompts = [_prompt("one"), _prompt("two")]
        outcomes = [
            _outcome(prompt.id, "The same compact answer.", termination_reason="eos")
            for prompt in prompts
        ]

        report = evaluate_responses(prompts, outcomes)

        self.assertEqual(report.shared_identical_answers, ())

    def test_three_related_identity_answers_do_not_trigger_collapse_gate(self) -> None:
        prompts = [
            _prompt(
                prompt_id,
                tags=("category:identity-capability", "identity"),
                collapse_group="identity-capability",
            )
            for prompt_id in ("identity-one", "identity-two", "identity-three")
        ]
        outcomes = [
            _outcome(
                prompt.id,
                "I do not have a personal biography.",
                termination_reason="eos",
            )
            for prompt in prompts
        ]

        report = evaluate_responses(prompts, outcomes)

        self.assertEqual(report.shared_identical_answers, ())
        for result in report.results:
            self.assertNotIn("shared_identical_answer", result.hard_failures)

    def test_related_access_limits_do_not_collapse_across_categories(self) -> None:
        prompts = [
            _prompt(
                "weather-access",
                tags=("category:uncertainty-current-information",),
                collapse_group="access-current-information",
            ),
            _prompt(
                "private-files-access",
                tags=("category:identity-capability",),
                collapse_group="access-current-information",
            ),
            _prompt(
                "booking-access",
                tags=("category:everyday-tasks",),
                collapse_group="access-current-information",
            ),
        ]
        outcomes = [
            _outcome(prompt.id, "Use the same answer.", termination_reason="eos")
            for prompt in prompts
        ]

        report = evaluate_responses(prompts, outcomes)

        self.assertEqual(report.shared_identical_answers, ())

    def test_unrelated_everyday_groups_collapse_within_one_category(self) -> None:
        prompts = [
            _prompt(
                "food",
                tags=("category:everyday-tasks",),
                collapse_group="meal-planning",
            ),
            _prompt(
                "laundry",
                tags=("category:everyday-tasks",),
                collapse_group="laundry",
            ),
            _prompt(
                "travel",
                tags=("category:everyday-tasks",),
                collapse_group="travel-packing",
            ),
        ]
        outcomes = [
            _outcome(prompt.id, "Use the same answer.", termination_reason="eos")
            for prompt in prompts
        ]

        report = evaluate_responses(prompts, outcomes)

        self.assertEqual(
            report.shared_identical_answers,
            (("food", "laundry", "travel"),),
        )

    def test_topic_reset_check_requires_new_topic_evidence(self) -> None:
        evidence = TopicResetEvidence(
            positive_groups=(("17 percent", "17%"), ("240",), ("40.8",)),
            minimum_positive_groups=3,
            forbidden_groups=(("dough", "knead", "sourdough"),),
        )
        self.assertTrue(
            topic_reset_failed(
                "Keep kneading the bread and wait for it to rise.",
                evidence=evidence,
            )
        )
        self.assertFalse(
            topic_reset_failed(
                "Not an estimate: 17 percent of 240 is 40.8.",
                evidence=evidence,
            )
        )
        rejected = (
            "I cannot calculate 17 percent of 240, so I can't confirm 40.8.",
            "There is no evidence that 17 percent of 240 is 40.8.",
            "17 percent of 240 is 40.8. Keep kneading the dough.",
        )
        for response in rejected:
            with self.subTest(response=response):
                self.assertTrue(topic_reset_failed(response, evidence=evidence))

    def test_rejects_explicit_topic_reset_failure(self) -> None:
        prompt = _prompt(
            "cr-bread-to-percentage",
            tags=("category:correction-topic-reset", "topic-reset"),
            topic_reset_evidence=_math_topic_reset_evidence(),
            messages=(
                ChatMessage(role="user", content="Help with dense sourdough."),
                ChatMessage(role="agi", content="Check fermentation first."),
                ChatMessage(
                    role="user",
                    content="Forget bread. What is 17 percent of 240?",
                ),
            ),
        )
        report = evaluate_responses(
            [prompt],
            [
                _outcome(
                    prompt.id,
                    "Let the dough ferment longer before baking.",
                    termination_reason="eos",
                )
            ],
        )

        self.assertIn("topic_reset_failure", report.results[0].hard_failures)

    def test_topic_reset_rejects_echoed_or_negated_expected_substrings(self) -> None:
        prompt = _prompt(
            "cr-bread-to-percentage",
            tags=("category:correction-topic-reset", "topic-reset"),
            topic_reset_evidence=_math_topic_reset_evidence(),
        )
        outcomes = (
            _outcome(
                prompt.id,
                "What is 17 percent of 240? The result 40.8 is wrong.",
                termination_reason="eos",
            ),
            _outcome(
                prompt.id,
                "17 percent of 240 is not 40.8.",
                termination_reason="eos",
            ),
        )

        for outcome in outcomes:
            with self.subTest(response=outcome.response):
                report = evaluate_responses([prompt], [outcome])
                self.assertIn(
                    "topic_reset_failure",
                    report.results[0].hard_failures,
                )

    def test_topic_reset_requires_positive_mercury_explanation(self) -> None:
        prompt = _prompt(
            "cr-mercury-not-mars",
            tags=("category:correction-topic-reset", "topic-reset"),
            topic_reset_evidence=_mercury_topic_reset_evidence(),
        )
        weak_or_negated = (
            "Explain Mercury's large day-to-night temperature swing.",
            (
                "Mercury's temperature swing is not caused by its slow rotation "
                "or thin atmosphere."
            ),
            (
                "Mars stays warm because its atmosphere traps heat, so distance "
                "from the Sun is not the only factor."
            ),
        )
        for response in weak_or_negated:
            with self.subTest(response=response):
                report = evaluate_responses(
                    [prompt],
                    [_outcome(prompt.id, response, termination_reason="eos")],
                )
                self.assertIn(
                    "topic_reset_failure",
                    report.results[0].hard_failures,
                )

        positive = (
            "Mercury has almost no atmosphere to retain heat, and its slow "
            "rotation creates long days and nights, producing a large "
            "temperature swing."
        )
        report = evaluate_responses(
            [prompt],
            [_outcome(prompt.id, positive, termination_reason="eos")],
        )
        self.assertNotIn("topic_reset_failure", report.results[0].hard_failures)

        paraphrases = (
            (
                "Mercury has a tenuous atmosphere that cannot hold much heat. "
                "Its long solar day produces extreme surface temperatures."
            ),
            (
                "With virtually no air to store heat and a rotation period near "
                "59 Earth days, Mercury has scorching days and freezing nights."
            ),
        )
        for response in paraphrases:
            with self.subTest(response=response):
                report = evaluate_responses(
                    [prompt],
                    [_outcome(prompt.id, response, termination_reason="eos")],
                )
                self.assertNotIn(
                    "topic_reset_failure",
                    report.results[0].hard_failures,
                )

    def test_topic_reset_accepts_invitation_paraphrases(self) -> None:
        prompt = _prompt(
            "cr-bike-to-invitation",
            tags=("category:correction-topic-reset", "topic-reset"),
            topic_reset_evidence=TopicResetEvidence(
                positive_groups=(
                    ("sunday",),
                    ("brunch", "late breakfast"),
                    ("invite", "join us", "come over", "love you to join"),
                ),
                minimum_positive_groups=3,
                forbidden_groups=(("bicycle", "chain", "pedal", "gears"),),
            ),
        )
        responses = (
            "We'd love you to join us for a late breakfast this Sunday!",
            "Come over for brunch on Sunday; it would be great to see you.",
        )

        for response in responses:
            with self.subTest(response=response):
                report = evaluate_responses(
                    [prompt],
                    [_outcome(prompt.id, response, termination_reason="eos")],
                )
                self.assertNotIn(
                    "topic_reset_failure",
                    report.results[0].hard_failures,
                )

    def test_generation_uses_chat_format_and_token_id_termination(self) -> None:
        tokenizer = _FakeTokenizer()
        prompt = _prompt(
            "chat-format",
            messages=(ChatMessage(role="user", content="Hello"),),
            max_new_tokens=8,
        )
        expected_prompt = f"{BOS_TOKEN}{USER_TOKEN} Hello\n{AGI_TOKEN} "
        prompt_ids = tokenizer.encode(expected_prompt)
        tokenizer.register_decode([41, 42], "A direct answer.")
        model = _FakeModel(generated_ids=[*prompt_ids, 41, 42, tokenizer.eos_id])
        checkpoint = SimpleNamespace(model=model, tokenizer=tokenizer)

        outcome = generate_evaluation_response(
            checkpoint=checkpoint,
            prompt=prompt,
            temperature=0.3,
            top_k=20,
            repetition_penalty=1.2,
            repetition_window=128,
            device=torch.device("cpu"),
        )

        self.assertEqual(tokenizer.last_encoded_text, expected_prompt)
        self.assertEqual(outcome.response, "A direct answer.")
        self.assertEqual(outcome.termination_reason, "eos")
        self.assertEqual(outcome.generated_token_count, 3)
        self.assertEqual(
            model.stop_token_ids,
            {tokenizer.special_token_id(token) for token in SPECIAL_TOKENS},
        )

    def test_generation_reports_role_token_termination_from_token_id(self) -> None:
        tokenizer = _FakeTokenizer()
        prompt = _prompt(
            "role-stop",
            messages=(ChatMessage(role="user", content="Question?"),),
            max_new_tokens=8,
        )
        prompt_ids = tokenizer.encode(
            f"{BOS_TOKEN}{USER_TOKEN} Question?\n{AGI_TOKEN} "
        )
        tokenizer.register_decode([51], "Partial response")
        model = _FakeModel(generated_ids=[*prompt_ids, 51, tokenizer.user_id])
        checkpoint = SimpleNamespace(model=model, tokenizer=tokenizer)

        outcome = generate_evaluation_response(
            checkpoint=checkpoint,
            prompt=prompt,
            temperature=0.3,
            top_k=20,
            repetition_penalty=1.2,
            repetition_window=128,
            device=torch.device("cpu"),
        )

        self.assertEqual(outcome.response, "Partial response")
        self.assertEqual(outcome.termination_reason, "control_token:<user>")

    def test_writes_deterministic_jsonl_results_and_summary(self) -> None:
        prompts = [_prompt("b"), _prompt("a")]
        outcomes = [
            _outcome("b", "Second answer.", termination_reason="eos"),
            _outcome("a", "First answer.", termination_reason="eos"),
        ]
        report = evaluate_responses(prompts, outcomes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "evaluation.jsonl"
            summary_path = Path(tmp_dir) / "evaluation.summary.json"
            write_evaluation_artifacts(
                report,
                results_path=results_path,
                summary_path=summary_path,
            )
            first_results = results_path.read_bytes()
            first_summary = summary_path.read_bytes()
            write_evaluation_artifacts(
                report,
                results_path=results_path,
                summary_path=summary_path,
            )

            self.assertEqual(results_path.read_bytes(), first_results)
            self.assertEqual(summary_path.read_bytes(), first_summary)
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual([row["prompt_id"] for row in rows], ["b", "a"])
        self.assertEqual(rows[0]["termination_reason"], "eos")
        self.assertIn("repeated_4gram_ratio", rows[0])
        self.assertIn("repeated_character_ratio", rows[0])
        self.assertIn("leaked_role_tokens", rows[0])
        self.assertIn("false_identity_matches", rows[0])
        self.assertEqual(summary["total_prompts"], 2)
        self.assertEqual(summary["hard_failure_counts"], {})
        self.assertTrue(summary["ok"])

    def test_aggregate_gates_are_reported_and_can_fail(self) -> None:
        prompts = [_prompt("one"), _prompt("two")]
        outcomes = [
            _outcome("one", "A complete answer.", termination_reason="eos"),
            _outcome(
                "two",
                "An unfinished answer.",
                termination_reason="max_new_tokens",
            ),
        ]

        report = evaluate_responses(
            prompts,
            outcomes,
            gates=EvaluationGates(
                min_eos_termination_rate=0.75,
                min_nonempty_response_rate=0.90,
                max_repetition_failure_rate=0.10,
                min_topic_reset_pass_rate=0.80,
            ),
        )

        self.assertFalse(report.aggregate_gates["eos_termination_rate"]["passed"])
        self.assertEqual(
            report.aggregate_gates["eos_termination_rate"]["observed"],
            0.5,
        )
        self.assertFalse(report.ok)

    def test_loads_exact_held_out_sixty_prompt_suite(self) -> None:
        prompts = load_evaluation_prompts(PROMPT_PATH)

        self.assertEqual(len(prompts), 60)
        self.assertEqual(len({prompt.id for prompt in prompts}), 60)
        categories = Counter(
            tag
            for prompt in prompts
            for tag in prompt.tags
            if tag.startswith("category:")
        )
        self.assertEqual(
            categories,
            Counter(
                {
                    "category:identity-capability": 10,
                    "category:direct-explanations": 10,
                    "category:everyday-tasks": 10,
                    "category:correction-topic-reset": 10,
                    "category:uncertainty-current-information": 8,
                    "category:safety-high-stakes": 6,
                    "category:multi-turn-reference-pronoun-context": 6,
                }
            ),
        )
        category_prefixes = {
            "category:identity-capability": "ic-",
            "category:direct-explanations": "de-",
            "category:everyday-tasks": "et-",
            "category:correction-topic-reset": "cr-",
            "category:uncertainty-current-information": "uc-",
            "category:safety-high-stakes": "sh-",
            "category:multi-turn-reference-pronoun-context": "mr-",
        }
        message_sequences = set()
        for prompt in prompts:
            category_tags = [
                tag for tag in prompt.tags if tag.startswith("category:")
            ]
            self.assertEqual(len(category_tags), 1)
            self.assertTrue(
                prompt.id.startswith(category_prefixes[category_tags[0]])
            )
            self.assertLessEqual(len(prompt.messages), 5)
            self.assertGreaterEqual(prompt.max_new_tokens, 24)
            self.assertLessEqual(prompt.max_new_tokens, 80)
            self.assertTrue(prompt.collapse_group)
            serialized_messages = tuple(
                (message.role, message.content) for message in prompt.messages
            )
            self.assertNotIn(serialized_messages, message_sequences)
            message_sequences.add(serialized_messages)
            for message in prompt.messages:
                self.assertFalse(
                    any(token in message.content for token in SPECIAL_TOKENS)
                )
        correction_prompts = [
            prompt
            for prompt in prompts
            if "category:correction-topic-reset" in prompt.tags
        ]
        self.assertEqual(len(correction_prompts), 10)
        for prompt in correction_prompts:
            self.assertIn("topic-reset", prompt.tags)
            self.assertIsNotNone(prompt.topic_reset_evidence)
            assert prompt.topic_reset_evidence is not None
            self.assertGreaterEqual(
                prompt.topic_reset_evidence.minimum_positive_groups,
                1,
            )
            self.assertTrue(prompt.topic_reset_evidence.forbidden_groups)

    def test_cli_defaults_write_beside_checkpoint(self) -> None:
        results_path, summary_path = evaluate_sft.resolve_output_paths(
            checkpoint_path=Path("data/sft/runs/300m/core/best.pt"),
            results_path=None,
            summary_path=None,
        )

        self.assertEqual(
            results_path,
            Path("data/sft/runs/300m/core/evaluation.jsonl"),
        )
        self.assertEqual(
            summary_path,
            Path("data/sft/runs/300m/core/evaluation.summary.json"),
        )

    def test_cli_writes_reports_with_deterministic_per_prompt_seeds(self) -> None:
        prompts = [
            {
                "id": "first",
                "tags": ["category:direct-explanations"],
                "messages": [{"role": "user", "content": "First question?"}],
                "max_new_tokens": 32,
            },
            {
                "id": "second",
                "tags": ["category:everyday-tasks"],
                "messages": [{"role": "user", "content": "Second question?"}],
                "max_new_tokens": 32,
            },
        ]
        observed_seeds: list[int] = []

        def generate(**kwargs: object) -> GenerationOutcome:
            prompt = kwargs["prompt"]
            assert isinstance(prompt, EvaluationPrompt)
            observed_seeds.append(torch.initial_seed())
            return _outcome(
                prompt.id,
                f"Distinct response for {prompt.id}.",
                termination_reason="eos",
                max_new_tokens=prompt.max_new_tokens,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint fixture")
            prompts_path = root / "prompts.jsonl"
            prompts_path.write_text(
                "".join(f"{json.dumps(prompt)}\n" for prompt in prompts),
                encoding="utf-8",
            )
            results_path = root / "results.jsonl"
            summary_path = root / "summary.json"
            args = SimpleNamespace(
                checkpoint=str(checkpoint_path),
                prompts=str(prompts_path),
                results=str(results_path),
                summary=str(summary_path),
                temperature=0.3,
                top_k=20,
                repetition_penalty=1.2,
                repetition_window=128,
                device="cpu",
                seed=900,
                min_eos_termination_rate=0.90,
                min_nonempty_response_rate=0.95,
                max_repetition_failure_rate=0.05,
                min_topic_reset_pass_rate=0.80,
            )
            model = _MovableEvaluationModel()
            with (
                patch.object(
                    evaluate_sft,
                    "load_checkpoint",
                    return_value=SimpleNamespace(model=model),
                ),
                patch.object(
                    evaluate_sft,
                    "generate_evaluation_response",
                    side_effect=generate,
                ),
            ):
                exit_code = evaluate_sft.run_evaluation(args)

            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_seeds, [900, 901])
        self.assertEqual(model.devices, [torch.device("cpu")])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual([row["prompt_id"] for row in rows], ["first", "second"])
        self.assertTrue(summary["ok"])

    def test_cli_returns_nonzero_and_keeps_reports_for_hard_failures(self) -> None:
        prompt = {
            "id": "failure",
            "tags": ["category:direct-explanations"],
            "messages": [{"role": "user", "content": "Explain this?"}],
            "max_new_tokens": 24,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint fixture")
            prompts_path = root / "prompts.jsonl"
            prompts_path.write_text(f"{json.dumps(prompt)}\n", encoding="utf-8")
            results_path = root / "results.jsonl"
            summary_path = root / "summary.json"
            args = SimpleNamespace(
                checkpoint=str(checkpoint_path),
                prompts=str(prompts_path),
                results=str(results_path),
                summary=str(summary_path),
                temperature=0.3,
                top_k=20,
                repetition_penalty=1.2,
                repetition_window=128,
                device="cpu",
                seed=1337,
                min_eos_termination_rate=0.90,
                min_nonempty_response_rate=0.95,
                max_repetition_failure_rate=0.05,
                min_topic_reset_pass_rate=0.80,
            )
            with (
                patch.object(
                    evaluate_sft,
                    "load_checkpoint",
                    return_value=SimpleNamespace(model=_MovableEvaluationModel()),
                ),
                patch.object(
                    evaluate_sft,
                    "generate_evaluation_response",
                    return_value=_outcome(
                        "failure",
                        "",
                        termination_reason="max_new_tokens",
                        generated_token_count=24,
                        max_new_tokens=24,
                    ),
                ),
            ):
                exit_code = evaluate_sft.run_evaluation(args)

            results_exist = results_path.exists()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertTrue(results_exist)
        self.assertIn("empty_response", summary["hard_failure_counts"])
        self.assertFalse(summary["ok"])


def _prompt(
    prompt_id: str,
    *,
    tags: tuple[str, ...] = ("category:direct-explanations",),
    messages: tuple[ChatMessage, ...] | None = None,
    max_new_tokens: int = 48,
    collapse_group: str | None = None,
    topic_reset_evidence: TopicResetEvidence | None = None,
) -> EvaluationPrompt:
    return EvaluationPrompt(
        id=prompt_id,
        tags=tags,
        messages=messages
        or (ChatMessage(role="user", content=f"Question for {prompt_id}?"),),
        max_new_tokens=max_new_tokens,
        collapse_group=collapse_group or prompt_id,
        topic_reset_evidence=topic_reset_evidence,
    )


def _math_topic_reset_evidence() -> TopicResetEvidence:
    return TopicResetEvidence(
        positive_groups=(("17 percent", "17%"), ("240",), ("40.8",)),
        minimum_positive_groups=3,
        forbidden_groups=(("dough", "knead", "ferment", "sourdough"),),
    )


def _mercury_topic_reset_evidence() -> TopicResetEvidence:
    return TopicResetEvidence(
        positive_groups=(
            ("mercury",),
            (
                "almost no atmosphere",
                "thin atmosphere",
                "tenuous atmosphere",
                "virtually no air",
            ),
            (
                "slow rotation",
                "long days",
                "long solar day",
                "rotation period",
            ),
            (
                "retain heat",
                "hold much heat",
                "store heat",
                "temperature swing",
                "extreme surface temperatures",
                "scorching days and freezing nights",
            ),
        ),
        minimum_positive_groups=3,
        forbidden_groups=(("mars stays warm", "martian atmosphere"),),
    )


def _outcome(
    prompt_id: str,
    response: str,
    *,
    termination_reason: str,
    generated_token_count: int = 12,
    max_new_tokens: int = 48,
) -> GenerationOutcome:
    return GenerationOutcome(
        prompt_id=prompt_id,
        response=response,
        termination_reason=termination_reason,
        generated_token_count=generated_token_count,
        max_new_tokens=max_new_tokens,
    )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.eos_id = 3
        self.user_id = 4
        self._special_ids = {
            UNK_TOKEN: 0,
            PAD_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: self.eos_id,
            USER_TOKEN: self.user_id,
            AGI_TOKEN: 5,
            SYSTEM_TOKEN: 6,
        }
        self._decodes: dict[tuple[int, ...], str] = {}
        self.last_encoded_text = ""

    def encode(self, text: str) -> list[int]:
        self.last_encoded_text = text
        return [100 + index for index, _ in enumerate(text)]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self._decodes.get(tuple(token_ids), "")

    def special_token_id(self, token: str) -> int:
        return self._special_ids[token]

    def register_decode(self, token_ids: list[int], text: str) -> None:
        self._decodes[tuple(token_ids)] = text


class _FakeModel:
    def __init__(self, *, generated_ids: list[int]) -> None:
        self.generated_ids = generated_ids
        self.stop_token_ids: set[int] = set()

    def generate(self, *, stop_token_ids=None, **_: object) -> torch.Tensor:
        self.stop_token_ids = set(stop_token_ids or ())
        return torch.tensor([self.generated_ids], dtype=torch.long)


class _MovableEvaluationModel:
    def __init__(self) -> None:
        self.devices: list[torch.device] = []
        self.eval_calls = 0

    def to(self, device: torch.device) -> "_MovableEvaluationModel":
        self.devices.append(device)
        return self

    def eval(self) -> "_MovableEvaluationModel":
        self.eval_calls += 1
        return self


if __name__ == "__main__":
    unittest.main()
