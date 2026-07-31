from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from superagi.chat.sft_audit import AuditConfig, audit_sft_corpus
from superagi.ingestion.tokenizer import (
    EOS_TOKEN,
    SPECIAL_TOKENS,
    BpeTokenizer,
    TokenEncoding,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_sft.py"
SPEC = importlib.util.spec_from_file_location("audit_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_sft)


class _ZeroLabelTokenizer:
    def special_token_id(self, token: str) -> int:
        return 0

    def encode_with_offsets(self, text: str) -> TokenEncoding:
        return TokenEncoding(
            ids=(1, 2, 3),
            offsets=((0, 0), (0, 0), (0, 0)),
        )


class _MissingEosTokenizer(_ZeroLabelTokenizer):
    def special_token_id(self, token: str) -> int:
        if token in {EOS_TOKEN, "<system>"}:
            raise ValueError(f"missing {token}")
        return 0


class SftAuditTests(unittest.TestCase):
    def test_fails_for_exact_and_near_duplicate_conversations(self) -> None:
        rows = [
            _conversation("What is AI?", "AI is software that learns patterns from data."),
            _conversation("What is AI?", "AI is software that learns patterns from data."),
            _conversation(
                "Explain machine learning.",
                "AI is software that learns patterns from data and examples.",
            ),
        ]

        report = self._audit(rows, mode="curated")

        self.assertFalse(report.ok)
        self.assertTrue(report.has_error("duplicate_conversation"))
        self.assertTrue(report.has_error("duplicate_agi_answer"))
        self.assertTrue(report.has_error("near_duplicate_agi_answer"))

    def test_fails_for_near_duplicate_prompt_answer_pairs(self) -> None:
        rows = [
            _conversation(
                "Explain how a heat pump works in a small apartment.",
                "A heat pump moves heat from outside air into the apartment using refrigerant and electricity.",
            ),
            _conversation(
                "Explain how a heat pump works in a tiny apartment.",
                "A heat pump moves heat from outside air into the apartment using refrigerant and electricity.",
            ),
        ]

        report = self._audit(rows, mode="style")

        self.assertTrue(report.has_error("near_duplicate_prompt_answer_pair"))

    def test_exact_duplicate_limits_are_configurable(self) -> None:
        rows = [
            _conversation("What is AI?", "AI is software that learns patterns from data."),
            _conversation("What is AI?", "AI is software that learns patterns from data."),
        ]

        report = self._audit(
            rows,
            mode="curated",
            config=AuditConfig(
                max_duplicate_conversations=1,
                max_duplicate_agi_answers=1,
                max_near_duplicate_answers=1,
                max_repeated_ngram_count=1000,
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertTrue(report.ok)

    def test_fails_for_invalid_roles_and_content_artifacts(self) -> None:
        rows = [
            {
                "source": "curated_core:quality",
                "messages": [
                    {"role": "agi", "content": "This starts in the wrong place."},
                    {"role": "user", "content": "Why?"},
                ],
            },
            _conversation("Markers?", "Do not emit <user> or <agi> in content."),
            _conversation("Tag?", "This has a synthetic [study-107 marker."),
            _conversation("Text?", "This contains a replacement character: \ufffd"),
        ]

        report = self._audit(rows, mode="curated")

        self.assertFalse(report.ok)
        self.assertTrue(report.has_error("invalid_role_sequence"))
        self.assertTrue(report.has_error("leaked_control_token"))
        self.assertTrue(report.has_error("synthetic_tag"))
        self.assertTrue(report.has_error("replacement_character"))

    def test_rejects_eos_and_every_other_special_token_in_source_content(self) -> None:
        rows = [
            _conversation(
                f"Special token fixture {index}?",
                f"This answer leaks {token} inside source content.",
                source=f"curated_core:everyday",
            )
            for index, token in enumerate(SPECIAL_TOKENS)
        ]

        report = self._audit(
            rows,
            mode="curated",
            config=AuditConfig(
                example_limit=len(SPECIAL_TOKENS),
                max_repeated_ngram_count=1000,
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        finding = next(
            finding
            for finding in report.findings
            if finding.code == "leaked_control_token"
        )
        self.assertEqual(len(finding.examples), len(SPECIAL_TOKENS))
        rendered_examples = " ".join(finding.examples)
        for token in SPECIAL_TOKENS:
            self.assertIn(token, rendered_examples)

    def test_checkpoint_reports_every_unresolved_special_token_id(self) -> None:
        report = self._audit(
            [_conversation("Question?", "A direct answer.")],
            mode="mixed",
            tokenizer=_MissingEosTokenizer(),
            context_length=32,
        )

        finding = next(
            finding
            for finding in report.findings
            if finding.code == "missing_special_token_ids"
        )
        self.assertIn("<eos>", finding.examples)
        self.assertIn("<system>", finding.examples)

    def test_fails_when_a_common_opening_exceeds_the_gate(self) -> None:
        rows = [
            _conversation(
                f"Question {index}?",
                f"The same opening explains a distinct detail number {index}.",
            )
            for index in range(101)
        ]

        report = self._audit(rows, mode="curated")

        self.assertFalse(report.ok)
        self.assertTrue(report.has_error("opening_frequency"))
        self.assertIn("the same opening", report.repeated_openings)

    def test_fails_when_identity_examples_exceed_curated_limit(self) -> None:
        rows = [
            _conversation(
                f"Question {index}?",
                f"A concrete ordinary answer with a distinct detail {index}.",
            )
            for index in range(96)
        ]
        rows.extend(
            _conversation(
                f"Identity question {index}?",
                f"I am SuperAGI, a small experimental model with limited training {index}.",
            )
            for index in range(4)
        )

        report = self._audit(rows, mode="curated")

        self.assertFalse(report.ok)
        self.assertGreater(report.identity_share, 0.03)
        self.assertTrue(report.has_error("identity_share"))

    def test_identity_share_counts_each_conversation_once(self) -> None:
        rows = [
            _conversation(
                f"Ordinary question {index}?",
                f"An ordinary response with unique detail {index}.",
            )
            for index in range(31)
        ]
        rows.append(
            {
                "source": "curated_core:identity",
                "messages": [
                    {"role": "user", "content": "Who are you?"},
                    {"role": "agi", "content": "I am SuperAGI, a small experimental model."},
                    {"role": "user", "content": "What can you do?"},
                    {"role": "agi", "content": "I can answer compact questions."},
                    {"role": "user", "content": "Anything else?"},
                    {"role": "agi", "content": "I can also explain simple concepts."},
                ],
            }
        )

        report = self._audit(
            rows,
            mode="curated",
            config=AuditConfig(
                max_repeated_ngram_count=1000,
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertAlmostEqual(report.identity_share, 1 / 32)
        self.assertTrue(report.has_error("identity_share"))

    def test_identity_share_uses_curated_and_style_identity_source_domains(self) -> None:
        rows = [
            _conversation(
                "What kind of software is this?",
                "The checkpoint generates a continuation from the text you provide.",
                source="curated_core:identity",
            ),
            _conversation(
                "Describe your role.",
                "Think of this as a compact text generator, not a mysterious oracle.",
                source="style_playful_direct:identity",
            ),
            _conversation(
                "Can you open the live page?",
                "I cannot browse or access current pages.",
                source="openassistant",
            ),
            _conversation(
                "What does a compiler do?",
                "A compiler translates source code into another executable form.",
                source="openassistant",
            ),
        ]

        report = self._audit(
            rows,
            mode="mixed",
            config=AuditConfig(
                identity_share_limit=1.0,
                max_repeated_ngram_count=1000,
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertAlmostEqual(report.identity_share, 3 / 4)

    def test_identity_share_does_not_double_count_source_and_text_detection(self) -> None:
        rows = [
            _conversation(
                "Who are you?",
                "I am SuperAGI, a small experimental language model.",
                source="curated_core:identity",
            ),
            _conversation(
                "What is a checksum?",
                "A checksum is a compact value used to detect changed data.",
                source="curated_core:technology",
            ),
        ]

        report = self._audit(
            rows,
            mode="mixed",
            config=AuditConfig(
                identity_share_limit=1.0,
                max_repeated_ngram_count=1000,
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertAlmostEqual(report.identity_share, 1 / 2)

    def test_topical_relevance_flags_obvious_unrelated_answer(self) -> None:
        report = self._audit(
            [
                _conversation(
                    "How do I boil pasta for dinner?",
                    "A mortgage is a loan secured by real estate property.",
                    source="curated_core:everyday",
                )
            ],
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertFalse(report.ok)
        self.assertTrue(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_mismatch"],
            1,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            0,
        )

    def test_topical_relevance_passes_representative_direct_answer(self) -> None:
        report = self._audit(
            [
                _conversation(
                    "How do I boil pasta for dinner?",
                    "Boil the pasta in salted water until tender, then drain it.",
                    source="curated_core:everyday",
                )
            ],
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertFalse(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            1,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_mismatch"],
            0,
        )

    def test_topical_relevance_flags_stale_answer_after_topic_reset(self) -> None:
        report = self._audit(
            [
                {
                    "source": "curated_core:repair",
                    "messages": [
                        {
                            "role": "user",
                            "content": "How do I boil pasta for dinner?",
                        },
                        {
                            "role": "agi",
                            "content": "Boil the pasta in salted water, then drain it.",
                        },
                        {
                            "role": "user",
                            "content": "Switch topics: how does a mortgage work?",
                        },
                        {
                            "role": "agi",
                            "content": "Boil the pasta until tender and add sauce.",
                        },
                    ],
                }
            ],
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertTrue(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_mismatch"],
            1,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            0,
        )

    def test_topical_relevance_supports_valid_reference_follow_up(self) -> None:
        report = self._audit(
            [
                {
                    "source": "curated_core:everyday",
                    "messages": [
                        {
                            "role": "user",
                            "content": "How do I boil pasta for dinner?",
                        },
                        {
                            "role": "agi",
                            "content": "Boil the pasta in salted water, then drain it.",
                        },
                        {
                            "role": "user",
                            "content": "Can I freeze it after dinner?",
                        },
                        {
                            "role": "agi",
                            "content": "Yes. Freeze the cooked pasta in a sealed container.",
                        },
                    ],
                }
            ],
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertFalse(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            1,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_unscored"],
            0,
        )

    def test_topical_relevance_avoids_false_mismatches_for_valid_context(
        self,
    ) -> None:
        rows = [
            _conversation(
                "I want to save money on groceries.",
                "Plan repeatable meals, compare unit prices, and shop your pantry first.",
                source="curated_core:everyday",
            ),
            _conversation(
                "What do I do if I miss my bus?",
                "Check the next departure in the transit app or an alternate route.",
                source="curated_core:everyday",
            ),
            _conversation(
                "How do I make a long car ride less tiring?",
                "Take breaks, share driving, drink water, and avoid heavy meals.",
                source="curated_core:everyday",
            ),
            {
                "source": "curated_core:health_safety",
                "messages": [
                    {
                        "role": "user",
                        "content": "I have a cough and other respiratory symptoms.",
                    },
                    {
                        "role": "agi",
                        "content": "Rest and monitor the cough and respiratory symptoms.",
                    },
                    {
                        "role": "user",
                        "content": "Can I go to my friend's dinner tonight?",
                    },
                    {
                        "role": "agi",
                        "content": "Skip close social plans while you may be contagious.",
                    },
                ],
            },
            {
                "source": "curated_core:technology",
                "messages": [
                    {
                        "role": "user",
                        "content": "My USB drive is unreadable.",
                    },
                    {
                        "role": "agi",
                        "content": "Do not format the USB drive if its files matter.",
                    },
                    {
                        "role": "user",
                        "content": "It contains my only tax documents.",
                    },
                    {
                        "role": "agi",
                        "content": "Stop writing to the drive and consider professional recovery.",
                    },
                ],
            },
            {
                "source": "curated_core:technology",
                "messages": [
                    {
                        "role": "user",
                        "content": "Should I pay extra for a 4K laptop display?",
                    },
                    {
                        "role": "agi",
                        "content": "A sharper display helps dense text and image work.",
                    },
                    {
                        "role": "user",
                        "content": "I mostly write and travel.",
                    },
                    {
                        "role": "agi",
                        "content": "A comfortable display with strong battery life is practical.",
                    },
                ],
            },
            _conversation(
                "Can I leave my computer plugged in overnight?",
                "Battery charging is managed by the computer, but use a quality charger.",
                source="curated_core:technology",
            ),
        ]

        report = self._audit(
            rows,
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertFalse(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            4,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_mismatch"],
            0,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_unscored"],
            3,
        )

    def test_topical_relevance_does_not_inherit_support_for_ambiguous_follow_up(
        self,
    ) -> None:
        report = self._audit(
            [
                {
                    "source": "curated_core:everyday",
                    "messages": [
                        {
                            "role": "user",
                            "content": "How do I boil pasta for dinner?",
                        },
                        {
                            "role": "agi",
                            "content": "Boil the pasta in salted water, then drain it.",
                        },
                        {
                            "role": "user",
                            "content": "Can you explain that more?",
                        },
                        {
                            "role": "agi",
                            "content": "Keep the water at a rolling boil until the pasta is tender.",
                        },
                    ],
                }
            ],
            mode="curated",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        self.assertFalse(report.has_error("topical_mismatch"))
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            0,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_unscored"],
            1,
        )

    def test_topical_mismatch_is_advisory_outside_curated_core(self) -> None:
        report = self._audit(
            [
                _conversation(
                    "How do I boil pasta for dinner?",
                    "A mortgage is a loan secured by real estate property.",
                    source="style_playful_direct:everyday",
                )
            ],
            mode="style",
            config=AuditConfig(
                required_curated_domains=(),
                require_curated_turn_coverage=False,
            ),
        )

        finding = next(
            finding
            for finding in report.findings
            if finding.code == "topical_mismatch"
        )
        self.assertEqual(finding.severity, "warning")
        self.assertFalse(report.has_error("topical_mismatch"))
        self.assertTrue(report.ok)

    def test_fails_when_a_repeated_ngram_exceeds_the_gate(self) -> None:
        repeated = "one repeated five word phrase appears here"
        rows = [
            _conversation(
                f"Question {index}?",
                f"{repeated} with a different ending {index}.",
            )
            for index in range(4)
        ]

        report = self._audit(
            rows,
            mode="style",
            config=AuditConfig(max_repeated_ngram_count=3),
        )

        self.assertFalse(report.ok)
        self.assertTrue(report.has_error("repeated_ngram"))

    def test_reports_zero_supervised_labels_and_context_overflow(self) -> None:
        zero_label_report = self._audit(
            [_conversation("Question?", "A direct answer.")],
            mode="curated",
            tokenizer=_ZeroLabelTokenizer(),
            context_length=32,
        )
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Question\n<agi> A direct answer with several words.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        overflow_report = self._audit(
            [_conversation("Question?", "A direct answer with several words.")],
            mode="curated",
            tokenizer=tokenizer,
            context_length=4,
        )

        self.assertTrue(zero_label_report.has_error("zero_supervised_labels"))
        self.assertTrue(overflow_report.has_error("context_overflow"))

    def test_fails_for_response_character_and_token_limits(self) -> None:
        rows = [
            _conversation(
                "Question?",
                "This response is deliberately longer than both configured limits.",
            )
        ]
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Question\n<agi> This response is deliberately longer than both configured limits.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )

        report = self._audit(
            rows,
            mode="mixed",
            tokenizer=tokenizer,
            context_length=1024,
            config=AuditConfig(
                max_response_chars=20,
                max_response_tokens=3,
                max_repeated_ngram_count=1000,
            ),
        )

        self.assertTrue(report.has_error("response_char_limit"))
        self.assertTrue(report.has_error("response_token_limit"))
        self.assertIsNotNone(report.token_quantiles)

    def test_fails_for_empty_agi_response(self) -> None:
        report = self._audit(
            [_conversation("Question?", "")],
            mode="curated",
        )

        self.assertTrue(report.has_error("empty_agi_response"))

    def test_fails_when_mixed_curated_sampling_mass_is_outside_bounds(self) -> None:
        rows = [
            _conversation(
                "Curated question?",
                "A curated answer with enough distinct detail.",
                source="curated_core:everyday",
            )
        ]
        rows.extend(
            _conversation(
                f"Public question {index}?",
                f"A public answer with enough distinct detail {index}.",
                source="dolly:unit",
            )
            for index in range(20)
        )

        report = self._audit(
            rows,
            mode="mixed",
            source_weights={"curated_core": 1.0, "dolly": 1.0},
        )

        self.assertFalse(report.ok)
        self.assertLess(report.curated_sampling_mass, 0.15)
        self.assertTrue(report.has_error("curated_sampling_mass"))

    def test_emits_deterministic_quantiles_and_json_payload(self) -> None:
        rows = [
            _conversation("One?", "One concise answer."),
            _conversation("Two?", "Two concise answers have more words."),
            _conversation("Three?", "Three concise answers have even more words here."),
        ]

        report = self._audit(rows, mode="style")
        payload = report.to_json_payload()

        self.assertEqual(report.response_length_quantiles["p50"], 6)
        self.assertEqual(report.word_quantiles["p50"], 7)
        self.assertIsNone(report.token_quantiles)
        self.assertIsNone(payload["token_quantiles"])
        self.assertEqual(payload["ok"], report.ok)
        self.assertEqual(payload["source_counts"], {"curated_core": 3})
        self.assertIn("findings", payload)

    def test_curated_mode_requires_all_domains_and_turn_structures(self) -> None:
        report = self._audit(
            [
                _conversation(
                    "What should I cook?",
                    "Make pasta with vegetables and a simple sauce.",
                    source="curated_core:everyday",
                )
            ],
            mode="curated",
        )

        self.assertTrue(report.has_error("missing_behavioral_coverage"))
        self.assertNotIn("direct_answer", report.coverage_categories)
        self.assertEqual(
            report.coverage_categories["topical_relevance_supported"],
            1,
        )
        self.assertEqual(
            report.coverage_categories["topical_relevance_mismatch"],
            0,
        )
        self.assertEqual(
            report.coverage_categories[
                "corrections_topic_changes_multi_turn_reference"
            ],
            0,
        )
        self.assertEqual(report.coverage_categories["uncertainty_safety"], 0)
        self.assertEqual(report.coverage_categories["identity_boundaries"], 0)

    def test_curated_mode_accepts_complete_domain_and_turn_coverage(self) -> None:
        rows = [
            _conversation(
                f"Question about {domain}?",
                f"A direct answer for the {domain} domain.",
                source=f"curated_core:{domain}",
            )
            for domain in AuditConfig().required_curated_domains
        ]
        rows.append(
            {
                "source": "curated_core:repair",
                "messages": [
                    {"role": "user", "content": "Explain the first topic."},
                    {"role": "agi", "content": "The first topic has a direct explanation."},
                    {"role": "user", "content": "Actually, switch topics."},
                    {"role": "agi", "content": "The new topic has a different explanation."},
                ],
            }
        )

        report = self._audit(rows, mode="curated")

        self.assertFalse(report.has_error("missing_behavioral_coverage"))
        self.assertFalse(report.has_error("invalid_curated_source"))

    def test_style_mode_requires_one_style_family_and_turn_mix_when_large(self) -> None:
        rows = [
            _conversation(
                f"Style question {index}?",
                f"A playful direct response with unique detail {index}.",
                source="style_playful_direct:everyday",
            )
            for index in range(20)
        ]

        report = self._audit(rows, mode="style")

        self.assertTrue(report.has_error("missing_behavioral_coverage"))
        self.assertFalse(report.has_error("invalid_style_source_family"))

    def test_cli_writes_json_report_and_returns_nonzero_for_hard_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "examples.jsonl"
            report_path = Path(tmp_dir) / "audit.json"
            data_path.write_text(
                json.dumps(_conversation("Question?", "Do not leak <agi> tokens.")) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data=str(data_path),
                checkpoint="",
                source_weights="",
                mode="curated",
                report=str(report_path),
            )
            result = audit_sft.run_audit(args)

            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["findings"][0]["code"], "leaked_control_token")

    def test_cli_uses_checkpoint_tokenizer_and_context_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "examples.jsonl"
            data_path.write_text(
                json.dumps(_conversation("Question?", "A direct answer.")) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data=str(data_path),
                checkpoint="checkpoint.pt",
                source_weights="",
                mode="curated",
                report="",
            )
            checkpoint = SimpleNamespace(
                tokenizer=_ZeroLabelTokenizer(),
                config=SimpleNamespace(context_length=2),
            )

            with patch.object(audit_sft, "load_checkpoint", return_value=checkpoint):
                result = audit_sft.run_audit(args)

        self.assertEqual(result, 1)

    def _audit(
        self,
        rows: list[dict[str, object]],
        *,
        mode: str,
        tokenizer: object | None = None,
        context_length: int | None = None,
        source_weights: dict[str, float] | None = None,
        config: AuditConfig | None = None,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "examples.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return audit_sft_corpus(
                [path],
                mode=mode,
                tokenizer=tokenizer,
                context_length=context_length,
                source_weights=source_weights or {},
                config=config or AuditConfig(max_repeated_ngram_count=1000),
            )


def _conversation(
    user: str,
    agi: str,
    *,
    source: str = "curated_core:quality",
) -> dict[str, object]:
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": user},
            {"role": "agi", "content": agi},
        ],
    }


if __name__ == "__main__":
    unittest.main()
