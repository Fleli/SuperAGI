from __future__ import annotations

import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from superagi.chat.sft import load_sft_records
from superagi.chat.sft_quality import canonical_text, conversation_fingerprint


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIRECTORY = REPOSITORY_ROOT / "data" / "sft" / "curated"
CORE_PATH = CURATED_DIRECTORY / "core.jsonl"
METADATA_PATH = CURATED_DIRECTORY / "core.metadata.json"
AUDIT_PATH = CURATED_DIRECTORY / "core.audit.json"
README_PATH = REPOSITORY_ROOT / "data" / "sft" / "README.md"

EXPECTED_DOMAIN_COUNTS = {
    "civics": 130,
    "creative": 120,
    "everyday": 180,
    "finance": 130,
    "health_safety": 120,
    "identity": 40,
    "relationships": 140,
    "repair": 170,
    "science_math": 140,
    "technology": 160,
    "work_study": 170,
}
EXPECTED_SOURCE_COUNTS = {
    f"curated_core:{domain}": count
    for domain, count in EXPECTED_DOMAIN_COUNTS.items()
}
FORBIDDEN_PHRASES = (
    "use that to choose the next step",
    "start smaller than feels necessary",
    "short answer short answer",
    "striptions",
    "email magic",
)
SPECIAL_TOKENS = ("<unk>", "<pad>", "<bos>", "<eos>", "<user>", "<agi>", "<system>")
SYNTHETIC_ID_RE = re.compile(r"\[[a-z][a-z0-9_-]*-\d+[a-z0-9_-]*\]", re.IGNORECASE)
NUMBERED_PROMPT_RE = re.compile(
    r"\b(?:be blunt about politics|explain finance)\s+\d+\b",
    re.IGNORECASE,
)


def _read_raw_records() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in CORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CuratedSftDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_records = _read_raw_records()
        cls.records = load_sft_records(CORE_PATH, default_source="curated_core")

    def test_has_exact_production_size_turn_balance_and_domain_quotas(self) -> None:
        self.assertEqual(len(self.records), 1_500)

        agi_turn_counts = [
            sum(message.role == "agi" for message in record.messages)
            for record in self.records
        ]
        self.assertEqual(Counter(agi_turn_counts), Counter({1: 750, 2: 750}))
        self.assertEqual(
            Counter(record.source for record in self.records),
            Counter(EXPECTED_SOURCE_COUNTS),
        )

    def test_uses_strict_nonempty_message_patterns(self) -> None:
        for raw_record, record in zip(self.raw_records, self.records, strict=True):
            self.assertEqual(set(raw_record), {"messages", "source"})
            self.assertIn(record.source, EXPECTED_SOURCE_COUNTS)
            self.assertTrue(record.messages)

            roles = [message.role for message in record.messages]
            self.assertIn(roles, (["user", "agi"], ["user", "agi", "user", "agi"]))
            for message in record.messages:
                self.assertTrue(message.content.strip())

    def test_conversations_and_final_answers_are_unique(self) -> None:
        fingerprints = [
            conversation_fingerprint(record.messages) for record in self.records
        ]
        self.assertEqual(len(set(fingerprints)), 1_500)

        final_answers = [
            canonical_text(record.messages[-1].content) for record in self.records
        ]
        self.assertTrue(all(final_answers))
        self.assertEqual(len(set(final_answers)), 1_500)

    def test_identity_examples_stay_within_the_production_cap(self) -> None:
        identity_count = sum(
            record.source == "curated_core:identity" for record in self.records
        )
        self.assertEqual(identity_count, 40)
        self.assertLessEqual(identity_count, 45)
        self.assertLessEqual(identity_count / len(self.records), 0.03)

    def test_contains_no_known_artifacts_or_control_token_leakage(self) -> None:
        for record in self.records:
            for message in record.messages:
                canonical = canonical_text(message.content)
                self.assertFalse(
                    any(phrase in canonical for phrase in FORBIDDEN_PHRASES),
                    message.content,
                )
                self.assertNotIn("\ufffd", message.content)
                self.assertIsNone(SYNTHETIC_ID_RE.search(message.content))
                self.assertIsNone(NUMBERED_PROMPT_RE.search(message.content))
                for token in SPECIAL_TOKENS:
                    self.assertNotIn(token, message.content.lower())

    def test_records_are_sorted_by_source_and_conversation_fingerprint(self) -> None:
        actual_order = [
            (record.source, conversation_fingerprint(record.messages))
            for record in self.records
        ]
        self.assertEqual(actual_order, sorted(actual_order))

    def test_metadata_is_deterministic_and_matches_the_corpus(self) -> None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        expected_metadata = {
            "schema_version": 1,
            "format": "superagi-curated-sft-v1",
            "conversation_count": 1_500,
            "single_turn_conversations": 750,
            "multi_turn_conversations": 750,
            "identity_conversations": 40,
            "source_domain_counts": EXPECTED_DOMAIN_COUNTS,
            "sha256": hashlib.sha256(CORE_PATH.read_bytes()).hexdigest(),
        }
        self.assertEqual(metadata, expected_metadata)
        self.assertFalse(any("time" in key.lower() for key in metadata))

    def test_committed_strict_audit_report_passes_for_this_corpus(self) -> None:
        report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "curated")
        self.assertEqual(report["conversation_count"], 1_500)
        self.assertEqual(report["coverage_categories"]["single_turn"], 750)
        self.assertEqual(report["coverage_categories"]["multi_turn"], 750)
        self.assertEqual(report["coverage_categories"]["identity_boundaries"], 40)
        self.assertFalse(
            any(finding["severity"] == "error" for finding in report["findings"])
        )

    def test_readme_identifies_curated_core_as_the_production_source(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "`curated/core.jsonl` is the production curated SFT source.",
            readme,
        )
        self.assertIn(
            "`stages/broad-mixed.jsonl` is legacy experimental data and is not a "
            "production input.",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
