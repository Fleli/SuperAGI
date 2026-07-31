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
STYLE_DIRECTORY = REPOSITORY_ROOT / "data" / "sft" / "styles"
METADATA_PATH = STYLE_DIRECTORY / "styles.metadata.json"

STYLE_SPECS = {
    "calm-precise": {
        "path": STYLE_DIRECTORY / "calm-precise.jsonl",
        "family": "style_calm_precise",
        "domains": {"knowledge": 250, "practical": 250},
        "audit_path": STYLE_DIRECTORY / "calm-precise.audit.json",
    },
    "playful-direct": {
        "path": STYLE_DIRECTORY / "playful-direct.jsonl",
        "family": "style_playful_direct",
        "domains": {"everyday": 250, "knowledge": 250},
        "audit_path": STYLE_DIRECTORY / "playful-direct.audit.json",
    },
}

SPECIAL_TOKENS = ("<unk>", "<pad>", "<bos>", "<eos>", "<user>", "<agi>", "<system>")
FORBIDDEN_ARTIFACTS = (
    "\ufffd",
    "email-magic",
    "striptions",
    "placeholder",
    "lorem ipsum",
    "insert answer",
    "todo:",
    "tbd",
)
FORBIDDEN_FRAMING_LABELS = (
    "blunt version:",
    "in simple terms:",
    "short answer:",
    "bottom line:",
)
IDENTITY_TRAINING_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+am|i['\u2019]m|as)\s+(?:an?\s+)?(?:ai|assistant|language\s+model|"
    r"superagi|chatgpt|model)|"
    r"(?:my|the\s+assistant['\u2019]?s)\s+(?:identity|capabilities|training)|"
    r"(?:what|who)\s+are\s+you|"
    r"(?:what|who)\s+am\s+i"
    r")\b",
    re.IGNORECASE,
)
SYNTHETIC_TAG_RE = re.compile(r"\[[a-z][a-z0-9_-]*-\d+[a-z0-9_-]*\]", re.IGNORECASE)
NUMBERED_TEMPLATE_RE = re.compile(
    r"\b(?:style|example|variant|response|prompt)\s*(?:number|no\.?)?\s*\d+\b",
    re.IGNORECASE,
)


def _read_raw_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StyleSftDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_records = {
            name: _read_raw_records(spec["path"])
            for name, spec in STYLE_SPECS.items()
        }
        cls.records = {
            name: load_sft_records(spec["path"], default_source=spec["family"])
            for name, spec in STYLE_SPECS.items()
        }

    def test_has_exact_size_turn_balance_and_source_domains(self) -> None:
        for name, spec in STYLE_SPECS.items():
            with self.subTest(style=name):
                records = self.records[name]
                self.assertEqual(len(records), 500)
                agi_turn_counts = Counter(
                    sum(message.role == "agi" for message in record.messages)
                    for record in records
                )
                self.assertEqual(agi_turn_counts, Counter({1: 250, 2: 250}))
                self.assertEqual(
                    Counter(record.source for record in records),
                    Counter(
                        {
                            f"{spec['family']}:{domain}": count
                            for domain, count in spec["domains"].items()
                        }
                    ),
                )

    def test_uses_only_strict_nonempty_chat_records(self) -> None:
        for name, raw_records in self.raw_records.items():
            with self.subTest(style=name):
                for raw_record in raw_records:
                    self.assertEqual(set(raw_record), {"messages", "source"})
                    self.assertIsInstance(raw_record["source"], str)
                    messages = raw_record["messages"]
                    self.assertIsInstance(messages, list)
                    roles = [message["role"] for message in messages]
                    self.assertIn(
                        roles,
                        (["user", "agi"], ["user", "agi", "user", "agi"]),
                    )
                    for message in messages:
                        self.assertEqual(set(message), {"role", "content"})
                        self.assertIsInstance(message["content"], str)
                        self.assertEqual(message["content"], message["content"].strip())
                        self.assertTrue(message["content"])

    def test_conversations_and_final_answers_are_unique_within_each_style(self) -> None:
        for name, records in self.records.items():
            with self.subTest(style=name):
                fingerprints = [
                    conversation_fingerprint(record.messages) for record in records
                ]
                final_answers = [
                    canonical_text(record.messages[-1].content) for record in records
                ]
                self.assertEqual(len(set(fingerprints)), 500)
                self.assertTrue(all(final_answers))
                self.assertEqual(len(set(final_answers)), 500)

    def test_styles_have_no_cross_style_exact_duplicates(self) -> None:
        calm_records = self.records["calm-precise"]
        playful_records = self.records["playful-direct"]
        calm_fingerprints = {
            conversation_fingerprint(record.messages) for record in calm_records
        }
        playful_fingerprints = {
            conversation_fingerprint(record.messages) for record in playful_records
        }
        self.assertFalse(calm_fingerprints.intersection(playful_fingerprints))

        calm_answers = {
            canonical_text(record.messages[-1].content) for record in calm_records
        }
        playful_answers = {
            canonical_text(record.messages[-1].content) for record in playful_records
        }
        self.assertFalse(calm_answers.intersection(playful_answers))

    def test_contains_no_identity_training_or_generated_artifacts(self) -> None:
        for name, records in self.records.items():
            with self.subTest(style=name):
                for record in records:
                    for message in record.messages:
                        content = message.content
                        canonical = canonical_text(content)
                        self.assertIsNone(IDENTITY_TRAINING_RE.search(content), content)
                        self.assertIsNone(SYNTHETIC_TAG_RE.search(content), content)
                        self.assertIsNone(NUMBERED_TEMPLATE_RE.search(content), content)
                        for token in SPECIAL_TOKENS:
                            self.assertNotIn(token, content.lower())
                        for artifact in FORBIDDEN_ARTIFACTS:
                            self.assertNotIn(artifact, canonical)

    def test_has_no_repeated_framing_labels_or_dominant_opening(self) -> None:
        for name, records in self.records.items():
            with self.subTest(style=name):
                final_answers = [record.messages[-1].content for record in records]
                for answer in final_answers:
                    canonical = canonical_text(answer)
                    for label in FORBIDDEN_FRAMING_LABELS:
                        self.assertNotIn(label, canonical)

                openings = Counter(
                    " ".join(canonical_text(answer).split()[:3])
                    for answer in final_answers
                )
                top_opening, top_count = openings.most_common(1)[0]
                self.assertLessEqual(
                    top_count / len(final_answers),
                    0.05,
                    f"{name}: opening {top_opening!r} occurs {top_count} times",
                )

    def test_records_are_sorted_by_source_and_conversation_fingerprint(self) -> None:
        for name, records in self.records.items():
            with self.subTest(style=name):
                actual_order = [
                    (record.source, conversation_fingerprint(record.messages))
                    for record in records
                ]
                self.assertEqual(actual_order, sorted(actual_order))

    def test_metadata_is_deterministic_and_matches_both_corpora(self) -> None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        expected_corpora = {}
        for name, spec in sorted(STYLE_SPECS.items()):
            expected_corpora[name] = {
                "conversation_count": 500,
                "multi_turn_conversations": 250,
                "sha256": hashlib.sha256(spec["path"].read_bytes()).hexdigest(),
                "single_turn_conversations": 250,
                "source_domain_counts": spec["domains"],
                "source_family": spec["family"],
            }
        self.assertEqual(
            metadata,
            {
                "corpora": expected_corpora,
                "format": "superagi-style-sft-v1",
                "schema_version": 1,
            },
        )
        self.assertFalse(
            any("time" in key.lower() for key in json.dumps(metadata).split('"'))
        )

    def test_committed_strict_style_audits_pass(self) -> None:
        for name, spec in STYLE_SPECS.items():
            with self.subTest(style=name):
                report = json.loads(spec["audit_path"].read_text(encoding="utf-8"))
                self.assertTrue(report["ok"])
                self.assertEqual(report["mode"], "style")
                self.assertEqual(report["conversation_count"], 500)
                self.assertEqual(report["coverage_categories"]["single_turn"], 250)
                self.assertEqual(report["coverage_categories"]["multi_turn"], 250)
                self.assertEqual(report["identity_share"], 0.0)
                self.assertEqual(
                    report["source_counts"],
                    {spec["family"]: 500},
                )
                self.assertFalse(
                    any(
                        finding["severity"] == "error"
                        for finding in report["findings"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
