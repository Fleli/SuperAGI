from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from superagi.ingestion.tokenizer import SPECIAL_TOKENS


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_PATHS = (
    ROOT / "data/sft/curated/behavior-identity-reset.jsonl",
    ROOT / "data/sft/curated/behavior-direct-current.jsonl",
)
EVAL_PATH = ROOT / "data/sft/eval_prompts.jsonl"
FORUM_ATTRIBUTION_RE = re.compile(
    r"[-\u2013\u2014]\s*[A-Z][A-Za-z0-9_. -]{1,48}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+"
    r"(?:'\d{2}|\d{4})\s+at\s+\d{1,2}:\d{2}\b"
)


class SftBehaviorDataTests(unittest.TestCase):
    def test_behavior_files_have_fixed_balanced_contract(self) -> None:
        for path in BEHAVIOR_PATHS:
            with self.subTest(path=path.name):
                records = _read_jsonl(path)
                self.assertEqual(len(records), 100)
                turn_counts = {2: 0, 4: 0}
                for record in records:
                    self.assertTrue(record["source"].startswith("curated_behavior:"))
                    messages = record["messages"]
                    self.assertIn(len(messages), turn_counts)
                    turn_counts[len(messages)] += 1
                    self.assertEqual(
                        [message["role"] for message in messages],
                        ["user", "agi"] * (len(messages) // 2),
                    )
                self.assertEqual(turn_counts, {2: 50, 4: 50})

    def test_behavior_answers_are_concise_unique_and_clean(self) -> None:
        records = [record for path in BEHAVIOR_PATHS for record in _read_jsonl(path)]
        answers: list[str] = []
        for record in records:
            for message in record["messages"]:
                if message["role"] != "agi":
                    continue
                answer = message["content"].strip()
                answers.append(answer)
                self.assertTrue(answer)
                self.assertLessEqual(len(answer.split()), 100)
                self.assertNotIn("```", answer)
                self.assertNotIn("\ufffd", answer)
                self.assertIsNone(FORUM_ATTRIBUTION_RE.search(answer))
                for token in SPECIAL_TOKENS:
                    self.assertNotIn(token, answer)
        normalized_answers = {_normalize(answer) for answer in answers}
        self.assertEqual(len(normalized_answers), len(answers))

    def test_behavior_prompts_do_not_copy_fixed_evaluation_prompts(self) -> None:
        eval_prompts = {
            _normalize(message["content"])
            for record in _read_jsonl(EVAL_PATH)
            for message in record["messages"]
            if message["role"] == "user"
        }
        behavior_prompts = {
            _normalize(message["content"])
            for path in BEHAVIOR_PATHS
            for record in _read_jsonl(path)
            for message in record["messages"]
            if message["role"] == "user"
        }
        self.assertFalse(eval_prompts & behavior_prompts)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


if __name__ == "__main__":
    unittest.main()
