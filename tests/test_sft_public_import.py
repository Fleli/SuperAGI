import json
import tempfile
import unittest
from pathlib import Path

from superagi.chat.sft_public_import import (
    ImportFilterConfig,
    PublicSftImporter,
    convert_dolly_row,
    convert_no_robots_row,
    convert_ultrachat_row,
    convert_wildchat_row,
    iter_openassistant_conversations,
)
from superagi.ingestion.tokenizer import BpeTokenizer


class PublicSftImportTests(unittest.TestCase):
    def test_converts_no_robots_messages_to_sft_roles(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "What are you?"},
                {"role": "assistant", "content": "I am a small model."},
            ]
        }

        messages = convert_no_robots_row(row)

        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "What are you?"), ("agi", "I am a small model.")],
        )

    def test_converts_dolly_context_into_single_user_turn(self) -> None:
        row = {
            "instruction": "When did it start?",
            "context": "The service started in 2020.",
            "response": "It started in 2020.",
        }

        messages = convert_dolly_row(row)

        self.assertEqual(messages[0].role, "user")
        self.assertIn("Context:\nThe service started in 2020.", messages[0].content)
        self.assertIn("Instruction:\nWhen did it start?", messages[0].content)
        self.assertEqual(messages[1].role, "agi")
        self.assertEqual(messages[1].content, "It started in 2020.")

    def test_converts_wildchat_and_limits_turns(self) -> None:
        row = {
            "conversation": [
                {"role": "user", "content": "One", "language": "English", "toxic": False, "redacted": False},
                {"role": "assistant", "content": "Two", "language": "English", "toxic": False, "redacted": False},
                {"role": "user", "content": "Three", "language": "English", "toxic": False, "redacted": False},
                {"role": "assistant", "content": "Four", "language": "English", "toxic": False, "redacted": False},
            ],
            "language": "English",
            "toxic": False,
            "redacted": False,
        }

        messages = convert_wildchat_row(row, max_turns=2)

        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "One"), ("agi", "Two")],
        )

    def test_converts_ultrachat_messages(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Explain AI."},
                {"role": "assistant", "content": "AI is software that learns patterns."},
            ]
        }

        messages = convert_ultrachat_row(row)

        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "agi")

    def test_reconstructs_openassistant_parent_child_path(self) -> None:
        rows = [
            {
                "message_id": "root",
                "parent_id": None,
                "role": "prompter",
                "text": "What is ML?",
                "lang": "en",
                "deleted": False,
                "review_result": True,
            },
            {
                "message_id": "child",
                "parent_id": "root",
                "role": "assistant",
                "text": "ML learns patterns from data.",
                "lang": "en",
                "deleted": False,
                "review_result": True,
            },
        ]

        conversations = list(iter_openassistant_conversations(rows))

        self.assertEqual(len(conversations), 1)
        self.assertEqual(
            [(message.role, message.content) for message in conversations[0]],
            [("user", "What is ML?"), ("agi", "ML learns patterns from data.")],
        )

    def test_importer_filters_long_repeated_and_duplicate_answers(self) -> None:
        tokenizer = BpeTokenizer.from_texts(
            [
                "<bos><user> short\n<agi> Direct answer about housing prices.<eos>\n",
                "<bos><user> long\n<agi> "
                + " ".join(f"word{index}" for index in range(200))
                + "<eos>\n",
                "<bos><user> repeated\n<agi> echo echo echo echo echo echo echo echo.<eos>\n",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(
            tokenizer=tokenizer,
            filter_config=ImportFilterConfig(
                max_context_tokens=40,
                min_agi_chars=5,
                max_agi_chars=3000,
                max_repeated_five_grams=1,
            ),
        )
        rows = [
            ("first", [{"role": "user", "content": "short"}, {"role": "agi", "content": "Direct answer about housing prices."}]),
            ("duplicate", [{"role": "user", "content": "short again"}, {"role": "agi", "content": "Direct answer about housing prices."}]),
            (
                "long",
                [
                    {"role": "user", "content": "long"},
                    {
                        "role": "agi",
                        "content": " ".join(f"word{index}" for index in range(200)),
                    },
                ],
            ),
            ("repeated", [{"role": "user", "content": "repeated"}, {"role": "agi", "content": "echo echo echo echo echo echo echo echo."}]),
        ]

        imported = importer.import_conversations(rows)

        self.assertEqual(len(imported.examples), 1)
        self.assertEqual(imported.examples[0].source, "first")
        self.assertEqual(imported.stats.accepted, 1)
        self.assertEqual(imported.stats.rejected_by_reason["duplicate_answer"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["too_long"], 1)
        self.assertEqual(imported.stats.rejected_by_reason["repeated_phrase"], 1)

    def test_writes_jsonl_and_metadata(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "<bos><user> Hello\n<agi> Hello back with a direct answer.<eos>\n",
            vocab_size=300,
            min_frequency=1,
        )
        importer = PublicSftImporter(tokenizer=tokenizer)
        imported = importer.import_conversations(
            [
                (
                    "unit",
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "agi", "content": "Hello back with a direct answer."},
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sft.jsonl"
            metadata_path = Path(tmp_dir) / "metadata.json"
            importer.write_import(imported, out_path=out_path, metadata_path=metadata_path)

            lines = out_path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[0])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"], "unit")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(metadata["stats"]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
