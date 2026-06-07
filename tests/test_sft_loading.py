import json
import tempfile
import unittest
from pathlib import Path

from superagi.chat.sft import load_sft_jsonl, load_sft_records


class SftLoadingTests(unittest.TestCase):
    def test_loads_sft_records_with_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source": "wildchat:42",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "agi", "content": "Hi."},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_sft_records(path, default_source="fallback")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "wildchat:42")
        self.assertEqual(records[0].messages[0].role, "user")

    def test_loads_sft_records_with_default_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "anchor.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What are you?"},
                            {"role": "agi", "content": "I am SuperAGI."},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_sft_records(path, default_source="anchor")
            conversations = load_sft_jsonl(path)

        self.assertEqual(records[0].source, "anchor")
        self.assertEqual(conversations[0], records[0].messages)


if __name__ == "__main__":
    unittest.main()
