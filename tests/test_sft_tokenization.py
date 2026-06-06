import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from superagi.chat.sft import (
    IGNORE_INDEX,
    load_sft_jsonl,
    tokenize_sft_messages,
)
from superagi.ingestion.tokenizer import BpeTokenizer, CharTokenizer


class SftTokenizationTests(unittest.TestCase):
    def test_tokenizes_chat_with_assistant_only_labels_using_char_offsets(self) -> None:
        messages = [
            {"role": "user", "content": "What are you?"},
            {"role": "agi", "content": "I am AGI."},
        ]
        text = "User: What are you?\nAGI: I am AGI.\n"
        tokenizer = CharTokenizer.from_text(text)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(tokenized.text, text)
        self.assertEqual(tokenizer.decode(supervised_ids), "I am AGI.")
        self.assertEqual(tokenized.supervised_token_count, len("I am AGI."))
        self.assertTrue(
            all(
                token_id == IGNORE_INDEX
                for token_id in tokenized.target_ids[: text.index("I am AGI.") - 1]
            )
        )

    def test_tokenizes_chat_with_assistant_only_labels_using_bpe_offsets(self) -> None:
        messages = [
            {"role": "user", "content": "Explain ML."},
            {"role": "agi", "content": "ML learns patterns from data."},
        ]
        text = "User: Explain ML.\nAGI: ML learns patterns from data.\n"
        tokenizer = BpeTokenizer.from_text(text, vocab_size=300, min_frequency=1)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(
            tokenizer.decode(supervised_ids).strip(),
            "ML learns patterns from data.",
        )
        self.assertGreater(tokenized.supervised_token_count, 0)

    def test_tokenizes_all_agi_turns_in_multi_turn_chat(self) -> None:
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "agi", "content": "AI is software."},
            {"role": "user", "content": "And ML?"},
            {"role": "agi", "content": "ML learns from data."},
        ]
        text = (
            "User: What is AI?\n"
            "AGI: AI is software.\n"
            "User: And ML?\n"
            "AGI: ML learns from data.\n"
        )
        tokenizer = CharTokenizer.from_text(text)

        tokenized = tokenize_sft_messages(messages, tokenizer)

        supervised_ids = [
            token_id
            for token_id in tokenized.target_ids
            if token_id != IGNORE_INDEX
        ]
        self.assertEqual(
            tokenizer.decode(supervised_ids),
            "AI is software.ML learns from data.",
        )

    def test_rejects_conversation_without_agi_labels(self) -> None:
        tokenizer = CharTokenizer.from_text("User: Hello\n")

        with self.assertRaisesRegex(ValueError, "no AGI response tokens"):
            tokenize_sft_messages([{"role": "user", "content": "Hello"}], tokenizer)

    def test_loads_sft_jsonl_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sft.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What are you?"},
                            {"role": "agi", "content": "A small model."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            conversations = load_sft_jsonl(path)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0][0].role, "user")
        self.assertEqual(conversations[0][1].role, "agi")

    def test_tracked_seed_sft_corpus_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "sft" / "seed.jsonl"

        conversations = load_sft_jsonl(path)

        self.assertGreaterEqual(len(conversations), 200)
        self.assertTrue(
            all(
                any(message.role == "agi" for message in conversation)
                for conversation in conversations
            )
        )

    def test_tracked_seed_sft_corpus_has_everyday_chat_coverage(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "sft" / "seed.jsonl"

        conversations = load_sft_jsonl(path)
        single_turn_count = sum(1 for conversation in conversations if len(conversation) == 2)
        multi_turn_count = sum(1 for conversation in conversations if len(conversation) > 2)
        agi_answer_lengths = [
            len(message.content)
            for conversation in conversations
            for message in conversation
            if message.role == "agi"
        ]
        agi_answers = [
            message.content
            for conversation in conversations
            for message in conversation
            if message.role == "agi"
        ]
        answer_counts = Counter(agi_answers)

        self.assertGreaterEqual(len(conversations), 20_700)
        self.assertGreaterEqual(single_turn_count, 10_000)
        self.assertGreaterEqual(multi_turn_count, 10_000)
        self.assertLessEqual(max(agi_answer_lengths), 180)
        self.assertGreaterEqual(len(answer_counts), 0.95 * len(agi_answers))
        self.assertLessEqual(max(answer_counts.values()), 3)


if __name__ == "__main__":
    unittest.main()
