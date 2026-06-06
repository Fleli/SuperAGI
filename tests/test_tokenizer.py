import unittest

from superagi.ingestion.tokenizer import BpeTokenizer, CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_encodes_and_decodes_characters_from_training_text(self) -> None:
        tokenizer = CharTokenizer.from_text("banana!")

        encoded = tokenizer.encode("ban!")

        self.assertEqual(tokenizer.decode(encoded), "ban!")
        self.assertEqual(tokenizer.vocab_size, 4)


class BpeTokenizerTests(unittest.TestCase):
    def test_trains_encodes_decodes_and_serializes_bpe(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "machine learning models learn machine learning patterns",
            vocab_size=300,
        )

        encoded = tokenizer.encode("machine learning")
        payload = tokenizer.to_payload()
        restored = BpeTokenizer.from_payload(payload)

        self.assertGreater(len(encoded), 0)
        self.assertLess(len(encoded), len("machine learning"))
        self.assertEqual(restored.decode(encoded), "machine learning")
        self.assertEqual(payload["tokenizer_type"], "bpe")
        self.assertEqual(payload["vocab_size"], tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
