import unittest

from superagi.ingestion.tokenizer import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_encodes_and_decodes_characters_from_training_text(self) -> None:
        tokenizer = CharTokenizer.from_text("banana!")

        encoded = tokenizer.encode("ban!")

        self.assertEqual(tokenizer.decode(encoded), "ban!")
        self.assertEqual(tokenizer.vocab_size, 4)


if __name__ == "__main__":
    unittest.main()
