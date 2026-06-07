import unittest

from superagi.ingestion.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    BpeTokenizer,
    CharTokenizer,
)


class CharTokenizerTests(unittest.TestCase):
    def test_encodes_and_decodes_characters_from_training_text(self) -> None:
        tokenizer = CharTokenizer.from_text("banana!")

        encoded = tokenizer.encode("ban!")

        self.assertEqual(tokenizer.decode(encoded), "ban!")
        self.assertEqual(tokenizer.vocab_size, 4)

    def test_encodes_characters_with_offsets(self) -> None:
        tokenizer = CharTokenizer.from_text("banana!")

        encoded = tokenizer.encode_with_offsets("ban!")

        self.assertEqual(tokenizer.decode(encoded.ids), "ban!")
        self.assertEqual(encoded.offsets, ((0, 1), (1, 2), (2, 3), (3, 4)))


class BpeTokenizerTests(unittest.TestCase):
    def test_bpe_reserves_special_tokens_as_atomic_ids(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "machine learning models learn machine learning patterns",
            vocab_size=300,
        )

        special_ids = [tokenizer.special_token_id(token) for token in SPECIAL_TOKENS]

        self.assertEqual(len(set(special_ids)), len(SPECIAL_TOKENS))
        self.assertEqual(tokenizer.encode(EOS_TOKEN), [tokenizer.special_token_id(EOS_TOKEN)])
        self.assertEqual(
            tokenizer.decode([tokenizer.special_token_id(BOS_TOKEN), tokenizer.special_token_id(EOS_TOKEN)]),
            f"{BOS_TOKEN}{EOS_TOKEN}",
        )

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

    def test_encodes_bpe_with_offsets(self) -> None:
        tokenizer = BpeTokenizer.from_text(
            "User: What are you?\nAGI: I am a model.\n",
            vocab_size=300,
            min_frequency=1,
        )

        encoded = tokenizer.encode_with_offsets("AGI: I am a model.")

        self.assertEqual(tokenizer.decode(encoded.ids), "AGI: I am a model.")
        self.assertEqual(len(encoded.ids), len(encoded.offsets))
        self.assertEqual(encoded.offsets[0][0], 0)
        self.assertEqual(encoded.offsets[-1][1], len("AGI: I am a model."))


if __name__ == "__main__":
    unittest.main()
