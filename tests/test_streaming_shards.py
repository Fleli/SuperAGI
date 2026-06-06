import json
import tempfile
import unittest
from pathlib import Path

import torch

from superagi.ingestion.streaming import (
    build_token_shards_from_stream,
    normalize_stream_document_text,
)
from superagi.training.train import TokenShardDataset


class StreamingShardTests(unittest.TestCase):
    def test_streaming_normalizer_does_not_depend_on_raw_corpus_builders(self) -> None:
        text = "  first line\r\n\r\n\r\n second line  "

        normalized = normalize_stream_document_text(text)

        self.assertEqual(normalized, "first line\n\nsecond line")

    def test_builds_bpe_token_shards_without_raw_text_files(self) -> None:
        examples = [
            {"text": "machine learning systems learn useful patterns"},
            {"text": "deep learning models use attention mechanisms"},
            {"text": "optimization adjusts parameters during training"},
            {"text": "validation loss estimates generalization"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_dir = Path(tmp_dir) / "processed"
            result = build_token_shards_from_stream(
                examples_factory=lambda: iter(examples),
                processed_dir=processed_dir,
                max_documents=4,
                tokenizer_sample_documents=2,
                shard_token_count=8,
                validation_token_count=4,
                bpe_vocab_size=300,
                bpe_min_frequency=1,
            )

            vocab = json.loads(
                (processed_dir / "train_vocab.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            validation_tokens = torch.load(processed_dir / "val_tokens.pt")
            train_shard_paths_exist = all(path.exists() for path in result.train_shard_paths)
            raw_text_files = list((processed_dir).glob("**/*.txt"))

            self.assertEqual(vocab["tokenizer_type"], "bpe")
            self.assertEqual(vocab["vocab_size"], result.tokenizer.vocab_size)
            self.assertEqual(vocab["train_tokens"], result.train_tokens)
            self.assertEqual(vocab["validation_tokens"], result.validation_tokens)
            self.assertGreaterEqual(len(result.train_shard_paths), 2)
            self.assertEqual(len(validation_tokens), 4)
            self.assertEqual(manifest["format"], "superagi-token-shards-v1")
            self.assertEqual(manifest["train_tokens"], result.train_tokens)
            self.assertTrue(train_shard_paths_exist)
            self.assertEqual(raw_text_files, [])

    def test_token_shard_dataset_samples_shifted_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_dir = root / "train_shards"
            shard_dir.mkdir()
            torch.save(torch.arange(20, dtype=torch.long), shard_dir / "train-000001.pt")
            manifest_path = shard_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "superagi-token-shards-v1",
                        "train_tokens": 20,
                        "shards": [
                            {"path": "train-000001.pt", "tokens": 20},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = TokenShardDataset.from_manifest(manifest_path)
            input_ids, target_ids = dataset.sample_next_token_batch(
                context_length=4,
                batch_size=3,
                device=torch.device("cpu"),
            )

        self.assertEqual(input_ids.shape, (3, 4))
        self.assertEqual(target_ids.shape, (3, 4))
        self.assertTrue(torch.equal(target_ids[:, :-1], input_ids[:, 1:]))


if __name__ == "__main__":
    unittest.main()
