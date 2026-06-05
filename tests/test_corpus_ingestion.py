import json
import tempfile
import unittest
from pathlib import Path

import torch

from superagi.ingestion.corpus import ingest_raw_corpus, read_raw_corpus


class CorpusIngestionTests(unittest.TestCase):
    def test_reads_text_files_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir)
            (raw_dir / "b.txt").write_text("second", encoding="utf-8")
            (raw_dir / "a.txt").write_text("first", encoding="utf-8")
            (raw_dir / "skip.bin").write_text("ignored", encoding="utf-8")

            corpus = read_raw_corpus(raw_dir)

        self.assertEqual(corpus.text, "first\nsecond")
        self.assertEqual([path.name for path in corpus.source_paths], ["a.txt", "b.txt"])

    def test_ingests_raw_corpus_to_processed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            raw_dir.mkdir()
            (raw_dir / "sample.txt").write_text("aba", encoding="utf-8")

            artifact = ingest_raw_corpus(raw_dir, processed_dir, artifact_name="train")

            token_file = processed_dir / "train_tokens.pt"
            vocab_file = processed_dir / "train_vocab.json"
            saved_tokens = torch.load(token_file)
            saved_vocab = json.loads(vocab_file.read_text(encoding="utf-8"))

        self.assertEqual(artifact.token_ids, [0, 1, 0])
        self.assertEqual(saved_tokens.tolist(), [0, 1, 0])
        self.assertEqual(saved_vocab["id_to_char"], ["a", "b"])
        self.assertEqual(artifact.tokenizer.decode(artifact.token_ids), "aba")

    def test_ingest_writes_fixed_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            raw_dir.mkdir()
            (raw_dir / "sample.txt").write_text("aaaabbbbcc", encoding="utf-8")

            ingest_raw_corpus(
                raw_dir,
                processed_dir,
                artifact_name="train",
                validation_fraction=0.3,
            )

            train_tokens = torch.load(processed_dir / "train_tokens.pt")
            validation_tokens = torch.load(processed_dir / "val_tokens.pt")

        self.assertEqual(len(train_tokens), 7)
        self.assertEqual(len(validation_tokens), 3)
        self.assertEqual(train_tokens.tolist() + validation_tokens.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
