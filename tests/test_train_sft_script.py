from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from superagi.ingestion.tokenizer import BpeTokenizer
from superagi.model.checkpoint import load_checkpoint, save_checkpoint
from superagi.model.transformer import TransformerConfig, TransformerLM


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_sft)


class TrainSftScriptTests(unittest.TestCase):
    def test_run_dir_keeps_best_checkpoint_and_prunes_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path = root / "base.pt"
            data_path = root / "examples.jsonl"
            run_dir = root / "run"
            tokenizer = BpeTokenizer.from_text(
                "<bos><user> What is AI?\n<agi> AI learns patterns from data.<eos>\n",
                vocab_size=300,
                min_frequency=1,
            )
            config = TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=128,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
            )
            save_checkpoint(
                base_path,
                model=TransformerLM(config),
                vocab=tokenizer.to_payload(),
            )
            base_hash = _sha256(base_path)
            rows = [
                {
                    "source": "curated_core:everyday",
                    "messages": [
                        {"role": "user", "content": "What is AI?"},
                        {"role": "agi", "content": "AI learns patterns from data."},
                    ],
                },
                {
                    "source": "curated_core:repair",
                    "messages": [
                        {"role": "user", "content": "What is math?"},
                        {"role": "agi", "content": "Math studies quantities and patterns."},
                    ],
                },
            ]
            data_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            argv = [
                "train_sft.py",
                "--base-checkpoint", str(base_path),
                "--data", str(data_path),
                "--run-dir", str(run_dir),
                "--steps", "3",
                "--batch", "1",
                "--lr", "1e-5",
                "--lr-min", "1e-6",
                "--checkpoint-interval", "1",
                "--checkpoint-keep", "2",
                "--validation-fraction", "0.5",
                "--validation-batches", "1",
                "--mixed-precision", "none",
                "--device", "cpu",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(train_sft, "evaluate_sft_loss", side_effect=[2.0, 1.4, 1.7]),
            ):
                self.assertEqual(train_sft.main(), 0)

            latest = load_checkpoint(run_dir / "latest.pt")
            best = load_checkpoint(run_dir / "best.pt")
            self.assertEqual(latest.metadata["sft_steps"], 3)
            self.assertEqual(best.metadata["best_validation_step"], 2)
            self.assertEqual(best.metadata["best_validation_loss"], 1.4)
            self.assertEqual((run_dir / "final.pt").read_bytes(), (run_dir / "best.pt").read_bytes())
            self.assertEqual(_sha256(base_path), base_hash)
            self.assertEqual(
                [path.name for path in sorted((run_dir / "snapshots").glob("*.pt"))],
                ["checkpoint-step-000000002.pt", "checkpoint-step-000000003.pt"],
            )
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([metric["step"] for metric in metrics], [1, 2, 3])

    def test_rejects_ambiguous_run_dir_and_legacy_output_arguments(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train_sft.py",
                "--base-checkpoint", "base.pt",
                "--data", "data.jsonl",
                "--run-dir", "run",
                "--out", "legacy.pt",
                "--metrics", "legacy.jsonl",
            ],
        ):
            with self.assertRaisesRegex(SystemExit, "exactly one"):
                train_sft.main()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
