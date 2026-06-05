import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from superagi.model.checkpoint import (
    generate_from_checkpoint,
    load_checkpoint,
    prepare_model_for_training,
    save_checkpoint,
)
from superagi.model.transformer import TransformerConfig, TransformerLM


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_portable_checkpoint(self) -> None:
        model = self._tiny_model()
        vocab = {"id_to_char": ["a", "b", " "]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            saved_path = save_checkpoint(
                checkpoint_path,
                model=model,
                vocab=vocab,
                losses=[1.0, 0.5],
                metrics=[
                    {
                        "step": 2,
                        "train_loss": 0.5,
                        "validation_loss": 0.75,
                        "elapsed_seconds": 12.0,
                    }
                ],
                metadata={"steps": 2},
            )

            loaded = load_checkpoint(saved_path)

        self.assertEqual(saved_path, checkpoint_path)
        self.assertEqual(loaded.config, model.config)
        self.assertEqual(loaded.tokenizer.decode([0, 2, 1]), "a b")
        self.assertEqual(loaded.losses, [1.0, 0.5])
        self.assertEqual(loaded.metrics[0]["validation_loss"], 0.75)
        self.assertEqual(loaded.metadata["steps"], 2)
        self.assertEqual(
            sum(parameter.numel() for parameter in loaded.model.parameters()),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_generate_from_checkpoint_returns_prompt_plus_new_chars(self) -> None:
        model = self._tiny_model(vocab_size=2)
        vocab = {"id_to_char": ["a", "b"]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=vocab,
            )

            generated = generate_from_checkpoint(
                checkpoint_path,
                prompt="a",
                max_new_tokens=3,
                seed=0,
            )

        self.assertEqual(len(generated), 4)
        self.assertTrue(generated.startswith("a"))
        self.assertTrue(set(generated).issubset({"a", "b"}))

    def test_run_model_script_generates_from_checkpoint(self) -> None:
        model = self._tiny_model(vocab_size=2)
        vocab = {"id_to_char": ["a", "b"]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=vocab,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_model.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "a",
                    "--new-tokens",
                    "2",
                    "--seed",
                    "0",
                ],
                check=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            )

        generated = result.stdout.strip()
        self.assertEqual(len(generated), 3)
        self.assertTrue(generated.startswith("a"))
        self.assertTrue(set(generated).issubset({"a", "b"}))

    def test_prepare_model_for_training_resumes_checkpoint(self) -> None:
        model = self._tiny_model(vocab_size=2)
        vocab = {"id_to_char": ["a", "b"]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=vocab,
                losses=[1.0, 0.5],
                metrics=[{"step": 2, "train_loss": 0.5, "validation_loss": 0.75}],
                metadata={"steps": 2},
            )

            state = prepare_model_for_training(
                vocab=vocab,
                config=TransformerConfig(
                    vocab_size=2,
                    context_length=4,
                    dim_embedding=4,
                    n_layers=1,
                    n_heads=1,
                ),
                resume_path=checkpoint_path,
            )

        self.assertEqual(state.config, model.config)
        self.assertEqual(state.previous_losses, [1.0, 0.5])
        self.assertEqual(state.previous_metrics[0]["step"], 2)
        self.assertEqual(state.metadata["steps"], 2)
        self.assertEqual(state.resumed_from, checkpoint_path)

    def test_prepare_model_for_training_rejects_vocab_mismatch(self) -> None:
        model = self._tiny_model(vocab_size=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab={"id_to_char": ["a", "b"]},
            )

            with self.assertRaisesRegex(ValueError, "vocab does not match"):
                prepare_model_for_training(
                    vocab={"id_to_char": ["a", "c"]},
                    config=model.config,
                    resume_path=checkpoint_path,
                )

    def _tiny_model(self, vocab_size: int = 3) -> TransformerLM:
        torch.manual_seed(0)
        return TransformerLM(
            TransformerConfig(
                vocab_size=vocab_size,
                context_length=8,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
