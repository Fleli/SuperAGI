import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from superagi.ingestion.tokenizer import EOS_TOKEN, BpeTokenizer
from superagi.model.checkpoint import (
    generate_from_checkpoint,
    load_checkpoint,
    prepare_model_for_training,
    save_checkpoint,
)
from superagi.model.transformer import TransformerConfig, TransformerLM


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_portable_checkpoint(self) -> None:
        tokenizer = self._tiny_tokenizer()
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)
        vocab = tokenizer.to_payload()

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
        self.assertEqual(loaded.tokenizer.decode(loaded.tokenizer.encode("a b")), "a b")
        self.assertEqual(loaded.vocab["tokenizer_type"], "bpe")
        self.assertEqual(loaded.losses, [1.0, 0.5])
        self.assertEqual(loaded.metrics[0]["validation_loss"], 0.75)
        self.assertEqual(loaded.metadata["steps"], 2)
        self.assertEqual(
            sum(parameter.numel() for parameter in loaded.model.parameters()),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_generate_from_checkpoint_returns_prompt_plus_new_text(self) -> None:
        tokenizer = self._tiny_tokenizer()
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=tokenizer.to_payload(),
            )

            generated = generate_from_checkpoint(
                checkpoint_path,
                prompt="a",
                max_new_tokens=3,
                seed=0,
            )

        self.assertTrue(generated.startswith("a"))
        self.assertGreaterEqual(len(generated), 1)

    def test_generate_from_checkpoint_streams_decoded_new_text(self) -> None:
        tokenizer = self._tiny_tokenizer()
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)
        streamed_chunks: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=tokenizer.to_payload(),
            )

            generated = generate_from_checkpoint(
                checkpoint_path,
                prompt="a",
                max_new_tokens=3,
                seed=0,
                on_text=streamed_chunks.append,
            )

        self.assertEqual("".join(streamed_chunks), generated[1:])

    def test_generate_from_checkpoint_stops_at_eos_token(self) -> None:
        tokenizer = self._tiny_tokenizer()
        eos_id = tokenizer.special_token_id(EOS_TOKEN)
        prompt_ids = tokenizer.encode("a")
        new_ids = tokenizer.encode(" b")
        streamed_chunks: list[str] = []
        test_case = self

        class FakeModel:
            def to(self, device):
                return self

            def generate(self, *, input_ids, stop_token_ids=None, on_token=None, **kwargs):
                test_case.assertIn(eos_id, stop_token_ids)
                for token_id in new_ids:
                    if on_token is not None:
                        on_token(token_id)
                return torch.tensor([prompt_ids + new_ids + [eos_id]])

        fake_checkpoint = mock.Mock()
        fake_checkpoint.model = FakeModel()
        fake_checkpoint.tokenizer = tokenizer

        with mock.patch(
            "superagi.model.checkpoint.load_checkpoint",
            return_value=fake_checkpoint,
        ):
            generated = generate_from_checkpoint(
                "unused.pt",
                prompt="a",
                max_new_tokens=10,
                on_text=streamed_chunks.append,
            )

        self.assertEqual(generated, "a b")
        self.assertEqual("".join(streamed_chunks), " b")

    def test_run_model_script_generates_from_checkpoint(self) -> None:
        tokenizer = self._tiny_tokenizer()
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=tokenizer.to_payload(),
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
                    "--top-k",
                    "1",
                    "--seed",
                    "0",
                ],
                check=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            )

        generated = result.stdout.strip()
        self.assertTrue(generated.startswith("a"))
        self.assertGreaterEqual(len(generated), 1)

    def test_run_model_script_streams_by_default(self) -> None:
        run_model = self._load_run_model_module()

        def fake_generate_from_checkpoint(*args, on_text=None, **kwargs):
            self.assertIsNotNone(on_text)
            on_text("b")
            on_text("c")
            return "abc"

        output = io.StringIO()
        with mock.patch.object(
            run_model,
            "generate_from_checkpoint",
            side_effect=fake_generate_from_checkpoint,
        ):
            with contextlib.redirect_stdout(output):
                exit_code = run_model.main(
                    [
                        "--checkpoint",
                        "unused.pt",
                        "--prompt",
                        "a",
                        "--new-tokens",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "abc\n")

    def test_run_model_script_passes_repetition_penalty_options(self) -> None:
        run_model = self._load_run_model_module()

        def fake_generate_from_checkpoint(*args, **kwargs):
            self.assertEqual(kwargs["repetition_penalty"], 1.2)
            self.assertEqual(kwargs["repetition_window"], 64)
            return "abc"

        output = io.StringIO()
        with mock.patch.object(
            run_model,
            "generate_from_checkpoint",
            side_effect=fake_generate_from_checkpoint,
        ):
            with contextlib.redirect_stdout(output):
                exit_code = run_model.main(
                    [
                        "--checkpoint",
                        "unused.pt",
                        "--prompt",
                        "a",
                        "--new-tokens",
                        "2",
                        "--stream",
                        "0",
                        "--repetition-penalty",
                        "1.2",
                        "--repetition-window",
                        "64",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "abc\n")

    def test_run_model_script_can_disable_streaming(self) -> None:
        run_model = self._load_run_model_module()

        def fake_generate_from_checkpoint(*args, on_text=None, **kwargs):
            self.assertIsNone(on_text)
            return "abc"

        output = io.StringIO()
        with mock.patch.object(
            run_model,
            "generate_from_checkpoint",
            side_effect=fake_generate_from_checkpoint,
        ):
            with contextlib.redirect_stdout(output):
                exit_code = run_model.main(
                    [
                        "--checkpoint",
                        "unused.pt",
                        "--prompt",
                        "a",
                        "--new-tokens",
                        "2",
                        "--stream",
                        "0",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "abc\n")

    def test_run_model_script_can_format_chat_prompt(self) -> None:
        run_model = self._load_run_model_module()

        def fake_generate_from_checkpoint(*args, prompt=None, **kwargs):
            self.assertEqual(prompt, "<bos><user> What are you?\n<agi> ")
            return "<bos><user> What are you?\n<agi> I am a small model.<eos>"

        output = io.StringIO()
        with mock.patch.object(
            run_model,
            "generate_from_checkpoint",
            side_effect=fake_generate_from_checkpoint,
        ):
            with contextlib.redirect_stdout(output):
                exit_code = run_model.main(
                    [
                        "--checkpoint",
                        "unused.pt",
                        "--prompt",
                        "What are you?",
                        "--new-tokens",
                        "2",
                        "--chat",
                        "1",
                        "--stream",
                        "0",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "User: What are you?\nAGI: I am a small model.\n")

    def test_prepare_model_for_training_resumes_checkpoint(self) -> None:
        tokenizer = self._tiny_tokenizer()
        vocab = tokenizer.to_payload()
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)

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
                    vocab_size=tokenizer.vocab_size,
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
        tokenizer = self._tiny_tokenizer()
        mismatched_tokenizer = self._tiny_tokenizer("x y x y")
        model = self._tiny_model(vocab_size=tokenizer.vocab_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = save_checkpoint(
                Path(tmp_dir) / "model.pt",
                model=model,
                vocab=tokenizer.to_payload(),
            )

            with self.assertRaisesRegex(ValueError, "vocab does not match"):
                prepare_model_for_training(
                    vocab=mismatched_tokenizer.to_payload(),
                    config=model.config,
                    resume_path=checkpoint_path,
                )

    def _tiny_tokenizer(self, text: str = "a b a b") -> BpeTokenizer:
        return BpeTokenizer.from_text(text, vocab_size=300, min_frequency=1)

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

    def _load_run_model_module(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_model.py"
        spec = importlib.util.spec_from_file_location("run_model_for_test", script_path)
        if spec is None or spec.loader is None:
            raise AssertionError("could not load run_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
