from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from superagi.ingestion.tokenizer import BpeTokenizer, CharTokenizer
from superagi.model.checkpoint import save_checkpoint
from superagi.model.transformer import TransformerConfig, TransformerLM


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "preflight_sft_300m.py"


class Sft300mPreflightTests(unittest.TestCase):
    def test_records_checkpoint_identity_and_nested_run_config(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            required_path = root / "data" / "sft" / "curated" / "core.jsonl"
            required_path.parent.mkdir(parents=True)
            required_path.write_text('{"messages":[]}\n', encoding="utf-8")
            sha_record = root / "runs" / "base-checkpoint.json"
            run_config = root / "runs" / "run-config.json"

            result = module.main(
                [
                    "--repository-root",
                    str(root),
                    "--base-checkpoint",
                    str(checkpoint_path),
                    "--sha-record",
                    str(sha_record),
                    "--run-config",
                    str(run_config),
                    "--require-path",
                    str(required_path),
                    "--config",
                    "core.batch=2",
                    "--config",
                    "core.mixed_precision=float16",
                    "--config",
                    "style.steps=500",
                ]
            )

            self.assertEqual(result, 0)
            sha_payload = json.loads(sha_record.read_text(encoding="utf-8"))
            config_payload = json.loads(run_config.read_text(encoding="utf-8"))
            self.assertRegex(sha_payload["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(sha_payload["context_length"], 1024)
            self.assertEqual(
                set(sha_payload["special_token_ids"]),
                {"<pad>", "<bos>", "<eos>", "<user>", "<agi>", "<system>"},
            )
            self.assertEqual(config_payload["settings"]["core"]["batch"], 2)
            self.assertEqual(
                config_payload["settings"]["core"]["mixed_precision"],
                "float16",
            )
            self.assertEqual(config_payload["settings"]["style"]["steps"], 500)
            self.assertEqual(
                config_payload["base_checkpoint"]["sha256"],
                sha_payload["sha256"],
            )

    def test_rejects_checkpoint_without_required_chat_special_tokens(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_char_checkpoint(root / "base.pt")

            with self.assertRaisesRegex(ValueError, "required special token"):
                module.main(
                    [
                        "--repository-root",
                        str(root),
                        "--base-checkpoint",
                        str(checkpoint_path),
                        "--sha-record",
                        str(root / "base-checkpoint.json"),
                        "--run-config",
                        str(root / "run-config.json"),
                    ]
                )

    def test_rejects_context_shorter_than_1024(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=512,
            )

            with self.assertRaisesRegex(ValueError, "at least 1024"):
                module.main(
                    [
                        "--repository-root",
                        str(root),
                        "--base-checkpoint",
                        str(checkpoint_path),
                        "--sha-record",
                        str(root / "base-checkpoint.json"),
                        "--run-config",
                        str(root / "run-config.json"),
                    ]
                )

    def test_rejects_missing_required_input_path(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )

            with self.assertRaisesRegex(ValueError, "missing required input"):
                module.main(
                    [
                        "--repository-root",
                        str(root),
                        "--base-checkpoint",
                        str(checkpoint_path),
                        "--sha-record",
                        str(root / "base-checkpoint.json"),
                        "--run-config",
                        str(root / "run-config.json"),
                        "--require-path",
                        str(root / "missing.jsonl"),
                    ]
                )

    def test_verify_only_detects_base_checkpoint_mutation(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            sha_record = root / "base-checkpoint.json"
            run_config = root / "run-config.json"
            module.main(
                [
                    "--repository-root",
                    str(root),
                    "--base-checkpoint",
                    str(checkpoint_path),
                    "--sha-record",
                    str(sha_record),
                    "--run-config",
                    str(run_config),
                ]
            )
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"changed")

            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                module.main(
                    [
                        "--repository-root",
                        str(root),
                        "--base-checkpoint",
                        str(checkpoint_path),
                        "--sha-record",
                        str(sha_record),
                        "--verify-only",
                    ]
                )

    def test_rejects_changed_run_configuration_on_restart(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            sha_record = root / "base-checkpoint.json"
            run_config = root / "run-config.json"
            arguments = [
                "--repository-root",
                str(root),
                "--base-checkpoint",
                str(checkpoint_path),
                "--sha-record",
                str(sha_record),
                "--run-config",
                str(run_config),
            ]
            module.main([*arguments, "--config", "core.batch=2"])

            with self.assertRaisesRegex(ValueError, "run configuration changed"):
                module.main([*arguments, "--config", "core.batch=4"])


def _load_preflight_module(test_case: unittest.TestCase) -> ModuleType:
    if not SCRIPT_PATH.is_file():
        test_case.fail(f"missing Task 9 preflight script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("preflight_sft_300m", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        test_case.fail("could not load Task 9 preflight script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bpe_checkpoint(path: Path, *, context_length: int) -> Path:
    tokenizer = BpeTokenizer.from_text(
        "The model answers useful questions with concise explanations.",
        vocab_size=300,
        min_frequency=1,
    )
    return _write_checkpoint(path, tokenizer, context_length=context_length)


def _write_char_checkpoint(path: Path) -> Path:
    tokenizer = CharTokenizer.from_text("abcdef")
    return _write_checkpoint(path, tokenizer, context_length=1024)


def _write_checkpoint(
    path: Path,
    tokenizer: BpeTokenizer | CharTokenizer,
    *,
    context_length: int,
) -> Path:
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        dim_embedding=8,
        n_layers=1,
        n_heads=1,
        dim_feed_forward=16,
    )
    return save_checkpoint(
        path,
        model=TransformerLM(config),
        vocab=tokenizer.to_payload(),
    )


if __name__ == "__main__":
    unittest.main()
