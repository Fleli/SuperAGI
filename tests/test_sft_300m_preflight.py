from __future__ import annotations

import importlib.util
import hashlib
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

    def test_verify_only_rejects_well_typed_run_configuration_mutation(self) -> None:
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
                "--config",
                "core.steps=3000",
            ]
            module.main(arguments)
            payload = json.loads(run_config.read_text(encoding="utf-8"))
            payload["settings"]["core"]["steps"] = 9999
            run_config.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "run configuration changed"):
                module.main([*arguments, "--verify-only"])

            payload["settings"]["core"]["steps"] = 3000
            run_config.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(module.main([*arguments, "--verify-only"]), 0)

    def test_records_public_import_identity_and_distinguishes_resume(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            sha_record = root / "runs" / "base-checkpoint.json"
            run_config = root / "runs" / "run-config.json"
            state_file = root / "runs" / "preflight-state.txt"
            public_data = root / "data" / "sft" / "imported" / "public.jsonl"
            public_metadata = (
                root / "data" / "sft" / "imported" / "public.metadata.json"
            )
            arguments = [
                "--repository-root",
                str(root),
                "--base-checkpoint",
                str(checkpoint_path),
                "--sha-record",
                str(sha_record),
                "--run-config",
                str(run_config),
                "--public-data",
                str(public_data),
                "--public-metadata",
                str(public_metadata),
                "--state-file",
                str(state_file),
                "--config",
                "core.batch=2",
            ]

            self.assertEqual(module.main(arguments), 0)
            self.assertEqual(state_file.read_text(encoding="utf-8"), "prepare\n")
            prepared_config = json.loads(run_config.read_text(encoding="utf-8"))
            self.assertEqual(prepared_config["schema_version"], 2)
            self.assertEqual(prepared_config["inputs"], {})

            public_data.parent.mkdir(parents=True)
            public_data.write_text('{"source":"dolly"}\n', encoding="utf-8")
            public_metadata.write_text(
                '{"written_count":1,"sources":{"dolly":{"selected":1}}}\n',
                encoding="utf-8",
            )
            self.assertEqual(module.main([*arguments, "--record-public"]), 0)

            sealed_config = json.loads(run_config.read_text(encoding="utf-8"))
            self.assertEqual(
                sealed_config["inputs"]["public_jsonl"]["sha256"],
                hashlib.sha256(public_data.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                sealed_config["inputs"]["public_metadata"]["sha256"],
                hashlib.sha256(public_metadata.read_bytes()).hexdigest(),
            )

            self.assertEqual(module.main(arguments), 0)
            self.assertEqual(state_file.read_text(encoding="utf-8"), "resume\n")

    def test_resume_rejects_mutated_public_import_bytes(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            public_data = root / "public.jsonl"
            public_metadata = root / "public.metadata.json"
            arguments = [
                "--repository-root",
                str(root),
                "--base-checkpoint",
                str(checkpoint_path),
                "--sha-record",
                str(root / "base-checkpoint.json"),
                "--run-config",
                str(root / "run-config.json"),
                "--public-data",
                str(public_data),
                "--public-metadata",
                str(public_metadata),
                "--state-file",
                str(root / "preflight-state.txt"),
                "--config",
                "core.batch=2",
            ]
            module.main(arguments)
            public_data.write_text('{"source":"dolly"}\n', encoding="utf-8")
            public_metadata.write_text(
                '{"written_count":1,"sources":{"dolly":{"selected":1}}}\n',
                encoding="utf-8",
            )
            module.main([*arguments, "--record-public"])
            public_data.write_text('{"source":"changed"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "public_jsonl.*changed.*new SFT run directory",
            ):
                module.main(arguments)

    def test_preparation_refuses_unsealed_existing_public_files(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            public_data = root / "runs" / "inputs" / "public.jsonl"
            public_metadata = root / "runs" / "inputs" / "public.metadata.json"
            public_data.parent.mkdir(parents=True)
            public_data.write_text('{"source":"partial"}\n', encoding="utf-8")
            public_metadata.write_text('{"partial":true}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "unsealed public import.*new SFT run directory",
            ):
                module.main(
                    [
                        "--repository-root",
                        str(root),
                        "--base-checkpoint",
                        str(checkpoint_path),
                        "--sha-record",
                        str(root / "runs" / "base-checkpoint.json"),
                        "--run-config",
                        str(root / "runs" / "run-config.json"),
                        "--public-data",
                        str(public_data),
                        "--public-metadata",
                        str(public_metadata),
                        "--state-file",
                        str(root / "runs" / "preflight-state.txt"),
                        "--config",
                        "core.batch=2",
                    ]
                )

    def test_verify_only_requires_sealed_public_import_identity(self) -> None:
        module = _load_preflight_module(self)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = _write_bpe_checkpoint(
                root / "base.pt",
                context_length=1024,
            )
            public_data = root / "public.jsonl"
            public_metadata = root / "public.metadata.json"
            arguments = [
                "--repository-root",
                str(root),
                "--base-checkpoint",
                str(checkpoint_path),
                "--sha-record",
                str(root / "base-checkpoint.json"),
                "--run-config",
                str(root / "run-config.json"),
                "--public-data",
                str(public_data),
                "--public-metadata",
                str(public_metadata),
                "--state-file",
                str(root / "preflight-state.txt"),
                "--config",
                "core.batch=2",
            ]
            module.main(arguments)

            with self.assertRaisesRegex(ValueError, "public import identity"):
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
                        "--public-data",
                        str(public_data),
                        "--public-metadata",
                        str(public_metadata),
                        "--verify-only",
                    ]
                )


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
