from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import import_public_sft, write_sft_manifest


RUN_NAMES = ("core", "playful", "calm")


class SftManifestWriterTests(unittest.TestCase):
    def test_writes_relative_deterministically_ordered_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = _write_complete_run(root)

            exit_code = write_sft_manifest.main(_arguments(root, paths))

            self.assertEqual(exit_code, 0)
            manifest_text = paths["output"].read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertRegex(
                manifest["created_at_utc"],
                re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
            )
            self.assertEqual(
                manifest_text,
                json.dumps(
                    manifest,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            self.assertNotIn(str(root), manifest_text)

            base_bytes = paths["base"].read_bytes()
            self.assertEqual(
                manifest["base_checkpoint"],
                {
                    "path": "base.pt",
                    "sha256": hashlib.sha256(base_bytes).hexdigest(),
                    "size_bytes": len(base_bytes),
                },
            )
            self.assertEqual(
                manifest["run_config"]["schema_version"],
                2,
            )
            self.assertEqual(
                manifest["runs"]["core"]["best_validation_metric"]["step"],
                20,
            )
            self.assertEqual(
                manifest["runs"]["core"]["best_validation_metric"][
                    "validation_loss"
                ],
                1.5,
            )
            self.assertTrue(
                manifest["runs"]["playful"]["evaluation_summary"]["ok"]
            )
            self.assertEqual(
                manifest["source_artifacts"]["public_metadata"]["path"],
                "data/sft/imported/public-mixed.metadata.json",
            )
            self.assertEqual(
                manifest["source_artifacts"]["mixed_audit"]["path"],
                "data/sft/runs/300m/audit.json",
            )
            self.assertEqual(
                set(manifest["source_artifacts"]),
                {
                    "calm_style_audit",
                    "calm_style_jsonl",
                    "curated_core_audit",
                    "curated_core_jsonl",
                    "curated_core_metadata",
                    "evaluation_prompts",
                    "mixed_audit",
                    "playful_style_audit",
                    "playful_style_jsonl",
                    "public_jsonl",
                    "public_metadata",
                    "style_metadata",
                },
            )
            for artifact in manifest["source_artifacts"].values():
                self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(artifact["size_bytes"], 0)

            expected_artifacts = {
                "best_checkpoint",
                "evaluation_results",
                "evaluation_summary",
                "final_checkpoint",
                "metrics",
            }
            for run_name in RUN_NAMES:
                self.assertEqual(
                    set(manifest["runs"][run_name]["artifacts"]),
                    expected_artifacts,
                )
                for artifact in manifest["runs"][run_name]["artifacts"].values():
                    self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
                    self.assertGreater(artifact["size_bytes"], 0)
                    self.assertFalse(Path(artifact["path"]).is_absolute())

    def test_refuses_each_missing_required_artifact_without_overwriting(self) -> None:
        missing_cases = {
            "core best checkpoint": "data/sft/runs/300m/core/best.pt",
            "core final checkpoint": "data/sft/runs/300m/core/final.pt",
            "core evaluation results": (
                "data/sft/runs/300m/core/evaluation.jsonl"
            ),
            "playful evaluation summary": (
                "data/sft/runs/300m/playful/evaluation.summary.json"
            ),
            "calm metrics": "data/sft/runs/300m/calm/metrics.jsonl",
            "public import metadata": (
                "data/sft/imported/public-mixed.metadata.json"
            ),
            "public import JSONL": "data/sft/imported/public-mixed.jsonl",
            "curated core JSONL": "data/sft/curated/core.jsonl",
            "curated core metadata": "data/sft/curated/core.metadata.json",
            "curated core audit": "data/sft/curated/core.audit.json",
            "playful style JSONL": "data/sft/styles/playful-direct.jsonl",
            "playful style audit": (
                "data/sft/styles/playful-direct.audit.json"
            ),
            "calm style JSONL": "data/sft/styles/calm-precise.jsonl",
            "calm style audit": "data/sft/styles/calm-precise.audit.json",
            "style metadata": "data/sft/styles/styles.metadata.json",
            "evaluation prompts": "data/sft/eval_prompts.jsonl",
            "base SHA record": "data/sft/runs/300m/base-checkpoint.json",
            "audit report": "data/sft/runs/300m/audit.json",
        }
        for label, relative_path in missing_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                paths["output"].parent.mkdir(parents=True, exist_ok=True)
                paths["output"].write_text("existing\n", encoding="utf-8")
                (root / relative_path).unlink()

                with self.assertRaisesRegex(
                    ValueError,
                    "required artifact",
                ):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertEqual(
                    paths["output"].read_text(encoding="utf-8"),
                    "existing\n",
                )

    def test_refuses_failed_or_malformed_reports(self) -> None:
        invalid_cases = {
            "failed evaluation": (
                "data/sft/runs/300m/core/evaluation.summary.json",
                _evaluation_summary(ok=False),
                "evaluation summary",
            ),
            "failed audit": (
                "data/sft/runs/300m/audit.json",
                _audit_report(ok=False),
                "audit report",
            ),
            "invalid import metadata": (
                "data/sft/imported/public-mixed.metadata.json",
                {"written_count": 0, "sources": {}},
                "public import metadata",
            ),
            "unpinned import metadata": (
                "data/sft/imported/public-mixed.metadata.json",
                {
                    "written_count": 200,
                    "sources": {
                        "dolly": {"selected": 100},
                        "no_robots": {"selected": 100},
                    },
                },
                "public import metadata",
            ),
        }
        for label, (relative_path, payload, message) in invalid_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                _write_json(root / relative_path, payload)
                if Path(relative_path) == paths["public_metadata"].relative_to(root):
                    _reseal_public_metadata(paths, root)

                with self.assertRaisesRegex(ValueError, message):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_internally_inconsistent_reports(self) -> None:
        inconsistent_evaluation = _evaluation_summary()
        inconsistent_evaluation["passed_prompts"] = 59
        inconsistent_evaluation["failed_prompts"] = 1
        inconsistent_audit = _audit_report()
        inconsistent_audit["findings"] = [
            {"severity": "error", "code": "hidden_failure"}
        ]
        inconsistent_import = {
            "written_count": 201,
            "sources": {
                "dolly": {"selected": 100},
                "no_robots": {"selected": 100},
            },
        }
        invalid_cases = {
            "evaluation": (
                "data/sft/runs/300m/core/evaluation.summary.json",
                inconsistent_evaluation,
                "evaluation summary",
            ),
            "audit": (
                "data/sft/runs/300m/audit.json",
                inconsistent_audit,
                "audit report",
            ),
            "import": (
                "data/sft/imported/public-mixed.metadata.json",
                inconsistent_import,
                "public import metadata",
            ),
        }
        for label, (relative_path, payload, message) in invalid_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                _write_json(root / relative_path, payload)
                if Path(relative_path) == paths["public_metadata"].relative_to(root):
                    _reseal_public_metadata(paths, root)

                with self.assertRaisesRegex(ValueError, message):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_corrupt_best_checkpoint_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = _write_complete_run(root)
            best_path = root / "data/sft/runs/300m/playful/best.pt"
            best_path.write_bytes(b"not a torch checkpoint")

            with self.assertRaisesRegex(
                ValueError,
                "playful best checkpoint",
            ):
                write_sft_manifest.main(_arguments(root, paths))

            self.assertFalse(paths["output"].exists())

    def test_refuses_base_record_and_run_config_identity_mismatches(self) -> None:
        mismatch_cases = {
            "base SHA record": (
                "base_record",
                ("sha256", "0" * 64),
                "base SHA record",
            ),
            "run config base path": (
                "run_config",
                ("base_checkpoint.path", "different.pt"),
                "run config base checkpoint",
            ),
            "run config public hash": (
                "run_config",
                ("inputs.public_jsonl.sha256", "f" * 64),
                "public_jsonl",
            ),
            "run config core path": (
                "run_config",
                ("settings.core.data", "different.jsonl"),
                "run configuration changed",
            ),
            "run config prompt path": (
                "run_config",
                ("settings.evaluation.prompts", "different.jsonl"),
                "run configuration changed",
            ),
        }
        for label, (path_key, mutation, message) in mismatch_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                target = paths[path_key]
                payload = json.loads(target.read_text(encoding="utf-8"))
                _set_dotted(payload, mutation[0], mutation[1])
                _write_json(target, payload)

                with self.assertRaisesRegex(ValueError, message):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_well_typed_run_config_setting_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = _write_complete_run(root)
            payload = json.loads(paths["run_config"].read_text(encoding="utf-8"))
            payload["settings"]["core"]["steps"] = 9999
            _write_json(paths["run_config"], payload)

            with self.assertRaisesRegex(
                ValueError,
                "run configuration changed",
            ):
                write_sft_manifest.main(_arguments(root, paths))

            self.assertFalse(paths["output"].exists())

    def test_refuses_resealed_public_metadata_source_identity_mutations(self) -> None:
        mutation_cases = {
            "source keys": _add_unconfigured_public_source,
            "dataset": lambda payload: _set_dotted(
                payload,
                "dataset_revisions.dolly.dataset",
                "example.invalid/dolly",
            ),
            "split": lambda payload: _set_dotted(
                payload,
                "dataset_revisions.dolly.split",
                "validation",
            ),
            "revision": lambda payload: _set_dotted(
                payload,
                "dataset_revisions.dolly.revision",
                "f" * 40,
            ),
        }
        for label, mutate in mutation_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                public_metadata = json.loads(
                    paths["public_metadata"].read_text(encoding="utf-8")
                )
                mutate(public_metadata)
                _write_json(paths["public_metadata"], public_metadata)
                run_config = json.loads(
                    paths["run_config"].read_text(encoding="utf-8")
                )
                run_config["inputs"]["public_metadata"] = _artifact_identity(
                    paths["public_metadata"],
                    root,
                )
                _write_json(paths["run_config"], run_config)

                with self.assertRaisesRegex(
                    ValueError,
                    "public import metadata",
                ):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_malformed_run_config_schema(self) -> None:
        invalid_cases = {
            "wrong schema": ("schema_version", 1),
            "missing settings": ("settings", None),
            "missing core section": ("settings.core", None),
            "invalid core steps": ("settings.core.steps", "3000"),
            "invalid source weights": ("settings.core.source_weights", ""),
        }
        for label, (field, value) in invalid_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                payload = json.loads(paths["run_config"].read_text(encoding="utf-8"))
                if value is None:
                    _delete_dotted(payload, field)
                else:
                    _set_dotted(payload, field, value)
                _write_json(paths["run_config"], payload)

                with self.assertRaisesRegex(ValueError, "run config"):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_metrics_without_finite_validation_loss(self) -> None:
        invalid_rows = (
            [{"step": 10, "train_loss": 2.0, "validation_loss": None}],
            [{"step": 10, "train_loss": 2.0, "validation_loss": float("nan")}],
            [{"step": "ten", "train_loss": 2.0, "validation_loss": 1.0}],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                _write_jsonl(
                    root / "data/sft/runs/300m/core/metrics.jsonl",
                    rows,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "validation metric",
                ):
                    write_sft_manifest.main(_arguments(root, paths))

                self.assertFalse(paths["output"].exists())

    def test_refuses_artifacts_outside_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "repo"
            root.mkdir()
            paths = _write_complete_run(root)
            outside = Path(tmp_dir) / "outside-base.pt"
            outside.write_bytes(b"outside")
            paths["base"] = outside

            with self.assertRaisesRegex(ValueError, "repository root"):
                write_sft_manifest.main(_arguments(root, paths))

            self.assertFalse(paths["output"].exists())


def _arguments(root: Path, paths: dict[str, Path]) -> list[str]:
    arguments = [
        "--repository-root",
        str(root),
        "--output",
        str(paths["output"]),
        "--base-checkpoint",
        str(paths["base"]),
        "--base-sha-record",
        str(paths["base_record"]),
        "--core-run-dir",
        str(root / "data/sft/runs/300m/core"),
        "--playful-run-dir",
        str(root / "data/sft/runs/300m/playful"),
        "--calm-run-dir",
        str(root / "data/sft/runs/300m/calm"),
        "--public-import-metadata",
        str(paths["public_metadata"]),
        "--public-import-data",
        str(paths["public_data"]),
        "--curated-core-data",
        str(paths["curated_core_data"]),
        "--curated-core-metadata",
        str(paths["curated_core_metadata"]),
        "--curated-core-audit",
        str(paths["curated_core_audit"]),
        "--playful-style-data",
        str(paths["playful_style_data"]),
        "--playful-style-audit",
        str(paths["playful_style_audit"]),
        "--calm-style-data",
        str(paths["calm_style_data"]),
        "--calm-style-audit",
        str(paths["calm_style_audit"]),
        "--style-metadata",
        str(paths["style_metadata"]),
        "--eval-prompts",
        str(paths["eval_prompts"]),
        "--audit-report",
        str(paths["audit"]),
        "--run-config",
        str(paths["run_config"]),
    ]
    for key, value in _flatten_settings(_production_settings()):
        arguments.extend(("--config", f"{key}={value}"))
    return arguments


def _write_complete_run(root: Path) -> dict[str, Path]:
    base = root / "base.pt"
    base.write_bytes(b"base-checkpoint")
    base_identity = {
        "path": "base.pt",
        "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "context_length": 1024,
        "special_token_ids": {
            "<agi>": 4,
            "<bos>": 1,
            "<eos>": 2,
            "<pad>": 0,
            "<system>": 5,
            "<user>": 3,
        },
    }
    base_record = root / "data/sft/runs/300m/base-checkpoint.json"
    _write_json(base_record, base_identity)

    public_data = root / "data/sft/imported/public-mixed.jsonl"
    _write_jsonl(
        public_data,
        [{"source": "dolly:1", "messages": [{"role": "agi", "content": "A"}]}],
    )
    public_metadata = root / "data/sft/imported/public-mixed.metadata.json"
    _write_json(
        public_metadata,
        {
            "written_count": 200,
            "sources": {
                source: {"selected": 50}
                for source in import_public_sft.DEFAULT_SOURCES
            },
            "selected_source_counts": {
                source: 50
                for source in import_public_sft.DEFAULT_SOURCES
            },
            "dataset_revisions": {
                source: {
                    "dataset": import_public_sft.SOURCE_DATASETS[source][0],
                    "split": import_public_sft.SOURCE_DATASETS[source][1],
                    "revision": import_public_sft.SOURCE_DATASETS[source][2],
                }
                for source in import_public_sft.DEFAULT_SOURCES
            },
        },
    )
    audit = root / "data/sft/runs/300m/audit.json"
    _write_json(audit, _audit_report())
    curated_core_data = root / "data/sft/curated/core.jsonl"
    _write_jsonl(
        curated_core_data,
        [{"source": "curated_core", "messages": [{"role": "agi", "content": "B"}]}],
    )
    curated_core_metadata = root / "data/sft/curated/core.metadata.json"
    _write_json(curated_core_metadata, {"schema_version": 1, "count": 1})
    curated_core_audit = root / "data/sft/curated/core.audit.json"
    _write_json(curated_core_audit, _audit_report(mode="curated"))
    playful_style_data = root / "data/sft/styles/playful-direct.jsonl"
    _write_jsonl(
        playful_style_data,
        [
            {
                "source": "style_playful_direct",
                "messages": [{"role": "agi", "content": "C"}],
            }
        ],
    )
    playful_style_audit = root / "data/sft/styles/playful-direct.audit.json"
    _write_json(playful_style_audit, _audit_report(mode="style"))
    calm_style_data = root / "data/sft/styles/calm-precise.jsonl"
    _write_jsonl(
        calm_style_data,
        [
            {
                "source": "style_calm_precise",
                "messages": [{"role": "agi", "content": "D"}],
            }
        ],
    )
    calm_style_audit = root / "data/sft/styles/calm-precise.audit.json"
    _write_json(calm_style_audit, _audit_report(mode="style"))
    style_metadata = root / "data/sft/styles/styles.metadata.json"
    _write_json(style_metadata, {"schema_version": 1, "count": 2})
    eval_prompts = root / "data/sft/eval_prompts.jsonl"
    _write_jsonl(eval_prompts, [{"id": "identity-1", "prompt": "Who are you?"}])

    run_config = root / "data/sft/runs/300m/run-config.json"
    settings = _production_settings()
    _write_json(
        run_config,
        {
            "schema_version": 2,
            "base_checkpoint": base_identity,
            "inputs": {
                "public_jsonl": _artifact_identity(public_data, root),
                "public_metadata": _artifact_identity(public_metadata, root),
            },
            "settings": settings,
        },
    )

    for index, run_name in enumerate(RUN_NAMES, start=1):
        run_dir = root / "data/sft/runs/300m" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_archive(
            run_dir / "best.pt",
            payload=f"{run_name}-best-checkpoint".encode("ascii"),
        )
        _write_checkpoint_archive(
            run_dir / "final.pt",
            payload=f"{run_name}-final-checkpoint".encode("ascii"),
        )
        _write_jsonl(
            run_dir / "metrics.jsonl",
            [
                {
                    "step": 10,
                    "train_loss": 2.5 + index,
                    "validation_loss": 2.0,
                    "learning_rate": 0.000006,
                },
                {
                    "step": 20,
                    "train_loss": 2.0 + index,
                    "validation_loss": 1.5,
                    "learning_rate": 0.000004,
                },
                {
                    "step": 30,
                    "train_loss": 1.5 + index,
                    "validation_loss": 1.8,
                    "learning_rate": 0.000002,
                },
            ],
        )
        _write_json(run_dir / "evaluation.summary.json", _evaluation_summary())
        _write_jsonl(
            run_dir / "evaluation.jsonl",
            [{"prompt_id": "identity-1", "passed": True}],
        )

    return {
        "output": root / "data/sft/runs/300m/manifest.json",
        "base": base,
        "base_record": base_record,
        "public_data": public_data,
        "public_metadata": public_metadata,
        "curated_core_data": curated_core_data,
        "curated_core_metadata": curated_core_metadata,
        "curated_core_audit": curated_core_audit,
        "playful_style_data": playful_style_data,
        "playful_style_audit": playful_style_audit,
        "calm_style_data": calm_style_data,
        "calm_style_audit": calm_style_audit,
        "style_metadata": style_metadata,
        "eval_prompts": eval_prompts,
        "audit": audit,
        "run_config": run_config,
    }


def _evaluation_summary(*, ok: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ok": ok,
        "total_prompts": 60,
        "passed_prompts": 60 if ok else 59,
        "failed_prompts": 0 if ok else 1,
        "hard_failure_counts": {} if ok else {"empty_response": 1},
        "termination_counts": {"eos": 60},
        "category_counts": {"category:identity": 10},
        "shared_identical_answers": [],
        "aggregate_gates": {
            "eos_termination_rate": {
                "observed": 1.0,
                "required": 0.9,
                "passed": ok,
            }
        },
    }


def _audit_report(*, ok: bool = True, mode: str = "mixed") -> dict[str, object]:
    return {
        "ok": ok,
        "mode": mode,
        "conversation_count": 1700,
        "response_count": 2100,
        "source_counts": {"curated_core": 1500, "dolly": 200},
        "findings": [] if ok else [{"severity": "error", "code": "bad"}],
    }


def _production_settings() -> dict[str, object]:
    return {
        "pipeline": {"device": "cuda", "seed": 1337},
        "import": {
            "sources": "no_robots,dolly,openassistant,ultrachat",
            "max_rows_per_source": 50000,
            "max_examples_per_source": 5000,
            "max_context_tokens": 900,
            "max_messages": 8,
            "max_agi_chars": 1200,
            "min_agi_chars": 20,
        },
        "core": {
            "data": (
                "data/sft/curated/core.jsonl,"
                "data/sft/imported/public-mixed.jsonl"
            ),
            "source_weights": (
                "curated_core=4,no_robots=1.5,openassistant=1.25,"
                "dolly=1,ultrachat=0.8,wildchat=0,default=1"
            ),
            "steps": 3000,
            "batch": 2,
            "grad_accum_steps": 8,
            "lr": 0.000006,
            "lr_min": 0.000001,
            "lr_warmup_steps": 150,
            "weight_decay": 0.01,
            "checkpoint_interval": 250,
            "checkpoint_keep": 3,
            "validation_interval": 250,
        },
        "style": {
            "playful_data": (
                "data/sft/curated/core.jsonl,"
                "data/sft/styles/playful-direct.jsonl"
            ),
            "calm_data": (
                "data/sft/curated/core.jsonl,"
                "data/sft/styles/calm-precise.jsonl"
            ),
            "playful_source_weights": (
                "curated_core=1,style_playful_direct=7,default=1"
            ),
            "calm_source_weights": (
                "curated_core=1,style_calm_precise=7,default=1"
            ),
            "steps": 500,
            "batch": 2,
            "grad_accum_steps": 8,
            "lr": 0.0000015,
            "lr_min": 0.0000005,
            "lr_warmup_steps": 50,
            "weight_decay": 0.01,
            "checkpoint_interval": 100,
            "checkpoint_keep": 3,
            "validation_interval": 50,
        },
        "validation": {"fraction": 0.05, "batches": 10},
        "optimizer": {
            "mixed_precision": "float16",
            "fused_adamw": "auto",
            "activation_checkpointing": 1,
        },
        "evaluation": {
            "prompts": "data/sft/eval_prompts.jsonl",
            "device": "cuda",
            "seed": 1337,
            "temperature": 0.3,
            "top_k": 20,
            "repetition_penalty": 1.25,
            "repetition_window": 128,
            "min_eos_termination_rate": 0.8,
            "min_nonempty_response_rate": 1.0,
            "max_repetition_failure_rate": 0.1,
            "min_topic_reset_pass_rate": 0.8,
        },
    }


def _artifact_identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _reseal_public_metadata(paths: dict[str, Path], root: Path) -> None:
    run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
    run_config["inputs"]["public_metadata"] = _artifact_identity(
        paths["public_metadata"],
        root,
    )
    _write_json(paths["run_config"], run_config)


def _flatten_settings(
    settings: dict[str, object],
    prefix: str = "",
) -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    for key in sorted(settings):
        value = settings[key]
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.extend(_flatten_settings(value, dotted_key))
        else:
            flattened.append((dotted_key, value))
    return flattened


def _set_dotted(payload: dict[str, object], key: str, value: object) -> None:
    parts = key.split(".")
    target = payload
    for part in parts[:-1]:
        next_target = target[part]
        if not isinstance(next_target, dict):
            raise AssertionError(f"{part} is not an object")
        target = next_target
    target[parts[-1]] = value


def _delete_dotted(payload: dict[str, object], key: str) -> None:
    parts = key.split(".")
    target = payload
    for part in parts[:-1]:
        next_target = target[part]
        if not isinstance(next_target, dict):
            raise AssertionError(f"{part} is not an object")
        target = next_target
    del target[parts[-1]]


def _add_unconfigured_public_source(payload: dict[str, object]) -> None:
    sources = payload["sources"]
    revisions = payload["dataset_revisions"]
    if not isinstance(sources, dict) or not isinstance(revisions, dict):
        raise AssertionError("public metadata source records must be objects")
    sources["wildchat"] = {"selected": 1}
    revisions["wildchat"] = {
        "dataset": "allenai/WildChat",
        "split": "train",
        "revision": "f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4",
    }
    written_count = payload["written_count"]
    if not isinstance(written_count, int):
        raise AssertionError("public metadata written_count must be an integer")
    payload["written_count"] = written_count + 1


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_checkpoint_archive(path: Path, *, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")


if __name__ == "__main__":
    unittest.main()
