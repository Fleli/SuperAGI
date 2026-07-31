from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import write_sft_manifest


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
            self.assertEqual(manifest["schema_version"], 1)
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
                manifest["run_config"],
                {
                    "device": "cuda",
                    "model": "300m",
                    "seed": 1337,
                },
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
                manifest["source_metadata"]["public_import"]["path"],
                "data/sft/imported/public-mixed.metadata.json",
            )
            self.assertEqual(
                manifest["source_metadata"]["audit_report"]["path"],
                "data/sft/runs/300m/audit.json",
            )

            expected_artifacts = {
                "best_checkpoint",
                "evaluation_results",
                "evaluation_summary",
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
            "playful evaluation summary": (
                "data/sft/runs/300m/playful/evaluation.summary.json"
            ),
            "calm metrics": "data/sft/runs/300m/calm/metrics.jsonl",
            "public import metadata": (
                "data/sft/imported/public-mixed.metadata.json"
            ),
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
        }
        for label, (relative_path, payload, message) in invalid_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                paths = _write_complete_run(root)
                (root / relative_path).write_text(
                    json.dumps(payload) + "\n",
                    encoding="utf-8",
                )

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
    return [
        "--repository-root",
        str(root),
        "--output",
        str(paths["output"]),
        "--base-checkpoint",
        str(paths["base"]),
        "--core-run-dir",
        str(root / "data/sft/runs/300m/core"),
        "--playful-run-dir",
        str(root / "data/sft/runs/300m/playful"),
        "--calm-run-dir",
        str(root / "data/sft/runs/300m/calm"),
        "--public-import-metadata",
        str(paths["public_metadata"]),
        "--audit-report",
        str(paths["audit"]),
        "--run-config",
        str(paths["run_config"]),
    ]


def _write_complete_run(root: Path) -> dict[str, Path]:
    base = root / "base.pt"
    base.write_bytes(b"base-checkpoint")

    public_metadata = root / "data/sft/imported/public-mixed.metadata.json"
    _write_json(
        public_metadata,
        {
            "written_count": 200,
            "sources": {
                "dolly": {"selected": 100},
                "no_robots": {"selected": 100},
            },
            "selected_source_counts": {"dolly": 100, "no_robots": 100},
        },
    )
    audit = root / "data/sft/runs/300m/audit.json"
    _write_json(audit, _audit_report())
    run_config = root / "data/sft/runs/300m/run-config.json"
    _write_json(
        run_config,
        {"seed": 1337, "device": "cuda", "model": "300m"},
    )

    for index, run_name in enumerate(RUN_NAMES, start=1):
        run_dir = root / "data/sft/runs/300m" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_archive(
            run_dir / "best.pt",
            payload=f"{run_name}-best-checkpoint".encode("ascii"),
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
        "public_metadata": public_metadata,
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


def _audit_report(*, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "mode": "mixed",
        "conversation_count": 1700,
        "response_count": 2100,
        "source_counts": {"curated_core": 1500, "dolly": 200},
        "findings": [] if ok else [{"severity": "error", "code": "bad"}],
    }


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
