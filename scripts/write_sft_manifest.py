from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
RUN_NAMES = ("core", "playful", "calm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a validated manifest for a completed 300M SFT run.",
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output",
        default="data/sft/runs/300m/manifest.json",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--core-run-dir",
        default="data/sft/runs/300m/core",
    )
    parser.add_argument(
        "--playful-run-dir",
        default="data/sft/runs/300m/playful",
    )
    parser.add_argument(
        "--calm-run-dir",
        default="data/sft/runs/300m/calm",
    )
    parser.add_argument(
        "--public-import-metadata",
        default="data/sft/imported/public-mixed.metadata.json",
    )
    parser.add_argument(
        "--audit-report",
        default="data/sft/runs/300m/audit.json",
    )
    parser.add_argument(
        "--run-config",
        default="data/sft/runs/300m/run-config.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(args.repository_root).resolve()
    output_path = _resolve_path(args.output, repository_root)
    base_path = _resolve_path(args.base_checkpoint, repository_root)
    public_metadata_path = _resolve_path(
        args.public_import_metadata,
        repository_root,
    )
    audit_path = _resolve_path(args.audit_report, repository_root)
    run_config_path = _resolve_path(args.run_config, repository_root)
    run_dirs = {
        "core": _resolve_path(args.core_run_dir, repository_root),
        "playful": _resolve_path(args.playful_run_dir, repository_root),
        "calm": _resolve_path(args.calm_run_dir, repository_root),
    }

    manifest = build_manifest(
        repository_root=repository_root,
        base_path=base_path,
        public_metadata_path=public_metadata_path,
        audit_path=audit_path,
        run_config_path=run_config_path,
        run_dirs=run_dirs,
    )
    _write_json_atomic(output_path, manifest)
    print(f"SFT manifest: {_relative_path(output_path, repository_root)}")
    return 0


def build_manifest(
    *,
    repository_root: Path,
    base_path: Path,
    public_metadata_path: Path,
    audit_path: Path,
    run_config_path: Path,
    run_dirs: Mapping[str, Path],
) -> dict[str, Any]:
    root = repository_root.resolve()
    _require_within_repository(root, base_path)
    _require_within_repository(root, public_metadata_path)
    _require_within_repository(root, audit_path)
    _require_within_repository(root, run_config_path)
    if set(run_dirs) != set(RUN_NAMES):
        raise ValueError(
            "run directories must contain exactly core, playful, and calm"
        )
    for run_dir in run_dirs.values():
        _require_within_repository(root, run_dir)

    base_artifact = _artifact_record(
        _require_file(base_path, "base checkpoint"),
        root,
    )
    public_metadata = _load_json_object(
        _require_file(public_metadata_path, "public import metadata"),
        "public import metadata",
    )
    _validate_public_metadata(public_metadata)
    audit_report = _load_json_object(
        _require_file(audit_path, "audit report"),
        "audit report",
    )
    _validate_audit_report(audit_report)
    run_config = _load_json_object(
        _require_file(run_config_path, "run config"),
        "run config",
    )
    if not run_config:
        raise ValueError("run config must be a non-empty JSON object")

    runs = {
        run_name: _run_manifest(
            run_name=run_name,
            run_dir=run_dirs[run_name],
            repository_root=root,
        )
        for run_name in RUN_NAMES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_timestamp(),
        "base_checkpoint": base_artifact,
        "run_config": run_config,
        "run_config_artifact": _artifact_record(run_config_path, root),
        "source_metadata": {
            "audit_report": {
                **_artifact_record(audit_path, root),
                "summary": audit_report,
            },
            "public_import": {
                **_artifact_record(public_metadata_path, root),
                "summary": public_metadata,
            },
        },
        "runs": runs,
    }


def _run_manifest(
    *,
    run_name: str,
    run_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    best_path = _require_file(
        run_dir / "best.pt",
        f"required artifact for {run_name}: best.pt",
    )
    _validate_checkpoint_archive(best_path, run_name=run_name)
    metrics_path = _require_file(
        run_dir / "metrics.jsonl",
        f"required artifact for {run_name}: metrics.jsonl",
    )
    evaluation_summary_path = _require_file(
        run_dir / "evaluation.summary.json",
        f"required artifact for {run_name}: evaluation summary",
    )
    evaluation_summary = _load_json_object(
        evaluation_summary_path,
        f"{run_name} evaluation summary",
    )
    _validate_evaluation_summary(evaluation_summary, run_name=run_name)
    best_validation_metric = _best_validation_metric(
        metrics_path,
        run_name=run_name,
    )

    artifacts = {
        "best_checkpoint": _artifact_record(best_path, repository_root),
        "evaluation_summary": _artifact_record(
            evaluation_summary_path,
            repository_root,
        ),
        "metrics": _artifact_record(metrics_path, repository_root),
    }
    optional_artifacts = (
        ("evaluation_results", run_dir / "evaluation.jsonl"),
        ("final_checkpoint", run_dir / "final.pt"),
    )
    for artifact_name, artifact_path in optional_artifacts:
        if artifact_path.exists():
            artifacts[artifact_name] = _artifact_record(
                _require_file(
                    artifact_path,
                    f"required artifact for {run_name}: {artifact_name}",
                ),
                repository_root,
            )

    return {
        "artifacts": artifacts,
        "best_validation_metric": best_validation_metric,
        "evaluation_summary": evaluation_summary,
    }


def _best_validation_metric(
    metrics_path: Path,
    *,
    run_name: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{run_name} validation metric at "
                f"{metrics_path}:{line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"{run_name} validation metric at "
                f"{metrics_path}:{line_number} must be an object"
            )
        step = value.get("step")
        validation_loss = value.get("validation_loss")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(
                f"{run_name} validation metric at "
                f"{metrics_path}:{line_number} has an invalid step"
            )
        if validation_loss is None:
            continue
        if (
            not isinstance(validation_loss, (int, float))
            or isinstance(validation_loss, bool)
            or not math.isfinite(float(validation_loss))
        ):
            raise ValueError(
                f"{run_name} validation metric at "
                f"{metrics_path}:{line_number} has a non-finite validation loss"
            )
        candidates.append(value)
    if not candidates:
        raise ValueError(
            f"{run_name} validation metric history contains no finite "
            "validation loss"
        )
    return min(
        candidates,
        key=lambda metric: (
            float(metric["validation_loss"]),
            int(metric["step"]),
        ),
    )


def _validate_public_metadata(payload: Mapping[str, Any]) -> None:
    written_count = payload.get("written_count")
    sources = payload.get("sources")
    if (
        not isinstance(written_count, int)
        or isinstance(written_count, bool)
        or written_count <= 0
        or not isinstance(sources, dict)
        or not sources
    ):
        raise ValueError(
            "public import metadata must contain a positive written_count "
            "and non-empty sources"
        )
    selected_total = 0
    for source_name, source_summary in sources.items():
        if (
            not isinstance(source_name, str)
            or not source_name
            or not isinstance(source_summary, dict)
        ):
            raise ValueError("public import metadata has invalid sources")
        selected = source_summary.get("selected")
        if (
            not isinstance(selected, int)
            or isinstance(selected, bool)
            or selected < 0
        ):
            raise ValueError(
                "public import metadata source is missing a valid selected count"
            )
        selected_total += selected
    if selected_total != written_count:
        raise ValueError(
            "public import metadata written_count does not match selected sources"
        )


def _validate_checkpoint_archive(path: Path, *, run_name: str) -> None:
    label = f"{run_name} best checkpoint"
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{label} is not a valid PyTorch archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{label} is not a valid PyTorch archive: {path}") from error
    if corrupt_member is not None:
        raise ValueError(
            f"{label} contains a corrupt archive member: {corrupt_member}"
        )
    required_members = ("data.pkl", "version")
    if any(
        not any(
            name == required or name.endswith(f"/{required}")
            for name in names
        )
        for required in required_members
    ):
        raise ValueError(
            f"{label} is missing required PyTorch archive members: {path}"
        )


def _validate_audit_report(payload: Mapping[str, Any]) -> None:
    if payload.get("ok") is not True:
        raise ValueError("audit report did not pass")
    if not isinstance(payload.get("mode"), str):
        raise ValueError("audit report is missing its mode")
    for field in ("conversation_count", "response_count"):
        value = payload.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"audit report has invalid {field}")
    if not isinstance(payload.get("source_counts"), dict) or not payload[
        "source_counts"
    ]:
        raise ValueError("audit report has invalid source_counts")
    findings = payload.get("findings")
    if not isinstance(findings, list) or any(
        not isinstance(finding, dict) for finding in findings
    ):
        raise ValueError("audit report has invalid findings")
    if any(finding.get("severity") == "error" for finding in findings):
        raise ValueError("audit report contains error findings")


def _validate_evaluation_summary(
    payload: Mapping[str, Any],
    *,
    run_name: str,
) -> None:
    prefix = f"{run_name} evaluation summary"
    if payload.get("schema_version") != 1:
        raise ValueError(f"{prefix} has an unsupported schema version")
    if payload.get("ok") is not True:
        raise ValueError(f"{prefix} did not pass")
    total = payload.get("total_prompts")
    passed = payload.get("passed_prompts")
    failed = payload.get("failed_prompts")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for value in (total, passed, failed)
    ):
        raise ValueError(f"{prefix} has invalid prompt counts")
    if total <= 0 or passed + failed != total:
        raise ValueError(f"{prefix} has inconsistent prompt counts")
    if failed != 0 or passed != total:
        raise ValueError(f"{prefix} reports failed prompts despite ok=true")
    hard_failure_counts = payload.get("hard_failure_counts")
    if not isinstance(hard_failure_counts, dict):
        raise ValueError(f"{prefix} has invalid hard failure counts")
    if hard_failure_counts:
        raise ValueError(f"{prefix} reports hard failures despite ok=true")
    aggregate_gates = payload.get("aggregate_gates")
    if not isinstance(aggregate_gates, dict) or not aggregate_gates:
        raise ValueError(f"{prefix} has invalid aggregate gates")
    if any(
        not isinstance(gate, dict) or gate.get("passed") is not True
        for gate in aggregate_gates.values()
    ):
        raise ValueError(f"{prefix} contains a failed aggregate gate")


def _artifact_record(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path, repository_root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"missing required artifact ({label}): {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"empty required artifact ({label}): {path}")
    return path


def _resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _require_within_repository(repository_root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"artifact path must be inside repository root: {path}"
        ) from error


def _relative_path(path: Path, repository_root: Path) -> str:
    _require_within_repository(repository_root, path)
    return path.resolve().relative_to(repository_root).as_posix()


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
    temporary_path.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
