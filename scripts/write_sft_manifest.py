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

if __package__:
    from scripts.import_public_sft import SOURCE_DATASETS
    from scripts.preflight_sft_300m import parse_config_entries
else:
    from import_public_sft import SOURCE_DATASETS
    from preflight_sft_300m import parse_config_entries


SCHEMA_VERSION = 2
RUN_CONFIG_SCHEMA_VERSION = 2
RUN_NAMES = ("core", "playful", "calm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a validated manifest for a completed 300M SFT run.",
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output",
        default="data/sft/runs/300m-v2/manifest.json",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-sha-record", required=True)
    parser.add_argument(
        "--core-run-dir",
        default="data/sft/runs/300m-v2/core",
    )
    parser.add_argument(
        "--playful-run-dir",
        default="data/sft/runs/300m-v2/playful",
    )
    parser.add_argument(
        "--calm-run-dir",
        default="data/sft/runs/300m-v2/calm",
    )
    parser.add_argument(
        "--public-import-data",
        default="data/sft/runs/300m-v2/inputs/public-mixed.jsonl",
    )
    parser.add_argument(
        "--public-import-metadata",
        default="data/sft/runs/300m-v2/inputs/public-mixed.metadata.json",
    )
    parser.add_argument(
        "--curated-core-data",
        default="data/sft/curated/core.jsonl",
    )
    parser.add_argument(
        "--curated-core-metadata",
        default="data/sft/curated/core.metadata.json",
    )
    parser.add_argument(
        "--curated-core-audit",
        default="data/sft/curated/core.audit.json",
    )
    parser.add_argument(
        "--behavior-identity-reset-data",
        default="data/sft/curated/behavior-identity-reset.jsonl",
    )
    parser.add_argument(
        "--behavior-direct-current-data",
        default="data/sft/curated/behavior-direct-current.jsonl",
    )
    parser.add_argument(
        "--playful-style-data",
        default="data/sft/styles/playful-direct.jsonl",
    )
    parser.add_argument(
        "--playful-style-audit",
        default="data/sft/styles/playful-direct.audit.json",
    )
    parser.add_argument(
        "--calm-style-data",
        default="data/sft/styles/calm-precise.jsonl",
    )
    parser.add_argument(
        "--calm-style-audit",
        default="data/sft/styles/calm-precise.audit.json",
    )
    parser.add_argument(
        "--style-metadata",
        default="data/sft/styles/styles.metadata.json",
    )
    parser.add_argument(
        "--eval-prompts",
        default="data/sft/eval_prompts.jsonl",
    )
    parser.add_argument(
        "--audit-report",
        default="data/sft/runs/300m-v2/audit.json",
    )
    parser.add_argument(
        "--run-config",
        default="data/sft/runs/300m-v2/run-config.json",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Expected active production setting as dotted key=value; repeat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config:
        raise ValueError(
            "manifest requires expected active production settings via --config"
        )
    expected_settings = parse_config_entries(args.config)
    repository_root = Path(args.repository_root).resolve()
    output_path = _resolve_path(args.output, repository_root)
    base_path = _resolve_path(args.base_checkpoint, repository_root)
    base_record_path = _resolve_path(args.base_sha_record, repository_root)
    audit_path = _resolve_path(args.audit_report, repository_root)
    run_config_path = _resolve_path(args.run_config, repository_root)
    source_paths = {
        "public_jsonl": _resolve_path(args.public_import_data, repository_root),
        "public_metadata": _resolve_path(
            args.public_import_metadata,
            repository_root,
        ),
        "curated_core_jsonl": _resolve_path(
            args.curated_core_data,
            repository_root,
        ),
        "curated_core_metadata": _resolve_path(
            args.curated_core_metadata,
            repository_root,
        ),
        "curated_core_audit": _resolve_path(
            args.curated_core_audit,
            repository_root,
        ),
        "behavior_identity_reset_jsonl": _resolve_path(
            args.behavior_identity_reset_data,
            repository_root,
        ),
        "behavior_direct_current_jsonl": _resolve_path(
            args.behavior_direct_current_data,
            repository_root,
        ),
        "playful_style_jsonl": _resolve_path(
            args.playful_style_data,
            repository_root,
        ),
        "playful_style_audit": _resolve_path(
            args.playful_style_audit,
            repository_root,
        ),
        "calm_style_jsonl": _resolve_path(
            args.calm_style_data,
            repository_root,
        ),
        "calm_style_audit": _resolve_path(
            args.calm_style_audit,
            repository_root,
        ),
        "style_metadata": _resolve_path(args.style_metadata, repository_root),
        "evaluation_prompts": _resolve_path(args.eval_prompts, repository_root),
        "mixed_audit": audit_path,
    }
    run_dirs = {
        "core": _resolve_path(args.core_run_dir, repository_root),
        "playful": _resolve_path(args.playful_run_dir, repository_root),
        "calm": _resolve_path(args.calm_run_dir, repository_root),
    }

    manifest = build_manifest(
        repository_root=repository_root,
        base_path=base_path,
        base_record_path=base_record_path,
        source_paths=source_paths,
        run_config_path=run_config_path,
        run_dirs=run_dirs,
        expected_settings=expected_settings,
    )
    _write_json_atomic(output_path, manifest)
    print(f"SFT manifest: {_relative_path(output_path, repository_root)}")
    return 0


def build_manifest(
    *,
    repository_root: Path,
    base_path: Path,
    base_record_path: Path,
    source_paths: Mapping[str, Path],
    run_config_path: Path,
    run_dirs: Mapping[str, Path],
    expected_settings: Mapping[str, Any],
) -> dict[str, Any]:
    root = repository_root.resolve()
    _require_within_repository(root, base_path)
    _require_within_repository(root, base_record_path)
    _require_within_repository(root, run_config_path)
    for source_path in source_paths.values():
        _require_within_repository(root, source_path)
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
    base_record = _load_json_object(
        _require_file(base_record_path, "base SHA record"),
        "base SHA record",
    )
    _validate_base_record(
        base_record,
        base_artifact=base_artifact,
    )
    source_artifacts = _source_artifacts(source_paths, repository_root=root)
    audit_report = _load_json_object(
        source_paths["mixed_audit"],
        "audit report",
    )
    _validate_audit_report(audit_report)
    run_config = _load_json_object(
        _require_file(run_config_path, "run config"),
        "run config",
    )
    _validate_run_config(
        run_config,
        base_record=base_record,
        source_artifacts=source_artifacts,
        source_paths=source_paths,
        repository_root=root,
        expected_settings=expected_settings,
    )
    public_metadata = _load_json_object(
        source_paths["public_metadata"],
        "public import metadata",
    )
    _validate_public_metadata(
        public_metadata,
        expected_sources=_configured_public_sources(expected_settings),
    )

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
        "base_checkpoint_record": {
            **_artifact_record(base_record_path, root),
            "identity": base_record,
        },
        "run_config": run_config,
        "run_config_artifact": _artifact_record(run_config_path, root),
        "source_artifacts": source_artifacts,
        "runs": runs,
    }


def _source_artifacts(
    source_paths: Mapping[str, Path],
    *,
    repository_root: Path,
) -> dict[str, dict[str, Any]]:
    expected_names = {
        "public_jsonl",
        "public_metadata",
        "curated_core_jsonl",
        "curated_core_metadata",
        "curated_core_audit",
        "behavior_identity_reset_jsonl",
        "behavior_direct_current_jsonl",
        "playful_style_jsonl",
        "playful_style_audit",
        "calm_style_jsonl",
        "calm_style_audit",
        "style_metadata",
        "evaluation_prompts",
        "mixed_audit",
    }
    if set(source_paths) != expected_names:
        raise ValueError("source paths do not match the manifest schema")

    artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        path = _require_file(
            source_paths[name],
            f"required artifact for source input: {name}",
        )
        artifacts[name] = _artifact_record(path, repository_root)

    metadata_names = (
        "curated_core_metadata",
        "style_metadata",
    )
    for name in metadata_names:
        payload = _load_json_object(source_paths[name], name.replace("_", " "))
        if not payload:
            raise ValueError(f"{name.replace('_', ' ')} must not be empty")
        artifacts[name]["summary"] = payload

    audit_names = (
        "curated_core_audit",
        "playful_style_audit",
        "calm_style_audit",
        "mixed_audit",
    )
    for name in audit_names:
        payload = _load_json_object(source_paths[name], name.replace("_", " "))
        _validate_audit_report(payload)
        artifacts[name]["summary"] = payload

    public_metadata = _load_json_object(
        source_paths["public_metadata"],
        "public import metadata",
    )
    artifacts["public_metadata"]["summary"] = public_metadata
    return artifacts


def _validate_base_record(
    payload: Mapping[str, Any],
    *,
    base_artifact: Mapping[str, Any],
) -> None:
    if payload.get("path") != base_artifact["path"]:
        raise ValueError("base SHA record path does not match the base checkpoint")
    if payload.get("sha256") != base_artifact["sha256"]:
        raise ValueError("base SHA record hash does not match the base checkpoint")
    context_length = payload.get("context_length")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < 1024
    ):
        raise ValueError("base SHA record has an invalid context length")
    special_token_ids = payload.get("special_token_ids")
    required_tokens = {"<pad>", "<bos>", "<eos>", "<user>", "<agi>", "<system>"}
    if (
        not isinstance(special_token_ids, dict)
        or set(special_token_ids) != required_tokens
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in special_token_ids.values()
        )
        or len(set(special_token_ids.values())) != len(required_tokens)
    ):
        raise ValueError("base SHA record has invalid special token identities")


def _validate_run_config(
    payload: Mapping[str, Any],
    *,
    base_record: Mapping[str, Any],
    source_artifacts: Mapping[str, Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    repository_root: Path,
    expected_settings: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"run config must use schema_version {RUN_CONFIG_SCHEMA_VERSION}"
        )
    if payload.get("base_checkpoint") != base_record:
        raise ValueError(
            "run config base checkpoint identity does not match the "
            "base SHA record"
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "public_jsonl",
        "public_metadata",
    }:
        raise ValueError(
            "run config inputs must contain public_jsonl and public_metadata"
        )
    for name in ("public_jsonl", "public_metadata"):
        expected = {
            key: source_artifacts[name][key]
            for key in ("path", "sha256", "size_bytes")
        }
        if inputs.get(name) != expected:
            raise ValueError(
                f"run config {name} identity does not match the actual artifact"
            )

    sealed_source_names = {
        "curated_core_jsonl",
        "behavior_identity_reset_jsonl",
        "behavior_direct_current_jsonl",
        "playful_style_jsonl",
        "calm_style_jsonl",
        "evaluation_prompts",
    }
    sealed_inputs = payload.get("sealed_inputs")
    if (
        not isinstance(sealed_inputs, dict)
        or set(sealed_inputs) != sealed_source_names
    ):
        raise ValueError(
            "run config sealed_inputs must contain every static training and "
            "evaluation input"
        )
    for name in sorted(sealed_source_names):
        expected = {
            key: source_artifacts[name][key]
            for key in ("path", "sha256", "size_bytes")
        }
        if sealed_inputs.get(name) != expected:
            raise ValueError(
                f"run config sealed input {name} does not match the actual artifact"
            )

    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("run config settings must be a JSON object")
    if settings != expected_settings:
        raise ValueError(
            "run configuration changed after preflight; "
            "use a new SFT run directory for different settings"
        )
    required_sections = {
        "pipeline",
        "import",
        "core",
        "style",
        "validation",
        "optimizer",
        "evaluation",
    }
    if not required_sections.issubset(settings):
        missing = sorted(required_sections - set(settings))
        raise ValueError(
            f"run config settings are missing sections: {', '.join(missing)}"
        )
    for section_name in required_sections:
        if not isinstance(settings[section_name], dict):
            raise ValueError(
                f"run config settings.{section_name} must be a JSON object"
            )

    nonempty_string_fields = (
        "pipeline.device",
        "import.sources",
        "core.data",
        "core.source_weights",
        "style.playful_data",
        "style.calm_data",
        "style.playful_source_weights",
        "style.calm_source_weights",
        "optimizer.mixed_precision",
        "optimizer.fused_adamw",
        "evaluation.prompts",
        "evaluation.device",
    )
    for field in nonempty_string_fields:
        value = _setting(payload, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"run config {field} must be a non-empty string")

    positive_integer_fields = (
        "pipeline.seed",
        "import.max_rows_per_source",
        "import.max_examples_per_source",
        "import.max_context_tokens",
        "import.max_messages",
        "import.max_agi_chars",
        "import.min_agi_chars",
        "core.steps",
        "core.batch",
        "core.grad_accum_steps",
        "core.lr_warmup_steps",
        "core.checkpoint_interval",
        "core.checkpoint_keep",
        "core.validation_interval",
        "style.steps",
        "style.batch",
        "style.grad_accum_steps",
        "style.lr_warmup_steps",
        "style.checkpoint_interval",
        "style.checkpoint_keep",
        "style.validation_interval",
        "validation.batches",
        "evaluation.seed",
        "evaluation.top_k",
        "evaluation.repetition_window",
    )
    for field in positive_integer_fields:
        value = _setting(payload, field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"run config {field} must be a positive integer")

    positive_number_fields = (
        "core.lr",
        "core.lr_min",
        "core.weight_decay",
        "style.lr",
        "style.lr_min",
        "style.weight_decay",
        "validation.fraction",
        "evaluation.temperature",
        "evaluation.repetition_penalty",
        "evaluation.min_eos_termination_rate",
        "evaluation.min_nonempty_response_rate",
        "evaluation.min_topic_reset_pass_rate",
    )
    for field in positive_number_fields:
        value = _setting(payload, field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"run config {field} must be a positive number")
    max_repetition_failure_rate = _setting(
        payload,
        "evaluation.max_repetition_failure_rate",
    )
    if (
        not isinstance(max_repetition_failure_rate, (int, float))
        or isinstance(max_repetition_failure_rate, bool)
        or not math.isfinite(float(max_repetition_failure_rate))
        or not 0 <= float(max_repetition_failure_rate) <= 1
    ):
        raise ValueError(
            "run config evaluation.max_repetition_failure_rate "
            "must be between zero and one"
        )
    activation_checkpointing = _setting(
        payload,
        "optimizer.activation_checkpointing",
    )
    if activation_checkpointing not in (0, 1, False, True):
        raise ValueError(
            "run config optimizer.activation_checkpointing must be boolean"
        )

    expected_paths = {
        "core data": (
            "core.data",
            ",".join(
                (
                    _relative_path(
                        source_paths["curated_core_jsonl"],
                        repository_root,
                    ),
                    _relative_path(source_paths["public_jsonl"], repository_root),
                )
            ),
        ),
        "playful style data": (
            "style.playful_data",
            ",".join(
                (
                    _relative_path(
                        source_paths["curated_core_jsonl"],
                        repository_root,
                    ),
                    _relative_path(
                        source_paths["playful_style_jsonl"],
                        repository_root,
                    ),
                )
            ),
        ),
        "calm style data": (
            "style.calm_data",
            ",".join(
                (
                    _relative_path(
                        source_paths["curated_core_jsonl"],
                        repository_root,
                    ),
                    _relative_path(
                        source_paths["calm_style_jsonl"],
                        repository_root,
                    ),
                )
            ),
        ),
        "evaluation prompts": (
            "evaluation.prompts",
            _relative_path(source_paths["evaluation_prompts"], repository_root),
        ),
    }
    for label, (field, expected) in expected_paths.items():
        if _setting(payload, field) != expected:
            raise ValueError(
                f"run config {label} path does not match the manifest input"
            )


def _setting(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload["settings"]
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"run config is missing settings.{dotted_key}")
        value = value[part]
    return value


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
    final_path = _require_file(
        run_dir / "final.pt",
        f"required artifact for {run_name}: final.pt",
    )
    _validate_checkpoint_archive(
        final_path,
        run_name=run_name,
        checkpoint_name="final",
    )
    metrics_path = _require_file(
        run_dir / "metrics.jsonl",
        f"required artifact for {run_name}: metrics.jsonl",
    )
    evaluation_results_path = _require_file(
        run_dir / "evaluation.jsonl",
        f"required artifact for {run_name}: evaluation.jsonl",
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

    best_checkpoint = _artifact_record(best_path, repository_root)
    final_checkpoint = _artifact_record(final_path, repository_root)
    if (
        final_checkpoint["sha256"] != best_checkpoint["sha256"]
        or final_checkpoint["size_bytes"] != best_checkpoint["size_bytes"]
    ):
        raise ValueError(
            f"{run_name} final checkpoint does not match evaluated best checkpoint"
        )

    artifacts = {
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "evaluation_results": _artifact_record(
            evaluation_results_path,
            repository_root,
        ),
        "evaluation_summary": _artifact_record(
            evaluation_summary_path,
            repository_root,
        ),
        "metrics": _artifact_record(metrics_path, repository_root),
    }

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


def _configured_public_sources(
    expected_settings: Mapping[str, Any],
) -> tuple[str, ...]:
    import_settings = expected_settings.get("import")
    if not isinstance(import_settings, dict):
        raise ValueError("expected settings import section must be a JSON object")
    raw_sources = import_settings.get("sources")
    if not isinstance(raw_sources, str):
        raise ValueError("expected settings import.sources must be a string")
    sources = tuple(
        source.strip()
        for source in raw_sources.split(",")
        if source.strip()
    )
    if not sources or len(sources) != len(set(sources)):
        raise ValueError(
            "expected settings import.sources must contain unique source names"
        )
    unknown_sources = sorted(set(sources) - set(SOURCE_DATASETS))
    if unknown_sources:
        raise ValueError(
            "expected settings import.sources contains unknown sources: "
            + ", ".join(unknown_sources)
        )
    return sources


def _validate_public_metadata(
    payload: Mapping[str, Any],
    *,
    expected_sources: tuple[str, ...],
) -> None:
    written_count = payload.get("written_count")
    sources = payload.get("sources")
    dataset_revisions = payload.get("dataset_revisions")
    if (
        not isinstance(written_count, int)
        or isinstance(written_count, bool)
        or written_count <= 0
        or not isinstance(sources, dict)
        or set(sources) != set(expected_sources)
    ):
        raise ValueError(
            "public import metadata must contain a positive written_count "
            "and exactly the configured sources"
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
    if (
        not isinstance(dataset_revisions, dict)
        or set(dataset_revisions) != set(expected_sources)
    ):
        raise ValueError(
            "public import metadata dataset revisions must match its sources"
        )
    for source_name in expected_sources:
        dataset, split, revision = SOURCE_DATASETS[source_name]
        expected_identity = {
            "dataset": dataset,
            "split": split,
            "revision": revision,
        }
        if dataset_revisions.get(source_name) != expected_identity:
            raise ValueError(
                "public import metadata dataset identity does not match "
                f"the pinned importer source {source_name}"
            )


def _validate_checkpoint_archive(
    path: Path,
    *,
    run_name: str,
    checkpoint_name: str = "best",
) -> None:
    label = f"{run_name} {checkpoint_name} checkpoint"
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
