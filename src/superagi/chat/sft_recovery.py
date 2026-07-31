from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from superagi.model.checkpoint import save_checkpoint
from superagi.model.transformer import TransformerLM


RECOVERY_POINTER_FORMAT = "superagi-sft-recovery-pointer-v1"
RECOVERY_MANIFEST_FORMAT = "superagi-sft-recovery-generation-v1"
RECOVERY_COMMIT_BOUNDARIES = (
    "best_object",
    "latest",
    "metrics",
    "trainer_state",
    "generation_manifest",
    "pointer",
)


@dataclass(frozen=True)
class RecoveryBundle:
    run_dir: Path
    recovery_dir: Path
    generation_name: str
    generation_dir: Path
    latest_path: Path
    best_path: Path
    metrics_path: Path
    trainer_state_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    trainer_state: dict[str, Any]


def recovery_pointer_path(run_dir: Path) -> Path:
    return run_dir / "recovery-current.json"


def commit_recovery_bundle(
    run_dir: Path,
    *,
    step: int,
    model: TransformerLM,
    vocab: dict[str, Any],
    losses: Sequence[float],
    metrics: Sequence[Mapping[str, Any]],
    latest_metadata: Mapping[str, Any],
    best_improved: bool,
    best_metadata: Mapping[str, Any] | None,
    previous_bundle: RecoveryBundle | None,
    trainer_state: Mapping[str, Any],
    boundary_hook: Callable[[str], None],
) -> RecoveryBundle:
    recovery_dir = run_dir / "recovery"
    generations_dir = recovery_dir / "generations"
    best_dir = recovery_dir / "best"
    generations_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    generation_name = f"generation-{step:09d}-{uuid.uuid4().hex}"
    generation_dir = generations_dir / generation_name
    generation_dir.mkdir()

    best_path = _resolve_best_checkpoint(
        best_dir,
        model=model,
        vocab=vocab,
        losses=losses,
        metrics=metrics,
        best_improved=best_improved,
        best_metadata=best_metadata,
        previous_bundle=previous_bundle,
        boundary_hook=boundary_hook,
    )

    latest_path = generation_dir / "latest.pt"
    _atomic_checkpoint_write(
        latest_path,
        model=model,
        vocab=vocab,
        losses=losses,
        metrics=metrics,
        metadata=latest_metadata,
        boundary="latest",
        boundary_hook=boundary_hook,
    )

    metrics_path = generation_dir / "metrics.jsonl"
    _atomic_metrics_write(
        metrics_path,
        metrics,
        boundary="metrics",
        boundary_hook=boundary_hook,
    )

    trainer_state_path = generation_dir / "trainer-state.pt"
    _atomic_torch_write(
        trainer_state_path,
        dict(trainer_state),
        boundary="trainer_state",
        boundary_hook=boundary_hook,
    )

    manifest_path = generation_dir / "manifest.json"
    manifest = {
        "format": RECOVERY_MANIFEST_FORMAT,
        "generation": generation_name,
        "step": step,
        "files": {
            "latest": _file_record(recovery_dir, latest_path),
            "best": _file_record(recovery_dir, best_path),
            "metrics": _file_record(recovery_dir, metrics_path),
            "trainer_state": _file_record(recovery_dir, trainer_state_path),
        },
    }
    _atomic_json_write(
        manifest_path,
        manifest,
        boundary="generation_manifest",
        boundary_hook=boundary_hook,
    )

    pointer = {
        "format": RECOVERY_POINTER_FORMAT,
        "generation": generation_name,
        "manifest": str(manifest_path.relative_to(recovery_dir)),
        "manifest_sha256": _sha256_file(manifest_path),
    }
    _atomic_json_write(
        recovery_pointer_path(run_dir),
        pointer,
        boundary="pointer",
        boundary_hook=boundary_hook,
    )
    return load_committed_recovery_bundle(run_dir)


def load_committed_recovery_bundle(run_dir: Path) -> RecoveryBundle:
    pointer_path = recovery_pointer_path(run_dir)
    pointer = _read_json_mapping(pointer_path, "recovery pointer")
    if pointer.get("format") != RECOVERY_POINTER_FORMAT:
        raise SystemExit("recovery pointer has an unsupported or corrupt format")

    recovery_dir = run_dir / "recovery"
    generation_name = _required_string(pointer, "generation", "recovery pointer")
    manifest_path = _resolve_record_path(
        recovery_dir,
        _required_string(pointer, "manifest", "recovery pointer"),
    )
    if _sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise SystemExit("recovery generation manifest does not match its pointer")

    manifest = _read_json_mapping(manifest_path, "recovery generation manifest")
    if manifest.get("format") != RECOVERY_MANIFEST_FORMAT:
        raise SystemExit(
            "recovery generation manifest has an unsupported or corrupt format"
        )
    if manifest.get("generation") != generation_name:
        raise SystemExit("recovery pointer and generation manifest disagree")
    generation_dir = recovery_dir / "generations" / generation_name
    if manifest_path.parent.resolve(strict=False) != generation_dir.resolve(
        strict=False
    ):
        raise SystemExit("recovery generation manifest is outside its generation")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SystemExit("recovery generation manifest is missing file records")
    resolved_files = {
        name: _validated_file_record(recovery_dir, files, name)
        for name in ("latest", "best", "metrics", "trainer_state")
    }
    try:
        trainer_state = torch.load(
            resolved_files["trainer_state"],
            map_location="cpu",
        )
    except Exception as error:
        raise SystemExit(f"failed to load recovery trainer state: {error}") from error
    if not isinstance(trainer_state, dict):
        raise SystemExit("recovery trainer state must contain a mapping")

    return RecoveryBundle(
        run_dir=run_dir,
        recovery_dir=recovery_dir,
        generation_name=generation_name,
        generation_dir=generation_dir,
        latest_path=resolved_files["latest"],
        best_path=resolved_files["best"],
        metrics_path=resolved_files["metrics"],
        trainer_state_path=resolved_files["trainer_state"],
        manifest_path=manifest_path,
        manifest=manifest,
        trainer_state=trainer_state,
    )


def publish_recovery_aliases(
    bundle: RecoveryBundle,
    *,
    latest_path: Path,
    best_path: Path,
    metrics_path: Path,
) -> None:
    _atomic_copy(bundle.latest_path, latest_path)
    _atomic_copy(bundle.best_path, best_path)
    _atomic_copy(bundle.metrics_path, metrics_path)


def prune_recovery_generations(
    run_dir: Path,
    *,
    keep: int,
) -> None:
    recovery_dir = run_dir / "recovery"
    generations_dir = recovery_dir / "generations"
    if not generations_dir.is_dir():
        return
    retained_count = max(2, keep)
    generation_dirs = sorted(
        (
            path
            for path in generations_dir.iterdir()
            if path.is_dir() and path.name.startswith("generation-")
        ),
        key=lambda path: path.name,
    )
    for stale_dir in generation_dirs[:-retained_count]:
        shutil.rmtree(stale_dir)
    _prune_unreferenced_best_objects(recovery_dir)


def _resolve_best_checkpoint(
    best_dir: Path,
    *,
    model: TransformerLM,
    vocab: dict[str, Any],
    losses: Sequence[float],
    metrics: Sequence[Mapping[str, Any]],
    best_improved: bool,
    best_metadata: Mapping[str, Any] | None,
    previous_bundle: RecoveryBundle | None,
    boundary_hook: Callable[[str], None],
) -> Path:
    if not best_improved:
        if previous_bundle is None:
            raise RuntimeError("first recovery generation must establish best.pt")
        return previous_bundle.best_path
    if best_metadata is None:
        raise RuntimeError("improved best checkpoint requires best metadata")

    candidate_path = best_dir / f".best-candidate-{uuid.uuid4().hex}.pt"
    try:
        save_checkpoint(
            candidate_path,
            model=model,
            vocab=vocab,
            losses=losses,
            metrics=metrics,
            metadata=dict(best_metadata),
        )
        digest = _sha256_file(candidate_path)
        best_path = best_dir / f"best-{digest}.pt"
        boundary_hook("best_object")
        if best_path.exists():
            if _sha256_file(best_path) != digest:
                raise SystemExit("content-addressed best checkpoint hash collision")
            candidate_path.unlink()
        else:
            os.replace(candidate_path, best_path)
        return best_path
    finally:
        candidate_path.unlink(missing_ok=True)


def _atomic_checkpoint_write(
    path: Path,
    *,
    model: TransformerLM,
    vocab: dict[str, Any],
    losses: Sequence[float],
    metrics: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    boundary: str,
    boundary_hook: Callable[[str], None],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        save_checkpoint(
            temporary_path,
            model=model,
            vocab=vocab,
            losses=losses,
            metrics=metrics,
            metadata=dict(metadata),
        )
        boundary_hook(boundary)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_metrics_write(
    path: Path,
    metrics: Sequence[Mapping[str, Any]],
    *,
    boundary: str,
    boundary_hook: Callable[[str], None],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for metric in metrics:
                file.write(json.dumps(dict(metric), sort_keys=True))
                file.write("\n")
        boundary_hook(boundary)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_write(
    path: Path,
    payload: Mapping[str, Any],
    *,
    boundary: str,
    boundary_hook: Callable[[str], None],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        torch.save(dict(payload), temporary_path)
        boundary_hook(boundary)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_write(
    path: Path,
    payload: Mapping[str, Any],
    *,
    boundary: str,
    boundary_hook: Callable[[str], None],
) -> None:
    temporary_path = _temporary_sibling(path)
    try:
        temporary_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        boundary_hook(boundary)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(destination)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _file_record(recovery_dir: Path, path: Path) -> dict[str, str]:
    resolved_root = recovery_dir.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    return {
        "path": str(resolved_path.relative_to(resolved_root)),
        "sha256": _sha256_file(resolved_path),
    }


def _validated_file_record(
    recovery_dir: Path,
    files: Mapping[str, Any],
    name: str,
) -> Path:
    record = files.get(name)
    if not isinstance(record, dict):
        raise SystemExit(f"recovery generation is missing {name} record")
    relative_path = _required_string(record, "path", f"{name} record")
    path = _resolve_record_path(recovery_dir, relative_path)
    if not path.is_file():
        raise SystemExit(f"recovery generation {name} file is missing")
    if _sha256_file(path) != record.get("sha256"):
        raise SystemExit(f"recovery generation {name} file hash does not match")
    return path


def _resolve_record_path(recovery_dir: Path, relative_path: str) -> Path:
    candidate = (recovery_dir / relative_path).resolve(strict=False)
    resolved_root = recovery_dir.resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        raise SystemExit("recovery manifest path escapes the recovery directory")
    return candidate


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to read {description}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{description} must contain a mapping")
    return payload


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{description} is missing {key}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prune_unreferenced_best_objects(recovery_dir: Path) -> None:
    referenced: set[Path] = set()
    generations_dir = recovery_dir / "generations"
    for manifest_path in generations_dir.glob("generation-*/manifest.json"):
        try:
            manifest = _read_json_mapping(
                manifest_path,
                "retained recovery generation manifest",
            )
            files = manifest.get("files")
            if not isinstance(files, dict):
                continue
            best = files.get("best")
            if not isinstance(best, dict) or not isinstance(best.get("path"), str):
                continue
            referenced.add(_resolve_record_path(recovery_dir, best["path"]))
        except SystemExit:
            continue
    best_dir = recovery_dir / "best"
    if not best_dir.is_dir():
        return
    for best_path in best_dir.glob("best-*.pt"):
        if best_path.resolve(strict=False) not in referenced:
            best_path.unlink(missing_ok=True)
