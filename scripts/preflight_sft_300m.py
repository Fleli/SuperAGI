#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from superagi.ingestion.tokenizer import (  # noqa: E402
    AGI_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
)
from superagi.model.checkpoint import load_checkpoint  # noqa: E402


SCHEMA_VERSION = 1
REQUIRED_SPECIAL_TOKENS = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    AGI_TOKEN,
    SYSTEM_TOKEN,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and record immutable inputs for the 300M SFT workflow."
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--sha-record", required=True)
    parser.add_argument("--run-config", default="")
    parser.add_argument("--minimum-context-length", type=int, default=1024)
    parser.add_argument("--require-path", action="append", default=[])
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Run setting as a dotted key=value pair; may be repeated.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.minimum_context_length <= 0:
        raise ValueError("--minimum-context-length must be positive")
    if not args.verify_only and not args.run_config.strip():
        raise ValueError("--run-config is required unless --verify-only is used")

    root = Path(args.repository_root).resolve()
    base_path = _resolve_inside_root(
        args.base_checkpoint,
        root,
        label="base checkpoint",
    )
    sha_record_path = _resolve_inside_root(
        args.sha_record,
        root,
        label="base SHA record",
    )
    required_paths = tuple(
        _resolve_inside_root(value, root, label="required input")
        for value in args.require_path
    )
    run_config_path = (
        _resolve_inside_root(args.run_config, root, label="run config")
        if args.run_config.strip()
        else None
    )

    checkpoint_identity = inspect_base_checkpoint(
        base_path,
        repository_root=root,
        minimum_context_length=args.minimum_context_length,
    )
    _require_input_paths(required_paths)
    _validate_or_write_sha_record(
        sha_record_path,
        checkpoint_identity,
        verify_only=args.verify_only,
    )

    if not args.verify_only:
        if run_config_path is None:
            raise RuntimeError("run config path was not resolved")
        settings = parse_config_entries(args.config)
        run_config = {
            "schema_version": SCHEMA_VERSION,
            "base_checkpoint": checkpoint_identity,
            "settings": settings,
        }
        _validate_or_write_run_config(run_config_path, run_config)

    print(
        "SFT base preflight: "
        f"context_length={checkpoint_identity['context_length']} "
        f"sha256={checkpoint_identity['sha256']}",
        flush=True,
    )
    print(f"SFT base SHA record: {_relative(sha_record_path, root)}", flush=True)
    if run_config_path is not None:
        print(f"SFT run config: {_relative(run_config_path, root)}", flush=True)
    return 0


def inspect_base_checkpoint(
    path: Path,
    *,
    repository_root: Path,
    minimum_context_length: int,
) -> dict[str, Any]:
    _require_nonempty_file(path, "base checkpoint")
    checkpoint = load_checkpoint(path, map_location="cpu")
    context_length = int(checkpoint.config.context_length)
    if context_length < minimum_context_length:
        raise ValueError(
            "base checkpoint context length must be at least "
            f"{minimum_context_length}, got {context_length}"
        )

    special_token_ids: dict[str, int] = {}
    for token in REQUIRED_SPECIAL_TOKENS:
        try:
            token_id = checkpoint.tokenizer.special_token_id(token)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"base checkpoint tokenizer is missing required special token {token!r}"
            ) from error
        special_token_ids[token] = int(token_id)
    if len(set(special_token_ids.values())) != len(special_token_ids):
        raise ValueError("base checkpoint required special token IDs must be unique")

    return {
        "path": _relative(path, repository_root),
        "sha256": _sha256(path),
        "context_length": context_length,
        "special_token_ids": special_token_ids,
    }


def parse_config_entries(entries: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in entries:
        key, separator, raw_value = entry.partition("=")
        key_parts = key.strip().split(".")
        if (
            not separator
            or not raw_value.strip()
            or any(not part.strip() for part in key_parts)
        ):
            raise ValueError(f"invalid --config entry: {entry!r}")
        value = _parse_scalar(raw_value.strip())
        target = result
        for part in key_parts[:-1]:
            normalized_part = part.strip()
            existing = target.get(normalized_part)
            if existing is None:
                existing = {}
                target[normalized_part] = existing
            if not isinstance(existing, dict):
                raise ValueError(f"conflicting --config key: {key!r}")
            target = existing
        final_key = key_parts[-1].strip()
        if final_key in target:
            raise ValueError(f"duplicate --config key: {key!r}")
        target[final_key] = value
    return result


def _validate_or_write_sha_record(
    path: Path,
    checkpoint_identity: Mapping[str, Any],
    *,
    verify_only: bool,
) -> None:
    if path.exists():
        existing = _load_json_object(path, "base SHA record")
        if existing.get("sha256") != checkpoint_identity["sha256"]:
            raise ValueError(
                "base checkpoint SHA-256 changed after preflight: "
                f"expected {existing.get('sha256')}, "
                f"got {checkpoint_identity['sha256']}"
            )
        if existing != checkpoint_identity:
            raise ValueError("base checkpoint identity changed after preflight")
        return
    if verify_only:
        raise ValueError(f"missing base SHA record for verification: {path}")
    _write_json_atomic(path, checkpoint_identity)


def _validate_or_write_run_config(
    path: Path,
    run_config: Mapping[str, Any],
) -> None:
    if path.exists():
        existing = _load_json_object(path, "run config")
        if existing != run_config:
            raise ValueError(
                "run configuration changed after preflight; "
                "use a new SFT run directory for different settings"
            )
        return
    _write_json_atomic(path, run_config)


def _require_input_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        _require_nonempty_file(path, "required input")


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"empty {label}: {path}")


def _resolve_inside_root(value: str, root: Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside repository root: {resolved}") from error
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _parse_scalar(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (dict, list)):
        raise ValueError("--config values must be JSON scalars")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


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
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary_path.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"preflight error: {error}") from error
