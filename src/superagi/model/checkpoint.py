from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from superagi.ingestion.tokenizer import (
    EOS_TOKEN,
    TokenizerLike,
    normalize_tokenizer_payload,
    tokenizer_from_payload,
)
from superagi.model.transformer import TransformerConfig, TransformerLM


CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: TransformerLM
    config: TransformerConfig
    tokenizer: TokenizerLike
    vocab: dict[str, Any]
    losses: list[float]
    metrics: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingState:
    model: TransformerLM
    config: TransformerConfig
    vocab: dict[str, Any]
    previous_losses: list[float]
    previous_metrics: list[dict[str, Any]]
    metadata: dict[str, Any]
    resumed_from: Path | None


def save_checkpoint(
    path: Path | str,
    *,
    model: TransformerLM,
    vocab: dict[str, Any],
    losses: Iterable[float] | None = None,
    metrics: Iterable[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "vocab": _normalize_vocab(vocab),
            "losses": list(losses or []),
            "metrics": list(metrics or []),
            "metadata": dict(metadata or {}),
        },
        checkpoint_path,
    )
    return checkpoint_path


def retain_checkpoint_snapshot(
    source_path: Path | str,
    snapshot_dir: Path | str,
    *,
    step: int,
    keep: int = 5,
) -> Path | None:
    if step < 0:
        raise ValueError("step must be non-negative")
    if keep < 0:
        raise ValueError("keep must be non-negative")
    if keep == 0:
        return None

    checkpoint_path = Path(source_path)
    snapshots_path = Path(snapshot_dir)
    snapshots_path.mkdir(parents=True, exist_ok=True)
    retained_path = snapshots_path / f"checkpoint-step-{step:09d}.pt"
    temporary_path = retained_path.with_suffix(f"{retained_path.suffix}.tmp")
    shutil.copy2(checkpoint_path, temporary_path)
    temporary_path.replace(retained_path)
    prune_checkpoint_snapshots(snapshots_path, keep=keep)
    return retained_path


def prune_checkpoint_snapshots(
    snapshot_dir: Path | str,
    *,
    keep: int,
    pattern: str = "checkpoint-step-*.pt",
) -> list[Path]:
    if keep < 0:
        raise ValueError("keep must be non-negative")

    snapshots_path = Path(snapshot_dir)
    snapshots = sorted(snapshots_path.glob(pattern), key=lambda path: path.name)
    stale_snapshots = snapshots if keep == 0 else snapshots[:-keep]
    deleted_paths = []
    for snapshot_path in stale_snapshots:
        snapshot_path.unlink(missing_ok=True)
        deleted_paths.append(snapshot_path)
    return deleted_paths


def prepare_model_for_training(
    *,
    vocab: dict[str, Any],
    config: TransformerConfig,
    resume_path: Path | str | None = None,
) -> TrainingState:
    normalized_vocab = _normalize_vocab(vocab)
    if resume_path is None or str(resume_path) == "":
        _validate_config_vocab_size(config, normalized_vocab)
        return TrainingState(
            model=TransformerLM(config),
            config=config,
            vocab=normalized_vocab,
            previous_losses=[],
            previous_metrics=[],
            metadata={},
            resumed_from=None,
        )

    checkpoint_path = Path(resume_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if _tokenizer_identity(checkpoint.vocab) != _tokenizer_identity(normalized_vocab):
        raise ValueError(
            "checkpoint vocab does not match processed vocab; "
            "resume with the same tokenizer vocabulary or train from scratch"
        )
    return TrainingState(
        model=checkpoint.model,
        config=checkpoint.config,
        vocab=normalized_vocab,
        previous_losses=checkpoint.losses,
        previous_metrics=checkpoint.metrics,
        metadata=checkpoint.metadata,
        resumed_from=checkpoint_path,
    )


def load_checkpoint(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")

    config = TransformerConfig(**_required_mapping(payload, "model_config"))
    vocab = _normalize_vocab(_required_mapping(payload, "vocab"))
    _validate_config_vocab_size(config, vocab)
    model = TransformerLM(config)
    model.load_state_dict(_migrate_model_state_dict(payload["model_state"]))
    model.eval()

    return LoadedCheckpoint(
        model=model,
        config=config,
        tokenizer=tokenizer_from_payload(vocab),
        vocab=vocab,
        losses=list(payload.get("losses", [])),
        metrics=list(payload.get("metrics", [])),
        metadata=dict(payload.get("metadata", {})),
    )


@torch.no_grad()
def generate_from_checkpoint(
    path: Path | str,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    on_text: Callable[[str], None] | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt must contain at least one character")
    if seed is not None:
        torch.manual_seed(seed)

    torch_device = _resolve_device(device)
    checkpoint = load_checkpoint(path, map_location="cpu")
    checkpoint.model.to(torch_device)
    prompt_ids = checkpoint.tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=torch_device,
    )
    streamed_token_ids = list(prompt_ids)
    streamed_text = checkpoint.tokenizer.decode(streamed_token_ids)
    eos_token_id = _special_token_id(checkpoint.tokenizer, EOS_TOKEN)

    def emit_text(token_id: int) -> None:
        nonlocal streamed_text
        if on_text is not None:
            streamed_token_ids.append(token_id)
            next_text = checkpoint.tokenizer.decode(streamed_token_ids)
            if next_text.startswith(streamed_text):
                on_text(next_text[len(streamed_text) :])
            else:
                on_text(checkpoint.tokenizer.decode([token_id]))
            streamed_text = next_text

    generated = checkpoint.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        stop_token_ids={eos_token_id} if eos_token_id is not None else None,
        on_token=emit_text if on_text is not None else None,
    )
    generated_ids = generated[0].cpu().tolist()
    generated_ids = _trim_after_token(
        generated_ids,
        token_id=eos_token_id,
        start_index=len(prompt_ids),
    )
    return checkpoint.tokenizer.decode(generated_ids)


def _normalize_vocab(vocab: dict[str, Any]) -> dict[str, Any]:
    return normalize_tokenizer_payload(vocab)


def _tokenizer_identity(vocab: dict[str, Any]) -> dict[str, Any]:
    return tokenizer_from_payload(vocab).to_payload()


def _validate_config_vocab_size(
    config: TransformerConfig,
    vocab: dict[str, Any],
) -> None:
    if config.vocab_size != int(vocab["vocab_size"]):
        raise ValueError("config vocab_size must match vocab size")


def _migrate_model_state_dict(model_state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint field 'model_state' must be a mapping")

    migrated_state = dict(model_state)
    qkv_suffix = ".attention.qkv_proj.weight"
    qkv_weight_keys = [
        key for key in migrated_state if isinstance(key, str) and key.endswith(qkv_suffix)
    ]
    block_prefixes = [key[: -len(qkv_suffix)] for key in qkv_weight_keys]
    legacy_q_suffix = ".attention.q_proj.weight"
    legacy_q_weight_keys = [
        key
        for key in migrated_state
        if isinstance(key, str) and key.endswith(legacy_q_suffix)
    ]
    block_prefixes.extend(key[: -len(legacy_q_suffix)] for key in legacy_q_weight_keys)

    for block_prefix in sorted(set(block_prefixes)):
        qkv_weight_key = f"{block_prefix}.attention.qkv_proj.weight"
        if qkv_weight_key in migrated_state:
            continue

        q_weight_key = f"{block_prefix}.attention.q_proj.weight"
        k_weight_key = f"{block_prefix}.attention.k_proj.weight"
        v_weight_key = f"{block_prefix}.attention.v_proj.weight"
        legacy_weight_keys = (q_weight_key, k_weight_key, v_weight_key)
        if not all(key in migrated_state for key in legacy_weight_keys):
            continue

        migrated_state[qkv_weight_key] = torch.cat(
            [migrated_state.pop(key) for key in legacy_weight_keys],
            dim=0,
        )

        q_bias_key = f"{block_prefix}.attention.q_proj.bias"
        k_bias_key = f"{block_prefix}.attention.k_proj.bias"
        v_bias_key = f"{block_prefix}.attention.v_proj.bias"
        legacy_bias_keys = (q_bias_key, k_bias_key, v_bias_key)
        if all(key in migrated_state for key in legacy_bias_keys):
            migrated_state[f"{block_prefix}.attention.qkv_proj.bias"] = torch.cat(
                [migrated_state.pop(key) for key in legacy_bias_keys],
                dim=0,
            )

    return migrated_state


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint field {key!r} must be a mapping")
    return value


def _resolve_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _special_token_id(tokenizer: TokenizerLike, token: str) -> int | None:
    token_id_lookup = getattr(tokenizer, "special_token_id", None)
    if token_id_lookup is None:
        return None
    try:
        return int(token_id_lookup(token))
    except ValueError:
        return None


def _trim_after_token(
    token_ids: list[int],
    *,
    token_id: int | None,
    start_index: int,
) -> list[int]:
    if token_id is None:
        return token_ids
    for index, current_token_id in enumerate(token_ids[start_index:], start=start_index):
        if current_token_id == token_id:
            return token_ids[:index]
    return token_ids
