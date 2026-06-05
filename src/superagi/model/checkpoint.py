from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from superagi.ingestion.tokenizer import CharTokenizer
from superagi.model.transformer import TransformerConfig, TransformerLM


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: TransformerLM
    config: TransformerConfig
    tokenizer: CharTokenizer
    vocab: dict[str, list[str]]
    losses: list[float]
    metadata: dict[str, Any]


def save_checkpoint(
    path: Path | str,
    *,
    model: TransformerLM,
    vocab: dict[str, Iterable[str]],
    losses: Iterable[float] | None = None,
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
            "metadata": dict(metadata or {}),
        },
        checkpoint_path,
    )
    return checkpoint_path


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
    model = TransformerLM(config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    return LoadedCheckpoint(
        model=model,
        config=config,
        tokenizer=_tokenizer_from_vocab(vocab),
        vocab=vocab,
        losses=list(payload.get("losses", [])),
        metadata=dict(payload.get("metadata", {})),
    )


@torch.no_grad()
def generate_from_checkpoint(
    path: Path | str,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
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
    input_ids = torch.tensor(
        [checkpoint.tokenizer.encode(prompt)],
        dtype=torch.long,
        device=torch_device,
    )
    generated = checkpoint.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return checkpoint.tokenizer.decode(generated[0].cpu().tolist())


def _tokenizer_from_vocab(vocab: dict[str, list[str]]) -> CharTokenizer:
    id_to_char = tuple(vocab["id_to_char"])
    char_to_id = {char: token_id for token_id, char in enumerate(id_to_char)}
    return CharTokenizer(char_to_id=char_to_id, id_to_char=id_to_char)


def _normalize_vocab(vocab: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    if "id_to_char" not in vocab:
        raise ValueError("vocab must contain id_to_char")
    id_to_char = list(vocab["id_to_char"])
    if not id_to_char:
        raise ValueError("vocab must contain at least one token")
    if len(set(id_to_char)) != len(id_to_char):
        raise ValueError("vocab id_to_char entries must be unique")
    return {"id_to_char": id_to_char}


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
