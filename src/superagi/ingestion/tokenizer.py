from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Literal

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters to token IDs; learned embeddings live in the model."""

    char_to_id: dict[str, int]
    id_to_char: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("text must contain at least one character")

        id_to_char = tuple(sorted(set(text)))
        char_to_id = {char: token_id for token_id, char in enumerate(id_to_char)}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"unknown character: {exc.args[0]!r}") from None

    def decode(self, token_ids: Iterable[int]) -> str:
        chars = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"unknown token id: {token_id}")
            chars.append(self.id_to_char[token_id])
        return "".join(chars)

    def to_payload(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "char",
            "id_to_char": list(self.id_to_char),
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CharTokenizer":
        if payload.get("tokenizer_type", "char") != "char":
            raise ValueError("char tokenizer payload must have tokenizer_type='char'")
        id_to_char_value = payload.get("id_to_char")
        if not isinstance(id_to_char_value, list):
            raise ValueError("char tokenizer payload must contain id_to_char")

        id_to_char = tuple(_validate_string_list(id_to_char_value, "id_to_char"))
        if not id_to_char:
            raise ValueError("char tokenizer payload must contain at least one token")
        if len(set(id_to_char)) != len(id_to_char):
            raise ValueError("char tokenizer id_to_char entries must be unique")

        vocab_size = payload.get("vocab_size", len(id_to_char))
        if not isinstance(vocab_size, int) or vocab_size != len(id_to_char):
            raise ValueError("char tokenizer vocab_size must match id_to_char length")

        char_to_id = {char: token_id for token_id, char in enumerate(id_to_char)}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)


@dataclass(frozen=True)
class BpeTokenizer:
    """Byte-level BPE tokenizer; learned embeddings still live in the model."""

    tokenizer: Tokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        if not text:
            raise ValueError("text must contain at least one character")
        return cls.from_texts(
            [text],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        text_iterator = iter(texts)
        first_text = next(text_iterator, None)
        if first_text is None:
            raise ValueError("texts must contain at least one document")
        if not first_text:
            raise ValueError("texts must not contain empty documents")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>"],
            initial_alphabet=ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(chain([first_text], text_iterator), trainer=trainer)
        return cls(tokenizer=tokenizer)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BpeTokenizer":
        if payload.get("tokenizer_type") != "bpe":
            raise ValueError("bpe tokenizer payload must have tokenizer_type='bpe'")
        tokenizer_json = payload.get("tokenizer_json")
        if not isinstance(tokenizer_json, str) or not tokenizer_json:
            raise ValueError("bpe tokenizer payload must contain tokenizer_json")

        tokenizer = Tokenizer.from_str(tokenizer_json)
        instance = cls(tokenizer=tokenizer)
        vocab_size = payload.get("vocab_size", instance.vocab_size)
        if not isinstance(vocab_size, int) or vocab_size != instance.vocab_size:
            raise ValueError("bpe tokenizer vocab_size must match tokenizer_json")
        return instance

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def to_payload(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "bpe",
            "tokenizer_json": self.tokenizer.to_str(),
            "vocab_size": self.vocab_size,
        }


TokenizerType = Literal["bpe", "char"]
TokenizerLike = BpeTokenizer | CharTokenizer


def tokenizer_from_payload(payload: dict[str, Any]) -> TokenizerLike:
    tokenizer_type = payload.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        return BpeTokenizer.from_payload(payload)
    if tokenizer_type == "char":
        return CharTokenizer.from_payload(payload)
    raise ValueError(f"unknown tokenizer_type: {tokenizer_type!r}")


def normalize_tokenizer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = tokenizer_from_payload(payload).to_payload()
    for key in ("source_paths", "train_tokens", "validation_tokens"):
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def _validate_string_list(value: list[Any], field_name: str) -> list[str]:
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must only contain strings")
    return value
