from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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
