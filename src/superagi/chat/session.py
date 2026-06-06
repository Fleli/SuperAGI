from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from superagi.chat.formatting import ChatMessage, format_chat_messages
from superagi.model.checkpoint import LoadedCheckpoint


def build_chat_prompt(messages: Sequence[ChatMessage]) -> str:
    return format_chat_messages(messages, add_generation_prompt=True).text


def extract_chat_reply(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        reply = generated[len(prompt) :]
    else:
        reply = generated

    for stop_marker in ("\nUser:", "\nUSER:", "\nAGI:", "\n"):
        marker_index = reply.find(stop_marker)
        if marker_index >= 0:
            reply = reply[:marker_index]
            break
    return reply.strip()


@torch.no_grad()
def generate_chat_reply(
    *,
    checkpoint: LoadedCheckpoint,
    messages: Sequence[ChatMessage],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
    device: torch.device,
    on_text: Callable[[str], None] | None = None,
) -> str:
    prompt = build_chat_prompt(messages)
    prompt_ids = checkpoint.tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("chat prompt must encode to at least one token")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    streamed_token_ids = list(prompt_ids)
    prompt_text = checkpoint.tokenizer.decode(streamed_token_ids)
    streamed_text = prompt_text
    streamed_reply = ""

    def emit_text(token_id: int) -> bool:
        nonlocal streamed_reply, streamed_text
        streamed_token_ids.append(token_id)
        next_text = checkpoint.tokenizer.decode(streamed_token_ids)
        next_reply = extract_chat_reply(prompt_text, next_text)
        if on_text is not None and next_reply.startswith(streamed_reply):
            on_text(next_reply[len(streamed_reply) :])
        streamed_reply = next_reply
        streamed_text = next_text
        return _chat_reply_should_stop(prompt_text, next_text)

    generated = checkpoint.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        on_token=emit_text,
    )
    generated_text = checkpoint.tokenizer.decode(generated[0].cpu().tolist())
    return extract_chat_reply(prompt_text, generated_text)


def _chat_reply_should_stop(prompt: str, generated: str) -> bool:
    if generated.startswith(prompt):
        reply = generated[len(prompt) :]
    else:
        reply = generated
    return any(marker in reply for marker in ("\nUser:", "\nUSER:", "\nAGI:", "\n"))
