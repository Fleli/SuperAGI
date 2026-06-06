# Supervised Fine-Tuning Data

This directory contains small, curated chat data for supervised fine-tuning (SFT).

Use JSONL files with one conversation per line:

```json
{"messages":[{"role":"user","content":"What are you?"},{"role":"agi","content":"I am a small experimental language model trained as a learning project."}]}
```

Guidelines:

- Keep `seed.jsonl` small, reviewed, and tracked in Git.
- Use `seed-playful-blunt.jsonl` for the separate personality variant. It starts
  with all `seed.jsonl` examples, then adds lightly teasing, direct examples.
- Use `stages/anchor.jsonl` first when a checkpoint needs to learn the basic
  chat contract: identity, limits, refusal to guess live facts, and topic repair.
- Use `stages/broad-mixed.jsonl` after anchor training. It keeps the reviewed
  seed corpus, repeats anchor rows, and adds broader low-stakes chat coverage.
- Use `stages/style-playful-direct.jsonl` last and lightly. It nudges tone
  without replacing the anchor and broad behavior.
- Use `diagnostics/overfit-50.jsonl` only as a pipeline sanity check. It is a
  tiny identity dataset designed to overfit hard, not a balanced chat dataset.
- Keep AGI answers concise and honest about model limitations.
- Use `generated/` for bulk synthetic data before review; it is ignored by Git.
- Use `runs/` for tokenized SFT artifacts, checkpoints, and experiment outputs; it is ignored by Git.
- Do not commit private website logs or personally identifying user content.

Run a supervised fine-tune from an existing checkpoint without overwriting it:

```bash
make sft-train \
  SFT_BASE_CHECKPOINT=./best-svl-current.pt \
  SFT_OUT=data/sft/runs/chat-sft.pt \
  SFT_STEPS=3000 \
  SFT_BATCH=8 \
  SFT_LR=3e-5 \
  SFT_DEVICE=auto
```

The trainer formats examples with literal role prefixes such as `User:` and `AGI:`. Future tokenizers can add special chat tokens.

Run the 50-example overfit diagnostic from a pretrained checkpoint:

```bash
make sft-overfit-50 \
  SFT_OVERFIT_BASE_CHECKPOINT=./best-current-cloud.pt \
  SFT_DEVICE=auto
```

If SFT is wired correctly, this run should make identity prompts such as
`What are you?` strongly prefer an answer starting with `SuperAGI`.

Run the staged chat fine-tune from a pretrained checkpoint:

```bash
make sft-staged \
  SFT_STAGED_BASE_CHECKPOINT=./best-current-cloud.pt \
  SFT_DEVICE=auto
```

This writes separate checkpoints for each phase:

- `data/sft/runs/chat-anchor.pt`
- `data/sft/runs/chat-broad.pt`
- `data/sft/runs/chat-playful-direct.pt`

For a conservative chat test, start with:

```bash
make chat \
  CHAT_CHECKPOINT=data/sft/runs/chat-playful-direct.pt \
  CHAT_MAX_TOKENS=80 \
  TEMPERATURE=0.45 \
  TOP_K=30 \
  REPETITION_PENALTY=1.2 \
  DEVICE=auto
```
