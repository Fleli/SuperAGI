# Supervised Fine-Tuning Data

This directory contains small, curated chat data for supervised fine-tuning (SFT).

Use JSONL files with one conversation per line:

```json
{"messages":[{"role":"user","content":"What are you?"},{"role":"agi","content":"I am a small experimental language model trained as a learning project."}]}
```

Guidelines:

- Keep `seed.jsonl` small, reviewed, and tracked in Git.
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
