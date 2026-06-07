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
- Use the broad stage after anchor training. By default it mixes
  `stages/anchor.jsonl`, `stages/broad-mixed.jsonl`, and
  `imported/public-mixed.jsonl`, then samples by source weights so reviewed
  anchor behavior is not drowned by high-volume public chat data.
- Use `stages/style-playful-direct.jsonl` last and lightly. It nudges tone
  without replacing the anchor and broad behavior.
- Use `diagnostics/overfit-50.jsonl` only as a pipeline sanity check. It is a
  tiny identity dataset designed to overfit hard, not a balanced chat dataset.
- Keep AGI answers concise and honest about model limitations.
- Use `generated/` for bulk synthetic data before review; it is ignored by Git.
- Use `imported/` for filtered public instruction/chat datasets; it is ignored
  by Git because the artifacts can become large and upstream licenses vary.
- Use `runs/` for tokenized SFT artifacts, checkpoints, and experiment outputs; it is ignored by Git.
- Do not commit private website logs or personally identifying user content.

Import filtered public SFT data from high-value instruction/chat datasets:

```bash
make sft-import-public \
  SFT_IMPORT_CHECKPOINT=./best-200m-current.pt \
  SFT_IMPORT_MAX_ROWS_PER_SOURCE=50000 \
  SFT_IMPORT_MAX_EXAMPLES_PER_SOURCE=5000
```

The default import sources are `no_robots`, `dolly`, `openassistant`,
`wildchat`, and `ultrachat`. The importer writes:

- `data/sft/imported/public-mixed.jsonl`
- `data/sft/imported/public-mixed.metadata.json`

The importer filters examples before writing them: over-context examples,
empty answers, very short answers, very long answers, duplicate answers,
known synthetic artifacts, and repeated five-gram loops are rejected.
Review upstream licenses before using imported data outside local learning
experiments.

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

`SFT_DATA` may contain one JSONL file or comma-separated JSONL files. Public
imports carry a `source` field such as `wildchat:123`; local staged files default
to the filename stem such as `anchor` or `broad-mixed`. Use `SFT_SOURCE_WEIGHTS`
to bias batch sampling without rewriting the corpus:

```bash
make sft-train \
  SFT_BASE_CHECKPOINT=./best-200m-current.pt \
  SFT_DATA=data/sft/stages/anchor.jsonl,data/sft/imported/public-mixed.jsonl \
  SFT_SOURCE_WEIGHTS=anchor=4,no_robots=1.5,openassistant=1.25,dolly=1,ultrachat=0.8,wildchat=0.35
```

Validation examples are still evaluated unweighted so the reported validation
loss remains a normal holdout estimate.

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

Run `make sft-import-public` first, or override `SFT_BROAD_DATA` to remove
`data/sft/imported/public-mixed.jsonl`.

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
