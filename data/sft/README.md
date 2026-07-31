# Supervised Fine-Tuning Data

This directory contains small, curated chat data for supervised fine-tuning (SFT).

Use JSONL files with one conversation per line:

```json
{"messages":[{"role":"user","content":"What are you?"},{"role":"agi","content":"I am a small experimental language model trained as a learning project."}]}
```

Guidelines:

- `curated/core.jsonl` is the production curated SFT source. It contains 1,500
  audited conversations across the production domain quotas; its deterministic
  metadata and strict audit report live beside it.
- `stages/broad-mixed.jsonl` is legacy experimental data and is not a production input.
- Keep `seed.jsonl` small, reviewed, and tracked in Git.
- Use `seed-playful-blunt.jsonl` for the separate personality variant. It starts
  with all `seed.jsonl` examples, then adds lightly teasing, direct examples.
- Use `stages/anchor.jsonl` first when a checkpoint needs to learn the basic
  chat contract: identity, limits, refusal to guess live facts, and topic repair.
- Use the production curated core with filtered public data during broad
  instruction training, then sample by source weights so reviewed behavior is
  not drowned by high-volume public chat data.
- `styles/playful-direct.jsonl` and `styles/calm-precise.jsonl` are the reviewed
  production personality corpora. Each contains 500 conversations with an
  exact 250 single-turn and 250 multi-turn split.
- Train either style from the same broad checkpoint. Each style phase mixes
  `curated/core.jsonl` with its personality corpus so tone does not replace
  the core instruction-following behavior.
- Use `diagnostics/overfit-50.jsonl` only as a pipeline sanity check. It is a
  tiny identity dataset designed to overfit hard, not a balanced chat dataset.
- Keep AGI answers concise and honest about model limitations.
- Use `generated/` for bulk synthetic data before review; it is ignored by Git.
- Use `imported/` for filtered public instruction/chat datasets; it is ignored
  by Git because the artifacts can become large and upstream licenses vary.
- Use `runs/` for tokenized SFT artifacts, checkpoints, and experiment outputs; it is ignored by Git.
- Do not commit private website logs or personally identifying user content.

The strict audit includes a conservative topical-relevance proxy. It scores
every adjacent user-to-AGI pair against the immediately preceding user request,
without concatenating older user turns. Any clear pair mismatch makes the
conversation a mismatch; every pair must be supported for the conversation to
be supported, and all other cases are explicitly unscored. Informative lexical
stems and a small declared topic vocabulary provide the signal. This proxy
catches obvious answer swaps but does not establish factual correctness,
completeness, usefulness, or full instruction following. Human review and
behavioral evaluation remain required.

Import filtered public SFT data from high-value instruction/chat datasets:

```bash
make sft-import-public \
  SFT_IMPORT_CHECKPOINT=./best-200m-current.pt \
  SFT_IMPORT_MAX_ROWS_PER_SOURCE=50000 \
  SFT_IMPORT_MAX_EXAMPLES_PER_SOURCE=5000
```

The default import sources are `no_robots`, `dolly`, `openassistant`, and
`ultrachat`. WildChat is opt-in rather than a production default. The importer
writes:

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
  SFT_MAX_EXAMPLES=0 \
  SFT_LR=3e-5 \
  SFT_DEVICE=auto
```

The trainer formats examples with the checkpoint tokenizer's chat special
tokens: `<bos>`, `<user>`, `<agi>`, and `<eos>`. Loss is only applied to AGI
answer tokens, so user prompts teach context rather than being predicted.
Set `SFT_MAX_EXAMPLES` above zero to run a deterministic subset before using
the full corpus.

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
- `data/sft/runs/chat-calm-precise.pt`

The final two checkpoints are sibling variants trained from
`data/sft/runs/chat-broad.pt`. Run either personality phase independently with:

```bash
make sft-style-playful \
  SFT_STYLE_PLAYFUL_BASE_CHECKPOINT=data/sft/runs/chat-broad.pt

make sft-style-calm \
  SFT_STYLE_CALM_BASE_CHECKPOINT=data/sft/runs/chat-broad.pt
```

For local testing of a downloaded 300M checkpoint, copy the cloud checkpoint to
`./best-300m-current.pt`, then run a small smoke pass:

```bash
make sft-local-smoke \
  SFT_LOCAL_BASE_CHECKPOINT=./best-300m-current.pt \
  SFT_LOCAL_DEVICE=auto
```

If the smoke run shows clear instruction-following movement, run the bounded
local staged pass:

```bash
make sft-local \
  SFT_LOCAL_BASE_CHECKPOINT=./best-300m-current.pt \
  SFT_LOCAL_DEVICE=auto
```

The local targets import public SFT data into
`data/sft/imported/local-public-mixed.jsonl`, then train three separate phases:

- anchor behavior into `data/sft/runs/chat-anchor-local.pt`
- public instruction/chat behavior into `data/sft/runs/chat-public-local.pt`
- a light reviewed playful style into `data/sft/runs/chat-style-local.pt`

The local public phase intentionally avoids `stages/broad-mixed.jsonl` by
default because that synthetic broad file can dominate small local runs. It
keeps the curated anchor data in the public phase with source weighting so the
model does not forget the basic chat contract. Override
`SFT_LOCAL_PUBLIC_MAX_EXAMPLES=0` for the full imported corpus, or keep the
default bounded subset while testing behavior on a Mac. The smoke target stops
after the public phase and writes `data/sft/runs/chat-public-local-smoke.pt`.

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
