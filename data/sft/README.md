# Supervised Fine-Tuning

This directory contains the reviewed and generated inputs for supervised
fine-tuning (SFT). Production SFT starts from an immutable pretrained
checkpoint, learns general instruction-following behavior, and then creates two
sibling personality variants from the same core checkpoint.

## One-Command RunPod Workflow

On a single RTX 4090 RunPod, run:

```bash
cd /workspace/SuperAGI
git pull
git checkout scale-300m
tmux new -s superagi-sft
make runpod-sft-300m \
  SFT_CLOUD_BASE_CHECKPOINT=data/checkpoints/best.pt \
  2>&1 | tee -a sft-300m.log
```

The target performs these phases in order:

1. installs project dependencies;
2. loads and validates the immutable base checkpoint and tokenizer;
3. imports filtered public instruction data;
4. audits the mixed core corpus;
5. trains and evaluates the core instruction checkpoint;
6. trains and evaluates the playful/direct variant from core `best.pt`;
7. trains and evaluates the calm/precise variant from the same core `best.pt`;
8. re-hashes the original base checkpoint; and
9. writes the validated artifact manifest.

The two personality phases each use approximately 70% personality examples and
30% curated-core replay by effective sampling mass. They do not train from one
another.

Detach from tmux with `Ctrl-b d`. Reattach later with:

```bash
tmux attach -t superagi-sft
```

Run all non-training checks, including public import and the mixed-corpus audit,
with:

```bash
make runpod-sft-300m-preflight \
  SFT_CLOUD_BASE_CHECKPOINT=data/checkpoints/best.pt
```

Preflight requires a checkpoint with at least 1,024 context tokens and unique
`<pad>`, `<bos>`, `<eos>`, `<user>`, `<agi>`, and `<system>` tokenizer entries.
It records the base SHA-256 before any training. A changed base checkpoint or
changed run configuration is rejected when the same run directory is reused.

## Monitoring

From another terminal:

```bash
cd /workspace/SuperAGI
tail -f sft-300m.log
```

Once a phase has started, inspect its metrics with:

```bash
tail -f data/sft/runs/300m/core/metrics.jsonl
tail -f data/sft/runs/300m/playful/metrics.jsonl
tail -f data/sft/runs/300m/calm/metrics.jsonl
```

List current recovery and checkpoint artifacts with:

```bash
find data/sft/runs/300m -maxdepth 3 \
  \( -name 'recovery-current.json' -o -name 'latest.pt' \
     -o -name 'best.pt' -o -name 'final.pt' \) -print
```

## Recovery

If the pod, shell, or training process stops, reattach to tmux or start a new
tmux session and rerun the exact `make runpod-sft-300m` command. The workflow:

- skips a phase when its `final.pt` exists;
- resumes an incomplete phase when its `recovery-current.json` exists; and
- starts a fresh phase only when its run directory is empty.

`latest.pt` is the convenient alias for the latest committed checkpoint.
`recovery-current.json` is the authoritative pointer to the complete recovery
generation, including trainer and optimizer state. Do not rename, move, or
delete either while a run may need recovery. The workflow supplies `--resume`
through `SFT_RESUME=1`; do not launch a fresh `sft-train` in a non-empty
production run directory.

To intentionally change the base checkpoint or production settings, use a new
run root rather than reusing incompatible recovery state:

```bash
make runpod-sft-300m \
  SFT_CLOUD_BASE_CHECKPOINT=data/checkpoints/new-best.pt \
  SFT_CLOUD_RUN_ROOT=data/sft/runs/300m-v2
```

## Production Artifacts

The default workflow writes:

- `data/sft/runs/300m/base-checkpoint.json`: immutable base identity and hash;
- `data/sft/runs/300m/run-config.json`: recorded production configuration;
- `data/sft/runs/300m/audit.json`: mixed-corpus audit report;
- `data/sft/runs/300m/core/{latest,best,final}.pt`;
- `data/sft/runs/300m/playful/{latest,best,final}.pt`;
- `data/sft/runs/300m/calm/{latest,best,final}.pt`;
- `metrics.jsonl`, `evaluation.jsonl`, and `evaluation.summary.json` in each
  phase directory; and
- `data/sft/runs/300m/manifest.json`: hashes and summaries for all durable
  production artifacts.

The public import is generated at:

- `data/sft/imported/public-mixed.jsonl`;
- `data/sft/imported/public-mixed.metadata.json`.

`imported/` and `runs/` are intentionally ignored by Git. Copy completed
checkpoints and the manifest to durable storage before terminating a pod.

## Data Contract

Each JSONL line contains one conversation:

```json
{"messages":[{"role":"user","content":"What are you?"},{"role":"agi","content":"I am a small experimental language model."}]}
```

Production tracked inputs are:

- `curated/core.jsonl`: 1,500 reviewed core conversations;
- `styles/playful-direct.jsonl`: 500 reviewed playful/direct conversations;
- `styles/calm-precise.jsonl`: 500 reviewed calm/precise conversations; and
- `eval_prompts.jsonl`: fixed behavioral evaluation prompts.

The matching audit and metadata files are checked during preflight. Legacy
`stages/broad-mixed.jsonl` and bulk files under `generated/` are not production
inputs.

Public SFT data defaults to `no_robots`, `dolly`, `openassistant`, and
`ultrachat`. WildChat is excluded from the default import and assigned zero
production sampling weight. The importer rejects empty, over-context, duplicate,
artifact-heavy, and repetitive responses.

The trainer formats examples with `<bos>`, `<user>`, `<agi>`, and `<eos>`.
Loss is applied only to AGI answer tokens, so prompts provide context without
being prediction targets. Every production phase uses a held-out validation
split, checkpoint retention, and fixed behavioral gates.

## Manual Commands

The low-level targets remain available for diagnostics:

```bash
make sft-import-public \
  SFT_IMPORT_CHECKPOINT=data/checkpoints/best.pt

make sft-audit \
  SFT_AUDIT_DATA=data/sft/curated/core.jsonl \
  SFT_AUDIT_CHECKPOINT=data/checkpoints/best.pt \
  SFT_AUDIT_MODE=curated

make sft-evaluate \
  SFT_EVAL_CHECKPOINT=data/sft/runs/300m/core/best.pt \
  SFT_EVAL_RESULTS=data/sft/runs/300m/core/evaluation.jsonl \
  SFT_EVAL_SUMMARY=data/sft/runs/300m/core/evaluation.summary.json \
  SFT_EVAL_DEVICE=cuda
```

Use the one-command target for real RunPod production runs so the immutable
base check, recovery policy, evaluations, and manifest cannot be skipped.
