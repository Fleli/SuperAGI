SHELL := /bin/bash

PYTHON := .venv/bin/python
PYTHONPATH := src
export PYTHONPATH

RAW_DIR := data/raw
PROCESSED_DIR := data/processed
CHECKPOINT_DIR := data/checkpoints
ARTIFACT := train
CHECKPOINT := $(CHECKPOINT_DIR)/latest.pt
BEST_CHECKPOINT := $(CHECKPOINT_DIR)/best.pt
MODEL_OUT := $(CHECKPOINT_DIR)/final-model.pt
RESUME :=
TOKENIZER := bpe
BPE_VOCAB_SIZE := 8000
BPE_MIN_FREQUENCY := 2

WIKI_QUERIES := machine learning,artificial intelligence
WIKI_MAX := 5
WIKI_MIN_CHARS := 1000
WIKI_USER_AGENT := SuperAGI-learning-corpus-builder/0.1 (local learning project; set WIKI_USER_AGENT=mailto:you@example.com)

C4_MAX := 100
C4_MIN_CHARS := 500
STREAM_C4_MAX := 10000
STREAM_C4_MIN_CHARS := 1000
STREAM_TOKENIZER_SAMPLE := 1000
STREAM_SHARD_TOKENS := 1000000
STREAM_VALIDATION_TOKENS := 200000
STREAM_TARGET_TOKENS := 0
SOURCES := fineweb,wikipedia,dolma,openwebmath,arxiv,pmc,stackexchange,gutenberg
STREAM_MAX_DOCUMENTS_PER_SOURCE := 10000

TRAIN_4090_RAW_DIR := data/raw/c4-4090-night
TRAIN_4090_C4_MAX := 150000
TRAIN_4090_C4_MIN_CHARS := 1000
TRAIN_4090_STEPS := 120000
TRAIN_4090_BATCH := 16
TRAIN_4090_GRAD_ACCUM_STEPS := 1
TRAIN_4090_ACTIVATION_CHECKPOINTING := 0
TRAIN_4090_DEVICE := cuda
TRAIN_4090_MIXED_PRECISION := auto
TRAIN_4090_PARAMETER_DTYPE := float32
TRAIN_4090_FUSED_ADAMW := auto
TRAIN_4090_COMPILE_MODEL := 0
TRAIN_4090_DROPOUT := 0.0
TRAIN_4090_SHARD_REFRESH_INTERVAL := 0
TRAIN_4090_EVAL_INTERVAL := 1000
TRAIN_4090_VAL_BATCHES := 20
TRAIN_4090_CHECKPOINT_INTERVAL := 1000
TRAIN_4090_CHECKPOINT_KEEP := 5
TRAIN_4090_PROMPT := In machine learning,
TRAIN_4090_NEW_TOKENS := 500
TRAIN_4090_TEMPERATURE := 0.6
TRAIN_4090_TOP_K := 30

TRAIN_200M_STREAM_C4_MAX := 1000000
TRAIN_200M_STREAM_C4_MIN_CHARS := 1000
TRAIN_200M_STREAM_TOKENIZER_SAMPLE := 20000
TRAIN_200M_STREAM_SHARD_TOKENS := 2000000
TRAIN_200M_STREAM_VALIDATION_TOKENS := 2000000
TRAIN_200M_BPE_VOCAB_SIZE := 8000
TRAIN_200M_BPE_MIN_FREQUENCY := 2
TRAIN_200M_STEPS := 300000
TRAIN_200M_BATCH := 8
TRAIN_200M_GRAD_ACCUM_STEPS := 1
TRAIN_200M_ACTIVATION_CHECKPOINTING := 0
TRAIN_200M_DEVICE := cuda
TRAIN_200M_MIXED_PRECISION := auto
TRAIN_200M_PARAMETER_DTYPE := float32
TRAIN_200M_FUSED_ADAMW := auto
TRAIN_200M_COMPILE_MODEL := 0
TRAIN_200M_DROPOUT := 0.0
TRAIN_200M_SHARD_REFRESH_INTERVAL := 0
TRAIN_200M_LR := 3e-4
TRAIN_200M_LR_MIN := 3e-5
TRAIN_200M_LR_WARMUP_STEPS := 3000
TRAIN_200M_EVAL_INTERVAL := 1000
TRAIN_200M_VAL_BATCHES := 20
TRAIN_200M_CHECKPOINT_INTERVAL := 1000
TRAIN_200M_CHECKPOINT_KEEP := 5
TRAIN_200M_PROMPT := In machine learning,
TRAIN_200M_NEW_TOKENS := 500
TRAIN_200M_TEMPERATURE := 0.7
TRAIN_200M_TOP_K := 30

TRAIN_H100_STREAM_C4_MAX := 50000000
TRAIN_H100_STREAM_C4_MIN_CHARS := 1000
TRAIN_H100_STREAM_TOKENIZER_SAMPLE := 50000
TRAIN_H100_STREAM_SHARD_TOKENS := 10000000
TRAIN_H100_STREAM_VALIDATION_TOKENS := 5000000
TRAIN_H100_SOURCES := fineweb,wikipedia,dolma,openwebmath,arxiv,pmc,stackexchange,gutenberg
TRAIN_H100_STREAM_MAX_DOCUMENTS_PER_SOURCE := 6250000
TRAIN_H100_BPE_VOCAB_SIZE := 32000
TRAIN_H100_BPE_MIN_FREQUENCY := 2
TRAIN_H100_CORPUS_TARGET_TOKENS := 20000000000
TRAIN_H100_START_TOKENS := 500000000
TRAIN_H100_READY_POLL_SECONDS := 30
TRAIN_H100_TOTAL_TRAINING_TOKENS := 20000000000
TRAIN_H100_BATCH := 1
TRAIN_H100_GRAD_ACCUM_STEPS := 128
TRAIN_H100_ACTIVATION_CHECKPOINTING := 1
TRAIN_H100_DEVICE := cuda
TRAIN_H100_MIXED_PRECISION := bfloat16
TRAIN_H100_PARAMETER_DTYPE := float32
TRAIN_H100_FUSED_ADAMW := auto
TRAIN_H100_COMPILE_MODEL := 1
TRAIN_H100_DROPOUT := 0.0
TRAIN_H100_SHARD_REFRESH_INTERVAL := 500
TRAIN_H100_LR := 3e-4
TRAIN_H100_LR_MIN := 3e-5
TRAIN_H100_LR_WARMUP_STEPS := 2000
TRAIN_H100_EVAL_INTERVAL := 1000
TRAIN_H100_VAL_BATCHES := 20
TRAIN_H100_CHECKPOINT_INTERVAL := 1000
TRAIN_H100_CHECKPOINT_KEEP := 5
TRAIN_H100_PROMPT := In machine learning,
TRAIN_H100_NEW_TOKENS := 500
TRAIN_H100_TEMPERATURE := 0.7
TRAIN_H100_TOP_K := 40

BATCH := 32
GRAD_ACCUM_STEPS := 1
ACTIVATION_CHECKPOINTING := 0
LR := 3e-4
LR_MIN := 3e-5
LR_WARMUP_STEPS := 100
MIXED_PRECISION := auto
PARAMETER_DTYPE := float32
FUSED_ADAMW := auto
COMPILE_MODEL := 0
DROPOUT := 0.0
SHARD_REFRESH_INTERVAL := 0
STEPS := 1000
DEVICE := auto
PROMPT := The
NEW_TOKENS := 100
TEMPERATURE := 1.0
TOP_K := 0
REPETITION_PENALTY := 1.0
REPETITION_WINDOW := 128
STREAM := 1
CHAT := 0
CHAT_MAX_TOKENS := 80
VALIDATION_FRACTION := 0.1
VAL_TOKENS := $(PROCESSED_DIR)/val_tokens.pt
EVAL_INTERVAL := 500
VAL_BATCHES := 10
METRICS := $(CHECKPOINT_DIR)/metrics.jsonl
CHECKPOINT_INTERVAL := 1000
CHECKPOINT_KEEP := 5
SFT_DATA := data/sft/seed.jsonl
SFT_BASE_CHECKPOINT := $(CHECKPOINT)
SFT_OUT := data/sft/runs/chat-sft.pt
CHAT_CHECKPOINT := $(SFT_OUT)
SFT_METRICS := data/sft/runs/metrics.jsonl
SFT_STEPS := 3000
SFT_BATCH := 8
SFT_LR := 3e-5
SFT_LR_MIN := 3e-6
SFT_LR_WARMUP_STEPS := 10
SFT_WEIGHT_DECAY := 0.01
SFT_GRAD_CLIP := 1.0
SFT_DEVICE := auto
SFT_CHECKPOINT_INTERVAL := 250
SFT_VALIDATION_FRACTION := 0.05
SFT_VALIDATION_BATCHES := 10
SFT_SOURCE_WEIGHTS :=
SFT_SEED := 1337
SFT_IMPORT_CHECKPOINT := $(SFT_BASE_CHECKPOINT)
SFT_IMPORT_OUT := data/sft/imported/public-mixed.jsonl
SFT_IMPORT_METADATA := data/sft/imported/public-mixed.metadata.json
SFT_IMPORT_SOURCES := no_robots,dolly,openassistant,wildchat,ultrachat
SFT_IMPORT_MAX_ROWS_PER_SOURCE := 50000
SFT_IMPORT_MAX_EXAMPLES_PER_SOURCE := 5000
SFT_IMPORT_MAX_CONTEXT_TOKENS := 900
SFT_IMPORT_MAX_MESSAGES := 8
SFT_IMPORT_MAX_AGI_CHARS := 1200
SFT_IMPORT_MIN_AGI_CHARS := 20
SFT_OVERFIT_DATA := data/sft/diagnostics/overfit-50.jsonl
SFT_OVERFIT_BASE_CHECKPOINT := $(SFT_BASE_CHECKPOINT)
SFT_OVERFIT_OUT := data/sft/runs/chat-sft-overfit-50.pt
SFT_OVERFIT_METRICS := data/sft/runs/chat-sft-overfit-50-metrics.jsonl
SFT_OVERFIT_STEPS := 2000
SFT_OVERFIT_BATCH := 16
SFT_OVERFIT_LR := 5e-5
SFT_OVERFIT_LR_MIN := 5e-6
SFT_OVERFIT_LR_WARMUP_STEPS := 25
SFT_OVERFIT_WEIGHT_DECAY := 0.0
SFT_OVERFIT_CHECKPOINT_INTERVAL := 250
SFT_STAGED_BASE_CHECKPOINT := $(SFT_BASE_CHECKPOINT)
SFT_ANCHOR_DATA := data/sft/stages/anchor.jsonl
SFT_ANCHOR_BASE_CHECKPOINT := $(SFT_STAGED_BASE_CHECKPOINT)
SFT_ANCHOR_OUT := data/sft/runs/chat-anchor.pt
SFT_ANCHOR_METRICS := data/sft/runs/chat-anchor-metrics.jsonl
SFT_ANCHOR_STEPS := 1200
SFT_ANCHOR_BATCH := 8
SFT_ANCHOR_LR := 2e-5
SFT_ANCHOR_LR_MIN := 5e-6
SFT_ANCHOR_LR_WARMUP_STEPS := 50
SFT_ANCHOR_WEIGHT_DECAY := 0.0
SFT_ANCHOR_CHECKPOINT_INTERVAL := 250
SFT_BROAD_DATA := data/sft/stages/anchor.jsonl,data/sft/stages/broad-mixed.jsonl,data/sft/imported/public-mixed.jsonl
SFT_BROAD_BASE_CHECKPOINT := $(SFT_ANCHOR_OUT)
SFT_BROAD_OUT := data/sft/runs/chat-broad.pt
SFT_BROAD_METRICS := data/sft/runs/chat-broad-metrics.jsonl
SFT_BROAD_STEPS := 2200
SFT_BROAD_BATCH := 8
SFT_BROAD_LR := 8e-6
SFT_BROAD_LR_MIN := 2e-6
SFT_BROAD_LR_WARMUP_STEPS := 100
SFT_BROAD_WEIGHT_DECAY := 0.01
SFT_BROAD_CHECKPOINT_INTERVAL := 250
SFT_BROAD_SOURCE_WEIGHTS := anchor=4,broad-mixed=2,no_robots=1.5,openassistant=1.25,dolly=1,ultrachat=0.8,wildchat=0.35,default=1
SFT_STYLE_DATA := data/sft/stages/style-playful-direct.jsonl
SFT_STYLE_BASE_CHECKPOINT := $(SFT_BROAD_OUT)
SFT_STYLE_OUT := data/sft/runs/chat-playful-direct.pt
SFT_STYLE_METRICS := data/sft/runs/chat-playful-direct-metrics.jsonl
SFT_STYLE_STEPS := 600
SFT_STYLE_BATCH := 8
SFT_STYLE_LR := 3e-6
SFT_STYLE_LR_MIN := 1e-6
SFT_STYLE_LR_WARMUP_STEPS := 50
SFT_STYLE_WEIGHT_DECAY := 0.01
SFT_STYLE_CHECKPOINT_INTERVAL := 200
SFT_STAGED_OUT := $(SFT_STYLE_OUT)

.PHONY: help setup data-dirs test wiki c4 ingest ingest-stream-c4 ingest-stream-sources train sft-import-public sft-train sft-overfit-50 sft-anchor sft-broad sft-style sft-staged params train-export-run train-4090 train-200m train-h100 std-train export-model generate run-model chat smoke-train clean-generated

help:
	@echo "SuperAGI pipeline targets"
	@echo ""
	@echo "Setup and verification:"
	@echo "  make setup             Create/update .venv and install requirements"
	@echo "  make test              Run the unittest suite"
	@echo "  make smoke-train       Run a tiny in-memory training smoke test"
	@echo ""
	@echo "Corpus building:"
	@echo "  make wiki WIKI_QUERIES=\"machine learning,linear algebra\" WIKI_MAX=3"
	@echo "  make wiki WIKI_USER_AGENT=\"SuperAGI-learning-corpus-builder/0.1 (mailto:you@example.com)\""
	@echo "  make c4 C4_MAX=20 C4_MIN_CHARS=500"
	@echo ""
	@echo "Ingest and train:"
	@echo "  make ingest            Tokenize data/raw into train/validation artifacts"
	@echo "  make ingest TOKENIZER=bpe BPE_VOCAB_SIZE=8000 BPE_MIN_FREQUENCY=2"
	@echo "  make ingest TOKENIZER=char"
	@echo "  make ingest-stream-c4  Stream C4 directly into token shards"
	@echo "  make ingest-stream-sources SOURCES=\"fineweb,wikipedia,openwebmath\""
	@echo "  make params            Count params using data/processed vocab"
	@echo "  make train STEPS=100 BATCH=16 GRAD_ACCUM_STEPS=4 ACTIVATION_CHECKPOINTING=1 DROPOUT=0.05 SHARD_REFRESH_INTERVAL=500 LR=3e-4 LR_MIN=3e-5 LR_WARMUP_STEPS=100 MIXED_PRECISION=bfloat16 PARAMETER_DTYPE=float32 FUSED_ADAMW=auto COMPILE_MODEL=1"
	@echo "  make train RESUME=data/checkpoints/latest.pt STEPS=1000 CHECKPOINT_INTERVAL=1000"
	@echo "  make train-export-run RESUME=data/checkpoints/latest.pt STEPS=1000 PROMPT=\"Attention is\""
	@echo "  make sft-import-public SFT_IMPORT_CHECKPOINT=./best-200m-current.pt"
	@echo "  make sft-train SFT_BASE_CHECKPOINT=data/checkpoints/best.pt SFT_STEPS=200 SFT_SOURCE_WEIGHTS=anchor=4,wildchat=0.35"
	@echo "  make sft-overfit-50 SFT_OVERFIT_BASE_CHECKPOINT=./best-current-cloud.pt"
	@echo "  make sft-staged SFT_STAGED_BASE_CHECKPOINT=./best-current-cloud.pt  # run sft-import-public first"
	@echo "  make train-4090        Fetch C4, rebuild artifacts, and start the RTX 4090 night run"
	@echo "  make train-200m        Clean, stream C4 shards, and start the 200M training run"
	@echo "  make train-h100        Zero-setup dynamic H100 C4 ingest/training run"
	@echo "  make run-model CHECKPOINT=data/checkpoints/best.pt PROMPT=\"The\""
	@echo "  make std-train         Resume latest, train 5k steps, export, sample"
	@echo "  make export-model      Validate/copy latest checkpoint to a portable .pt file"
	@echo "  make run-model PROMPT=\"The\" NEW_TOKENS=100 TOP_K=50"
	@echo "  make run-model PROMPT=\"The\" REPETITION_PENALTY=1.15 REPETITION_WINDOW=128"
	@echo "  make run-model CHAT=1 PROMPT=\"What are you?\""
	@echo "  make run-model STREAM=0 PROMPT=\"The\" NEW_TOKENS=100"
	@echo "  make chat CHAT_CHECKPOINT=data/sft/runs/chat-sft.pt"
	@echo "  make generate PROMPT=\"The\" NEW_TOKENS=100"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-generated   Remove generated processed artifacts/checkpoints"

$(PYTHON):
	python3 -m venv .venv

.venv/.installed: requirements.txt | $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch .venv/.installed

setup: .venv/.installed

data-dirs:
	mkdir -p "$(RAW_DIR)" "$(PROCESSED_DIR)" "$(CHECKPOINT_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests

wiki: setup data-dirs
	@printf '==> [wiki] Fetching Wikipedia text\n'
	@printf '%s\n' \
		'from superagi.ingestion.builders.wikipedia import build_wikipedia_corpus' \
		'' \
		'queries = [query.strip() for query in "$(WIKI_QUERIES)".split(",") if query.strip()]' \
		'result = build_wikipedia_corpus(' \
		'    queries=queries,' \
		'    raw_root="$(RAW_DIR)",' \
		'    max_articles_per_query=int("$(WIKI_MAX)"),' \
		'    min_chars=int("$(WIKI_MIN_CHARS)"),' \
		'    user_agent="$(WIKI_USER_AGENT)",' \
		')' \
		'print(f"Wrote {result.documents_written} Wikipedia documents to {result.output_dir}")' \
		'print(f"Metadata: {result.metadata_path}")' \
	| $(PYTHON)
	@printf '==> [wiki] Finished fetching Wikipedia text\n'

c4: setup data-dirs
	@printf '==> [c4] Fetching C4 text\n'
	@printf '%s\n' \
		'from superagi.ingestion.builders.c4 import build_c4_corpus' \
		'' \
		'result = build_c4_corpus(' \
		'    raw_root="$(RAW_DIR)",' \
		'    max_documents=int("$(C4_MAX)"),' \
		'    min_chars=int("$(C4_MIN_CHARS)"),' \
		')' \
		'print(f"Wrote {result.documents_written} C4 documents to {result.output_dir}")' \
		'print(f"Metadata: {result.metadata_path}")' \
	| $(PYTHON)
	@printf '==> [c4] Finished fetching C4 text\n'

ingest: setup data-dirs
	@printf '==> [ingest] Ingesting raw text\n'
	@printf '%s\n' \
		'from superagi.ingestion.corpus import ingest_raw_corpus' \
		'' \
		'artifact = ingest_raw_corpus(' \
		'    raw_dir="$(RAW_DIR)",' \
		'    processed_dir="$(PROCESSED_DIR)",' \
		'    artifact_name="$(ARTIFACT)",' \
		'    validation_fraction=float("$(VALIDATION_FRACTION)"),' \
		'    tokenizer_type="$(TOKENIZER)",' \
		'    bpe_vocab_size=int("$(BPE_VOCAB_SIZE)"),' \
		'    bpe_min_frequency=int("$(BPE_MIN_FREQUENCY)"),' \
		')' \
		'print(f"Tokenized {len(artifact.token_ids)} tokens")' \
		'print(f"Vocab size: {artifact.tokenizer.vocab_size}")' \
		'print("Artifacts: $(PROCESSED_DIR)/$(ARTIFACT)_tokens.pt, $(VAL_TOKENS), and $(PROCESSED_DIR)/$(ARTIFACT)_vocab.json")' \
	| $(PYTHON)
	@printf '==> [ingest] Finished ingesting raw text\n'

ingest-stream-c4: setup data-dirs
	@printf '==> [ingest-stream-c4] Streaming C4 into token shards\n'
	@printf '%s\n' \
		'from superagi.ingestion.streaming import build_c4_token_shards' \
		'' \
		'result = build_c4_token_shards(' \
		'    processed_dir="$(PROCESSED_DIR)",' \
		'    max_documents=int("$(STREAM_C4_MAX)"),' \
		'    tokenizer_sample_documents=int("$(STREAM_TOKENIZER_SAMPLE)"),' \
		'    shard_token_count=int("$(STREAM_SHARD_TOKENS)"),' \
		'    validation_token_count=int("$(STREAM_VALIDATION_TOKENS)"),' \
		'    target_train_tokens=int("$(STREAM_TARGET_TOKENS)") or None,' \
		'    min_chars=int("$(STREAM_C4_MIN_CHARS)"),' \
		'    bpe_vocab_size=int("$(BPE_VOCAB_SIZE)"),' \
		'    bpe_min_frequency=int("$(BPE_MIN_FREQUENCY)"),' \
		')' \
		'print(f"Tokenized {result.train_tokens} train tokens into {len(result.train_shard_paths)} shards")' \
		'print(f"Validation tokens: {result.validation_tokens}")' \
		'print(f"Target train tokens: {result.target_train_tokens}")' \
		'print(f"Vocab size: {result.tokenizer.vocab_size}")' \
		'print(f"Manifest: {result.manifest_path}")' \
		'print(f"Vocab: {result.vocab_path}")' \
	| $(PYTHON)
	@printf '==> [ingest-stream-c4] Finished streaming C4 into token shards\n'

ingest-stream-sources: setup data-dirs
	@printf '==> [ingest-stream-sources] Streaming mixed sources into token shards\n'
	@printf '%s\n' \
		'from superagi.ingestion.sources import build_multi_source_token_shards' \
		'' \
		'result = build_multi_source_token_shards(' \
		'    processed_dir="$(PROCESSED_DIR)",' \
		'    sources="$(SOURCES)",' \
		'    max_documents_per_source=int("$(STREAM_MAX_DOCUMENTS_PER_SOURCE)"),' \
		'    tokenizer_sample_documents=int("$(STREAM_TOKENIZER_SAMPLE)"),' \
		'    shard_token_count=int("$(STREAM_SHARD_TOKENS)"),' \
		'    validation_token_count=int("$(STREAM_VALIDATION_TOKENS)"),' \
		'    target_train_tokens=int("$(STREAM_TARGET_TOKENS)") or None,' \
		'    min_chars=int("$(STREAM_C4_MIN_CHARS)"),' \
		'    bpe_vocab_size=int("$(BPE_VOCAB_SIZE)"),' \
		'    bpe_min_frequency=int("$(BPE_MIN_FREQUENCY)"),' \
		')' \
		'print(f"Tokenized {result.train_tokens} train tokens into {len(result.train_shard_paths)} shards")' \
		'print(f"Validation tokens: {result.validation_tokens}")' \
		'print(f"Target train tokens: {result.target_train_tokens}")' \
		'print(f"Documents tokenized: {result.documents_tokenized}")' \
		'print(f"Vocab size: {result.tokenizer.vocab_size}")' \
		'print(f"Manifest: {result.manifest_path}")' \
		'print(f"Vocab: {result.vocab_path}")' \
	| $(PYTHON)
	@printf '==> [ingest-stream-sources] Finished streaming mixed sources into token shards\n'

params: setup
	@printf '%s\n' \
		'import json' \
		'from pathlib import Path' \
		'' \
		'from superagi.initialization.read_config import load_project_config' \
		'from superagi.model.transformer import TransformerLM' \
		'' \
		'vocab_path = Path("$(PROCESSED_DIR)") / "$(ARTIFACT)_vocab.json"' \
		'if not vocab_path.exists():' \
		'    raise SystemExit("Missing vocab artifact. Run `make ingest` first.")' \
		'vocab = json.loads(vocab_path.read_text(encoding="utf-8"))' \
		'config = load_project_config().to_transformer_config(vocab_size=int(vocab["vocab_size"]))' \
		'model = TransformerLM(config)' \
		'print(sum(parameter.numel() for parameter in model.parameters()))' \
	| $(PYTHON)

train: setup data-dirs
	@printf '==> [train] Training model\n'
	@printf '%s\n' \
		'import json' \
		'from dataclasses import asdict, replace' \
		'from pathlib import Path' \
		'' \
		'import torch' \
		'' \
		'from superagi.model.checkpoint import prepare_model_for_training, retain_checkpoint_snapshot, save_checkpoint' \
		'from superagi.initialization.read_config import load_project_config' \
		'from superagi.training.train import TokenShardDataset, TrainConfig, append_metrics_jsonl, train_model_with_metrics' \
		'' \
		'def choose_device() -> str:' \
		'    configured = "$(DEVICE)"' \
		'    if configured != "auto":' \
		'        return configured' \
		'    if torch.cuda.is_available():' \
		'        return "cuda"' \
		'    mps = getattr(torch.backends, "mps", None)' \
		'    if mps is not None and mps.is_available():' \
		'        return "mps"' \
		'    return "cpu"' \
		'' \
		'token_path = Path("$(PROCESSED_DIR)") / "$(ARTIFACT)_tokens.pt"' \
		'shard_manifest_path = Path("$(PROCESSED_DIR)") / "train_shards" / "manifest.json"' \
		'vocab_path = Path("$(PROCESSED_DIR)") / "$(ARTIFACT)_vocab.json"' \
		'val_token_path = Path("$(VAL_TOKENS)")' \
		'if not vocab_path.exists():' \
		'    raise SystemExit("Missing vocab artifact. Run `make ingest`, `make ingest-stream-c4`, or `make ingest-stream-sources` first.")' \
		'' \
		'if token_path.exists():' \
		'    token_ids = torch.load(token_path)' \
		'elif shard_manifest_path.exists():' \
		'    token_ids = TokenShardDataset.from_manifest(shard_manifest_path)' \
		'    print(f"Training shards: {token_ids.shard_count} shards, {token_ids.total_tokens} tokens")' \
		'else:' \
		'    raise SystemExit("Missing train tokens. Run `make ingest`, `make ingest-stream-c4`, or `make ingest-stream-sources` first. Expected train_tokens.pt or train_shards/manifest.json.")' \
		'validation_token_ids = None' \
		'if val_token_path.exists():' \
		'    validation_token_ids = torch.load(val_token_path)' \
		'    print(f"Validation tokens: {len(validation_token_ids)}")' \
		'else:' \
		'    print("Validation tokens: not found; validation_loss will be null")' \
		'activation_checkpointing = "$(ACTIVATION_CHECKPOINTING)".lower() in {"1", "true", "yes", "on"}' \
		'compile_model = "$(COMPILE_MODEL)".lower() in {"1", "true", "yes", "on"}' \
		'vocab = json.loads(vocab_path.read_text(encoding="utf-8"))' \
		'model_config = load_project_config().to_transformer_config(vocab_size=int(vocab["vocab_size"]))' \
		'model_config = replace(' \
		'    model_config,' \
		'    activation_checkpointing=activation_checkpointing,' \
		'    dropout=float("$(DROPOUT)"),' \
		')' \
		'training_state = prepare_model_for_training(' \
		'    vocab=vocab,' \
		'    config=model_config,' \
		'    resume_path="$(RESUME)" or None,' \
		')' \
		'if training_state.resumed_from is not None:' \
		'    print(f"Resuming from: {training_state.resumed_from}")' \
		'else:' \
		'    print("Training from scratch")' \
		'checkpoint_interval = int("$(CHECKPOINT_INTERVAL)")' \
		'checkpoint_keep = int("$(CHECKPOINT_KEEP)")' \
		'best_state = {"validation_loss": None, "step": None}' \
		'for previous_metric in training_state.previous_metrics:' \
		'    validation_loss = previous_metric.get("validation_loss")' \
		'    if validation_loss is not None and (best_state["validation_loss"] is None or validation_loss < best_state["validation_loss"]):' \
		'        best_state["validation_loss"] = float(validation_loss)' \
		'        best_state["step"] = previous_metric.get("step")' \
		'metadata_best_loss = training_state.metadata.get("best_validation_loss")' \
		'metadata_best_step = training_state.metadata.get("best_validation_step")' \
		'if metadata_best_loss is not None:' \
		'    metadata_best_loss = float(metadata_best_loss)' \
		'    if best_state["validation_loss"] is None or metadata_best_loss < best_state["validation_loss"]:' \
		'        best_state["validation_loss"] = metadata_best_loss' \
		'        best_state["step"] = metadata_best_step' \
		'def build_training_metadata(total_losses: int, status: str, saved_step: int | None = None) -> dict:' \
		'    metadata = dict(training_state.metadata)' \
		'    metadata.update(' \
		'        {' \
		'            "artifact": "$(ARTIFACT)",' \
		'            "steps": total_losses,' \
		'            "last_run_steps": int("$(STEPS)"),' \
		'            "batch_size": int("$(BATCH)"),' \
		'            "grad_accum_steps": int("$(GRAD_ACCUM_STEPS)"),' \
		'            "effective_batch_size": int("$(BATCH)") * int("$(GRAD_ACCUM_STEPS)"),' \
		'            "activation_checkpointing": activation_checkpointing,' \
		'            "shard_refresh_interval": int("$(SHARD_REFRESH_INTERVAL)"),' \
		'            "learning_rate": float("$(LR)"),' \
		'            "min_learning_rate": float("$(LR_MIN)"),' \
		'            "warmup_steps": int("$(LR_WARMUP_STEPS)"),' \
		'            "mixed_precision": "$(MIXED_PRECISION)",' \
		'            "parameter_dtype": "$(PARAMETER_DTYPE)",' \
		'            "fused_adamw": "$(FUSED_ADAMW)",' \
		'            "compile_model": compile_model,' \
		'            "dropout": float("$(DROPOUT)"),' \
		'            "eval_interval": int("$(EVAL_INTERVAL)"),' \
		'            "validation_batches": int("$(VAL_BATCHES)"),' \
		'            "checkpoint_interval": checkpoint_interval,' \
		'            "checkpoint_keep": checkpoint_keep,' \
		'            "metrics_path": "$(METRICS)",' \
		'            "status": status,' \
		'        }' \
		'    )' \
		'    if saved_step is not None:' \
		'        metadata["last_saved_step"] = saved_step' \
		'    if best_state["validation_loss"] is not None:' \
		'        metadata["best_validation_loss"] = best_state["validation_loss"]' \
		'        metadata["best_validation_step"] = best_state["step"]' \
		'        metadata["best_checkpoint"] = "$(BEST_CHECKPOINT)"' \
		'    if training_state.resumed_from is not None:' \
		'        metadata["resumed_from"] = str(training_state.resumed_from)' \
		'    return metadata' \
		'def save_best_checkpoint(metric: object, run_losses: list[float], run_metrics: list[object]) -> None:' \
		'    if metric.validation_loss is None:' \
		'        return' \
		'    validation_loss = float(metric.validation_loss)' \
		'    if best_state["validation_loss"] is not None and validation_loss >= best_state["validation_loss"]:' \
		'        return' \
		'    best_state["validation_loss"] = validation_loss' \
		'    best_state["step"] = metric.step' \
		'    run_metric_dicts = [asdict(item) for item in run_metrics]' \
		'    best_losses = training_state.previous_losses + run_losses' \
		'    best_metrics = training_state.previous_metrics + run_metric_dicts' \
		'    checkpoint_path = save_checkpoint(' \
		'        Path("$(BEST_CHECKPOINT)"),' \
		'        model=training_state.model,' \
		'        vocab=training_state.vocab,' \
		'        losses=best_losses,' \
		'        metrics=best_metrics,' \
		'        metadata=build_training_metadata(len(best_losses), "best", metric.step),' \
		'    )' \
		'    print(f"Best checkpoint: {checkpoint_path} at step {metric.step} validation_loss={validation_loss:.6f}", flush=True)' \
		'def save_periodic_checkpoint(step: int, run_losses: list[float], run_metrics: list[object]) -> None:' \
		'    run_metric_dicts = [asdict(metric) for metric in run_metrics]' \
		'    periodic_losses = training_state.previous_losses + run_losses' \
		'    periodic_metrics = training_state.previous_metrics + run_metric_dicts' \
		'    checkpoint_path = save_checkpoint(' \
		'        Path("$(CHECKPOINT)"),' \
		'        model=training_state.model,' \
		'        vocab=training_state.vocab,' \
		'        losses=periodic_losses,' \
		'        metrics=periodic_metrics,' \
		'        metadata=build_training_metadata(len(periodic_losses), "in_progress", step),' \
		'    )' \
		'    print(f"Periodic checkpoint: {checkpoint_path} at step {step}", flush=True)' \
		'    retained_path = retain_checkpoint_snapshot(' \
		'        checkpoint_path,' \
		'        Path("$(CHECKPOINT_DIR)") / "snapshots",' \
		'        step=step,' \
		'        keep=checkpoint_keep,' \
		'    )' \
		'    if retained_path is not None:' \
		'        print(f"Retained checkpoint: {retained_path}", flush=True)' \
		'checkpoint_callback = save_periodic_checkpoint if checkpoint_interval > 0 else None' \
		'new_losses, new_metrics = train_model_with_metrics(' \
		'    model=training_state.model,' \
		'    token_ids=token_ids,' \
		'    config=TrainConfig(' \
		'        batch_size=int("$(BATCH)"),' \
		'        grad_accum_steps=int("$(GRAD_ACCUM_STEPS)"),' \
		'        learning_rate=float("$(LR)"),' \
		'        min_learning_rate=float("$(LR_MIN)"),' \
		'        warmup_steps=int("$(LR_WARMUP_STEPS)"),' \
		'        mixed_precision="$(MIXED_PRECISION)",' \
		'        parameter_dtype="$(PARAMETER_DTYPE)",' \
		'        fused_adamw="$(FUSED_ADAMW)",' \
		'        compile_model=compile_model,' \
		'        shard_refresh_interval=int("$(SHARD_REFRESH_INTERVAL)"),' \
		'        max_steps=int("$(STEPS)"),' \
		'    ),' \
		'    device=choose_device(),' \
		'    validation_token_ids=validation_token_ids,' \
		'    eval_interval=int("$(EVAL_INTERVAL)"),' \
		'    validation_batches=int("$(VAL_BATCHES)"),' \
		'    start_step=len(training_state.previous_losses),' \
		'    checkpoint_interval=checkpoint_interval,' \
		'    checkpoint_callback=checkpoint_callback,' \
		'    metric_callback=save_best_checkpoint,' \
		')' \
		'losses = training_state.previous_losses + new_losses' \
		'new_metric_dicts = [asdict(metric) for metric in new_metrics]' \
		'metrics = training_state.previous_metrics + new_metric_dicts' \
		'append_metrics_jsonl(Path("$(METRICS)"), new_metrics)' \
		'metadata = build_training_metadata(len(losses), "complete")' \
		'checkpoint_path = save_checkpoint(' \
		'    Path("$(CHECKPOINT)"),' \
		'    model=training_state.model,' \
		'    vocab=training_state.vocab,' \
		'    losses=losses,' \
		'    metrics=metrics,' \
		'    metadata=metadata,' \
		')' \
		'final_loss = new_losses[-1] if new_losses else "n/a"' \
		'print(f"Final loss: {final_loss}")' \
		'if new_metrics:' \
		'    last_metric = new_metrics[-1]' \
		'    print(f"Last metric: step={last_metric.step} train_loss={last_metric.train_loss} validation_loss={last_metric.validation_loss}")' \
		'    print("Metrics: $(METRICS)")' \
		'print(f"Total recorded steps: {len(losses)}")' \
		'print(f"Checkpoint: {checkpoint_path}")' \
	| $(PYTHON)
	@printf '==> [train] Finished training model\n'

sft-train: setup
	@printf '==> [sft-train] Training supervised chat model\n'
	$(PYTHON) scripts/train_sft.py \
		--base-checkpoint "$(SFT_BASE_CHECKPOINT)" \
		--data "$(SFT_DATA)" \
		--out "$(SFT_OUT)" \
		--metrics "$(SFT_METRICS)" \
		--steps "$(SFT_STEPS)" \
		--batch "$(SFT_BATCH)" \
		--lr "$(SFT_LR)" \
		--lr-min "$(SFT_LR_MIN)" \
		--lr-warmup-steps "$(SFT_LR_WARMUP_STEPS)" \
		--weight-decay "$(SFT_WEIGHT_DECAY)" \
		--grad-clip "$(SFT_GRAD_CLIP)" \
		--checkpoint-interval "$(SFT_CHECKPOINT_INTERVAL)" \
		--validation-fraction "$(SFT_VALIDATION_FRACTION)" \
		--validation-batches "$(SFT_VALIDATION_BATCHES)" \
		--source-weights "$(SFT_SOURCE_WEIGHTS)" \
		--device "$(SFT_DEVICE)" \
		--seed "$(SFT_SEED)"
	@printf '==> [sft-train] Finished supervised chat training\n'

sft-import-public: setup
	@printf '==> [sft-import-public] Importing public SFT datasets\n'
	$(PYTHON) scripts/import_public_sft.py \
		--checkpoint "$(SFT_IMPORT_CHECKPOINT)" \
		--out "$(SFT_IMPORT_OUT)" \
		--metadata "$(SFT_IMPORT_METADATA)" \
		--sources "$(SFT_IMPORT_SOURCES)" \
		--max-rows-per-source "$(SFT_IMPORT_MAX_ROWS_PER_SOURCE)" \
		--max-examples-per-source "$(SFT_IMPORT_MAX_EXAMPLES_PER_SOURCE)" \
		--max-context-tokens "$(SFT_IMPORT_MAX_CONTEXT_TOKENS)" \
		--max-messages "$(SFT_IMPORT_MAX_MESSAGES)" \
		--max-agi-chars "$(SFT_IMPORT_MAX_AGI_CHARS)" \
		--min-agi-chars "$(SFT_IMPORT_MIN_AGI_CHARS)"
	@printf '==> [sft-import-public] Finished importing public SFT datasets\n'

sft-overfit-50:
	@printf '==> [sft-overfit-50] Training hard-overfit SFT diagnostic\n'
	$(MAKE) sft-train \
		SFT_BASE_CHECKPOINT="$(SFT_OVERFIT_BASE_CHECKPOINT)" \
		SFT_DATA="$(SFT_OVERFIT_DATA)" \
		SFT_OUT="$(SFT_OVERFIT_OUT)" \
		SFT_METRICS="$(SFT_OVERFIT_METRICS)" \
		SFT_STEPS="$(SFT_OVERFIT_STEPS)" \
		SFT_BATCH="$(SFT_OVERFIT_BATCH)" \
		SFT_LR="$(SFT_OVERFIT_LR)" \
		SFT_LR_MIN="$(SFT_OVERFIT_LR_MIN)" \
		SFT_LR_WARMUP_STEPS="$(SFT_OVERFIT_LR_WARMUP_STEPS)" \
		SFT_WEIGHT_DECAY="$(SFT_OVERFIT_WEIGHT_DECAY)" \
		SFT_CHECKPOINT_INTERVAL="$(SFT_OVERFIT_CHECKPOINT_INTERVAL)" \
		SFT_VALIDATION_FRACTION="0"
	@printf '==> [sft-overfit-50] Finished hard-overfit SFT diagnostic\n'

sft-anchor:
	@printf '==> [sft-anchor] Training anchor behavior SFT phase\n'
	$(MAKE) sft-train \
		SFT_BASE_CHECKPOINT="$(SFT_ANCHOR_BASE_CHECKPOINT)" \
		SFT_DATA="$(SFT_ANCHOR_DATA)" \
		SFT_OUT="$(SFT_ANCHOR_OUT)" \
		SFT_METRICS="$(SFT_ANCHOR_METRICS)" \
		SFT_STEPS="$(SFT_ANCHOR_STEPS)" \
		SFT_BATCH="$(SFT_ANCHOR_BATCH)" \
		SFT_LR="$(SFT_ANCHOR_LR)" \
		SFT_LR_MIN="$(SFT_ANCHOR_LR_MIN)" \
		SFT_LR_WARMUP_STEPS="$(SFT_ANCHOR_LR_WARMUP_STEPS)" \
		SFT_WEIGHT_DECAY="$(SFT_ANCHOR_WEIGHT_DECAY)" \
		SFT_CHECKPOINT_INTERVAL="$(SFT_ANCHOR_CHECKPOINT_INTERVAL)"
	@printf '==> [sft-anchor] Finished anchor behavior SFT phase\n'

sft-broad:
	@printf '==> [sft-broad] Training broad chat SFT phase\n'
	$(MAKE) sft-train \
		SFT_BASE_CHECKPOINT="$(SFT_BROAD_BASE_CHECKPOINT)" \
		SFT_DATA="$(SFT_BROAD_DATA)" \
		SFT_OUT="$(SFT_BROAD_OUT)" \
		SFT_METRICS="$(SFT_BROAD_METRICS)" \
		SFT_STEPS="$(SFT_BROAD_STEPS)" \
		SFT_BATCH="$(SFT_BROAD_BATCH)" \
		SFT_LR="$(SFT_BROAD_LR)" \
		SFT_LR_MIN="$(SFT_BROAD_LR_MIN)" \
		SFT_LR_WARMUP_STEPS="$(SFT_BROAD_LR_WARMUP_STEPS)" \
		SFT_WEIGHT_DECAY="$(SFT_BROAD_WEIGHT_DECAY)" \
		SFT_CHECKPOINT_INTERVAL="$(SFT_BROAD_CHECKPOINT_INTERVAL)" \
		SFT_SOURCE_WEIGHTS="$(SFT_BROAD_SOURCE_WEIGHTS)"
	@printf '==> [sft-broad] Finished broad chat SFT phase\n'

sft-style:
	@printf '==> [sft-style] Training playful direct style SFT phase\n'
	$(MAKE) sft-train \
		SFT_BASE_CHECKPOINT="$(SFT_STYLE_BASE_CHECKPOINT)" \
		SFT_DATA="$(SFT_STYLE_DATA)" \
		SFT_OUT="$(SFT_STYLE_OUT)" \
		SFT_METRICS="$(SFT_STYLE_METRICS)" \
		SFT_STEPS="$(SFT_STYLE_STEPS)" \
		SFT_BATCH="$(SFT_STYLE_BATCH)" \
		SFT_LR="$(SFT_STYLE_LR)" \
		SFT_LR_MIN="$(SFT_STYLE_LR_MIN)" \
		SFT_LR_WARMUP_STEPS="$(SFT_STYLE_LR_WARMUP_STEPS)" \
		SFT_WEIGHT_DECAY="$(SFT_STYLE_WEIGHT_DECAY)" \
		SFT_CHECKPOINT_INTERVAL="$(SFT_STYLE_CHECKPOINT_INTERVAL)"
	@printf '==> [sft-style] Finished playful direct style SFT phase\n'

sft-staged:
	@printf '==> [sft-staged] Starting staged supervised chat training\n'
	$(MAKE) sft-anchor \
		SFT_ANCHOR_BASE_CHECKPOINT="$(SFT_STAGED_BASE_CHECKPOINT)"
	$(MAKE) sft-broad \
		SFT_BROAD_BASE_CHECKPOINT="$(SFT_ANCHOR_OUT)"
	$(MAKE) sft-style \
		SFT_STYLE_BASE_CHECKPOINT="$(SFT_BROAD_OUT)"
	@printf 'Final staged checkpoint: $(SFT_STAGED_OUT)\n'
	@printf '==> [sft-staged] Finished staged supervised chat training\n'

train-export-run:
	@set -e; \
	printf '==> [pipeline] Starting train/export/run\n'; \
	start_time=$$(date +%s); \
	$(MAKE) train; \
	end_time=$$(date +%s); \
	elapsed=$$((end_time - start_time)); \
	printf 'Training elapsed: %02d:%02d:%02d (%s seconds)\n' \
		$$((elapsed / 3600)) \
		$$(((elapsed % 3600) / 60)) \
		$$((elapsed % 60)) \
		$$elapsed
	@printf '==> [pipeline] Training finished; exporting model\n'
	$(MAKE) export-model
	@printf '==> [pipeline] Export finished; generating sample\n'
	$(MAKE) run-model CHECKPOINT="$(MODEL_OUT)"
	@printf '==> [pipeline] Finished train/export/run\n'

train-4090:
	@printf '==> [train-4090] Fetching night-run C4 corpus\n'
	$(MAKE) c4 \
		RAW_DIR="$(TRAIN_4090_RAW_DIR)" \
		C4_MAX="$(TRAIN_4090_C4_MAX)" \
		C4_MIN_CHARS="$(TRAIN_4090_C4_MIN_CHARS)"
	@printf '==> [train-4090] Clearing generated processed artifacts and checkpoints\n'
	$(MAKE) clean-generated
	@printf '==> [train-4090] Ingesting night-run corpus\n'
	$(MAKE) ingest RAW_DIR="$(TRAIN_4090_RAW_DIR)"
	@printf '==> [train-4090] Starting fresh RTX 4090 night run\n'
	$(MAKE) train-export-run \
		RESUME= \
		STEPS="$(TRAIN_4090_STEPS)" \
		BATCH="$(TRAIN_4090_BATCH)" \
		GRAD_ACCUM_STEPS="$(TRAIN_4090_GRAD_ACCUM_STEPS)" \
		ACTIVATION_CHECKPOINTING="$(TRAIN_4090_ACTIVATION_CHECKPOINTING)" \
		DEVICE="$(TRAIN_4090_DEVICE)" \
		MIXED_PRECISION="$(TRAIN_4090_MIXED_PRECISION)" \
		PARAMETER_DTYPE="$(TRAIN_4090_PARAMETER_DTYPE)" \
		FUSED_ADAMW="$(TRAIN_4090_FUSED_ADAMW)" \
		COMPILE_MODEL="$(TRAIN_4090_COMPILE_MODEL)" \
		DROPOUT="$(TRAIN_4090_DROPOUT)" \
		SHARD_REFRESH_INTERVAL="$(TRAIN_4090_SHARD_REFRESH_INTERVAL)" \
		EVAL_INTERVAL="$(TRAIN_4090_EVAL_INTERVAL)" \
		VAL_BATCHES="$(TRAIN_4090_VAL_BATCHES)" \
		CHECKPOINT_INTERVAL="$(TRAIN_4090_CHECKPOINT_INTERVAL)" \
		CHECKPOINT_KEEP="$(TRAIN_4090_CHECKPOINT_KEEP)" \
		PROMPT="$(TRAIN_4090_PROMPT)" \
		NEW_TOKENS="$(TRAIN_4090_NEW_TOKENS)" \
		TEMPERATURE="$(TRAIN_4090_TEMPERATURE)" \
		TOP_K="$(TRAIN_4090_TOP_K)"
	@printf '==> [train-4090] Finished RTX 4090 night run\n'

train-200m:
	@printf '==> [train-200m] Starting 200M training pipeline\n'
	@printf '==> [train-200m] Clearing generated processed artifacts and checkpoints\n'
	$(MAKE) clean-generated
	@printf '==> [train-200m] Streaming and tokenizing C4 corpus\n'
	$(MAKE) ingest-stream-c4 \
		STREAM_C4_MAX="$(TRAIN_200M_STREAM_C4_MAX)" \
		STREAM_C4_MIN_CHARS="$(TRAIN_200M_STREAM_C4_MIN_CHARS)" \
		STREAM_TOKENIZER_SAMPLE="$(TRAIN_200M_STREAM_TOKENIZER_SAMPLE)" \
		STREAM_SHARD_TOKENS="$(TRAIN_200M_STREAM_SHARD_TOKENS)" \
		STREAM_VALIDATION_TOKENS="$(TRAIN_200M_STREAM_VALIDATION_TOKENS)" \
		BPE_VOCAB_SIZE="$(TRAIN_200M_BPE_VOCAB_SIZE)" \
		BPE_MIN_FREQUENCY="$(TRAIN_200M_BPE_MIN_FREQUENCY)"
	@printf '==> [train-200m] Starting fresh 200M train/export/run\n'
	$(MAKE) train-export-run \
		RESUME= \
		STEPS="$(TRAIN_200M_STEPS)" \
		BATCH="$(TRAIN_200M_BATCH)" \
		GRAD_ACCUM_STEPS="$(TRAIN_200M_GRAD_ACCUM_STEPS)" \
		ACTIVATION_CHECKPOINTING="$(TRAIN_200M_ACTIVATION_CHECKPOINTING)" \
		DEVICE="$(TRAIN_200M_DEVICE)" \
		MIXED_PRECISION="$(TRAIN_200M_MIXED_PRECISION)" \
		PARAMETER_DTYPE="$(TRAIN_200M_PARAMETER_DTYPE)" \
		FUSED_ADAMW="$(TRAIN_200M_FUSED_ADAMW)" \
		COMPILE_MODEL="$(TRAIN_200M_COMPILE_MODEL)" \
		DROPOUT="$(TRAIN_200M_DROPOUT)" \
		SHARD_REFRESH_INTERVAL="$(TRAIN_200M_SHARD_REFRESH_INTERVAL)" \
		LR="$(TRAIN_200M_LR)" \
		LR_MIN="$(TRAIN_200M_LR_MIN)" \
		LR_WARMUP_STEPS="$(TRAIN_200M_LR_WARMUP_STEPS)" \
		EVAL_INTERVAL="$(TRAIN_200M_EVAL_INTERVAL)" \
		VAL_BATCHES="$(TRAIN_200M_VAL_BATCHES)" \
		CHECKPOINT_INTERVAL="$(TRAIN_200M_CHECKPOINT_INTERVAL)" \
		CHECKPOINT_KEEP="$(TRAIN_200M_CHECKPOINT_KEEP)" \
		PROMPT="$(TRAIN_200M_PROMPT)" \
		NEW_TOKENS="$(TRAIN_200M_NEW_TOKENS)" \
		TEMPERATURE="$(TRAIN_200M_TEMPERATURE)" \
		TOP_K="$(TRAIN_200M_TOP_K)"
	@printf '==> [train-200m] Finished 200M training pipeline\n'

train-h100:
	@printf '==> [train-h100] Starting zero-setup H100 dynamic training pipeline\n'
	$(MAKE) setup
	@printf '==> [train-h100] Clearing generated processed artifacts and checkpoints\n'
	$(MAKE) clean-generated
	@set -e; \
	manifest_path="$(PROCESSED_DIR)/train_shards/manifest.json"; \
		printf '==> [train-h100] Starting background mixed-source streaming/tokenization\n'; \
		$(MAKE) ingest-stream-sources \
			SOURCES="$(TRAIN_H100_SOURCES)" \
			STREAM_MAX_DOCUMENTS_PER_SOURCE="$(TRAIN_H100_STREAM_MAX_DOCUMENTS_PER_SOURCE)" \
			STREAM_C4_MAX="$(TRAIN_H100_STREAM_C4_MAX)" \
			STREAM_C4_MIN_CHARS="$(TRAIN_H100_STREAM_C4_MIN_CHARS)" \
			STREAM_TOKENIZER_SAMPLE="$(TRAIN_H100_STREAM_TOKENIZER_SAMPLE)" \
			STREAM_SHARD_TOKENS="$(TRAIN_H100_STREAM_SHARD_TOKENS)" \
		STREAM_VALIDATION_TOKENS="$(TRAIN_H100_STREAM_VALIDATION_TOKENS)" \
		STREAM_TARGET_TOKENS="$(TRAIN_H100_CORPUS_TARGET_TOKENS)" \
		BPE_VOCAB_SIZE="$(TRAIN_H100_BPE_VOCAB_SIZE)" \
		BPE_MIN_FREQUENCY="$(TRAIN_H100_BPE_MIN_FREQUENCY)" & \
	ingest_pid=$$!; \
	cleanup() { \
		if kill -0 "$$ingest_pid" 2>/dev/null; then \
			printf '==> [train-h100] Stopping background ingestion\n'; \
			kill "$$ingest_pid" 2>/dev/null || true; \
			wait "$$ingest_pid" 2>/dev/null || true; \
		fi; \
	}; \
	trap cleanup EXIT INT TERM; \
	printf '==> [train-h100] Waiting for %s prepared train tokens before training\n' "$(TRAIN_H100_START_TOKENS)"; \
	while true; do \
		if [ -f "$$manifest_path" ]; then \
			ready_tokens=$$($(PYTHON) -c 'import json, sys; from pathlib import Path; print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("train_tokens", 0)))' "$$manifest_path"); \
			printf '==> [train-h100] Prepared train tokens: %s / %s\n' "$$ready_tokens" "$(TRAIN_H100_START_TOKENS)"; \
			if [ "$$ready_tokens" -ge "$(TRAIN_H100_START_TOKENS)" ]; then \
				break; \
			fi; \
		else \
			printf '==> [train-h100] Waiting for first token shard manifest\n'; \
		fi; \
		if ! kill -0 "$$ingest_pid" 2>/dev/null; then \
			wait "$$ingest_pid"; \
			printf '==> [train-h100] Ingestion exited before start-token threshold was reached\n'; \
			exit 1; \
		fi; \
		sleep "$(TRAIN_H100_READY_POLL_SECONDS)"; \
	done; \
	train_steps=$$($(PYTHON) -c 'import math, yaml; from pathlib import Path; config = yaml.safe_load(Path("specs/config.yaml").read_text(encoding="utf-8")); ctx = int(config["parameters"]["ctx_window"]); tokens_per_step = int("$(TRAIN_H100_BATCH)") * int("$(TRAIN_H100_GRAD_ACCUM_STEPS)") * ctx; print(max(1, math.ceil(int("$(TRAIN_H100_TOTAL_TRAINING_TOKENS)") / tokens_per_step)))'); \
	printf '==> [train-h100] Computed training steps: %s for %s target training tokens\n' "$$train_steps" "$(TRAIN_H100_TOTAL_TRAINING_TOKENS)"; \
	set +e; \
	$(MAKE) train-export-run \
		RESUME= \
		STEPS="$$train_steps" \
		BATCH="$(TRAIN_H100_BATCH)" \
		GRAD_ACCUM_STEPS="$(TRAIN_H100_GRAD_ACCUM_STEPS)" \
		ACTIVATION_CHECKPOINTING="$(TRAIN_H100_ACTIVATION_CHECKPOINTING)" \
		DEVICE="$(TRAIN_H100_DEVICE)" \
		MIXED_PRECISION="$(TRAIN_H100_MIXED_PRECISION)" \
		PARAMETER_DTYPE="$(TRAIN_H100_PARAMETER_DTYPE)" \
		FUSED_ADAMW="$(TRAIN_H100_FUSED_ADAMW)" \
		COMPILE_MODEL="$(TRAIN_H100_COMPILE_MODEL)" \
		DROPOUT="$(TRAIN_H100_DROPOUT)" \
		SHARD_REFRESH_INTERVAL="$(TRAIN_H100_SHARD_REFRESH_INTERVAL)" \
		LR="$(TRAIN_H100_LR)" \
		LR_MIN="$(TRAIN_H100_LR_MIN)" \
		LR_WARMUP_STEPS="$(TRAIN_H100_LR_WARMUP_STEPS)" \
		EVAL_INTERVAL="$(TRAIN_H100_EVAL_INTERVAL)" \
		VAL_BATCHES="$(TRAIN_H100_VAL_BATCHES)" \
		CHECKPOINT_INTERVAL="$(TRAIN_H100_CHECKPOINT_INTERVAL)" \
		CHECKPOINT_KEEP="$(TRAIN_H100_CHECKPOINT_KEEP)" \
		PROMPT="$(TRAIN_H100_PROMPT)" \
		NEW_TOKENS="$(TRAIN_H100_NEW_TOKENS)" \
		TEMPERATURE="$(TRAIN_H100_TEMPERATURE)" \
		TOP_K="$(TRAIN_H100_TOP_K)"; \
	train_status=$$?; \
	set -e; \
	if kill -0 "$$ingest_pid" 2>/dev/null; then \
		printf '==> [train-h100] Training finished before ingestion target; stopping ingestion\n'; \
		kill "$$ingest_pid" 2>/dev/null || true; \
		wait "$$ingest_pid" 2>/dev/null || true; \
	else \
		wait "$$ingest_pid" || true; \
	fi; \
	trap - EXIT INT TERM; \
	printf '==> [train-h100] Finished H100 dynamic training pipeline\n'; \
	exit "$$train_status"

std-train:
	$(MAKE) train-export-run \
		RESUME=data/checkpoints/latest.pt \
		STEPS=5000 \
		BATCH=16 \
		DEVICE=auto \
		PROMPT="Attention is" \
		NEW_TOKENS=300 \
		TEMPERATURE=0.6

export-model: setup
	@printf '==> [export-model] Exporting portable model\n'
	@printf '%s\n' \
		'import shutil' \
		'from pathlib import Path' \
		'' \
		'from superagi.model.checkpoint import load_checkpoint' \
		'' \
		'checkpoint_path = Path("$(CHECKPOINT)")' \
		'if not checkpoint_path.exists():' \
		'    raise SystemExit("Missing checkpoint. Run `make train` first.")' \
		'load_checkpoint(checkpoint_path)' \
		'output_path = Path("$(MODEL_OUT)")' \
		'output_path.parent.mkdir(parents=True, exist_ok=True)' \
		'shutil.copy2(checkpoint_path, output_path)' \
		'print(f"Portable model artifact: {output_path.resolve()}")' \
	| $(PYTHON)
	@printf '==> [export-model] Finished exporting portable model\n'

generate: run-model

run-model: setup
	@printf '==> [run-model] Generating sample text\n'
	$(PYTHON) scripts/run_model.py \
		--checkpoint "$(CHECKPOINT)" \
		--prompt "$(PROMPT)" \
		--new-tokens "$(NEW_TOKENS)" \
		--temperature "$(TEMPERATURE)" \
		--top-k "$(TOP_K)" \
		--repetition-penalty "$(REPETITION_PENALTY)" \
		--repetition-window "$(REPETITION_WINDOW)" \
		--chat "$(CHAT)" \
		--stream "$(STREAM)" \
		--device "$(DEVICE)"
	@printf '==> [run-model] Finished generating sample text\n'

chat: setup
	@printf '==> [chat] Starting interactive chat\n'
	$(PYTHON) scripts/chat.py \
		--checkpoint "$(CHAT_CHECKPOINT)" \
		--new-tokens "$(CHAT_MAX_TOKENS)" \
		--temperature "$(TEMPERATURE)" \
		--top-k "$(TOP_K)" \
		--repetition-penalty "$(REPETITION_PENALTY)" \
		--repetition-window "$(REPETITION_WINDOW)" \
		--device "$(DEVICE)"

smoke-train: setup
	@printf '%s\n' \
		'from superagi.model.transformer import TransformerConfig, TransformerLM' \
		'from superagi.training.train import TrainConfig, train_model' \
		'' \
		'token_ids = [0, 1, 2, 3, 4, 5] * 8' \
		'model = TransformerLM(' \
		'    TransformerConfig(' \
		'        vocab_size=6,' \
		'        context_length=8,' \
		'        dim_embedding=16,' \
		'        n_layers=1,' \
		'        n_heads=4,' \
		'    )' \
		')' \
		'losses = train_model(' \
		'    model=model,' \
		'    token_ids=token_ids,' \
		'    config=TrainConfig(batch_size=4, learning_rate=1e-3, max_steps=2),' \
		')' \
		'print(losses)' \
	| $(PYTHON)

clean-generated:
	rm -rf "$(PROCESSED_DIR)"/* "$(CHECKPOINT_DIR)"/*
