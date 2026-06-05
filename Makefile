SHELL := /bin/bash

PYTHON := .venv/bin/python
PYTHONPATH := src
export PYTHONPATH

RAW_DIR := data/raw
PROCESSED_DIR := data/processed
CHECKPOINT_DIR := data/checkpoints
ARTIFACT := train
CHECKPOINT := $(CHECKPOINT_DIR)/latest.pt
MODEL_OUT := $(CHECKPOINT_DIR)/final-model.pt

WIKI_QUERIES := machine learning,artificial intelligence
WIKI_MAX := 5
WIKI_MIN_CHARS := 1000

C4_MAX := 100
C4_MIN_CHARS := 500

BATCH := 32
LR := 3e-4
STEPS := 1000
DEVICE := auto
PROMPT := The
NEW_TOKENS := 100
TEMPERATURE := 1.0

.PHONY: help setup data-dirs test wiki c4 ingest train params export-model generate run-model smoke-train clean-generated

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
	@echo "  make c4 C4_MAX=20 C4_MIN_CHARS=500"
	@echo ""
	@echo "Ingest and train:"
	@echo "  make ingest            Tokenize data/raw into data/processed"
	@echo "  make params            Count params using data/processed vocab"
	@echo "  make train STEPS=100 BATCH=16"
	@echo "  make export-model      Validate/copy latest checkpoint to a portable .pt file"
	@echo "  make run-model PROMPT=\"The\" NEW_TOKENS=100"
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
	@printf '%s\n' \
		'from superagi.ingestion.builders.wikipedia import build_wikipedia_corpus' \
		'' \
		'queries = [query.strip() for query in "$(WIKI_QUERIES)".split(",") if query.strip()]' \
		'result = build_wikipedia_corpus(' \
		'    queries=queries,' \
		'    raw_root="$(RAW_DIR)",' \
		'    max_articles_per_query=int("$(WIKI_MAX)"),' \
		'    min_chars=int("$(WIKI_MIN_CHARS)"),' \
		')' \
		'print(f"Wrote {result.documents_written} Wikipedia documents to {result.output_dir}")' \
		'print(f"Metadata: {result.metadata_path}")' \
	| $(PYTHON)

c4: setup data-dirs
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

ingest: setup data-dirs
	@printf '%s\n' \
		'from superagi.ingestion.corpus import ingest_raw_corpus' \
		'' \
		'artifact = ingest_raw_corpus(' \
		'    raw_dir="$(RAW_DIR)",' \
		'    processed_dir="$(PROCESSED_DIR)",' \
		'    artifact_name="$(ARTIFACT)",' \
		')' \
		'print(f"Tokenized {len(artifact.token_ids)} tokens")' \
		'print(f"Vocab size: {artifact.tokenizer.vocab_size}")' \
		'print("Artifacts: $(PROCESSED_DIR)/$(ARTIFACT)_tokens.pt and $(PROCESSED_DIR)/$(ARTIFACT)_vocab.json")' \
	| $(PYTHON)

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
		'config = load_project_config().to_transformer_config(vocab_size=len(vocab["id_to_char"]))' \
		'model = TransformerLM(config)' \
		'print(sum(parameter.numel() for parameter in model.parameters()))' \
	| $(PYTHON)

train: setup data-dirs
	@printf '%s\n' \
		'import json' \
		'from pathlib import Path' \
		'' \
		'import torch' \
		'' \
		'from superagi.model.checkpoint import save_checkpoint' \
		'from superagi.initialization.read_config import load_project_config' \
		'from superagi.model.transformer import TransformerLM' \
		'from superagi.training.train import TrainConfig, train_model' \
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
		'vocab_path = Path("$(PROCESSED_DIR)") / "$(ARTIFACT)_vocab.json"' \
		'if not token_path.exists() or not vocab_path.exists():' \
		'    raise SystemExit("Missing processed artifacts. Run `make ingest` first.")' \
		'' \
		'token_ids = torch.load(token_path).tolist()' \
		'vocab = json.loads(vocab_path.read_text(encoding="utf-8"))' \
		'model_config = load_project_config().to_transformer_config(vocab_size=len(vocab["id_to_char"]))' \
		'model = TransformerLM(model_config)' \
		'losses = train_model(' \
		'    model=model,' \
		'    token_ids=token_ids,' \
		'    config=TrainConfig(' \
		'        batch_size=int("$(BATCH)"),' \
		'        learning_rate=float("$(LR)"),' \
		'        max_steps=int("$(STEPS)"),' \
		'    ),' \
		'    device=choose_device(),' \
		')' \
		'checkpoint_path = save_checkpoint(' \
		'    Path("$(CHECKPOINT)"),' \
		'    model=model,' \
		'    vocab=vocab,' \
		'    losses=losses,' \
		'    metadata={"artifact": "$(ARTIFACT)", "steps": int("$(STEPS)")},' \
		')' \
		'final_loss = losses[-1] if losses else "n/a"' \
		'print(f"Final loss: {final_loss}")' \
		'print(f"Checkpoint: {checkpoint_path}")' \
	| $(PYTHON)

export-model: setup
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

generate: run-model

run-model: setup
	$(PYTHON) scripts/run_model.py \
		--checkpoint "$(CHECKPOINT)" \
		--prompt "$(PROMPT)" \
		--new-tokens "$(NEW_TOKENS)" \
		--temperature "$(TEMPERATURE)" \
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
	rm -f "$(PROCESSED_DIR)"/* "$(CHECKPOINT_DIR)"/*
