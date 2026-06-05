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

WIKI_QUERIES := machine learning,artificial intelligence
WIKI_MAX := 5
WIKI_MIN_CHARS := 1000
WIKI_USER_AGENT := SuperAGI-learning-corpus-builder/0.1 (local learning project; set WIKI_USER_AGENT=mailto:you@example.com)

C4_MAX := 100
C4_MIN_CHARS := 500

BATCH := 32
LR := 3e-4
STEPS := 1000
DEVICE := auto
PROMPT := The
NEW_TOKENS := 100
TEMPERATURE := 1.0
TOP_K := 0
STREAM := 1
VALIDATION_FRACTION := 0.1
VAL_TOKENS := $(PROCESSED_DIR)/val_tokens.pt
EVAL_INTERVAL := 500
VAL_BATCHES := 10
METRICS := $(CHECKPOINT_DIR)/metrics.jsonl
CHECKPOINT_INTERVAL := 1000

.PHONY: help setup data-dirs test wiki c4 ingest train params train-export-run std-train export-model generate run-model smoke-train clean-generated

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
	@echo "  make params            Count params using data/processed vocab"
	@echo "  make train STEPS=100 BATCH=16"
	@echo "  make train RESUME=data/checkpoints/latest.pt STEPS=1000 CHECKPOINT_INTERVAL=1000"
	@echo "  make train-export-run RESUME=data/checkpoints/latest.pt STEPS=1000 PROMPT=\"Attention is\""
	@echo "  make run-model CHECKPOINT=data/checkpoints/best.pt PROMPT=\"The\""
	@echo "  make std-train         Resume latest, train 5k steps, export, sample"
	@echo "  make export-model      Validate/copy latest checkpoint to a portable .pt file"
	@echo "  make run-model PROMPT=\"The\" NEW_TOKENS=100 TOP_K=50"
	@echo "  make run-model STREAM=0 PROMPT=\"The\" NEW_TOKENS=100"
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
		')' \
		'print(f"Tokenized {len(artifact.token_ids)} tokens")' \
		'print(f"Vocab size: {artifact.tokenizer.vocab_size}")' \
		'print("Artifacts: $(PROCESSED_DIR)/$(ARTIFACT)_tokens.pt, $(VAL_TOKENS), and $(PROCESSED_DIR)/$(ARTIFACT)_vocab.json")' \
	| $(PYTHON)
	@printf '==> [ingest] Finished ingesting raw text\n'

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
	@printf '==> [train] Training model\n'
	@printf '%s\n' \
		'import json' \
		'from dataclasses import asdict' \
		'from pathlib import Path' \
		'' \
		'import torch' \
		'' \
		'from superagi.model.checkpoint import prepare_model_for_training, save_checkpoint' \
		'from superagi.initialization.read_config import load_project_config' \
		'from superagi.training.train import TrainConfig, append_metrics_jsonl, train_model_with_metrics' \
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
		'val_token_path = Path("$(VAL_TOKENS)")' \
		'if not token_path.exists() or not vocab_path.exists():' \
		'    raise SystemExit("Missing processed artifacts. Run `make ingest` first.")' \
		'' \
		'token_ids = torch.load(token_path).tolist()' \
		'validation_token_ids = None' \
		'if val_token_path.exists():' \
		'    validation_token_ids = torch.load(val_token_path).tolist()' \
		'    print(f"Validation tokens: {len(validation_token_ids)}")' \
		'else:' \
		'    print("Validation tokens: not found; validation_loss will be null")' \
		'vocab = json.loads(vocab_path.read_text(encoding="utf-8"))' \
		'model_config = load_project_config().to_transformer_config(vocab_size=len(vocab["id_to_char"]))' \
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
		'            "eval_interval": int("$(EVAL_INTERVAL)"),' \
		'            "validation_batches": int("$(VAL_BATCHES)"),' \
		'            "checkpoint_interval": checkpoint_interval,' \
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
		'checkpoint_callback = save_periodic_checkpoint if checkpoint_interval > 0 else None' \
		'new_losses, new_metrics = train_model_with_metrics(' \
		'    model=training_state.model,' \
		'    token_ids=token_ids,' \
		'    config=TrainConfig(' \
		'        batch_size=int("$(BATCH)"),' \
		'        learning_rate=float("$(LR)"),' \
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
		--stream "$(STREAM)" \
		--device "$(DEVICE)"
	@printf '==> [run-model] Finished generating sample text\n'

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
