import unittest
from pathlib import Path


class MakefileTests(unittest.TestCase):
    def test_train_export_run_target_trains_times_exports_and_runs(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("train-export-run", contents)
        self.assertIn("make train-export-run", contents)
        self.assertIn("date +%s", contents)
        self.assertIn("Training elapsed:", contents)
        self.assertIn("$(MAKE) train", contents)
        self.assertIn("$(MAKE) export-model", contents)
        self.assertIn('$(MAKE) run-model CHECKPOINT="$(MODEL_OUT)"', contents)

    def test_run_model_target_wires_top_k_sampling(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("TOP_K :=", contents)
        self.assertIn('--top-k "$(TOP_K)"', contents)

    def test_run_model_target_wires_repetition_penalty_sampling(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("REPETITION_PENALTY :=", contents)
        self.assertIn("REPETITION_WINDOW :=", contents)
        self.assertIn('--repetition-penalty "$(REPETITION_PENALTY)"', contents)
        self.assertIn('--repetition-window "$(REPETITION_WINDOW)"', contents)

    def test_run_model_target_streams_by_default(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("STREAM := 1", contents)
        self.assertIn('--stream "$(STREAM)"', contents)

    def test_run_model_target_wires_chat_formatting(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("CHAT :=", contents)
        self.assertIn('--chat "$(CHAT)"', contents)

    def test_sft_train_target_writes_separate_checkpoint(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-train", contents)
        self.assertIn("SFT_DATA := data/sft/seed.jsonl", contents)
        self.assertIn("SFT_BASE_CHECKPOINT := $(CHECKPOINT)", contents)
        self.assertIn("SFT_OUT := data/sft/runs/chat-sft.pt", contents)
        self.assertIn("SFT_METRICS := data/sft/runs/metrics.jsonl", contents)
        self.assertIn("scripts/train_sft.py", contents)
        self.assertIn('--base-checkpoint "$(SFT_BASE_CHECKPOINT)"', contents)
        self.assertIn('--out "$(SFT_OUT)"', contents)
        self.assertIn('--metrics "$(SFT_METRICS)"', contents)
        self.assertIn("==> [sft-train] Training supervised chat model", contents)
        self.assertIn("==> [sft-train] Finished supervised chat training", contents)

    def test_sft_import_public_target_downloads_and_filters_public_data(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-import-public", contents)
        self.assertIn("SFT_IMPORT_SOURCES := no_robots,dolly,openassistant,wildchat,ultrachat", contents)
        self.assertIn("SFT_IMPORT_OUT := data/sft/imported/public-mixed.jsonl", contents)
        self.assertIn("SFT_IMPORT_METADATA := data/sft/imported/public-mixed.metadata.json", contents)
        self.assertIn("scripts/import_public_sft.py", contents)
        self.assertIn('--checkpoint "$(SFT_IMPORT_CHECKPOINT)"', contents)
        self.assertIn('--sources "$(SFT_IMPORT_SOURCES)"', contents)
        self.assertIn('--max-context-tokens "$(SFT_IMPORT_MAX_CONTEXT_TOKENS)"', contents)
        self.assertIn("==> [sft-import-public] Importing public SFT datasets", contents)

    def test_sft_overfit_50_target_trains_diagnostic_checkpoint(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-overfit-50", contents)
        self.assertIn("SFT_OVERFIT_DATA := data/sft/diagnostics/overfit-50.jsonl", contents)
        self.assertIn("SFT_OVERFIT_OUT := data/sft/runs/chat-sft-overfit-50.pt", contents)
        self.assertIn("SFT_OVERFIT_STEPS := 2000", contents)
        self.assertIn("SFT_OVERFIT_LR := 5e-5", contents)
        self.assertIn("SFT_OVERFIT_WEIGHT_DECAY := 0.0", contents)
        self.assertIn('SFT_DATA="$(SFT_OVERFIT_DATA)"', contents)
        self.assertIn('SFT_OUT="$(SFT_OVERFIT_OUT)"', contents)
        self.assertIn('SFT_WEIGHT_DECAY="$(SFT_OVERFIT_WEIGHT_DECAY)"', contents)
        self.assertIn("==> [sft-overfit-50] Training hard-overfit SFT diagnostic", contents)

    def test_staged_sft_targets_chain_anchor_broad_and_style_runs(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-anchor", contents)
        self.assertIn("sft-broad", contents)
        self.assertIn("sft-style", contents)
        self.assertIn("sft-staged", contents)
        self.assertIn("SFT_ANCHOR_DATA := data/sft/stages/anchor.jsonl", contents)
        self.assertIn("SFT_BROAD_DATA := data/sft/stages/broad-mixed.jsonl", contents)
        self.assertIn("SFT_STYLE_DATA := data/sft/stages/style-playful-direct.jsonl", contents)
        self.assertIn("SFT_ANCHOR_OUT := data/sft/runs/chat-anchor.pt", contents)
        self.assertIn("SFT_BROAD_BASE_CHECKPOINT := $(SFT_ANCHOR_OUT)", contents)
        self.assertIn("SFT_STYLE_BASE_CHECKPOINT := $(SFT_BROAD_OUT)", contents)
        self.assertIn("SFT_STAGED_OUT := $(SFT_STYLE_OUT)", contents)
        self.assertIn('SFT_DATA="$(SFT_ANCHOR_DATA)"', contents)
        self.assertIn('SFT_DATA="$(SFT_BROAD_DATA)"', contents)
        self.assertIn('SFT_DATA="$(SFT_STYLE_DATA)"', contents)
        self.assertIn('$(MAKE) sft-anchor', contents)
        self.assertIn('$(MAKE) sft-broad', contents)
        self.assertIn('$(MAKE) sft-style', contents)
        self.assertIn("==> [sft-staged] Finished staged supervised chat training", contents)

    def test_chat_target_starts_interactive_chat_wrapper(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("chat", contents)
        self.assertIn("CHAT_MAX_TOKENS :=", contents)
        self.assertIn("CHAT_CHECKPOINT := $(SFT_OUT)", contents)
        self.assertIn("scripts/chat.py", contents)
        self.assertIn('--checkpoint "$(CHAT_CHECKPOINT)"', contents)
        self.assertIn('--new-tokens "$(CHAT_MAX_TOKENS)"', contents)
        self.assertIn('--repetition-penalty "$(REPETITION_PENALTY)"', contents)
        self.assertIn("==> [chat] Starting interactive chat", contents)

    def test_std_train_prefills_standard_training_command(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("std-train", contents)
        self.assertIn("make std-train", contents)
        self.assertIn("$(MAKE) train-export-run", contents)
        self.assertIn("RESUME=data/checkpoints/latest.pt", contents)
        self.assertIn("STEPS=5000", contents)
        self.assertIn("BATCH=16", contents)
        self.assertIn("DEVICE=auto", contents)
        self.assertIn('PROMPT="Attention is"', contents)
        self.assertIn("NEW_TOKENS=300", contents)
        self.assertIn("TEMPERATURE=0.6", contents)

    def test_train_4090_fetches_ingests_and_starts_night_run(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("train-4090", contents)
        self.assertIn("make train-4090", contents)
        self.assertIn("TRAIN_4090_RAW_DIR := data/raw/c4-4090-night", contents)
        self.assertIn("TRAIN_4090_C4_MAX := 150000", contents)
        self.assertIn("TRAIN_4090_C4_MIN_CHARS := 1000", contents)
        self.assertIn("TRAIN_4090_STEPS := 120000", contents)
        self.assertIn("TRAIN_4090_BATCH := 16", contents)
        self.assertIn("TRAIN_4090_DEVICE := cuda", contents)
        self.assertIn("$(MAKE) c4 \\", contents)
        self.assertIn('RAW_DIR="$(TRAIN_4090_RAW_DIR)"', contents)
        self.assertIn("$(MAKE) clean-generated", contents)
        self.assertIn('$(MAKE) ingest RAW_DIR="$(TRAIN_4090_RAW_DIR)"', contents)
        self.assertIn("$(MAKE) train-export-run", contents)
        self.assertIn('RESUME=', contents)
        self.assertIn('PROMPT="$(TRAIN_4090_PROMPT)"', contents)
        self.assertIn('TOP_K="$(TRAIN_4090_TOP_K)"', contents)

    def test_train_200m_streams_tokenizes_and_starts_serious_run(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("train-200m", contents)
        self.assertIn("make train-200m", contents)
        self.assertIn("TRAIN_200M_STREAM_C4_MAX := 1000000", contents)
        self.assertIn("TRAIN_200M_STREAM_TOKENIZER_SAMPLE := 20000", contents)
        self.assertIn("TRAIN_200M_STREAM_SHARD_TOKENS := 2000000", contents)
        self.assertIn("TRAIN_200M_STREAM_VALIDATION_TOKENS := 2000000", contents)
        self.assertIn("TRAIN_200M_STEPS := 300000", contents)
        self.assertIn("TRAIN_200M_BATCH := 8", contents)
        self.assertIn("TRAIN_200M_DEVICE := cuda", contents)
        self.assertIn("TRAIN_200M_MIXED_PRECISION := auto", contents)
        self.assertIn("$(MAKE) clean-generated", contents)
        self.assertIn("$(MAKE) ingest-stream-c4", contents)
        self.assertIn('STREAM_C4_MAX="$(TRAIN_200M_STREAM_C4_MAX)"', contents)
        self.assertIn('STREAM_TOKENIZER_SAMPLE="$(TRAIN_200M_STREAM_TOKENIZER_SAMPLE)"', contents)
        self.assertIn('BPE_VOCAB_SIZE="$(TRAIN_200M_BPE_VOCAB_SIZE)"', contents)
        self.assertIn("$(MAKE) train-export-run", contents)
        self.assertIn('LR_WARMUP_STEPS="$(TRAIN_200M_LR_WARMUP_STEPS)"', contents)
        self.assertIn('MIXED_PRECISION="$(TRAIN_200M_MIXED_PRECISION)"', contents)

    def test_validation_metrics_are_wired_into_ingest_and_train(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("VALIDATION_FRACTION := 0.1", contents)
        self.assertIn("EVAL_INTERVAL := 500", contents)
        self.assertIn("VAL_BATCHES := 10", contents)
        self.assertIn("VAL_TOKENS := $(PROCESSED_DIR)/val_tokens.pt", contents)
        self.assertIn("METRICS := $(CHECKPOINT_DIR)/metrics.jsonl", contents)
        self.assertIn('validation_fraction=float("$(VALIDATION_FRACTION)")', contents)
        self.assertIn("train_model_with_metrics", contents)
        self.assertIn("append_metrics_jsonl", contents)
        self.assertIn('Path("$(METRICS)")', contents)

    def test_train_target_wires_learning_rate_schedule(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("LR_MIN :=", contents)
        self.assertIn("LR_WARMUP_STEPS :=", contents)
        self.assertIn('min_learning_rate=float("$(LR_MIN)")', contents)
        self.assertIn('warmup_steps=int("$(LR_WARMUP_STEPS)")', contents)
        self.assertIn('"min_learning_rate": float("$(LR_MIN)")', contents)
        self.assertIn('"warmup_steps": int("$(LR_WARMUP_STEPS)")', contents)

    def test_train_target_wires_mixed_precision(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("MIXED_PRECISION := auto", contents)
        self.assertIn('mixed_precision="$(MIXED_PRECISION)"', contents)
        self.assertIn('"mixed_precision": "$(MIXED_PRECISION)"', contents)
        self.assertIn('MIXED_PRECISION="$(TRAIN_4090_MIXED_PRECISION)"', contents)

    def test_ingest_defaults_to_bpe_tokenization(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("TOKENIZER := bpe", contents)
        self.assertIn("BPE_VOCAB_SIZE := 8000", contents)
        self.assertIn("BPE_MIN_FREQUENCY := 2", contents)
        self.assertIn('tokenizer_type="$(TOKENIZER)"', contents)
        self.assertIn('bpe_vocab_size=int("$(BPE_VOCAB_SIZE)")', contents)
        self.assertIn('bpe_min_frequency=int("$(BPE_MIN_FREQUENCY)")', contents)
        self.assertIn('vocab_size=int(vocab["vocab_size"])', contents)

    def test_stream_c4_target_builds_token_shards(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("ingest-stream-c4", contents)
        self.assertIn("STREAM_SHARD_TOKENS :=", contents)
        self.assertIn("STREAM_VALIDATION_TOKENS :=", contents)
        self.assertIn("build_c4_token_shards", contents)
        self.assertIn("train_shards/manifest.json", contents)
        self.assertIn("TokenShardDataset.from_manifest", contents)

    def test_train_target_wires_periodic_checkpointing(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("CHECKPOINT_INTERVAL := 1000", contents)
        self.assertIn('checkpoint_interval = int("$(CHECKPOINT_INTERVAL)")', contents)
        self.assertIn("def save_periodic_checkpoint", contents)
        self.assertIn("checkpoint_interval=checkpoint_interval", contents)
        self.assertIn("checkpoint_callback=checkpoint_callback", contents)
        self.assertIn("Periodic checkpoint:", contents)

    def test_train_target_wires_best_validation_checkpointing(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("BEST_CHECKPOINT := $(CHECKPOINT_DIR)/best.pt", contents)
        self.assertIn("def save_best_checkpoint", contents)
        self.assertIn("metric_callback=save_best_checkpoint", contents)
        self.assertIn("best_validation_loss", contents)
        self.assertIn("best_validation_step", contents)
        self.assertIn("Best checkpoint:", contents)

    def test_wiki_target_exposes_configurable_user_agent(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("WIKI_USER_AGENT :=", contents)
        self.assertIn("SuperAGI-learning-corpus-builder/0.1 (mailto:you@example.com)", contents)
        self.assertIn('user_agent="$(WIKI_USER_AGENT)"', contents)

    def test_pipeline_targets_print_phase_markers(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        expected_markers = [
            "==> [wiki] Fetching Wikipedia text",
            "==> [wiki] Finished fetching Wikipedia text",
            "==> [c4] Fetching C4 text",
            "==> [c4] Finished fetching C4 text",
            "==> [ingest] Ingesting raw text",
            "==> [ingest] Finished ingesting raw text",
            "==> [train] Training model",
            "==> [train] Finished training model",
            "==> [export-model] Exporting portable model",
            "==> [export-model] Finished exporting portable model",
            "==> [run-model] Generating sample text",
            "==> [run-model] Finished generating sample text",
            "==> [pipeline] Starting train/export/run",
            "==> [pipeline] Finished train/export/run",
            "==> [train-200m] Streaming and tokenizing C4 corpus",
            "==> [train-200m] Finished 200M training pipeline",
        ]

        for marker in expected_markers:
            self.assertIn(marker, contents)


if __name__ == "__main__":
    unittest.main()
