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

    def test_run_model_target_streams_by_default(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("STREAM := 1", contents)
        self.assertIn('--stream "$(STREAM)"', contents)

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
        ]

        for marker in expected_markers:
            self.assertIn(marker, contents)


if __name__ == "__main__":
    unittest.main()
