import unittest
from pathlib import Path


class MakefileTests(unittest.TestCase):
    def test_runpod_sft_300m_defines_production_4090_defaults(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        expected_defaults = [
            "SFT_CLOUD_RUN_ROOT := data/sft/runs/300m-v2",
            "SFT_CLOUD_CORE_STEPS := 3000",
            "SFT_CLOUD_CORE_BATCH := 2",
            "SFT_CLOUD_GRAD_ACCUM_STEPS := 8",
            "SFT_CLOUD_MIXED_PRECISION := float16",
            "SFT_CLOUD_FUSED_ADAMW := auto",
            "SFT_CLOUD_ACTIVATION_CHECKPOINTING := 1",
            "SFT_CLOUD_CORE_LR := 6e-6",
            "SFT_CLOUD_CORE_LR_MIN := 1e-6",
            "SFT_CLOUD_CORE_LR_WARMUP_STEPS := 150",
            "SFT_CLOUD_CORE_CHECKPOINT_INTERVAL := 250",
            "SFT_CLOUD_CORE_LOG_INTERVAL := 250",
            "SFT_CLOUD_CHECKPOINT_KEEP := 3",
            "SFT_CLOUD_STYLE_STEPS := 500",
            "SFT_CLOUD_STYLE_BATCH := 2",
            "SFT_CLOUD_STYLE_LR := 1.5e-6",
            "SFT_CLOUD_STYLE_LR_MIN := 5e-7",
            "SFT_CLOUD_STYLE_LR_WARMUP_STEPS := 50",
            (
                "SFT_CLOUD_PUBLIC_DATA := "
                "$(SFT_CLOUD_RUN_ROOT)/inputs/public-mixed.jsonl"
            ),
            (
                "SFT_CLOUD_PUBLIC_METADATA := "
                "$(SFT_CLOUD_RUN_ROOT)/inputs/public-mixed.metadata.json"
            ),
            (
                "SFT_CLOUD_PREFLIGHT_STATE := "
                "$(SFT_CLOUD_RUN_ROOT)/preflight-state.txt"
            ),
            (
                "SFT_CLOUD_PLAYFUL_SOURCE_WEIGHTS := "
                "curated_core=1,style_playful_direct=7,default=1"
            ),
            (
                "SFT_CLOUD_CALM_SOURCE_WEIGHTS := "
                "curated_core=1,style_calm_precise=7,default=1"
            ),
            "SFT_IMPORT_MAX_MESSAGES := 6",
            "SFT_IMPORT_MAX_AGI_CHARS := 700",
            "SFT_IMPORT_MAX_AGI_TOKENS := 192",
            "SFT_IMPORT_MAX_CROSS_EXAMPLE_NGRAM_COUNT := 3",
            (
                "SFT_CLOUD_CORE_SOURCE_WEIGHTS := "
                "curated_core=8,curated_behavior=8,no_robots=0.5,"
                "openassistant=0.75,dolly=0.75,ultrachat=0.6,"
                "wildchat=0,default=1"
            ),
            (
                "SFT_CLOUD_BEHAVIOR_IDENTITY_DATA := "
                "data/sft/curated/behavior-identity-reset.jsonl"
            ),
            (
                "SFT_CLOUD_BEHAVIOR_DIRECT_DATA := "
                "data/sft/curated/behavior-direct-current.jsonl"
            ),
            (
                "SFT_CLOUD_BEHAVIOR_DATA := "
                "$(SFT_CLOUD_EXPECTED_BEHAVIOR_DATA)"
            ),
            (
                '--sealed-input "behavior_identity_reset_jsonl='
                '$(SFT_CLOUD_BEHAVIOR_IDENTITY_DATA)"'
            ),
            (
                '--sealed-input "behavior_direct_current_jsonl='
                '$(SFT_CLOUD_BEHAVIOR_DIRECT_DATA)"'
            ),
            '--sealed-input "evaluation_prompts=$(SFT_EVAL_PROMPTS)"',
        ]
        for expected in expected_defaults:
            self.assertIn(expected, contents)

    def test_runpod_sft_300m_preflight_orders_non_training_gates(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")
        recipe = _target_recipe(contents, "runpod-sft-300m-preflight")

        expected_order = [
            "$(MAKE) setup",
            "scripts/preflight_sft_300m.py",
            "$(MAKE) sft-import-public",
            "$(MAKE) sft-audit",
        ]
        positions = [recipe.index(value) for value in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('SFT_AUDIT_MODE="mixed"', recipe)
        self.assertIn(
            'SFT_AUDIT_REPORT="$(SFT_CLOUD_AUDIT_REPORT)"',
            recipe,
        )
        self.assertIn(
            'SFT_IMPORT_NGRAM_REFERENCE_DATA="$(SFT_CLOUD_CURATED_DATA)"',
            recipe,
        )
        self.assertIn('SFT_AUDIT_CURATED_SAMPLING_MASS_MIN="0.45"', recipe)
        self.assertIn('SFT_AUDIT_CURATED_SAMPLING_MASS_MAX="0.65"', recipe)
        self.assertIn(
            'SFT_AUDIT_CURATED_SOURCE_FAMILIES="curated_core,curated_behavior,curated"',
            recipe,
        )
        self.assertIn('--public-data "$(SFT_CLOUD_PUBLIC_DATA)"', recipe)
        self.assertIn(
            '--public-metadata "$(SFT_CLOUD_PUBLIC_METADATA)"',
            recipe,
        )
        self.assertIn('--state-file "$(SFT_CLOUD_PREFLIGHT_STATE)"', recipe)
        self.assertIn(
            "Aggregate SFT data overrides are not allowed",
            recipe,
        )
        self.assertIn(
            'if [ "$$preflight_state" = "prepare" ]; then',
            recipe,
        )
        self.assertIn("--record-public", recipe)
        self.assertIn(
            "Immutable public import already sealed; skipping download",
            recipe,
        )
        self.assertLess(
            recipe.index('if [ "$$preflight_state" = "prepare" ]; then'),
            recipe.index("$(MAKE) sft-import-public"),
        )
        self.assertLess(
            recipe.index("$(MAKE) sft-import-public"),
            recipe.index("--record-public"),
        )
        self.assertEqual(recipe.count("$(MAKE) sft-import-public"), 1)
        self.assertEqual(recipe.count("$(SFT_CLOUD_CONFIG_ARGS)"), 3)
        self.assertEqual(recipe.count("$(SFT_CLOUD_SEALED_INPUT_ARGS)"), 3)
        self.assertNotIn("$(MAKE) sft-train", recipe)

    def test_runpod_sft_300m_orders_training_evaluation_and_manifest(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")
        recipe = _target_recipe(contents, "runpod-sft-300m")

        expected_order = [
            "$(MAKE) runpod-sft-300m-preflight",
            'SFT_RUN_DIR="$(SFT_CLOUD_CORE_RUN_DIR)"',
            'SFT_EVAL_CHECKPOINT="$(SFT_CLOUD_CORE_RUN_DIR)/best.pt"',
            'SFT_RUN_DIR="$(SFT_CLOUD_PLAYFUL_RUN_DIR)"',
            'SFT_EVAL_CHECKPOINT="$(SFT_CLOUD_PLAYFUL_RUN_DIR)/best.pt"',
            'SFT_RUN_DIR="$(SFT_CLOUD_CALM_RUN_DIR)"',
            'SFT_EVAL_CHECKPOINT="$(SFT_CLOUD_CALM_RUN_DIR)/best.pt"',
            "scripts/write_sft_manifest.py",
        ]
        positions = [recipe.index(value) for value in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            'SFT_BASE_CHECKPOINT="$(SFT_CLOUD_CORE_RUN_DIR)/best.pt"',
            recipe,
        )
        self.assertEqual(
            recipe.count(
                'SFT_BASE_CHECKPOINT="$(SFT_CLOUD_CORE_RUN_DIR)/best.pt"'
            ),
            2,
        )
        self.assertIn(
            'SFT_SOURCE_WEIGHTS="$(SFT_CLOUD_PLAYFUL_SOURCE_WEIGHTS)"',
            recipe,
        )
        self.assertIn(
            'SFT_SOURCE_WEIGHTS="$(SFT_CLOUD_CALM_SOURCE_WEIGHTS)"',
            recipe,
        )
        expected_evaluation_paths = [
            'SFT_EVAL_RESULTS="$(SFT_CLOUD_CORE_RUN_DIR)/evaluation.jsonl"',
            (
                'SFT_EVAL_SUMMARY="$(SFT_CLOUD_CORE_RUN_DIR)'
                '/evaluation.summary.json"'
            ),
            (
                'SFT_EVAL_RESULTS="$(SFT_CLOUD_PLAYFUL_RUN_DIR)'
                '/evaluation.jsonl"'
            ),
            (
                'SFT_EVAL_SUMMARY="$(SFT_CLOUD_PLAYFUL_RUN_DIR)'
                '/evaluation.summary.json"'
            ),
            'SFT_EVAL_RESULTS="$(SFT_CLOUD_CALM_RUN_DIR)/evaluation.jsonl"',
            (
                'SFT_EVAL_SUMMARY="$(SFT_CLOUD_CALM_RUN_DIR)'
                '/evaluation.summary.json"'
            ),
        ]
        for path in expected_evaluation_paths:
            self.assertIn(path, recipe)
        self.assertEqual(recipe.count('recovery-current.json'), 3)
        self.assertEqual(recipe.count("$(SFT_CLOUD_SEALED_INPUT_ARGS)"), 4)
        self.assertEqual(recipe.count('SFT_RESUME="$$resume"'), 3)
        self.assertEqual(recipe.count('final.pt"'), 3)
        self.assertIn(
            '--base-checkpoint "$(SFT_CLOUD_BASE_CHECKPOINT)"',
            recipe,
        )
        self.assertIn('--run-config "$(SFT_CLOUD_RUN_CONFIG)"', recipe)
        self.assertIn(
            '--core-run-dir "$(SFT_CLOUD_CORE_RUN_DIR)"',
            recipe,
        )
        self.assertIn(
            '--playful-run-dir "$(SFT_CLOUD_PLAYFUL_RUN_DIR)"',
            recipe,
        )
        self.assertIn(
            '--calm-run-dir "$(SFT_CLOUD_CALM_RUN_DIR)"',
            recipe,
        )
        self.assertEqual(recipe.count("--verify-only"), 4)
        self.assertEqual(recipe.count("$(SFT_CLOUD_CONFIG_ARGS)"), 5)
        self.assertEqual(
            recipe.count('--public-data "$(SFT_CLOUD_PUBLIC_DATA)"'),
            4,
        )
        self.assertEqual(
            recipe.count(
                '--public-metadata "$(SFT_CLOUD_PUBLIC_METADATA)"'
            ),
            4,
        )
        self.assertIn(
            '--base-sha-record "$(SFT_CLOUD_BASE_SHA_RECORD)"',
            recipe,
        )
        manifest_inputs = [
            '--public-import-data "$(SFT_CLOUD_PUBLIC_DATA)"',
            '--public-import-metadata "$(SFT_CLOUD_PUBLIC_METADATA)"',
            '--curated-core-data "$(SFT_CLOUD_CURATED_CORE_DATA)"',
            '--curated-core-metadata "data/sft/curated/core.metadata.json"',
            '--curated-core-audit "data/sft/curated/core.audit.json"',
            (
                '--behavior-identity-reset-data '
                '"$(SFT_CLOUD_BEHAVIOR_IDENTITY_DATA)"'
            ),
            (
                '--behavior-direct-current-data '
                '"$(SFT_CLOUD_BEHAVIOR_DIRECT_DATA)"'
            ),
            '--playful-style-data "$(SFT_CLOUD_PLAYFUL_STYLE_DATA)"',
            (
                '--playful-style-audit '
                '"data/sft/styles/playful-direct.audit.json"'
            ),
            '--calm-style-data "$(SFT_CLOUD_CALM_STYLE_DATA)"',
            (
                '--calm-style-audit '
                '"data/sft/styles/calm-precise.audit.json"'
            ),
            '--style-metadata "data/sft/styles/styles.metadata.json"',
            '--eval-prompts "$(SFT_EVAL_PROMPTS)"',
            '--audit-report "$(SFT_CLOUD_AUDIT_REPORT)"',
        ]
        for manifest_input in manifest_inputs:
            self.assertIn(manifest_input, recipe)

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

    def test_sft_train_target_wires_validation_split_and_batches(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("SFT_VALIDATION_FRACTION := 0.05", contents)
        self.assertIn("SFT_VALIDATION_BATCHES := 10", contents)
        self.assertIn("SFT_MAX_EXAMPLES := 0", contents)
        self.assertIn("SFT_SOURCE_WEIGHTS :=", contents)
        self.assertIn("SFT_LOG_INTERVAL := 50", contents)
        self.assertIn('--validation-fraction "$(SFT_VALIDATION_FRACTION)"', contents)
        self.assertIn('--validation-batches "$(SFT_VALIDATION_BATCHES)"', contents)
        self.assertIn('--max-examples "$(SFT_MAX_EXAMPLES)"', contents)
        self.assertIn('--source-weights "$(SFT_SOURCE_WEIGHTS)"', contents)
        self.assertIn('--log-interval "$(SFT_LOG_INTERVAL)"', contents)

    def test_sft_train_target_wires_explicit_resume(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("SFT_RESUME :=", contents)
        self.assertIn(
            "$(if $(filter 1 true yes on,$(SFT_RESUME)),--resume,)",
            contents,
        )

    def test_sft_import_public_target_downloads_and_filters_public_data(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-import-public", contents)
        self.assertIn("SFT_IMPORT_SOURCES := no_robots,dolly,openassistant,ultrachat", contents)
        self.assertIn("SFT_IMPORT_SEED := 1337", contents)
        self.assertIn("SFT_IMPORT_OUT := data/sft/imported/public-mixed.jsonl", contents)
        self.assertIn("SFT_IMPORT_METADATA := data/sft/imported/public-mixed.metadata.json", contents)
        self.assertIn("scripts/import_public_sft.py", contents)
        self.assertIn('--checkpoint "$(SFT_IMPORT_CHECKPOINT)"', contents)
        self.assertIn('--sources "$(SFT_IMPORT_SOURCES)"', contents)
        self.assertIn('--max-context-tokens "$(SFT_IMPORT_MAX_CONTEXT_TOKENS)"', contents)
        self.assertIn('--max-agi-tokens "$(SFT_IMPORT_MAX_AGI_TOKENS)"', contents)
        self.assertIn(
            '--max-cross-example-ngram-count '
            '"$(SFT_IMPORT_MAX_CROSS_EXAMPLE_NGRAM_COUNT)"',
            contents,
        )
        self.assertIn(
            '$(if $(strip $(SFT_IMPORT_NGRAM_REFERENCE_DATA)),'
            '--ngram-reference-data "$(SFT_IMPORT_NGRAM_REFERENCE_DATA)",)',
            contents,
        )
        self.assertIn('--seed "$(SFT_IMPORT_SEED)"', contents)
        self.assertIn("==> [sft-import-public] Importing public SFT datasets", contents)

    def test_sft_audit_target_wires_machine_enforced_corpus_audit(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-audit", contents)
        self.assertIn("SFT_AUDIT_DATA :=", contents)
        self.assertIn("SFT_AUDIT_CHECKPOINT :=", contents)
        self.assertIn("SFT_AUDIT_SOURCE_WEIGHTS :=", contents)
        self.assertIn("SFT_AUDIT_MODE :=", contents)
        self.assertIn("SFT_AUDIT_REPORT :=", contents)
        self.assertIn("SFT_AUDIT_CURATED_SAMPLING_MASS_MIN :=", contents)
        self.assertIn("SFT_AUDIT_CURATED_SAMPLING_MASS_MAX :=", contents)
        self.assertIn("SFT_AUDIT_CURATED_SOURCE_FAMILIES :=", contents)
        self.assertIn("scripts/audit_sft.py", contents)
        self.assertIn('--data "$(SFT_AUDIT_DATA)"', contents)
        self.assertIn('--checkpoint "$(SFT_AUDIT_CHECKPOINT)"', contents)
        self.assertIn('--source-weights "$(SFT_AUDIT_SOURCE_WEIGHTS)"', contents)
        self.assertIn('--mode "$(SFT_AUDIT_MODE)"', contents)
        self.assertIn('--report "$(SFT_AUDIT_REPORT)"', contents)
        self.assertIn(
            '--curated-sampling-mass-min '
            '"$(SFT_AUDIT_CURATED_SAMPLING_MASS_MIN)"',
            contents,
        )
        self.assertIn(
            '--curated-sampling-mass-max '
            '"$(SFT_AUDIT_CURATED_SAMPLING_MASS_MAX)"',
            contents,
        )
        self.assertIn(
            '--curated-source-families '
            '"$(SFT_AUDIT_CURATED_SOURCE_FAMILIES)"',
            contents,
        )

    def test_sft_evaluate_target_wires_fixed_behavioral_gates(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-evaluate", contents)
        self.assertIn(
            "SFT_EVAL_CHECKPOINT := data/sft/runs/300m-v2/core/best.pt",
            contents,
        )
        self.assertIn(
            "SFT_EVAL_PROMPTS := data/sft/eval_prompts.jsonl",
            contents,
        )
        self.assertIn("SFT_EVAL_RESULTS :=", contents)
        self.assertIn("SFT_EVAL_SUMMARY :=", contents)
        self.assertIn("SFT_EVAL_TEMPERATURE := 0.3", contents)
        self.assertIn("SFT_EVAL_TOP_K := 20", contents)
        self.assertIn("SFT_EVAL_REPETITION_PENALTY := 1.2", contents)
        self.assertIn("SFT_EVAL_REPETITION_WINDOW := 128", contents)
        self.assertIn("SFT_EVAL_DEVICE := auto", contents)
        self.assertIn("SFT_EVAL_SEED := 1337", contents)
        self.assertIn("SFT_EVAL_MIN_EOS_TERMINATION_RATE := 0.90", contents)
        self.assertIn("SFT_EVAL_MIN_NONEMPTY_RESPONSE_RATE := 0.95", contents)
        self.assertIn("SFT_EVAL_MAX_REPETITION_FAILURE_RATE := 0.05", contents)
        self.assertIn("SFT_EVAL_MIN_TOPIC_RESET_PASS_RATE := 0.80", contents)
        self.assertIn("scripts/evaluate_sft.py", contents)
        self.assertIn('--checkpoint "$(SFT_EVAL_CHECKPOINT)"', contents)
        self.assertIn('--prompts "$(SFT_EVAL_PROMPTS)"', contents)
        self.assertIn('--temperature "$(SFT_EVAL_TEMPERATURE)"', contents)
        self.assertIn('--top-k "$(SFT_EVAL_TOP_K)"', contents)
        self.assertIn(
            '--repetition-penalty "$(SFT_EVAL_REPETITION_PENALTY)"',
            contents,
        )
        self.assertIn(
            '--repetition-window "$(SFT_EVAL_REPETITION_WINDOW)"',
            contents,
        )
        self.assertIn('--device "$(SFT_EVAL_DEVICE)"', contents)
        self.assertIn('--seed "$(SFT_EVAL_SEED)"', contents)
        self.assertIn(
            '--min-eos-termination-rate '
            '"$(SFT_EVAL_MIN_EOS_TERMINATION_RATE)"',
            contents,
        )
        self.assertIn(
            '--min-nonempty-response-rate '
            '"$(SFT_EVAL_MIN_NONEMPTY_RESPONSE_RATE)"',
            contents,
        )
        self.assertIn(
            '--max-repetition-failure-rate '
            '"$(SFT_EVAL_MAX_REPETITION_FAILURE_RATE)"',
            contents,
        )
        self.assertIn(
            '--min-topic-reset-pass-rate '
            '"$(SFT_EVAL_MIN_TOPIC_RESET_PASS_RATE)"',
            contents,
        )
        self.assertIn(
            "$(if $(strip $(SFT_EVAL_RESULTS)),--results "
            '"$(SFT_EVAL_RESULTS)",)',
            contents,
        )
        self.assertIn(
            "$(if $(strip $(SFT_EVAL_SUMMARY)),--summary "
            '"$(SFT_EVAL_SUMMARY)",)',
            contents,
        )
        self.assertIn(
            "==> [sft-evaluate] Running fixed behavioral evaluation gates",
            contents,
        )

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
        self.assertIn('SFT_VALIDATION_FRACTION="0"', contents)
        self.assertIn("==> [sft-overfit-50] Training hard-overfit SFT diagnostic", contents)

    def test_staged_sft_targets_chain_anchor_broad_and_style_runs(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-anchor", contents)
        self.assertIn("sft-broad", contents)
        self.assertIn("sft-style-playful", contents)
        self.assertIn("sft-style-calm", contents)
        self.assertIn("sft-staged", contents)
        self.assertIn("SFT_ANCHOR_DATA := data/sft/stages/anchor.jsonl", contents)
        self.assertIn(
            "SFT_BROAD_DATA := data/sft/stages/anchor.jsonl,data/sft/curated/core.jsonl,data/sft/imported/public-mixed.jsonl",
            contents,
        )
        self.assertIn(
            "SFT_BROAD_SOURCE_WEIGHTS := anchor=4,curated_core=4,no_robots=1.5,openassistant=1.25,dolly=1,ultrachat=0.8,wildchat=0.35,default=1",
            contents,
        )
        self.assertNotIn(
            "SFT_BROAD_DATA := data/sft/stages/anchor.jsonl,data/sft/stages/broad-mixed.jsonl",
            contents,
        )
        self.assertNotIn("broad-mixed=2", contents)
        self.assertIn(
            "SFT_STYLE_PLAYFUL_DATA := data/sft/curated/core.jsonl,data/sft/styles/playful-direct.jsonl",
            contents,
        )
        self.assertIn(
            "SFT_STYLE_CALM_DATA := data/sft/curated/core.jsonl,data/sft/styles/calm-precise.jsonl",
            contents,
        )
        self.assertIn(
            "SFT_STYLE_PLAYFUL_SOURCE_WEIGHTS := curated_core=1,style_playful_direct=7,default=1",
            contents,
        )
        self.assertIn(
            "SFT_STYLE_CALM_SOURCE_WEIGHTS := curated_core=1,style_calm_precise=7,default=1",
            contents,
        )
        self.assertNotIn("data/sft/stages/style-playful-direct.jsonl", contents)
        self.assertIn("SFT_ANCHOR_OUT := data/sft/runs/chat-anchor.pt", contents)
        self.assertIn("SFT_BROAD_BASE_CHECKPOINT := $(SFT_ANCHOR_OUT)", contents)
        self.assertIn(
            "SFT_STYLE_PLAYFUL_BASE_CHECKPOINT := $(SFT_BROAD_OUT)",
            contents,
        )
        self.assertIn(
            "SFT_STYLE_CALM_BASE_CHECKPOINT := $(SFT_BROAD_OUT)",
            contents,
        )
        self.assertIn(
            "SFT_STAGED_PLAYFUL_OUT := $(SFT_STYLE_PLAYFUL_OUT)",
            contents,
        )
        self.assertIn("SFT_STAGED_CALM_OUT := $(SFT_STYLE_CALM_OUT)", contents)
        self.assertIn('SFT_DATA="$(SFT_ANCHOR_DATA)"', contents)
        self.assertIn('SFT_DATA="$(SFT_BROAD_DATA)"', contents)
        self.assertIn('SFT_SOURCE_WEIGHTS="$(SFT_BROAD_SOURCE_WEIGHTS)"', contents)
        self.assertIn('SFT_DATA="$(SFT_STYLE_PLAYFUL_DATA)"', contents)
        self.assertIn('SFT_DATA="$(SFT_STYLE_CALM_DATA)"', contents)
        self.assertIn(
            'SFT_SOURCE_WEIGHTS="$(SFT_STYLE_PLAYFUL_SOURCE_WEIGHTS)"',
            contents,
        )
        self.assertIn(
            'SFT_SOURCE_WEIGHTS="$(SFT_STYLE_CALM_SOURCE_WEIGHTS)"',
            contents,
        )
        self.assertIn('$(MAKE) sft-anchor', contents)
        self.assertIn('$(MAKE) sft-broad', contents)
        self.assertIn('$(MAKE) sft-style-playful', contents)
        self.assertIn('$(MAKE) sft-style-calm', contents)
        self.assertIn("==> [sft-staged] Finished staged supervised chat training", contents)

    def test_style_sampling_mass_is_seventy_percent_per_variant(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        makefile = repository_root / "Makefile"
        contents = makefile.read_text(encoding="utf-8")
        core_count = _jsonl_count(
            repository_root / "data" / "sft" / "curated" / "core.jsonl"
        )
        self.assertEqual(core_count, 1500)

        variants = {
            "playful": (
                "SFT_STYLE_PLAYFUL_SOURCE_WEIGHTS",
                "style_playful_direct",
                repository_root
                / "data"
                / "sft"
                / "styles"
                / "playful-direct.jsonl",
            ),
            "calm": (
                "SFT_STYLE_CALM_SOURCE_WEIGHTS",
                "style_calm_precise",
                repository_root
                / "data"
                / "sft"
                / "styles"
                / "calm-precise.jsonl",
            ),
        }
        for name, (assignment, style_family, style_path) in variants.items():
            with self.subTest(style=name):
                style_count = _jsonl_count(style_path)
                self.assertEqual(style_count, 500)
                weights = _parse_source_weights(
                    _make_assignment(contents, assignment)
                )
                style_mass = style_count * weights[style_family]
                core_mass = core_count * weights["curated_core"]
                effective_style_share = style_mass / (style_mass + core_mass)
                self.assertAlmostEqual(effective_style_share, 0.70)

    def test_staged_sft_evaluates_both_personalities_with_gates(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")
        staged_recipe = _target_recipe(contents, "sft-staged")
        evaluation_recipe = _target_recipe(contents, "sft-evaluate-styles")

        self.assertIn("make sft-evaluate-styles", contents)
        self.assertLess(
            staged_recipe.index("$(MAKE) sft-style-calm"),
            staged_recipe.index("$(MAKE) sft-evaluate-styles"),
        )
        self.assertEqual(evaluation_recipe.count("$(MAKE) sft-evaluate"), 2)
        expected_routes = [
            'SFT_EVAL_CHECKPOINT="$(SFT_STYLE_PLAYFUL_OUT)"',
            'SFT_EVAL_RESULTS="$(SFT_STYLE_PLAYFUL_EVAL_RESULTS)"',
            'SFT_EVAL_SUMMARY="$(SFT_STYLE_PLAYFUL_EVAL_SUMMARY)"',
            'SFT_EVAL_CHECKPOINT="$(SFT_STYLE_CALM_OUT)"',
            'SFT_EVAL_RESULTS="$(SFT_STYLE_CALM_EVAL_RESULTS)"',
            'SFT_EVAL_SUMMARY="$(SFT_STYLE_CALM_EVAL_SUMMARY)"',
        ]
        for route in expected_routes:
            self.assertIn(route, evaluation_recipe)
        self.assertNotEqual(
            _make_assignment(contents, "SFT_STYLE_PLAYFUL_EVAL_RESULTS"),
            _make_assignment(contents, "SFT_STYLE_CALM_EVAL_RESULTS"),
        )
        self.assertNotEqual(
            _make_assignment(contents, "SFT_STYLE_PLAYFUL_EVAL_SUMMARY"),
            _make_assignment(contents, "SFT_STYLE_CALM_EVAL_SUMMARY"),
        )
        self.assertNotIn("-$(MAKE) sft-evaluate", evaluation_recipe)
        self.assertNotIn("|| true", evaluation_recipe)

    def test_local_sft_targets_run_staged_behavior_then_public_then_style(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("sft-prepare-local", contents)
        self.assertIn("sft-anchor-local", contents)
        self.assertIn("sft-public-local", contents)
        self.assertIn("sft-style-local", contents)
        self.assertIn("sft-local-smoke", contents)
        self.assertIn("SFT_LOCAL_BASE_CHECKPOINT := ./best-300m-current.pt", contents)
        self.assertIn("SFT_LOCAL_IMPORT_OUT := data/sft/imported/local-public-mixed.jsonl", contents)
        self.assertIn("SFT_LOCAL_ANCHOR_DATA := data/sft/stages/anchor.jsonl", contents)
        self.assertIn("SFT_LOCAL_PUBLIC_DATA := data/sft/stages/anchor.jsonl,$(SFT_LOCAL_IMPORT_OUT)", contents)
        self.assertIn(
            "SFT_LOCAL_STYLE_DATA := data/sft/curated/core.jsonl,data/sft/styles/playful-direct.jsonl",
            contents,
        )
        self.assertIn("SFT_LOCAL_PUBLIC_SOURCE_WEIGHTS := anchor=4,no_robots=1.5,openassistant=1.25,dolly=1,ultrachat=0.8,wildchat=0.25,default=1", contents)
        self.assertIn("SFT_LOCAL_PUBLIC_MAX_EXAMPLES := 4000", contents)
        self.assertIn("SFT_LOCAL_SMOKE_MAX_EXAMPLES_PER_SOURCE := 100", contents)
        self.assertIn("SFT_LOCAL_SMOKE_ANCHOR_STEPS := 120", contents)
        self.assertIn("SFT_LOCAL_SMOKE_PUBLIC_STEPS := 160", contents)
        self.assertIn('SFT_IMPORT_CHECKPOINT="$(SFT_LOCAL_BASE_CHECKPOINT)"', contents)
        self.assertIn('SFT_IMPORT_OUT="$(SFT_LOCAL_IMPORT_OUT)"', contents)
        self.assertIn('SFT_BASE_CHECKPOINT="$(SFT_LOCAL_BASE_CHECKPOINT)"', contents)
        self.assertIn('SFT_DATA="$(SFT_LOCAL_ANCHOR_DATA)"', contents)
        self.assertIn("SFT_LOCAL_PUBLIC_BASE_CHECKPOINT := $(SFT_LOCAL_ANCHOR_OUT)", contents)
        self.assertIn('SFT_BASE_CHECKPOINT="$(SFT_LOCAL_PUBLIC_BASE_CHECKPOINT)"', contents)
        self.assertIn('SFT_DATA="$(SFT_LOCAL_PUBLIC_DATA)"', contents)
        self.assertIn('SFT_SOURCE_WEIGHTS="$(SFT_LOCAL_PUBLIC_SOURCE_WEIGHTS)"', contents)
        self.assertIn('SFT_MAX_EXAMPLES="$(SFT_LOCAL_PUBLIC_MAX_EXAMPLES)"', contents)
        self.assertIn('SFT_DEVICE="$(SFT_LOCAL_DEVICE)"', contents)
        self.assertIn("==> [sft-local] Finished local SFT pipeline", contents)

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

    def test_train_target_wires_parameter_dtype(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("PARAMETER_DTYPE := float32", contents)
        self.assertIn("TRAIN_200M_PARAMETER_DTYPE :=", contents)
        self.assertIn('parameter_dtype="$(PARAMETER_DTYPE)"', contents)
        self.assertIn('"parameter_dtype": "$(PARAMETER_DTYPE)"', contents)
        self.assertIn('PARAMETER_DTYPE="$(TRAIN_200M_PARAMETER_DTYPE)"', contents)

    def test_train_target_wires_configurable_dropout(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("DROPOUT := 0.0", contents)
        self.assertIn("TRAIN_200M_DROPOUT :=", contents)
        self.assertIn("dropout=float(\"$(DROPOUT)\")", contents)
        self.assertIn('"dropout": float("$(DROPOUT)")', contents)
        self.assertIn('DROPOUT="$(TRAIN_200M_DROPOUT)"', contents)

    def test_train_target_wires_gpu_speed_flags(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("FUSED_ADAMW := auto", contents)
        self.assertIn("COMPILE_MODEL := 0", contents)
        self.assertIn('fused_adamw="$(FUSED_ADAMW)"', contents)
        self.assertIn('compile_model=compile_model', contents)
        self.assertIn('"fused_adamw": "$(FUSED_ADAMW)"', contents)
        self.assertIn('"compile_model": compile_model', contents)
        self.assertIn('FUSED_ADAMW="$(TRAIN_200M_FUSED_ADAMW)"', contents)
        self.assertIn('COMPILE_MODEL="$(TRAIN_H100_COMPILE_MODEL)"', contents)

    def test_train_target_keeps_token_artifacts_as_tensors(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("token_ids = torch.load(token_path)", contents)
        self.assertIn("validation_token_ids = torch.load(val_token_path)", contents)
        self.assertNotIn("torch.load(token_path).tolist()", contents)
        self.assertNotIn("torch.load(val_token_path).tolist()", contents)

    def test_train_target_wires_gradient_accumulation(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("GRAD_ACCUM_STEPS := 1", contents)
        self.assertIn("TRAIN_200M_GRAD_ACCUM_STEPS :=", contents)
        self.assertIn('grad_accum_steps=int("$(GRAD_ACCUM_STEPS)")', contents)
        self.assertIn('"grad_accum_steps": int("$(GRAD_ACCUM_STEPS)")', contents)
        self.assertIn('GRAD_ACCUM_STEPS="$(TRAIN_200M_GRAD_ACCUM_STEPS)"', contents)

    def test_train_target_wires_activation_checkpointing(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("ACTIVATION_CHECKPOINTING := 0", contents)
        self.assertIn("TRAIN_200M_ACTIVATION_CHECKPOINTING :=", contents)
        self.assertIn('activation_checkpointing=activation_checkpointing', contents)
        self.assertIn('"activation_checkpointing": activation_checkpointing', contents)
        self.assertIn(
            'ACTIVATION_CHECKPOINTING="$(TRAIN_200M_ACTIVATION_CHECKPOINTING)"',
            contents,
        )

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

    def test_train_target_wires_dynamic_shard_refresh(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("SHARD_REFRESH_INTERVAL := 0", contents)
        self.assertIn("TRAIN_200M_SHARD_REFRESH_INTERVAL :=", contents)
        self.assertIn(
            'shard_refresh_interval=int("$(SHARD_REFRESH_INTERVAL)")',
            contents,
        )
        self.assertIn('"shard_refresh_interval": int("$(SHARD_REFRESH_INTERVAL)")', contents)
        self.assertIn(
            'SHARD_REFRESH_INTERVAL="$(TRAIN_200M_SHARD_REFRESH_INTERVAL)"',
            contents,
        )

    def test_train_h100_target_runs_zero_setup_dynamic_pipeline(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("train-h100", contents)
        self.assertIn("TRAIN_H100_SOURCES :=", contents)
        self.assertIn("TRAIN_H100_CORPUS_TARGET_TOKENS := 20000000000", contents)
        self.assertIn("TRAIN_H100_START_TOKENS :=", contents)
        self.assertIn("TRAIN_H100_STREAM_SHARD_TOKENS :=", contents)
        self.assertIn("TRAIN_H100_TOTAL_TRAINING_TOKENS := 20000000000", contents)
        self.assertIn("TRAIN_H100_BATCH := 1", contents)
        self.assertIn("TRAIN_H100_GRAD_ACCUM_STEPS := 128", contents)
        self.assertIn("TRAIN_H100_MIXED_PRECISION := bfloat16", contents)
        self.assertIn("TRAIN_H100_SHARD_REFRESH_INTERVAL :=", contents)
        self.assertIn("$(MAKE) setup", contents)
        self.assertIn("$(MAKE) ingest-stream-sources", contents)
        self.assertIn('SOURCES="$(TRAIN_H100_SOURCES)"', contents)
        self.assertIn('STREAM_TARGET_TOKENS="$(TRAIN_H100_CORPUS_TARGET_TOKENS)"', contents)
        self.assertIn("ingest_pid=$$!", contents)
        self.assertIn("Waiting for", contents)
        self.assertIn("TRAIN_H100_TOTAL_TRAINING_TOKENS", contents)
        self.assertIn('SHARD_REFRESH_INTERVAL="$(TRAIN_H100_SHARD_REFRESH_INTERVAL)"', contents)

    def test_train_300m_target_runs_weighted_mixed_source_pipeline(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("train-300m", contents)
        self.assertIn("runpod-train-300m", contents)
        self.assertIn("TRAIN_300M_SOURCES :=", contents)
        self.assertIn("TRAIN_300M_SOURCE_WEIGHTS :=", contents)
        self.assertIn("TRAIN_300M_BPE_VOCAB_SIZE := 16000", contents)
        self.assertIn("TRAIN_300M_CORPUS_TARGET_TOKENS := 6000000000", contents)
        self.assertIn("TRAIN_300M_START_TOKENS := 250000000", contents)
        self.assertIn("TRAIN_300M_TOTAL_TRAINING_TOKENS := 6000000000", contents)
        self.assertIn("$(MAKE) ingest-stream-sources", contents)
        self.assertIn('SOURCES="$(TRAIN_300M_SOURCES)"', contents)
        self.assertIn('SOURCE_WEIGHTS="$(TRAIN_300M_SOURCE_WEIGHTS)"', contents)
        self.assertIn('STREAM_TARGET_TOKENS="$(TRAIN_300M_CORPUS_TARGET_TOKENS)"', contents)
        self.assertIn('SHARD_REFRESH_INTERVAL="$(TRAIN_300M_SHARD_REFRESH_INTERVAL)"', contents)

    def test_ingest_stream_sources_target_wires_multiple_corpus_sources(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("ingest-stream-sources", contents)
        self.assertIn("SOURCES := fineweb,wikipedia,dolma,openwebmath,arxiv,pmc,stackexchange,gutenberg", contents)
        self.assertIn("STREAM_MAX_DOCUMENTS_PER_SOURCE :=", contents)
        self.assertIn("build_multi_source_token_shards", contents)
        self.assertIn('sources="$(SOURCES)"', contents)
        self.assertIn('source_weights="$(SOURCE_WEIGHTS)"', contents)
        self.assertIn('max_documents_per_source=int("$(STREAM_MAX_DOCUMENTS_PER_SOURCE)")', contents)
        self.assertIn("Source documents:", contents)
        self.assertIn("Source tokens:", contents)
        self.assertIn('MIXED_PRECISION="$(TRAIN_H100_MIXED_PRECISION)"', contents)

    def test_train_target_wires_periodic_checkpointing(self) -> None:
        makefile = Path(__file__).resolve().parents[1] / "Makefile"
        contents = makefile.read_text(encoding="utf-8")

        self.assertIn("CHECKPOINT_INTERVAL := 1000", contents)
        self.assertIn("CHECKPOINT_KEEP := 5", contents)
        self.assertIn('checkpoint_interval = int("$(CHECKPOINT_INTERVAL)")', contents)
        self.assertIn('checkpoint_keep = int("$(CHECKPOINT_KEEP)")', contents)
        self.assertIn("retain_checkpoint_snapshot", contents)
        self.assertIn('"checkpoint_keep": checkpoint_keep', contents)
        self.assertIn("def save_periodic_checkpoint", contents)
        self.assertIn("checkpoint_interval=checkpoint_interval", contents)
        self.assertIn("checkpoint_callback=checkpoint_callback", contents)
        self.assertIn("Periodic checkpoint:", contents)
        self.assertIn("Retained checkpoint:", contents)
        self.assertIn('CHECKPOINT_KEEP="$(TRAIN_200M_CHECKPOINT_KEEP)"', contents)

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


def _target_recipe(contents: str, target: str) -> str:
    marker = f"\n{target}:"
    start = contents.index(marker) + 1
    remainder = contents[start:]
    return remainder.split("\n\n", 1)[0]


def _make_assignment(contents: str, name: str) -> str:
    prefix = f"{name} :="
    for line in contents.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise AssertionError(f"Missing Make assignment: {name}")


def _parse_source_weights(value: str) -> dict[str, float]:
    return {
        name.strip(): float(weight)
        for item in value.split(",")
        for name, weight in (item.split("=", 1),)
    }


def _jsonl_count(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


if __name__ == "__main__":
    unittest.main()
