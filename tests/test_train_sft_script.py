from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

import torch

from superagi.chat.sft import IGNORE_INDEX
from superagi.ingestion.tokenizer import BpeTokenizer
from superagi.model.checkpoint import load_checkpoint, save_checkpoint
from superagi.model.transformer import TransformerConfig, TransformerLM


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_sft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_sft
SPEC.loader.exec_module(train_sft)


class TrainSftScriptTests(unittest.TestCase):
    def test_rejects_base_checkpoint_collision_with_production_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            artifact_paths = [
                run_dir / "latest.pt",
                run_dir / "best.pt",
                run_dir / "final.pt",
                run_dir / "metrics.jsonl",
                run_dir / "snapshots" / "base.pt",
            ]

            for base_path in artifact_paths:
                with self.subTest(base_path=base_path):
                    with self.assertRaisesRegex(SystemExit, "base checkpoint"):
                        train_sft._validate_artifact_paths(
                            base_path=base_path,
                            latest_path=run_dir / "latest.pt",
                            best_path=run_dir / "best.pt",
                            final_path=run_dir / "final.pt",
                            metrics_path=run_dir / "metrics.jsonl",
                            snapshots_path=run_dir / "snapshots",
                            final_alias_path=None,
                        )

    def test_rejects_base_checkpoint_collision_with_legacy_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.pt"
            with self.assertRaisesRegex(SystemExit, "base checkpoint"):
                train_sft._validate_artifact_paths(
                    base_path=base_path,
                    latest_path=Path(tmp_dir) / "latest.pt",
                    best_path=Path(tmp_dir) / "best.pt",
                    final_path=Path(tmp_dir) / "final.pt",
                    metrics_path=base_path,
                    snapshots_path=Path(tmp_dir) / "snapshots",
                    final_alias_path=Path(tmp_dir) / "legacy.pt",
                )

    def test_production_run_requires_held_out_validation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "held-out validation"):
            train_sft._require_production_validation(
                production_mode=True,
                validation_fraction=0.0,
                validation_examples=(),
            )

        with self.assertRaisesRegex(SystemExit, "held-out validation"):
            train_sft._require_production_validation(
                production_mode=True,
                validation_fraction=0.1,
                validation_examples=(),
            )

        train_sft._require_production_validation(
            production_mode=False,
            validation_fraction=0.0,
            validation_examples=(),
        )

    def test_seeds_cpu_and_cuda_torch_rngs(self) -> None:
        with (
            patch.object(torch, "manual_seed") as manual_seed,
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "manual_seed_all") as manual_seed_all,
        ):
            train_sft._seed_global_torch_rng(1234)

        manual_seed.assert_called_once_with(1234)
        manual_seed_all.assert_called_once_with(1234)

    def test_fused_adamw_auto_falls_back_but_on_is_strict(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        fallback_optimizer = object()

        with patch.object(
            torch.optim,
            "AdamW",
            side_effect=[TypeError("fused unsupported"), fallback_optimizer],
        ) as adamw:
            optimizer = train_sft._build_adamw_optimizer(
                [parameter],
                learning_rate=1e-5,
                weight_decay=0.01,
                fused_mode="auto",
                device=torch.device("cuda"),
            )

        self.assertIs(optimizer, fallback_optimizer)
        self.assertEqual(
            adamw.call_args_list,
            [
                call([parameter], lr=1e-5, weight_decay=0.01, fused=True),
                call([parameter], lr=1e-5, weight_decay=0.01),
            ],
        )

        with (
            patch.object(
                torch.optim,
                "AdamW",
                side_effect=TypeError("fused unsupported"),
            ),
            self.assertRaisesRegex(TypeError, "fused unsupported"),
        ):
            train_sft._build_adamw_optimizer(
                [parameter],
                learning_rate=1e-5,
                weight_decay=0.01,
                fused_mode="on",
                device=torch.device("cuda"),
            )

        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            train_sft._build_adamw_optimizer(
                [parameter],
                learning_rate=1e-5,
                weight_decay=0.01,
                fused_mode="on",
                device=torch.device("cpu"),
            )

    def test_sampled_microbatch_counts_use_actual_supervised_tokens(self) -> None:
        first = (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[IGNORE_INDEX, 3]], dtype=torch.long),
        )
        second = (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            torch.tensor([[IGNORE_INDEX, 3, 4, 5]], dtype=torch.long),
        )
        with patch.object(
            train_sft,
            "sample_sft_batch",
            side_effect=[first, second],
        ):
            batches, supervised_tokens, examples = train_sft._sample_sft_microbatches(
                (),
                batch_size=1,
                grad_accum_steps=2,
                pad_token_id=0,
                device=torch.device("cpu"),
                generator=torch.Generator(),
                source_weights={},
            )

        self.assertEqual(batches, [first, second])
        self.assertEqual(supervised_tokens, 4)
        self.assertEqual(examples, 2)

    def test_run_dir_keeps_best_checkpoint_and_prunes_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path = root / "base.pt"
            data_path = root / "examples.jsonl"
            run_dir = root / "run"
            tokenizer = BpeTokenizer.from_text(
                "<bos><user> What is AI?\n<agi> AI learns patterns from data.<eos>\n",
                vocab_size=300,
                min_frequency=1,
            )
            config = TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=128,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
            )
            save_checkpoint(
                base_path,
                model=TransformerLM(config),
                vocab=tokenizer.to_payload(),
            )
            base_hash = _sha256(base_path)
            rows = [
                {
                    "source": "curated_core:everyday",
                    "messages": [
                        {"role": "user", "content": "What is AI?"},
                        {"role": "agi", "content": "AI learns patterns from data."},
                    ],
                },
                {
                    "source": "curated_core:repair",
                    "messages": [
                        {"role": "user", "content": "What is math?"},
                        {"role": "agi", "content": "Math studies quantities and patterns."},
                    ],
                },
            ]
            data_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            argv = [
                "train_sft.py",
                "--base-checkpoint", str(base_path),
                "--data", str(data_path),
                "--run-dir", str(run_dir),
                "--steps", "3",
                "--batch", "1",
                "--lr", "1e-5",
                "--lr-min", "1e-6",
                "--checkpoint-interval", "1",
                "--checkpoint-keep", "2",
                "--validation-fraction", "0.5",
                "--validation-batches", "1",
                "--mixed-precision", "none",
                "--device", "cpu",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(train_sft, "evaluate_sft_loss", side_effect=[2.0, 1.4, 1.7]),
            ):
                self.assertEqual(train_sft.main(), 0)

            latest = load_checkpoint(run_dir / "latest.pt")
            best = load_checkpoint(run_dir / "best.pt")
            self.assertEqual(latest.metadata["sft_steps"], 3)
            self.assertEqual(best.metadata["best_validation_step"], 2)
            self.assertEqual(best.metadata["best_validation_loss"], 1.4)
            self.assertEqual((run_dir / "final.pt").read_bytes(), (run_dir / "best.pt").read_bytes())
            self.assertEqual(_sha256(base_path), base_hash)
            self.assertEqual(
                [path.name for path in sorted((run_dir / "snapshots").glob("*.pt"))],
                ["checkpoint-step-000000002.pt", "checkpoint-step-000000003.pt"],
            )
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([metric["step"] for metric in metrics], [1, 2, 3])

    def test_resume_restores_training_state_and_matches_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path, data_path = _write_tiny_training_fixture(root)
            uninterrupted_dir = root / "uninterrupted"
            resumed_dir = root / "resumed"

            uninterrupted_argv = _training_argv(
                base_path=base_path,
                data_path=data_path,
                run_dir=uninterrupted_dir,
                steps=4,
            )
            with (
                patch.object(sys, "argv", uninterrupted_argv),
                patch.object(
                    train_sft,
                    "evaluate_sft_loss",
                    side_effect=[2.0, 1.8, 1.6, 1.7],
                ),
            ):
                self.assertEqual(train_sft.main(), 0)

            original_train_step = train_sft.train_accumulated_step
            completed_steps = 0

            def interrupt_after_two_steps(*args: object, **kwargs: object) -> float:
                nonlocal completed_steps
                if completed_steps == 2:
                    raise RuntimeError("simulated interruption")
                completed_steps += 1
                return original_train_step(*args, **kwargs)

            interrupted_argv = _training_argv(
                base_path=base_path,
                data_path=data_path,
                run_dir=resumed_dir,
                steps=4,
            )
            with (
                patch.object(sys, "argv", interrupted_argv),
                patch.object(
                    train_sft,
                    "evaluate_sft_loss",
                    side_effect=[2.0, 1.8],
                ),
                patch.object(
                    train_sft,
                    "train_accumulated_step",
                    side_effect=interrupt_after_two_steps,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                train_sft.main()

            metrics_before_invalid_resume = (
                resumed_dir / "metrics.jsonl"
            ).read_bytes()
            latest_before_corruption = (resumed_dir / "latest.pt").read_bytes()
            (resumed_dir / "latest.pt").write_bytes(
                latest_before_corruption + b"corrupt"
            )
            with patch.object(sys, "argv", interrupted_argv + ["--resume"]):
                with self.assertRaisesRegex(SystemExit, "corrupt reuse"):
                    train_sft.main()
            self.assertEqual(
                (resumed_dir / "metrics.jsonl").read_bytes(),
                metrics_before_invalid_resume,
            )
            (resumed_dir / "latest.pt").write_bytes(latest_before_corruption)

            changed_schedule_argv = list(interrupted_argv)
            changed_schedule_argv[changed_schedule_argv.index("--steps") + 1] = "5"
            with patch.object(
                sys,
                "argv",
                changed_schedule_argv + ["--resume"],
            ):
                with self.assertRaisesRegex(SystemExit, "schedule"):
                    train_sft.main()

            resume_argv = interrupted_argv + ["--resume"]
            trainer_state_path = resumed_dir / "trainer-state.pt"
            trainer_state_bytes = trainer_state_path.read_bytes()
            committed_metrics_bytes = (
                resumed_dir / "metrics.jsonl"
            ).read_bytes()
            stale_metrics_bytes = (
                committed_metrics_bytes + b'{"step": 999, "uncommitted": true}\n'
            )
            (resumed_dir / "metrics.jsonl").write_bytes(stale_metrics_bytes)
            corrupt_state = torch.load(trainer_state_path, map_location="cpu")
            corrupt_state["optimizer_state"] = {
                "state": {},
                "param_groups": [],
            }
            torch.save(corrupt_state, trainer_state_path)
            with patch.object(sys, "argv", resume_argv):
                with self.assertRaisesRegex(ValueError, "parameter group"):
                    train_sft.main()
            self.assertEqual(
                (resumed_dir / "metrics.jsonl").read_bytes(),
                stale_metrics_bytes,
            )
            trainer_state_path.write_bytes(trainer_state_bytes)
            (resumed_dir / "metrics.jsonl").write_bytes(
                committed_metrics_bytes
            )

            with (
                patch.object(sys, "argv", resume_argv),
                patch.object(
                    train_sft,
                    "evaluate_sft_loss",
                    side_effect=[1.6, 1.7],
                ),
            ):
                self.assertEqual(train_sft.main(), 0)

            uninterrupted = load_checkpoint(uninterrupted_dir / "latest.pt")
            resumed = load_checkpoint(resumed_dir / "latest.pt")
            self.assertEqual(resumed.metadata["sft_steps"], 4)
            self.assertEqual(
                [row["step"] for row in resumed.metrics],
                [1, 2, 3, 4],
            )
            for name, tensor in uninterrupted.model.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, resumed.model.state_dict()[name]),
                    msg=name,
                )
            self.assertEqual(
                (resumed_dir / "final.pt").read_bytes(),
                (resumed_dir / "best.pt").read_bytes(),
            )
            self.assertTrue((resumed_dir / "trainer-state.pt").is_file())
            self.assertEqual(
                load_checkpoint(
                    uninterrupted_dir / "best.pt"
                ).metadata["best_validation_step"],
                load_checkpoint(
                    resumed_dir / "best.pt"
                ).metadata["best_validation_step"],
            )

    def test_rejects_non_empty_fresh_run_and_corrupt_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path, data_path = _write_tiny_training_fixture(root)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
            argv = _training_argv(
                base_path=base_path,
                data_path=data_path,
                run_dir=run_dir,
                steps=1,
            )

            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "non-empty run directory"):
                    train_sft.main()

            (run_dir / "unrelated.txt").unlink()
            (run_dir / "latest.pt").write_bytes(b"corrupt")
            with patch.object(sys, "argv", argv + ["--resume"]):
                with self.assertRaisesRegex(SystemExit, "trainer state"):
                    train_sft.main()

    def test_log_intervals_report_actual_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path, data_path = _write_tiny_training_fixture(root)
            argv = [
                "train_sft.py",
                "--base-checkpoint", str(base_path),
                "--data", str(data_path),
                "--out", str(root / "legacy.pt"),
                "--metrics", str(root / "legacy-metrics.jsonl"),
                "--steps", "2",
                "--batch", "1",
                "--grad-accum-steps", "1",
                "--checkpoint-interval", "2",
                "--log-interval", "1",
                "--validation-fraction", "0",
                "--mixed-precision", "none",
                "--device", "cpu",
            ]
            output = io.StringIO()

            with patch.object(sys, "argv", argv), redirect_stdout(output):
                self.assertEqual(train_sft.main(), 0)

            first_log = next(
                line
                for line in output.getvalue().splitlines()
                if line.startswith("sft_step=1 ")
            )
            self.assertIn("supervised_tokens_per_second=", first_log)
            self.assertIn("examples_per_second=", first_log)
            metrics = [
                json.loads(line)
                for line in (root / "legacy-metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([metric["step"] for metric in metrics], [1, 2])
            self.assertGreater(metrics[0]["supervised_tokens_seen"], 0)
            self.assertGreater(metrics[0]["examples_seen"], 0)
            self.assertGreater(metrics[0]["supervised_tokens_per_second"], 0)
            self.assertGreater(metrics[0]["examples_per_second"], 0)

    def test_rejects_ambiguous_run_dir_and_legacy_output_arguments(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train_sft.py",
                "--base-checkpoint", "base.pt",
                "--data", "data.jsonl",
                "--run-dir", "run",
                "--out", "legacy.pt",
                "--metrics", "legacy.jsonl",
            ],
        ):
            with self.assertRaisesRegex(SystemExit, "exactly one"):
                train_sft.main()


def _write_tiny_training_fixture(root: Path) -> tuple[Path, Path]:
    base_path = root / "base.pt"
    data_path = root / "examples.jsonl"
    tokenizer = BpeTokenizer.from_text(
        "<bos><user> What is AI?\n<agi> AI learns patterns from data.<eos>\n"
        "<bos><user> What is math?\n<agi> Math studies quantities and patterns.<eos>\n",
        vocab_size=300,
        min_frequency=1,
    )
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        dim_embedding=8,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
    )
    save_checkpoint(
        base_path,
        model=TransformerLM(config),
        vocab=tokenizer.to_payload(),
    )
    rows = [
        {
            "source": "curated_core:everyday",
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "agi", "content": "AI learns patterns from data."},
            ],
        },
        {
            "source": "curated_core:science_math",
            "messages": [
                {"role": "user", "content": "What is math?"},
                {"role": "agi", "content": "Math studies quantities and patterns."},
            ],
        },
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return base_path, data_path


def _training_argv(
    *,
    base_path: Path,
    data_path: Path,
    run_dir: Path,
    steps: int,
) -> list[str]:
    return [
        "train_sft.py",
        "--base-checkpoint", str(base_path),
        "--data", str(data_path),
        "--run-dir", str(run_dir),
        "--steps", str(steps),
        "--batch", "1",
        "--grad-accum-steps", "2",
        "--lr", "1e-5",
        "--lr-min", "1e-6",
        "--lr-warmup-steps", "1",
        "--checkpoint-interval", "1",
        "--checkpoint-keep", "2",
        "--validation-fraction", "0.5",
        "--validation-batches", "1",
        "--mixed-precision", "none",
        "--device", "cpu",
        "--seed", "1234",
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
