import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile

import torch

from superagi.model.transformer import TransformerConfig, TransformerLM
from superagi.training.train import (
    NextTokenDataset,
    TrainConfig,
    append_metrics_jsonl,
    sample_next_token_batch,
    train_model,
    train_model_with_metrics,
    train_step,
)


class TrainingTests(unittest.TestCase):
    def test_next_token_dataset_returns_shifted_windows(self) -> None:
        dataset = NextTokenDataset(token_ids=[0, 1, 2, 3, 4], context_length=3)

        x0, y0 = dataset[0]
        x1, y1 = dataset[1]

        self.assertEqual(x0.tolist(), [0, 1, 2])
        self.assertEqual(y0.tolist(), [1, 2, 3])
        self.assertEqual(x1.tolist(), [1, 2, 3])
        self.assertEqual(y1.tolist(), [2, 3, 4])
        self.assertEqual(len(dataset), 2)

    def test_sample_next_token_batch_returns_shifted_windows_on_device(self) -> None:
        tokens = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        generator = torch.Generator(device=tokens.device).manual_seed(0)

        input_ids, target_ids = sample_next_token_batch(
            tokens=tokens,
            context_length=3,
            batch_size=4,
            generator=generator,
        )

        self.assertEqual(input_ids.shape, (4, 3))
        self.assertEqual(target_ids.shape, (4, 3))
        self.assertEqual(input_ids.device, tokens.device)
        self.assertEqual(target_ids.device, tokens.device)
        self.assertTrue(torch.equal(target_ids[:, :-1], input_ids[:, 1:]))

    def test_train_step_updates_model_parameters(self) -> None:
        torch.manual_seed(0)
        config = TransformerConfig(
            vocab_size=6,
            context_length=4,
            dim_embedding=8,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        model = TransformerLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        target_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        before = model.embeddings.token_embedding.weight.detach().clone()

        loss = train_step(model, optimizer, input_ids, target_ids)

        after = model.embeddings.token_embedding.weight.detach()
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.equal(before, after))

    def test_train_model_runs_requested_number_of_steps(self) -> None:
        torch.manual_seed(0)
        config = TransformerConfig(
            vocab_size=6,
            context_length=4,
            dim_embedding=8,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        model = TransformerLM(config)

        losses = train_model(
            model,
            token_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2],
            config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=3),
        )

        self.assertEqual(len(losses), 3)
        self.assertTrue(all(loss > 0.0 for loss in losses))

    def test_train_model_with_metrics_records_validation_loss(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(
            TransformerConfig(
                vocab_size=6,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )

        with redirect_stdout(StringIO()):
            losses, metrics = train_model_with_metrics(
                model,
                token_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2],
                config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=3),
                validation_token_ids=[0, 1, 2, 3, 4, 5, 0],
                eval_interval=2,
                validation_batches=1,
            )

        self.assertEqual(len(losses), 3)
        self.assertEqual([metric.step for metric in metrics], [2, 3])
        self.assertTrue(all(metric.train_loss > 0.0 for metric in metrics))
        self.assertTrue(all(metric.validation_loss is not None for metric in metrics))
        self.assertTrue(all(metric.validation_loss > 0.0 for metric in metrics if metric.validation_loss is not None))

    def test_train_model_with_metrics_prints_validation_loss(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(
            TransformerConfig(
                vocab_size=6,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            train_model_with_metrics(
                model,
                token_ids=[0, 1, 2, 3, 4, 5, 0],
                config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=1),
                validation_token_ids=[0, 1, 2, 3, 4, 5, 0],
                eval_interval=1,
                validation_batches=1,
            )

        printed = output.getvalue()
        self.assertIn("step=1", printed)
        self.assertIn("train_loss=", printed)
        self.assertIn("validation_loss=", printed)

    def test_train_model_with_metrics_calls_checkpoint_callback_periodically(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(
            TransformerConfig(
                vocab_size=6,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )
        callback_steps = []
        callback_loss_counts = []

        def capture_checkpoint(step: int, losses: list[float], metrics: list[object]) -> None:
            callback_steps.append(step)
            callback_loss_counts.append(len(losses))

        with redirect_stdout(StringIO()):
            train_model_with_metrics(
                model,
                token_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2],
                config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=5),
                eval_interval=2,
                validation_batches=1,
                start_step=10,
                checkpoint_interval=2,
                checkpoint_callback=capture_checkpoint,
            )

        self.assertEqual(callback_steps, [12, 14, 15])
        self.assertEqual(callback_loss_counts, [2, 4, 5])

    def test_train_model_with_metrics_calls_metric_callback_on_evaluation(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(
            TransformerConfig(
                vocab_size=6,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )
        callback_steps = []
        callback_loss_counts = []

        def capture_metric(metric: object, losses: list[float], metrics: list[object]) -> None:
            callback_steps.append(metric.step)
            callback_loss_counts.append(len(losses))

        with redirect_stdout(StringIO()):
            train_model_with_metrics(
                model,
                token_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2],
                config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=5),
                validation_token_ids=[0, 1, 2, 3, 4, 5, 0],
                eval_interval=2,
                validation_batches=1,
                start_step=10,
                metric_callback=capture_metric,
            )

        self.assertEqual(callback_steps, [12, 14, 15])
        self.assertEqual(callback_loss_counts, [2, 4, 5])

    def test_append_metrics_jsonl_writes_one_record_per_metric(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(
            TransformerConfig(
                vocab_size=6,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
            )
        )
        with redirect_stdout(StringIO()):
            _, metrics = train_model_with_metrics(
                model,
                token_ids=[0, 1, 2, 3, 4, 5, 0],
                config=TrainConfig(batch_size=2, learning_rate=1e-2, max_steps=1),
                validation_token_ids=[0, 1, 2, 3, 4, 5, 0],
                eval_interval=1,
                validation_batches=1,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "metrics.jsonl"
            append_metrics_jsonl(metrics_path, metrics)
            lines = metrics_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn('"step": 1', lines[0])
        self.assertIn('"train_loss":', lines[0])
        self.assertIn('"validation_loss":', lines[0])


if __name__ == "__main__":
    unittest.main()
