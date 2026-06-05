import unittest

import torch

from superagi.model.transformer import TransformerConfig, TransformerLM
from superagi.training.train import NextTokenDataset, TrainConfig, train_model, train_step


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


if __name__ == "__main__":
    unittest.main()
