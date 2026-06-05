import unittest

import torch

from superagi.model.transformer import TransformerConfig, TransformerLM


class TransformerLMTests(unittest.TestCase):
    def test_forward_returns_logits_and_loss(self) -> None:
        config = TransformerConfig(
            vocab_size=11,
            context_length=8,
            dim_embedding=16,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
        )
        model = TransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        target_ids = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

        logits, loss = model(input_ids, target_ids)

        self.assertEqual(logits.shape, (2, 4, 11))
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))

    def test_parameters_are_registered_recursively(self) -> None:
        config = TransformerConfig(
            vocab_size=7,
            context_length=5,
            dim_embedding=12,
            n_layers=1,
            n_heads=3,
            dropout=0.0,
        )
        model = TransformerLM(config)
        parameter_names = set(dict(model.named_parameters()))

        self.assertIn("embeddings.token_embedding.weight", parameter_names)
        self.assertIn("blocks.0.attention.q_proj.weight", parameter_names)
        self.assertIn("blocks.0.feed_forward.net.0.weight", parameter_names)
        self.assertGreater(sum(param.numel() for param in model.parameters()), 0)

    def test_generate_appends_tokens(self) -> None:
        torch.manual_seed(1)
        config = TransformerConfig(
            vocab_size=5,
            context_length=4,
            dim_embedding=8,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        model = TransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3]])

        generated = model.generate(input_ids, max_new_tokens=3)

        self.assertEqual(generated.shape, (1, 6))
        self.assertTrue(torch.all((generated >= 0) & (generated < config.vocab_size)))


if __name__ == "__main__":
    unittest.main()
