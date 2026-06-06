import unittest

import torch

from superagi.model.inference.attend import CausalSelfAttention
from superagi.model.inference.percept import FeedForward
from superagi.model.transformer import TransformerConfig, TransformerLM


class TransformerLMTests(unittest.TestCase):
    def test_attention_and_perception_modules_live_directly_under_inference(self) -> None:
        self.assertEqual(CausalSelfAttention.__name__, "CausalSelfAttention")
        self.assertEqual(FeedForward.__name__, "FeedForward")

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

    def test_output_projection_reuses_token_embedding_weights(self) -> None:
        config = TransformerConfig(
            vocab_size=7,
            context_length=5,
            dim_embedding=12,
            n_layers=1,
            n_heads=3,
            dropout=0.0,
        )
        model = TransformerLM(config)
        input_ids = torch.tensor([[0, 1, 2, 3]])
        target_ids = torch.tensor([[1, 2, 3, 4]])

        _, loss = model(input_ids, target_ids)
        if loss is None:
            self.fail("expected training loss")
        loss.backward()

        self.assertIs(
            model.output_projection.weight,
            model.embeddings.token_embedding.weight,
        )
        self.assertIsNone(model.output_projection.bias)
        self.assertIsNotNone(model.embeddings.token_embedding.weight.grad)

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

    def test_generate_with_top_k_only_samples_from_top_logits(self) -> None:
        class FixedLogitModel(TransformerLM):
            def __init__(self) -> None:
                super().__init__(
                    TransformerConfig(
                        vocab_size=5,
                        context_length=4,
                        dim_embedding=8,
                        n_layers=1,
                        n_heads=2,
                        dropout=0.0,
                    )
                )

            def forward(self, input_ids, target_ids=None):
                batch_size, sequence_length = input_ids.shape
                logits = torch.zeros(batch_size, sequence_length, self.config.vocab_size)
                logits[..., 0] = 10.0
                logits[..., 1] = 9.0
                logits[..., 2] = 8.0
                logits[..., 3] = 7.0
                logits[..., 4] = 6.0
                return logits, None

        torch.manual_seed(0)
        model = FixedLogitModel()
        input_ids = torch.tensor([[1, 2, 3]])

        generated = model.generate(input_ids, max_new_tokens=20, top_k=2)
        new_tokens = generated[0, 3:]

        self.assertTrue(set(new_tokens.tolist()).issubset({0, 1}))

    def test_generate_penalizes_repeated_tokens_before_sampling(self) -> None:
        class RepeatFavoriteModel(TransformerLM):
            def __init__(self) -> None:
                super().__init__(
                    TransformerConfig(
                        vocab_size=5,
                        context_length=4,
                        dim_embedding=8,
                        n_layers=1,
                        n_heads=2,
                        dropout=0.0,
                    )
                )

            def forward(self, input_ids, target_ids=None):
                batch_size, sequence_length = input_ids.shape
                logits = torch.zeros(batch_size, sequence_length, self.config.vocab_size)
                logits[..., 1] = 9.0
                logits[..., 2] = 10.0
                return logits, None

        torch.manual_seed(0)
        model = RepeatFavoriteModel()
        input_ids = torch.tensor([[2]])

        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            top_k=1,
            repetition_penalty=2.0,
            repetition_window=4,
        )

        self.assertEqual(generated[0, -1].item(), 1)

    def test_generate_repetition_window_only_penalizes_recent_tokens(self) -> None:
        class RepeatFavoriteModel(TransformerLM):
            def __init__(self) -> None:
                super().__init__(
                    TransformerConfig(
                        vocab_size=5,
                        context_length=4,
                        dim_embedding=8,
                        n_layers=1,
                        n_heads=2,
                        dropout=0.0,
                    )
                )

            def forward(self, input_ids, target_ids=None):
                batch_size, sequence_length = input_ids.shape
                logits = torch.zeros(batch_size, sequence_length, self.config.vocab_size)
                logits[..., 1] = 9.0
                logits[..., 2] = 10.0
                return logits, None

        torch.manual_seed(0)
        model = RepeatFavoriteModel()
        input_ids = torch.tensor([[2, 3]])

        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            top_k=1,
            repetition_penalty=2.0,
            repetition_window=1,
        )

        self.assertEqual(generated[0, -1].item(), 2)

    def test_generate_calls_on_token_for_each_new_token(self) -> None:
        torch.manual_seed(0)
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
        streamed_tokens: list[int] = []

        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            on_token=streamed_tokens.append,
        )

        self.assertEqual(streamed_tokens, generated[0, -3:].tolist())

    def test_generate_stops_when_token_callback_requests_stop(self) -> None:
        class FixedLogitModel(TransformerLM):
            def __init__(self) -> None:
                super().__init__(
                    TransformerConfig(
                        vocab_size=5,
                        context_length=4,
                        dim_embedding=8,
                        n_layers=1,
                        n_heads=2,
                        dropout=0.0,
                    )
                )

            def forward(self, input_ids, target_ids=None):
                batch_size, sequence_length = input_ids.shape
                logits = torch.zeros(batch_size, sequence_length, self.config.vocab_size)
                logits[..., 1] = 10.0
                return logits, None

        model = FixedLogitModel()
        input_ids = torch.tensor([[0]])
        streamed_tokens: list[int] = []

        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            top_k=1,
            on_token=lambda token_id: streamed_tokens.append(token_id) or True,
        )

        self.assertEqual(streamed_tokens, [1])
        self.assertEqual(generated.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
