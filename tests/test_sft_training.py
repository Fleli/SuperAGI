import unittest

import torch

from superagi.chat.sft import IGNORE_INDEX, TokenizedSftExample
from superagi.chat.sft_training import (
    collate_sft_batch,
    evaluate_sft_loss,
    parse_sft_source_weights,
    sample_sft_batch,
    split_sft_examples,
    source_sampling_weights,
    source_summary,
)
from superagi.model.transformer import TransformerConfig, TransformerLM


class SftTrainingTests(unittest.TestCase):
    def test_collates_variable_length_sft_examples_with_label_padding(self) -> None:
        examples = [
            TokenizedSftExample(
                text="short",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
            ),
            TokenizedSftExample(
                text="long",
                input_ids=(4, 5, 6),
                target_ids=(IGNORE_INDEX, 7, 8),
                supervised_token_count=2,
            ),
        ]

        input_ids, target_ids = collate_sft_batch(
            examples,
            pad_token_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(input_ids.tolist(), [[1, 2, 0], [4, 5, 6]])
        self.assertEqual(
            target_ids.tolist(),
            [[IGNORE_INDEX, 3, IGNORE_INDEX], [IGNORE_INDEX, 7, 8]],
        )

    def test_rejects_empty_sft_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one SFT example"):
            collate_sft_batch([], pad_token_id=0, device=torch.device("cpu"))

    def test_splits_sft_examples_deterministically(self) -> None:
        examples = [
            TokenizedSftExample(
                text=f"example-{index}",
                input_ids=(index, index + 1),
                target_ids=(IGNORE_INDEX, index + 2),
                supervised_token_count=1,
            )
            for index in range(10)
        ]

        train_examples, validation_examples = split_sft_examples(
            examples,
            validation_fraction=0.2,
            seed=123,
        )
        train_examples_again, validation_examples_again = split_sft_examples(
            examples,
            validation_fraction=0.2,
            seed=123,
        )

        self.assertEqual(len(train_examples), 8)
        self.assertEqual(len(validation_examples), 2)
        self.assertEqual(train_examples, train_examples_again)
        self.assertEqual(validation_examples, validation_examples_again)
        self.assertTrue(set(train_examples).isdisjoint(validation_examples))

    def test_sft_split_keeps_all_examples_for_training_when_validation_disabled(self) -> None:
        examples = [
            TokenizedSftExample(
                text="only",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
            )
        ]

        train_examples, validation_examples = split_sft_examples(
            examples,
            validation_fraction=0.0,
            seed=123,
        )

        self.assertEqual(train_examples, tuple(examples))
        self.assertEqual(validation_examples, ())

    def test_evaluates_sft_validation_loss_without_updating_model(self) -> None:
        model = TransformerLM(
            TransformerConfig(
                vocab_size=8,
                context_length=4,
                dim_embedding=8,
                n_layers=1,
                n_heads=2,
            )
        )
        examples = [
            TokenizedSftExample(
                text="a",
                input_ids=(1, 2, 3),
                target_ids=(IGNORE_INDEX, 4, 5),
                supervised_token_count=2,
            ),
            TokenizedSftExample(
                text="b",
                input_ids=(2, 3),
                target_ids=(IGNORE_INDEX, 6),
                supervised_token_count=1,
            ),
        ]
        before_parameters = [parameter.detach().clone() for parameter in model.parameters()]

        validation_loss = evaluate_sft_loss(
            model,
            examples,
            batch_size=2,
            pad_token_id=0,
            device=torch.device("cpu"),
            max_batches=2,
        )

        self.assertGreater(validation_loss, 0.0)
        for before, after in zip(before_parameters, model.parameters(), strict=True):
            self.assertTrue(torch.equal(before, after))

    def test_parses_sft_source_weights(self) -> None:
        weights = parse_sft_source_weights("anchor=4,wildchat=0.35,default=1")

        self.assertEqual(weights["anchor"], 4.0)
        self.assertEqual(weights["wildchat"], 0.35)
        self.assertEqual(weights["default"], 1.0)

    def test_rejects_invalid_sft_source_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "source weight"):
            parse_sft_source_weights("anchor=-1")
        with self.assertRaisesRegex(ValueError, "source=weight"):
            parse_sft_source_weights("anchor")

    def test_builds_source_sampling_weights_from_source_prefix(self) -> None:
        examples = [
            TokenizedSftExample(
                text="anchor",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="anchor",
            ),
            TokenizedSftExample(
                text="wildchat",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="wildchat:42",
            ),
            TokenizedSftExample(
                text="other",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="unknown:7",
            ),
        ]

        weights = source_sampling_weights(
            examples,
            {"anchor": 4.0, "wildchat": 0.25, "default": 0.5},
        )

        self.assertEqual(weights.tolist(), [4.0, 0.25, 0.5])

    def test_summarizes_sft_source_counts_and_weighted_mass(self) -> None:
        examples = [
            TokenizedSftExample(
                text="anchor",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="anchor",
            ),
            TokenizedSftExample(
                text="wildchat",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="wildchat:1",
            ),
            TokenizedSftExample(
                text="wildchat",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="wildchat:2",
            ),
        ]

        summary = source_summary(examples, {"anchor": 4.0, "wildchat": 0.5})

        self.assertIn("anchor=1", summary.counts_text)
        self.assertIn("wildchat=2", summary.counts_text)
        self.assertIn("anchor=80.0%", summary.sampling_mass_text)
        self.assertIn("wildchat=20.0%", summary.sampling_mass_text)

    def test_weighted_sft_sampling_can_exclude_a_source(self) -> None:
        examples = [
            TokenizedSftExample(
                text="anchor",
                input_ids=(1, 2),
                target_ids=(IGNORE_INDEX, 3),
                supervised_token_count=1,
                source="anchor",
            ),
            TokenizedSftExample(
                text="wildchat",
                input_ids=(9, 10),
                target_ids=(IGNORE_INDEX, 11),
                supervised_token_count=1,
                source="wildchat:1",
            ),
        ]

        input_ids, _ = sample_sft_batch(
            examples,
            batch_size=4,
            pad_token_id=0,
            device=torch.device("cpu"),
            source_weights={"anchor": 1.0, "wildchat": 0.0},
        )

        self.assertEqual(input_ids.tolist(), [[1, 2], [1, 2], [1, 2], [1, 2]])


if __name__ == "__main__":
    unittest.main()
