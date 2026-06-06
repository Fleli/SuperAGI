import unittest

import torch

from superagi.chat.sft import IGNORE_INDEX, TokenizedSftExample
from superagi.chat.sft_training import collate_sft_batch


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


if __name__ == "__main__":
    unittest.main()
