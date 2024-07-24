import unittest
from unittest.mock import MagicMock

import torch

from llm_studio.src.augmentations.nlp_aug import BaseNLPAug


class TestBaseNLPAug(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock()
        self.cfg.tokenizer._tokenizer_mask_token_id = 1337

    def test_init(self):
        aug = BaseNLPAug(self.cfg)
        self.assertEqual(aug.cfg, self.cfg)

    def test_forward_no_augmentation(self):
        aug = BaseNLPAug(self.cfg)
        self.cfg.augmentation.token_mask_probability = 0.0

        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                ]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            ),
            "labels": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                ]
            ),
        }

        result = aug.forward(batch.copy())
        self.assertTrue(torch.equal(result["input_ids"], batch["input_ids"]))
        self.assertTrue(torch.equal(result["attention_mask"], batch["attention_mask"]))
        self.assertTrue(torch.equal(result["labels"], batch["labels"]))

    def test_forward_with_augmentation(self):
        aug = BaseNLPAug(self.cfg)
        self.cfg.augmentation.token_mask_probability = 0.5
        torch.manual_seed(42)  # For reproducibility

        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                ]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            ),
            "labels": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                ]
            ),
        }

        result = aug.forward(batch.copy())

        # Check that some tokens have been masked
        self.assertFalse(torch.equal(result["input_ids"], batch["input_ids"]))

        # Check that masked tokens are replaced with mask token ID
        mask = result["input_ids"] == self.cfg.tokenizer._tokenizer_mask_token_id
        self.assertTrue(mask.any())

        # Check that attention mask is updated for masked tokens
        self.assertTrue(
            torch.equal(result["attention_mask"][mask], torch.zeros(mask.sum()))
        )

        # Check that labels are updated to -100 for masked tokens
        self.assertTrue(
            torch.equal(result["labels"][mask], torch.ones(mask.sum()) * -100)
        )
