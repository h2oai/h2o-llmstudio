import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as CausalLMCustomDataset,
)
from llm_studio.src.datasets.text_utils import TEXT_SEPARATOR

logger = logging.getLogger(__name__)


class CustomDataset(CausalLMCustomDataset):
    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        assert (
            cfg.dataset.system_column == "None"
        ), "RLHF is not compatible with system column."
        assert (
            cfg.dataset.limit_chained_samples is False
        ), "RLHF is not compatible with limit_chained_samples."
        assert (
            cfg.dataset.mask_prompt_labels is True
        ), "RLHF is not compatible with mask_prompt_labels."
        super().__init__(df, cfg, mode)

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = super().__getitem__(idx)
        sample["reward_model_prompt_text"] = TEXT_SEPARATOR.join(
            self.get_chained_prompt_text_list(idx)
        )
        return sample

    def get_labels(self, prompt_encodings, answer_encodings):
        if self.mode == "train":  # no labels required for RLHF during training
            return dict()
        else:
            return super().get_labels(prompt_encodings, answer_encodings)

    def get_encodings(self, input_text_dict):
        system_encoding, prompt_encodings, answer_encodings = super().get_encodings(
            input_text_dict
        )
        # remove last ground truth answer,
        # as RLHF will generate the answer from the prompt
        answer_encodings[-1] = torch.empty(0)
        return system_encoding, prompt_encodings, answer_encodings

    def postprocess_batch_predictions(self, output: Dict) -> Dict:
        if "predicted_answer_ids" in output.keys():
            predicted_text = [
                self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in output["predicted_answer_ids"]
            ]

            output["predicted_text"] = np.array(predicted_text)
            output["predicted_answer_ids"] = output["predicted_answer_ids"].detach()
        return output

    def augment_data(self, encodings):
        return encodings

    def get_chained_prompt_text_list(self, idx: int) -> List[str]:
        text_dict = self.conversation_chain_handler[idx]
        chat_history = "".join(
            [
                prompt + TEXT_SEPARATOR + answer + TEXT_SEPARATOR
                for prompt, answer in zip(
                    text_dict["prompts"][:-1], text_dict["answers"][:-1]
                )
            ]
        )
        prompt_text = text_dict["systems"][0] + chat_history + text_dict["prompts"][-1]
        return prompt_text.split(TEXT_SEPARATOR)
