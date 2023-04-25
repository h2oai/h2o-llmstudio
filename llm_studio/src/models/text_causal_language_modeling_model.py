import logging
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging as transformers_logging

from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import create_nlp_backbone

logger = logging.getLogger(__name__)


class TokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on tokens.
    Will stop generation when each generated sample contains at least one of the stop_word_ids.
    """

    def __init__(self, stop_word_ids, prompt_input_ids_len):
        super().__init__()
        self.prompt_input_ids_len = prompt_input_ids_len
        if stop_word_ids is None:
            stop_word_ids = []
        self.stop_word_ids = stop_word_ids

    def should_stop(
        self, generated_ids: torch.LongTensor, stop_word_id: torch.FloatTensor
    ):
        return (
            torch.mean(((generated_ids == stop_word_id).sum(1) > 0).float()) == 1
        ).item()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        generated_ids = input_ids[:, len(self.prompt_input_ids_len)]
        for stop_word_id in self.stop_word_ids:
            if self.should_stop(generated_ids, stop_word_id.to(generated_ids.device)):
                return True
        return False


class Model(nn.Module):
    """
    Model for causal language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__()

        self.cfg = cfg
        kwargs = {}

        self.backbone = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM, kwargs=kwargs
        )

        if cfg.training.lora:
            lora_config = LoraConfig(
                r=cfg.training.lora_r,
                lora_alpha=cfg.training.lora_alpha,
                target_modules=cfg.training.lora_target_modules.split(",")
                if cfg.training.lora_target_modules
                else None,
                lora_dropout=cfg.training.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if cfg.architecture.gradient_checkpointing:
                self.backbone.enable_input_require_grads()
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        self.loss_fn = self.cfg.training.loss_class.get(self.cfg.training.loss_function)

    def generate(self, batch: Dict, cfg: Any):
        pad_token_id = (
            self.backbone.config.pad_token_id or self.backbone.config.eos_token_id
        )

        batch = batch_padding(
            self.cfg,
            batch,
            self.training,
            mask_key="prompt_attention_mask",
            pad_keys=[
                "prompt_input_ids",
                "prompt_attention_mask",
            ],
        )

        # Adding GenerationMixin type annotation for faster lookup
        generation_function: GenerationMixin.generate = self.backbone.generate

        verbosity = transformers_logging.get_verbosity()
        stopping_criteria = StoppingCriteriaList[
            TokenStoppingCriteria(
                stop_word_ids=self.cfg.tokenizer._stop_words_ids,
                prompt_input_ids_len=batch["prompt_input_ids"].shape[1],
            )
        ]

        transformers_logging.set_verbosity_error()
        output = generation_function(
            inputs=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            pad_token_id=pad_token_id,
            min_new_tokens=cfg.prediction.min_length_inference,
            max_new_tokens=cfg.prediction.max_length_inference,
            do_sample=cfg.prediction.do_sample,
            num_beams=cfg.prediction.num_beams,
            temperature=float(cfg.prediction.temperature),
            repetition_penalty=float(cfg.prediction.repetition_penalty),
            stopping_criteria=stopping_criteria,
            renormalize_logits=True,
            return_dict_in_generate=False,
            use_cache=True,
        )
        transformers_logging.set_verbosity(verbosity)

        # Mask the prompt tokens
        output[:, : batch["prompt_input_ids"].shape[1]] = pad_token_id

        return output

    def forward(
        self,
        batch: Dict,
        calculate_loss: bool = True,
    ) -> Dict:
        outputs: Dict = {}

        # model's forward only works with labels
        if "labels" in batch:

            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                pad_keys=[
                    "input_ids",
                    "attention_mask",
                    "special_tokens_mask",
                    "labels",
                ],
            )
            output = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            if calculate_loss:
                assert self.cfg.training.loss_function == "CrossEntropy"
                outputs["loss"] = output.loss

        if not self.training:
            outputs["predicted_answer_ids"] = self.generate(batch, self.cfg)

        return outputs
