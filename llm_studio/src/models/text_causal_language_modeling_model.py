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


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, cfg, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to(cfg.environment._device) for stop in stops]
        if cfg.environment._local_rank == 0:
            logger.info(f"Stopping criteria tokens: {self.stops}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
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

        if (
            hasattr(self.cfg.tokenizer, "_stop_words_ids")
            and len(self.cfg.tokenizer._stop_words_ids) > 0
        ):
            self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(cfg=cfg, stops=self.cfg.tokenizer._stop_words_ids)]
            )
        else:
            self.stopping_criteria = None

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
            stopping_criteria=self.stopping_criteria,
            renormalize_logits=True,
            return_dict_in_generate=False,
            use_cache=True,
        )
        transformers_logging.set_verbosity(verbosity)

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
            # if `return_dict_in_generate` is false, `generate` just returns
            # a LongTensor, so we have to ignore the type annotation of the method.

            outputs["predicted_answer_ids"] = self.generate(batch, self.cfg)

        return outputs
