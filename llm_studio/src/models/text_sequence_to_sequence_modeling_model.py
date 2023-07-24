import logging
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch import nn
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration,AutoModelForSeq2SeqLM, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging as transformers_logging

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import create_nlp_backbone

from llm_studio.src.models.text_causal_language_modeling_model import TokenStoppingCriteria

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """
    Model for sequence to sequence modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__()

        self.cfg = cfg

        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForSeq2SeqLM
        )

        if cfg.training.lora:
            self.prepare_lora()

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)


    def prepare_lora(self):
        target_modules = (
            [
                lora_target_module.strip()
                for lora_target_module in self.cfg.training.lora_target_modules.strip().split(  # noqa: E501
                    ","
                )
            ]
            if self.cfg.training.lora_target_modules
            else None
        )
        if (
            not target_modules
            and self.backbone.config.model_type
            not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
        ):
            # extend LORA automatic target module mapping.
            target_modules = {
                "RefinedWebModel": [
                    "query_key_value",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                    "dense",
                ],
            }.get(self.backbone.config.model_type)
        lora_config = LoraConfig(
            r=self.cfg.training.lora_r,
            lora_alpha=self.cfg.training.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.cfg.training.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.enable_input_require_grads()
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        mask_key = "prompt_attention_mask"
        pad_keys = [
            "prompt_input_ids",
            "prompt_attention_mask",
        ]

        batch = batch_padding(
            self.cfg,
            batch,
            self.training,
            mask_key=mask_key,
            pad_keys=pad_keys,
        )

        input_ids = batch["prompt_input_ids"]
        attention_mask = batch["prompt_attention_mask"]

        # Adding GenerationMixin type annotation for faster lookup
        generation_function: GenerationMixin.generate = self.backbone.generate

        verbosity = transformers_logging.get_verbosity()
        stopping_criteria = StoppingCriteriaList(
            [
                TokenStoppingCriteria(
                    stop_word_ids=self.cfg.tokenizer._stop_words_ids,
                    prompt_input_ids_len=input_ids.shape[1],
                )
            ]
        )

        do_sample = cfg.prediction.do_sample
        temperature = float(cfg.prediction.temperature)
        top_k = cfg.prediction.top_k
        top_p = float(cfg.prediction.top_p)
        repetition_penalty = float(cfg.prediction.repetition_penalty)

        # force to use cache and disable gradient checkpointing if enabled
        self.backbone.config.use_cache = True
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.gradient_checkpointing_disable()

        transformers_logging.set_verbosity_error()
        output = generation_function(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=self.backbone.generation_config,
            min_new_tokens=cfg.prediction.min_length_inference,
            max_new_tokens=cfg.prediction.max_length_inference,
            do_sample=do_sample,
            num_beams=cfg.prediction.num_beams,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            stopping_criteria=stopping_criteria,
            renormalize_logits=True,
            return_dict_in_generate=False,
            use_cache=True,
            streamer=streamer,
        )
        transformers_logging.set_verbosity(verbosity)

        # enable checkpointing again
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        return output

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        kwargs = {}

        mask_key = "attention_mask"
        pad_keys = [
            "input_ids",
            "attention_mask",
            "special_tokens_mask",
            "labels",
        ]

        if padding:
            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
            )

        labels = batch["answer_input_ids"]
        labels[batch["answer_attention_mask"]==0] = -100

        output = self.backbone(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            labels=labels,
            **kwargs,
        )

        outputs["loss"] = output.loss

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
