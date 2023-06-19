import logging
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch import nn
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging as transformers_logging

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import create_nlp_backbone

logger = logging.getLogger(__name__)


class TokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on tokens.
    Will stop generation when each generated sample contains at least one of the
    stop_word_ids.
    """

    def __init__(self, stop_word_ids, prompt_input_ids_len):
        super().__init__()
        self.prompt_input_ids_len = prompt_input_ids_len
        if stop_word_ids is None:
            stop_word_ids = []
        self.stop_word_ids = stop_word_ids

    def should_stop(
        self,
        generated_ids: torch.LongTensor,
        stop_word_id: torch.Tensor,
    ):
        if len(stop_word_id.shape) == 0:
            return (
                torch.mean(((generated_ids == stop_word_id).sum(1) > 0).float()) == 1
            ).item()
        else:
            return (
                self.get_num_vector_found_in_matrix_rows(stop_word_id, generated_ids)
                == generated_ids.shape[0]
            )

    @staticmethod
    def get_num_vector_found_in_matrix_rows(vector, matrix):
        """
        Count the number of times a vector is found in a matrix row.
        If the vector is found in a row, the search stops and the next row is searched.
        """
        assert len(vector.shape) == 1
        assert len(matrix.shape) == 2

        found = 0
        for row in matrix:
            # stride through the vector
            for i in range(len(row) - len(vector) + 1):
                # check if the vector contains the tensor
                if torch.all(row[i : i + len(vector)] == vector):
                    found += 1
                    break

        return found

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        generated_ids = input_ids[:, self.prompt_input_ids_len :]
        for stop_word_id in self.stop_word_ids:
            if self.should_stop(generated_ids, stop_word_id.to(generated_ids.device)):
                if generated_ids.shape[1] == 1:
                    logger.warning(
                        f"Stopping criteria triggered for {stop_word_id} at first "
                        "generated token."
                    )
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
            self.prepare_lora()

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def prepare_lora(self):
        target_modules = (
            self.cfg.training.lora_target_modules.split(",")
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
            task_type="CAUSAL_LM",
        )
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.enable_input_require_grads()
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

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
        stopping_criteria = StoppingCriteriaList(
            [
                TokenStoppingCriteria(
                    stop_word_ids=self.cfg.tokenizer._stop_words_ids,
                    prompt_input_ids_len=batch["prompt_input_ids"].shape[1],
                )
            ]
        )

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
            top_k=cfg.prediction.top_k,
            top_p=float(cfg.prediction.top_p),
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
            )
            if calculate_loss:
                outputs["loss"] = self.loss_fn(output.logits, batch["labels"])

            if self.cfg.prediction.metric == "Perplexity":
                outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        if not self.training and self.cfg.prediction.metric != "Perplexity":
            outputs["predicted_answer_ids"] = self.generate(batch, self.cfg)

        return outputs
