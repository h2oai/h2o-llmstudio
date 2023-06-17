import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerationMixin
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)
from transformers.utils import ModelOutput
from transformers.utils import logging as transformers_logging

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import create_nlp_backbone

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    The ValueHead class implements a head for GPT2 that returns a scalar for each
    output token.

    Based on the implementation of trl library:
    https://github.com/lvwerra/trl/blob/main/trl/models/modeling_value_head.py
    """

    def __init__(self, config):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = 0.1
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = (
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        )

        # some models such as OPT have a projection layer before the word embeddings
        # e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


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

        self.backbone, self.backbone_config = create_nlp_backbone(
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

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

        if self.cfg.training.use_rlhf:
            self.value_head = ValueHead(self.backbone_config)
            self.value_head.summary.bias.data.zero_()

    @torch.inference_mode(mode=True)
    def generate(self, batch: Dict, cfg: Any, remove_prompt=False):
        pad_token_id = (
            self.backbone.config.pad_token_id or self.backbone.config.eos_token_id
        )

        if "prompt_attention_mask" in batch:
            mask_key = "prompt_attention_mask"
            pad_keys = [
                "input_ids",
                "attention_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
            ]
        else:
            mask_key = "attention_mask"
            pad_keys = [
                "input_ids",
                "attention_mask",
            ]

        batch = batch_padding(
            self.cfg,
            batch,
            self.training,
            mask_key=mask_key,
            pad_keys=pad_keys,
        )

        if "prompt_attention_mask" in batch:
            input_ids = batch["prompt_input_ids"]
            attention_mask = batch["prompt_attention_mask"]
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

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

        # The KL-div estimation assumes sampling and specific settings
        if self.training and cfg.training.use_rlhf:
            do_sample = True
            temperature = cfg.training.ppo_generate_temperature
            top_k = 0.0
            top_p = 1.0
            repetition_penalty = 1.0
        else:
            do_sample = cfg.prediction.do_sample
            temperature = float(cfg.prediction.temperature)
            top_k = cfg.prediction.top_k
            top_p = float(cfg.prediction.top_p)
            repetition_penalty = float(cfg.prediction.repetition_penalty)

        transformers_logging.set_verbosity_error()
        output = generation_function(
            inputs=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
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
        )
        transformers_logging.set_verbosity(verbosity)

        if remove_prompt:
            # remove the prompt tokens
            output = output[:, input_ids.shape[1] :]
        else:
            # Mask the prompt tokens
            output[:, : input_ids.shape[1]] = pad_token_id

        return output

    def forward(
        self,
        batch: Dict,
        generate: bool = False,
        padding: bool = True,
        is_ref_model: bool = False,
    ) -> Dict:
        outputs: Dict = {}

        kwargs = {}

        if self.training and self.cfg.training.use_rlhf:
            kwargs["output_hidden_states"] = True

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

        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )

        if "labels" in batch and not is_ref_model:
            loss = self.loss_fn(output.logits, batch["labels"])
            outputs["loss"] = loss

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        if self.training and self.cfg.training.use_rlhf:
            last_hidden_state = output.hidden_states[-1]

            # force upcast in fp32 if logits are in half-precision
            if output.logits.dtype != torch.float32:
                output.logits = output.logits.float()

            outputs["logits"] = output.logits
            outputs["value"] = (
                self.value_head(last_hidden_state).squeeze(-1)
                if not is_ref_model
                else None
            )

        if self.cfg.prediction.metric != "Perplexity":
            # do not generate new text in forward if perplexity is the metric
            generate = False

        if generate:
            outputs["predicted_answer_ids"] = (
                self.generate(batch, self.cfg).detach().cpu()
            )
        return outputs


class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

    pooling: Literal["mean", "last"]

    def __init__(
        self,
        pooling: Literal["mean", "last"] = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling = pooling or "last"


@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None


class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if type(config) == GPTNeoXConfig:
            # When a normal GPTNeoX was loaded it will be converted into a reward model.
            # The direct `type(config) == GPTNeoXConfig` comparison is used (instead of
            # `isinstance()`) since the configuration class of the reward model is also
            # derived form `GPTNeoXConfig`.
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> GPTNeoXRewardModelOutput:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask).sum(
                    dim=1
                ) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1]
            else:
                last_idx = attention_mask.cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states.gather(
                    1, last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
                ).squeeze(1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)


class RewardModel(nn.Module):
    def __init__(self, cfg):
        super(RewardModel, self).__init__()

        AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
        AutoModelForSequenceClassification.register(
            GPTNeoXRewardModelConfig, GPTNeoXRewardModel
        )

        self.cfg = cfg
        self.model_name = cfg.training.reward_model
        self.device = cfg.environment._device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, max_model_input_sizes=2048
        )

    def get_score(
        self,
        prompts=None,
        answers=None,
    ):
        scores = []
        for prompt, answer in zip(prompts, answers):
            if self.model_name == "OpenAssistant/reward-model-deberta-v3-large-v2":
                inputs = self.tokenizer(
                    " ".join(prompt.split("<|endoftext|>")),
                    answer,
                    return_tensors="pt",
                    max_length=2048,
                ).to(self.device)
            elif self.model_name in [
                "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
                "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            ]:
                prompt = prompt.split("<|endoftext|>")

                input_text = ""

                for i, prompt_part in enumerate(prompt[::-1]):
                    if i % 2 == 0:
                        prefix = "<|prompter|>"
                    else:
                        prefix = "<|assistant|>"
                    input_text = f"{prefix}{prompt_part}<|endoftext|>" + input_text

                input_text = input_text + f"<|assistant|>{answer}<|endoftext|>"

                inputs = self.tokenizer(
                    input_text, return_tensors="pt", max_length=2048
                ).to(self.device)

            scores.append(self.model(**inputs).logits[0].cpu().detach().item())
            del inputs
        return scores
