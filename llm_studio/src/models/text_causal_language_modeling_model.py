import logging
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging as transformers_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import create_nlp_backbone

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    The ValueHead class implements a head for GPT2 that returns a scalar for each
    output token.
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

        if self.cfg.training.use_rlhf:
            # logger.info("Loading reference model for RLHF")
            # self.ref_model, self.ref_model_config = create_nlp_backbone(
            #     cfg, model_class=AutoModelForCausalLM, kwargs=kwargs
            # )
            # self.ref_model.eval()
            # self.ref_model.requires_grad_(False)

            self.v_head = ValueHead(self.backbone_config)
            # random init by default
            self.v_head.summary.weight.data.normal_(mean=0.0, std=0.2)
            self.v_head.summary.bias.data.zero_()

        self.loss_fn = self.cfg.training.loss_class.get(self.cfg.training.loss_function)

    def generate(self, batch: Dict, cfg: Any, remove_prompt=False):
        pad_token_id = (
            self.backbone.config.pad_token_id or self.backbone.config.eos_token_id
        )

        if "prompt_attention_mask" in batch:
            mask_key = "prompt_attention_mask"
            pad_keys = [
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

        transformers_logging.set_verbosity_error()
        output = generation_function(
            inputs=input_ids,
            attention_mask=attention_mask,
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

        if remove_prompt:
            # remove the prompt tokens
            output = output[:, input_ids.shape[1] :]
        else:
            # Mask the prompt tokens
            output[:, : input_ids.shape[1]] = pad_token_id

        logger.info(f"SHAPE: input {input_ids.shape}, output {output.shape}")
        return output

    def forward(
        self,
        batch: Dict,
        calculate_loss: bool = True,
        generate: bool = False,
    ) -> Dict:
        outputs: Dict = {}

        kwargs = {}

        if self.cfg.training.use_rlhf:
            kwargs["output_hidden_states"] = True

        # model's forward only works with labels
        if "labels" in batch:
            if "prompt_attention_mask" in batch:
                mask_key = "prompt_attention_mask"
                pad_keys = [
                    "prompt_input_ids",
                    "prompt_attention_mask",
                    "special_tokens_mask",
                    "labels",
                ]
            else:
                mask_key = "attention_mask"
                pad_keys = [
                    "input_ids",
                    "attention_mask",
                    "special_tokens_mask",
                    "labels",
                ]

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
                labels=batch["labels"],
                **kwargs,
            )
            if calculate_loss:
                assert self.cfg.training.loss_function == "CrossEntropy"
                outputs["loss"] = output.loss
        else:
            if "prompt_attention_mask" in batch:
                mask_key = "prompt_attention_mask"
                pad_keys = [
                    "prompt_input_ids",
                    "prompt_attention_mask",
                    "special_tokens_mask",
                ]
            else:
                mask_key = "attention_mask"
                pad_keys = [
                    "input_ids",
                    "attention_mask",
                    "special_tokens_mask",
                ]

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

        if self.cfg.training.use_rlhf:
            last_hidden_state = output.hidden_states[-1]

            # force upcast in fp32 if logits are in half-precision
            if output.logits.dtype != torch.float32:
                output.logits = output.logits.float()

            outputs["logits"] = output.logits
            outputs["value"] = self.v_head(last_hidden_state).squeeze(-1)
            print("value", outputs["value"].shape)
            print("logits", outputs["logits"].shape)
        if not self.training or generate:
            outputs["predicted_answer_ids"] = self.generate(batch, self.cfg).detach()
        return outputs


class RefModel(nn.Module):
    """
    Model for causal language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(RefModel, self).__init__()

        self.cfg = cfg

        logger.info("Loading reference model for RLHF")
        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM
        )
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(
        self,
        batch: Dict,
        calculate_loss: bool = False,
    ) -> Dict:
        outputs: Dict = {}

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

        # force upcast in fp32 if logits are in half-precision
        if output.logits.dtype != torch.float32:
            output.logits = output.logits.float()

        outputs["logits"] = output.logits
        outputs["value"] = None

        return outputs


class RewardModel(nn.Module):
    def __init__(self, reward_model="OpenAssistant/reward-model-deberta-v3-large-v2"):
        super(RewardModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model)

    def get_score(
        self,
        questions=None,
        answers=None,
    ):
        scores = []
        for question, answer in zip(questions, answers):
            inputs = self.tokenizer(question, answer, return_tensors="pt").to(
                self.device
            )
            scores.append(self.model(**inputs).logits[0].cpu().detach().item())
            del inputs
        return scores
