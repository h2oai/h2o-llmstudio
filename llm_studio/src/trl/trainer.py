# This file borrows large pieces from the trl library, which is licensed under
# the Apache 2.0 license.
# https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_trainer.py


# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import PyTorchModelHubMixin
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from llm_studio.src.utils.modeling_utils import unwrap_model


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat: Dict[str, Any] = {}
    rec(nested, "", flat)
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=-1)
    return results


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        bessel_correction = mask.sum() / (mask.sum() - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict: Dict[str, Any] = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu()
            if new_dict[k].dtype == torch.bfloat16:
                new_dict[k] = new_dict[k].float()
            new_dict[k] = new_dict[k].numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.value = self.cfg.training.initial_kl_coefficient
        self.target = self.cfg.training.kl_target
        self.horizon = self.cfg.training.kl_horizon

    def update(self, current, n_steps):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.value = self.cfg.training.initial_kl_coefficient

    def update(self, current, n_steps):
        pass


class PPOTrainer(PyTorchModelHubMixin):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize
    work here: https://github.com/openai/summarize-from-feedback

    Attributes:
        **cfg** (`LLM Studio Config`) -- Experiment configuration object. Check the
            documentation of `LLM Studio Config` for more details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face
            transformer model with a value head. Check the documentation of
            `PreTrainedModelWrapper` for more details.
        **tokenizer** (`Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`)
            Tokenizer to be used for encoding the data. Check the documentation of
            `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **optimizer** (`torch.optim.Optimizer`) -- Optimizer to be used for training.
        **lr_scheduler** (`torch.optim.lr_scheduler`) -- Learning rate scheduler to be
            used for training.
        **scaler** (`torch.cuda.amp.GradScaler`) -- Gradient scaler to be used for
            training.
    """

    def __init__(
        self,
        cfg=None,
        model=None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler=None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            cfg (`LLM Studio Config`):
                experiment configuration object. Check the documentation of
                `LLM Studio Config` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            tokenizer (`transformers.PreTrainedTokenizer`):
                Hugging Face tokenizer
            optimizer (`torch.optim.Optimizer`):
                Optimizer used for training.
            lr_scheduler (`torch.optim.lr_scheduler`):
                Learning rate scheduler used for training.
            scaler (`torch.cuda.amp.GradScaler`):
                Gradient scaler used for training.
        """
        self.cfg = cfg

        # Step 1: Initialize Model
        self.model = model
        self.tokenizer = tokenizer

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler

        self.kl_ctl: AdaptiveKLController | FixedKLController
        if self.cfg.training.adaptive_kl_control:
            self.kl_ctl = AdaptiveKLController(cfg)
        else:
            self.kl_ctl = FixedKLController(cfg)

        self.current_device = self.cfg.environment._device

        # init the current step
        self.current_step = 0

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.Tensor],
    ):
        """
        Check if the input data is valid for training and move the data to the correct
        device.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape
                (`response_length`)
            scores (List[`torch.Tensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(
            ["queries", "responses", "scores"], [queries, responses, scores]
        ):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"{name} must be a list of tensors - got {type(tensor_list)}"
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                )
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples"
                    f" - but got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(
                    f"Scores must be 1-dimensional - got {score.dim()} for {score}"
                )
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.Tensor],
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and
        rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape
                (`response_length`)
            scores (List[`torch.Tensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.cfg.training.batch_size

        queries, responses, scores = self._step_safety_checker(
            bs, queries, responses, scores
        )

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        model_inputs_names = list(model_inputs.keys())

        with torch.no_grad():
            all_logprobs, _, values, masks = self.batched_forward_pass(
                self.model, queries, responses, model_inputs
            )
            with unwrap_model(self.model).backbone.disable_adapter():
                ref_logprobs, _, _, _ = self.batched_forward_pass(
                    self.model,
                    queries,
                    responses,
                    model_inputs,
                    return_values=False,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(
            scores, all_logprobs, ref_logprobs, masks
        )
        timing["time/ppo/compute_rewards"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        mini_batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "rewards": rewards,
            "masks": masks,
        }

        def collator(data: List[Dict[str, torch.Tensor]]):
            return_dict: Dict[str, Any] = dict()
            keys = data[0].keys()
            for key in keys:
                if key in ["queries", "responses"]:
                    return_dict[key] = [d[key] for d in data]
                else:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(
                        self.current_device
                    )
            return return_dict

        mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = DataLoader(
            mini_batch_data,
            batch_size=self.cfg.training.ppo_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        t = time.time()
        all_stats = []
        num_updates = 0

        if (
            self.cfg.training.ppo_epochs * self.cfg.training.ppo_batch_size
        ) % self.cfg.training.grad_accumulation != 0:
            raise ValueError(
                "ppo_epochs*ppo_batch_size must be multiply of grad_accumulation"
            )

        for _ in range(self.cfg.training.ppo_epochs):
            for batch in mini_batch_dataloader:
                num_updates += 1

                model_inputs = {k: batch[k] for k in model_inputs_names}
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model,
                    batch["queries"],
                    batch["responses"],
                    model_inputs,
                    return_logits=True,
                )

                loss_p, loss_v, train_stats = self.loss(
                    batch["logprobs"],
                    batch["values"],
                    batch["rewards"],
                    logits,
                    vpreds,
                    logprobs,
                    batch["masks"],
                )
                loss = loss_p + loss_v

                # loss is a mean loss per batch/sample
                # as grad_accumulations sums up the gradients, this loss must be scaled
                # by the number of grad_accumulations, to have similar behavior for
                # BS * grad_accumulations = const.
                if self.cfg.training.grad_accumulation != 1:
                    loss = loss / self.cfg.training.grad_accumulation

                # Backward pass
                if self.cfg.environment.mixed_precision:
                    self.scaler.scale(loss).backward()
                    if num_updates % self.cfg.training.grad_accumulation == 0:
                        if self.cfg.training.gradient_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.cfg.training.gradient_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if num_updates % self.cfg.training.grad_accumulation == 0:
                        if self.cfg.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.cfg.training.gradient_clip
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                if self.cfg.environment._distributed:
                    torch.cuda.synchronize(device=self.current_device)

                del logprobs, logits, vpreds

                all_stats.append(train_stats)

        timing["time/ppo/ppo_steps"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], -1
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(stats["objective/kl"], self.cfg.training.batch_size)

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator(
            [
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                for ids in input_ids
            ]
        ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data

    def batched_forward_pass(
        self,
        model,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        return_values: bool = True,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`,
                `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`,
                `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed
                to reduce memory consumption.
            return_values (`bool`, *optional*, defaults to `True`):
                Whether to return values. Set to `False` if values are not needed to
                reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the
                    responses, shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the
                    responses, shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape
                    (`batch_size`, `response_length`)
        """

        bs = len(queries)
        ppo_bs = self.cfg.training.ppo_batch_size

        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / ppo_bs)):
            model_inputs_batch = {
                key: value[i * ppo_bs : (i + 1) * ppo_bs]
                for key, value in model_inputs.items()
            }

            query_batch = queries[i * ppo_bs : (i + 1) * ppo_bs]
            response_batch = responses[i * ppo_bs : (i + 1) * ppo_bs]

            with autocast(enabled=self.cfg.environment.mixed_precision):
                outputs = model(
                    model_inputs_batch,
                    padding=False,
                )

            logits = outputs["logits"]
            values = outputs["value"]

            input_ids = model_inputs_batch["input_ids"]
            attention_mask = model_inputs_batch["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(ppo_bs):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError(
                        "Responses are too short. Make sure they are at least 2"
                        " tokens long."
                    )

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            if return_values:
                all_values.append(values)
            else:
                del values
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        del outputs

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1] if return_values else None,
            torch.cat(all_masks)[:, :-1],
        )

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`,
                `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(
            scores, logprobs, ref_logprobs, masks
        ):
            # compute KL penalty (from difference in logprobs)
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`,
                `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = torch.tensor(0.0)
        advantages_reversed: List[torch.Tensor] = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else torch.tensor(0.0)
            delta: torch.Tensor = (
                rewards[:, t]
                + torch.tensor(self.cfg.training.advantages_gamma) * nextvalues
                - values[:, t]
            )
            lastgaelam = (
                delta
                + torch.tensor(self.cfg.training.advantages_gamma)
                * torch.tensor(self.cfg.training.advantages_lambda)
                * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds,
            values - self.cfg.training.ppo_clip_value,
            values + self.cfg.training.ppo_clip_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cfg.training.ppo_clip_policy,
            1.0 + self.cfg.training.ppo_clip_policy,
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.cfg.training.scaling_factor_value_loss * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(
                policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()
            ),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return (
            pg_loss,
            self.cfg.training.scaling_factor_value_loss * vf_loss,
            flatten_dict(stats),
        )

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        mask = data.pop("masks")

        kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward is size `batch_size`, `response_length`
        mean_scores = torch.stack(data["scores"]).mean()  # scores is size `batch_size`
        std_scores = torch.stack(data["scores"]).std()

        if mean_kl.item() < -1.0:
            warnings.warn(
                f"KL divergence is starting to become negative: {mean_kl.item():.2f} -"
                " this might be a precursor for failed training."
                " sometimes this happens because the generation kwargs are not"
                " correctly set. Please make sure that the generation kwargs are set"
                " correctly, or review your training hyperparameters."
            )

        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        }

        # Log text properties
        query_lens = torch.tensor(
            [len(query) for query in data["queries"]], dtype=torch.float
        )
        response_lens = torch.tensor(
            [len(response) for response in data["responses"]], dtype=torch.float
        )

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = (
            torch.mean(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_len_std"] = (
            torch.std(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, dim=0)
        stats["ppo/val/var_explained"] = (
            1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        )
        return stats

    def _save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
