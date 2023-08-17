import asyncio
import gc
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from h2o_wave import Q
from h2o_wave import data as chat_data
from h2o_wave import ui
from transformers import AutoTokenizer, TextStreamer

from app_utils.utils import get_experiments, get_ui_elements, parse_ui_elements, set_env
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.models.text_causal_language_modeling_model import Model
from llm_studio.src.utils.config_utils import load_config_yaml
from llm_studio.src.utils.modeling_utils import load_checkpoint

logger = logging.getLogger(__name__)

USER = True
BOT = False


class WaveChatStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        q: Q,
        text_cleaner: Optional[Callable] = None,
        **decode_kwargs,
    ):
        """
        Updates the chabot card in a streaming fashion
        """
        super().__init__(tokenizer, skip_prompt=True, **decode_kwargs)
        self.text_cleaner = text_cleaner
        self.words_predicted_answer: List[str] = []
        self.q = q
        self.loop = asyncio.get_event_loop()
        self.finished = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.words_predicted_answer += [text]
        self.loop.create_task(self.push_to_chat())

    async def push_to_chat(self):
        self.q.page["experiment/display/chat"].data[-1] = [self.answer, BOT]
        await self.q.page.save()

    @property
    def answer(self):
        """
        Create the answer by joining the generated words.
        By this, self.text_cleaner does not need to be idempotent.
        """
        answer = "".join(self.words_predicted_answer)
        if answer.endswith(self.tokenizer.eos_token):
            # text generation is stopped
            answer = answer.replace(self.tokenizer.eos_token, "")
        if self.text_cleaner:
            answer = self.text_cleaner(answer)
        return answer

    def end(self):
        super().end()
        self.finished = True


async def chat_tab(q: Q, load_model=True):
    # gpu id in UI is offset by 1 to be in sync with experiment UI
    gpu_id = q.client["gpu_used_for_chat"] - 1
    if gpu_is_blocked(q, gpu_id):
        return

    if load_model:
        q.page["experiment/display/chat"] = ui.form_card(
            box="first",
            items=[ui.progress(label="Loading the model...")],
        )

    q.client["experiment/display/chat/messages"] = []
    q.client.delete_cards.add("experiment/display/chat")

    q.page["experiment/display/chat/settings"] = ui.form_card(
        box="second",
        items=[
            ui.expander(
                name="chat_settings",
                label="Chat Settings",
                items=[ui.progress(label="Loading model configuration...")],
                expanded=True,
            )
        ],
    )
    q.client.delete_cards.add("experiment/display/chat/settings")

    await q.page.save()
    logger.info(torch.cuda.memory_allocated())

    if load_model:
        with set_env(HUGGINGFACE_TOKEN=q.client["default_huggingface_api_token"]):
            cfg, model, tokenizer = load_cfg_model_tokenizer(
                q.client["experiment/display/experiment_path"], device=f"cuda:{gpu_id}"
            )
        q.client["experiment/display/chat/cfg"] = cfg
        q.client["experiment/display/chat/model"] = model
        q.client["experiment/display/chat/tokenizer"] = tokenizer
        initial_message = "Model successfully loaded, how can I help you?"

    else:
        cfg = q.client["experiment/display/chat/cfg"]
        assert q.client["experiment/display/chat/model"] is not None
        assert q.client["experiment/display/chat/tokenizer"] is not None
        initial_message = "Chat History cleaned. How can I help you?"

    # Hide fields that are should not be visible in the UI
    cfg.prediction._visibility["metric"] = -1
    cfg.prediction._visibility["batch_size_inference"] = -1
    cfg.prediction._visibility["min_length_inference"] = -1
    cfg.prediction._visibility["stop_tokens"] = -1

    logger.info(torch.cuda.memory_allocated())
    q.page["experiment/display/chat"] = ui.chatbot_card(
        box="first",
        data=chat_data(fields="content from_user", t="list"),  # type: ignore
        name="experiment/display/chat/chatbot",
    )
    q.page["experiment/display/chat"].data += [initial_message, BOT]

    option_items = get_ui_elements(
        cfg=q.client["experiment/display/chat/cfg"].prediction,
        q=q,
        pre="chat/cfg_predictions",
    )
    q.page["experiment/display/chat/settings"] = ui.form_card(
        box="second",
        items=[
            ui.button(
                name="experiment/display/chat/clear_history",
                label="Clear History",
                primary=True,
            ),
            ui.expander(
                name="chat_settings",
                label="Chat Settings",
                items=option_items,
                expanded=True,
            ),
        ],
    )


def gpu_is_blocked(q, gpu_id):
    experiments = get_experiments(q=q)
    running_experiments = experiments[experiments.status.isin(["running"])]
    gpu_blocked = any(
        [
            str(gpu_id) in gpu_list
            for gpu_list in running_experiments["gpu_list"]
            .apply(lambda x: x.split(","))
            .to_list()
        ]
    )
    if gpu_blocked:
        q.page["experiment/display/chat"] = ui.form_card(
            box="first",
            items=[
                ui.text(
                    f"""Chatbot is not available when GPU{q.client["gpu_used_for_chat"]}
                     is blocked by another experiment.
                     You can change "Gpu used for Chat" in the settings tab
                     to use another GPU for the chatbot. """
                )
            ],
            title="",
        )
        q.client.delete_cards.add("experiment/display/chat")
        return True
    return False


@torch.inference_mode(mode=True)
async def chat_update(q: Q) -> None:
    """
    Update the chatbot with the new message.
    """
    q.client["experiment/display/chat/finished"] = False

    cfg_prediction = parse_ui_elements(
        cfg=q.client["experiment/display/chat/cfg"].prediction,
        q=q,
        pre="chat/cfg_predictions/cfg/",
    )
    logger.info(f"Using chatbot config: {cfg_prediction}")
    q.client["experiment/display/chat/cfg"].prediction = cfg_prediction

    prompt = q.client["experiment/display/chat/chatbot"]

    message = [prompt, USER]
    q.client["experiment/display/chat/messages"].append(message)
    q.page["experiment/display/chat"].data += message
    q.page["experiment/display/chat"].data += ["", BOT]
    await q.page.save()

    cfg = q.client["experiment/display/chat/cfg"]
    model: Model = q.client["experiment/display/chat/model"]
    tokenizer = q.client["experiment/display/chat/tokenizer"]

    full_prompt = ""
    if len(q.client["experiment/display/chat/messages"]):
        for prev_message in q.client["experiment/display/chat/messages"][
            -(cfg.prediction.num_history + 1) :
        ]:
            if prev_message[1] is USER:
                prev_message = cfg.dataset.dataset_class.parse_prompt(
                    cfg, prev_message[0]
                )
            else:
                prev_message = prev_message[0]
                if cfg.dataset.add_eos_token_to_answer:
                    prev_message += cfg._tokenizer_eos_token

            full_prompt += prev_message
    logger.info(f"Full prompt: {full_prompt}")
    inputs = cfg.dataset.dataset_class.encode(
        tokenizer, full_prompt, cfg.tokenizer.max_length_prompt, "left"
    )
    inputs["prompt_input_ids"] = (
        inputs.pop("input_ids").unsqueeze(0).to(cfg.environment._device)
    )
    inputs["prompt_attention_mask"] = (
        inputs.pop("attention_mask").unsqueeze(0).to(cfg.environment._device)
    )

    def text_cleaner(text: str) -> str:
        return cfg.dataset.dataset_class.clean_output(
            output={"predicted_text": np.array([text])}, prompts=[full_prompt], cfg=cfg
        )["predicted_text"][0]

    if cfg.prediction.num_beams == 1:
        streamer = WaveChatStreamer(tokenizer=tokenizer, q=q, text_cleaner=text_cleaner)
        # Need to start generation in a separate thread, otherwise streaming is blocked
        thread = threading.Thread(
            target=generate,
            kwargs=dict(model=model, inputs=inputs, cfg=cfg, streamer=streamer),
        )
        try:
            thread.start()
        finally:
            while True:
                if streamer.finished:
                    thread.join()
                    predicted_text = streamer.answer
                    break
                await q.sleep(1)
    else:
        # ValueError: `streamer` cannot be used with beam search (yet!).
        # Make sure that `num_beams` is set to 1.
        logger.info("Not streaming output, as it cannot be used with beam search.")
        q.page["experiment/display/chat"].data[-1] = ["...", BOT]
        await q.page.save()
        predicted_answer_ids = generate(model, inputs, cfg)[0]
        predicted_text = tokenizer.decode(
            predicted_answer_ids, skip_special_tokens=True
        )
        predicted_text = text_cleaner(predicted_text)

    logger.info(f"Predicted Answer: {predicted_text}")
    message = [predicted_text, BOT]
    q.client["experiment/display/chat/messages"].append(message)
    q.page["experiment/display/chat"].data[-1] = message

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    q.client["experiment/display/chat/finished"] = True


def generate(model: Model, inputs: Dict, cfg: Any, streamer: TextStreamer = None):
    with torch.cuda.amp.autocast():
        output = model.generate(batch=inputs, cfg=cfg, streamer=streamer).detach().cpu()
    return output


def load_cfg_model_tokenizer(
    experiment_path: str, merge: bool = False, device: str = "cuda:0"
):
    cfg = load_config_yaml(os.path.join(experiment_path, "cfg.yaml"))
    cfg.architecture.pretrained = False
    cfg.architecture.gradient_checkpointing = False
    cfg.environment._device = device.replace("_shard", "")
    cfg.environment._local_rank = 0
    cfg.prediction._visibility["num_history"] = 1

    tokenizer = get_tokenizer(cfg)

    gc.collect()
    torch.cuda.empty_cache()

    if (
        merge
        and cfg.training.lora
        and cfg.architecture.backbone_dtype in ("int4", "int8")
    ):
        logger.info("Loading backbone in float16 for merging LORA weights.")
        cfg.architecture.backbone_dtype = "float16"
        cfg.architecture.pretrained = True

    with torch.device(cfg.environment._device):
        model = cfg.architecture.model_class(cfg)
        cfg.architecture.pretrained_weights = os.path.join(
            experiment_path, "checkpoint.pth"
        )
        load_checkpoint(cfg, model, strict=False)

    if merge and cfg.training.lora:
        # merges the LoRa layers into the base model.
        # This is needed if one wants to use the base model as a standalone model.
        logger.info("Merging LORA layers with base model.")
        model.backbone = model.backbone.merge_and_unload()

    if device == "cpu_shard":
        max_memory = get_balanced_memory(
            model,
        )
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        model = dispatch_model(
            model,
            device_map=device_map,
        )

    model = model.eval()
    model.backbone.use_cache = True

    return cfg, model, tokenizer


async def show_chat_is_running_dialog(q):
    q.page["meta"].dialog = ui.dialog(
        title="Text Generation is streaming.",
        name="chatbot_running_dialog",
        items=[
            ui.text("Please wait till the text generation has stopped."),
        ],
        closable=True,
    )
    await q.page.save()
