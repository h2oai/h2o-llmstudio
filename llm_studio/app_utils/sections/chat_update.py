import asyncio
import gc
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from h2o_wave import Q, ui
from transformers import AutoTokenizer, TextStreamer

from llm_studio.app_utils.utils import parse_ui_elements
from llm_studio.src.models.text_causal_language_modeling_model import Model

logger = logging.getLogger(__name__)

__all__ = ["chat_update", "is_app_blocked_while_streaming"]

USER = True
BOT = False


@torch.inference_mode(mode=True)
async def chat_update(q: Q) -> None:
    """
    Update the chatbot with the new message.
    """
    q.client["experiment/display/chat/finished"] = False
    try:
        await update_chat_window(q)
    finally:
        q.client["experiment/display/chat/finished"] = True


async def update_chat_window(q):
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

    predicted_text = await answer_chat(q)
    logger.info(f"Predicted Answer: {predicted_text}")
    message = [predicted_text, BOT]
    q.client["experiment/display/chat/messages"].append(message)
    q.page["experiment/display/chat"].data[-1] = message


class WaveChatStreamer(TextStreamer):
    """
    Utility class that updates the chabot card in a streaming fashion
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        q: Q,
        text_cleaner: Optional[Callable] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt=True, **decode_kwargs)
        self.text_cleaner = text_cleaner
        self.words_predicted_answer: List[str] = []
        self.q = q
        self.loop = asyncio.get_event_loop()
        self.finished = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.words_predicted_answer += [text]
        self.loop.create_task(self.update_chat_page())

    async def update_chat_page(self):
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


async def answer_chat(q: Q) -> str:
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
            output={"predicted_text": np.array([text])}, cfg=cfg
        )["predicted_text"][0]

    if cfg.prediction.num_beams == 1:
        streamer = WaveChatStreamer(tokenizer=tokenizer, q=q, text_cleaner=text_cleaner)
        q.client["chat_streamer"] = streamer
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
                    q.client["chat_streamer"] = None
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

    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return predicted_text


def generate(model: Model, inputs: Dict, cfg: Any, streamer: TextStreamer = None):
    with torch.cuda.amp.autocast():
        output = model.generate(batch=inputs, cfg=cfg, streamer=streamer).detach().cpu()
    return output


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


async def show_stream_is_aborted_dialog(q):
    q.page["meta"].dialog = ui.dialog(
        title="Text Generation will be stopped.",
        name="chatbot_stopping_dialog",
        items=[
            ui.text("Please wait"),
        ],
        closable=False,
    )
    await q.page.save()


async def is_app_blocked_while_streaming(q: Q):
    """
    Check whether the app is blocked with current answer generation.
    """
    if (
        q.args["experiment/display/chat/abort_stream"]
        and q.client["chat_streamer"] is not None
    ):
        # User clicks abort button while the chat is currently streaming
        # - Set the streamer to finished state
        # - wait till the text generation thread has stopped, i.e.
        #   q.client["chat_streamer"] is None.
        #   In that case, chat_update function will finish
        #   and "experiment/display/chat/finished" will be set to True.

        logger.info("Stopping Chat Stream")
        q.client["chat_streamer"].finished = True
        await show_stream_is_aborted_dialog(q)
        await q.page.save()
        for _ in range(20):  # don't wait longer than 10 seconds
            await q.sleep(0.5)
            if q.client["chat_streamer"] is None:
                q.page["meta"].dialog = None
                await q.page.save()
                return True
        else:
            logger.warning("Could not terminate stream")
            return True

    elif q.client["experiment/display/chat/finished"] is False:
        await show_chat_is_running_dialog(q)
        return True
    return False
