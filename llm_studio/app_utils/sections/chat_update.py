import asyncio
import threading
from typing import Optional, Callable, List, Dict, Any

import torch
from h2o_wave import Q
from transformers import TextStreamer, AutoTokenizer

from llm_studio.app_utils.sections.chat import logger
from llm_studio.src.models.text_causal_language_modeling_model import Model

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
        self.loop.create_task(self.update_chat_page())

    async def update_chat_page(self):
        if self.q.args["experiment/display/chat/abort_stream"]:
            self.finished = True
            return

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


async def predict_text(cfg, inputs, model, q, text_cleaner, tokenizer):
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
    return predicted_text


def generate(model: Model, inputs: Dict, cfg: Any, streamer: TextStreamer = None):
    with torch.cuda.amp.autocast():
        output = model.generate(batch=inputs, cfg=cfg, streamer=streamer).detach().cpu()
    return output
