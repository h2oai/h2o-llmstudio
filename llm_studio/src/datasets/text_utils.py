import codecs
import logging
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def get_texts(df, cfg, separator=None):
    if isinstance(cfg.dataset.prompt_column, str):
        # single column dataset
        texts = df[cfg.dataset.prompt_column].astype(str)
        texts = texts.values
    else:
        # multi-column dataset - prepend (if necessary) and join
        columns = list(cfg.dataset.prompt_column)

        for column in columns:
            df[column] = df[column].astype(str)

        if separator is None:
            if hasattr(cfg.dataset, "separator") and len(cfg.dataset.separator):
                separator = cfg.dataset.separator
            else:
                separator = getattr(cfg, "_tokenizer_sep_token", "<SEPARATOR>")

        join_str = f" {separator} "
        texts = df[columns].astype(str)
        texts = texts.apply(lambda x: join_str.join(x), axis=1).values

    return texts


def get_tokenizer(cfg: Any):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm_backbone,
        add_prefix_space=cfg.tokenizer.add_prefix_space,
        use_fast=True,
        trust_remote_code=cfg.environment.trust_remote_code,
    )
    tokenizer.padding_side = getattr(
        cfg.tokenizer, "_padding_side", tokenizer.padding_side
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = " "
    if hasattr(cfg.dataset, "separator") and len(cfg.dataset.separator):
        cfg._tokenizer_sep_token = cfg.dataset.separator
    else:
        cfg._tokenizer_sep_token = tokenizer.sep_token
    if tokenizer.unk_token_id is not None:
        cfg._tokenizer_mask_token_id = tokenizer.unk_token_id
    elif tokenizer.mask_token_id is not None:
        cfg._tokenizer_mask_token_id = tokenizer.mask_token_id
    elif tokenizer.mask_token_id is not None:
        cfg._tokenizer_mask_token_id = tokenizer.pad_token_id
    else:
        # setting the mask token id to the last token in the vocabulary
        # this usually is a safe choice and mostly refers to eos token
        cfg._tokenizer_mask_token_id = len(tokenizer) - 1

    cfg._tokenizer_eos_token = tokenizer.eos_token

    cfg.tokenizer._stop_words = list(
        filter(None, cfg.prediction.stop_tokens.split(","))
    )

    for stop_word in [cfg.dataset.text_prompt_start, cfg.dataset.text_answer_separator]:
        stop_word = codecs.decode(stop_word, "unicode_escape").strip()
        if stop_word != "" and stop_word not in tokenizer.get_vocab():
                tokenizer.add_tokens([stop_word])
            cfg.tokenizer._stop_words.append(stop_word)

    cfg.tokenizer._vocab_length = len(tokenizer.vocab)

    cfg.tokenizer._stop_words_ids = []
    for stop_word in set(cfg.tokenizer._stop_words):
        cfg.tokenizer._stop_words_ids.append(
            tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0]
        )

    return tokenizer
