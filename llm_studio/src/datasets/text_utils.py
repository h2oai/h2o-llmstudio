import codecs
import json
import logging
import os

from pandas import DataFrame
from transformers import AutoTokenizer

from llm_studio.python_configs.base import DefaultConfigProblemBase

logger = logging.getLogger(__name__)


def get_texts(df: DataFrame, cfg: DefaultConfigProblemBase):
    if isinstance(cfg.dataset.prompt_column, str):
        # single column dataset
        texts = df[cfg.dataset.prompt_column].astype(str)
        texts = texts.values
    else:
        # multi-column dataset - prepend (if necessary) and join
        columns = list(cfg.dataset.prompt_column)

        for column in columns:
            df[column] = df[column].astype(str)

        join_str = codecs.decode(cfg.dataset.prompt_column_separator, "unicode_escape")

        texts = df[columns].astype(str)
        texts = texts.apply(lambda x: join_str.join(x), axis=1).values

    return texts


def get_tokenizer(cfg: DefaultConfigProblemBase):

    kwargs = dict(
        revision=cfg.environment.huggingface_branch,
        trust_remote_code=cfg.environment.trust_remote_code,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )

    # We will be able to remove this after
    # https://github.com/huggingface/transformers/pull/30964
    tokenizer_class = AutoTokenizer.from_pretrained(
        cfg.llm_backbone, **kwargs
    ).__class__
    if tokenizer_class.__name__ in ["LlamaTokenizer", "LlamaTokenizerFast"]:
        kwargs["from_slow"] = True

    kwargs.update(json.loads(cfg.tokenizer.tokenizer_kwargs.strip()))

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_backbone, **kwargs)
    except TypeError as e:
        error_message = str(e)
        if "token" in error_message:
            # TypeError: RWForCausalLM.__init__() got
            # an unexpected keyword argument 'token'
            kwargs.pop("token")
            tokenizer = AutoTokenizer.from_pretrained(cfg.llm_backbone, **kwargs)
        elif "not a string" in error_message:
            # https://github.com/h2oai/h2o-llmstudio/issues/623
            kwargs.pop("add_prefix_space", None)
            kwargs.pop("from_slow", None)
            tokenizer = AutoTokenizer.from_pretrained(cfg.llm_backbone, **kwargs)
        else:
            raise e

    tokenizer.padding_side = getattr(
        cfg.tokenizer, "_padding_side", tokenizer.padding_side
    )

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    # if the eos token is an empty string, we assign it to a token
    if tokenizer.eos_token == "":
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.eos_token

    if tokenizer.unk_token_id is not None:
        cfg.tokenizer._tokenizer_mask_token_id = tokenizer.unk_token_id
    elif tokenizer.mask_token_id is not None:
        cfg.tokenizer._tokenizer_mask_token_id = tokenizer.mask_token_id
    elif tokenizer.pad_token_id is not None:
        cfg.tokenizer._tokenizer_mask_token_id = tokenizer.pad_token_id
    else:
        # setting the mask token id to the last token in the vocabulary
        # this usually is a safe choice and mostly refers to eos token
        cfg.tokenizer._tokenizer_mask_token_id = len(tokenizer) - 1

    cfg.tokenizer._tokenizer_eos_token = tokenizer.eos_token

    if hasattr(cfg.prediction, "stop_tokens"):
        set_stop_token_ids(cfg, tokenizer)
    cfg.tokenizer._vocab_length = len(tokenizer)

    return tokenizer


def set_stop_token_ids(cfg, tokenizer):
    cfg.tokenizer._stop_words = list(
        filter(None, cfg.prediction.stop_tokens.split(","))
    )
    for stop_word in [
        cfg.dataset.text_system_start,
        cfg.dataset.text_prompt_start,
        cfg.dataset.text_answer_separator,
    ]:
        stop_word = codecs.decode(stop_word, "unicode_escape").strip()
        if (
            stop_word != ""
            and cfg.tokenizer.add_prompt_answer_tokens
            and (stop_word not in tokenizer.get_vocab())
        ):
            tokenizer.add_tokens([stop_word])
        cfg.tokenizer._stop_words.append(stop_word)
    cfg.tokenizer._stop_words = [
        stop_word for stop_word in cfg.tokenizer._stop_words if stop_word != ""
    ]
    cfg.tokenizer._stop_words_ids = []
    for stop_word in set(cfg.tokenizer._stop_words):
        cfg.tokenizer._stop_words_ids.append(
            tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0]
        )
    logger.info(f"Stop token ids: {cfg.tokenizer._stop_words_ids}")
