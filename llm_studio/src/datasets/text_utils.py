import codecs
import json
import logging
import os

from pandas import DataFrame
from tokenizers import normalizers, pre_tokenizers
from transformers import AutoTokenizer, TokenizersBackend

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


def remove_prefix_space(tokenizer) -> bool:
    """Stop a tokenizer from prepending a space to every encoded text.

    LLM Studio encodes conversation components separately and concatenates their
    token ids. Transformers 5 no longer synchronizes add_prefix_space with
    every loaded tokenizer backend, so an ignored value would insert word
    boundaries between system, prompt, and answer components.

    Returns:
        Whether a prefix-space behavior was removed from the backend.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False

    removed = _remove_pre_tokenizer_prefix_space(backend)

    # A Prepend normalizer cannot be reconfigured, so remove it.
    normalizer = backend.normalizer
    if type(normalizer).__name__ == "Prepend":
        backend.normalizer = None
        removed = True
    elif isinstance(normalizer, normalizers.Sequence):
        kept = [
            component
            for component in normalizer
            if type(component).__name__ != "Prepend"
        ]
        if len(kept) < len(normalizer):
            backend.normalizer = normalizers.Sequence(kept)
            removed = True

    return removed


def _remove_pre_tokenizer_prefix_space(backend) -> bool:
    """Reconfigure a tokenizers backend not to add a prefix space."""
    pre_tokenizer = backend.pre_tokenizer
    if pre_tokenizer is None:
        return False

    # Sequence is iterable but exposes no useful public components attribute.
    try:
        components = list(pre_tokenizer)
        is_sequence = True
    except TypeError:
        components = [pre_tokenizer]
        is_sequence = False

    removed = False
    prepending = [
        component
        for component in components
        if getattr(component, "prepend_scheme", "never") != "never"
    ]
    for component in prepending:
        component.prepend_scheme = "never"
        removed = True

    for component in components:
        if getattr(component, "add_prefix_space", False):
            component.add_prefix_space = False
            removed = True

    if prepending:
        # WhitespaceSplit hides whether the first word really had leading
        # whitespace. Let Metaspace see the original whitespace instead.
        components = [
            component
            for component in components
            if type(component).__name__ != "WhitespaceSplit"
        ]

    if is_sequence:
        backend.pre_tokenizer = pre_tokenizers.Sequence(components)

    return removed


def _as_serializable_tokenizer_backend(tokenizer):
    """Preserve an edited backend when model-specific classes rebuild on load."""
    tokenizer_class = type(tokenizer)
    if (
        tokenizer_class is TokenizersBackend
        or "__init__" not in tokenizer_class.__dict__
    ):
        return tokenizer

    kwargs = dict(tokenizer.init_kwargs)
    kwargs.update(tokenizer.special_tokens_map)
    for key in (
        "added_tokens_decoder",
        "is_local",
        "local_files_only",
        "name_or_path",
        "tokenizer_file",
        "vocab_file",
    ):
        kwargs.pop(key, None)

    return TokenizersBackend(
        tokenizer_object=tokenizer.backend_tokenizer,
        **kwargs,
    )


def get_tokenizer(cfg: DefaultConfigProblemBase):
    kwargs = dict(
        revision=cfg.environment.huggingface_branch,
        trust_remote_code=cfg.environment.trust_remote_code,
        token=os.getenv("HF_TOKEN") or None,
    )

    kwargs.update(json.loads(cfg.tokenizer.tokenizer_kwargs.strip()))
    add_prefix_space = kwargs.get("add_prefix_space", True)

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
            tokenizer = AutoTokenizer.from_pretrained(cfg.llm_backbone, **kwargs)
        else:
            raise e

    if not add_prefix_space and remove_prefix_space(tokenizer):
        tokenizer = _as_serializable_tokenizer_backend(tokenizer)
        logger.info(
            "Tokenizer of %s ignored add_prefix_space=False; its backend was "
            "updated and made serialization-safe.",
            cfg.llm_backbone,
        )

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
