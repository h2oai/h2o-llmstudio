import pytest
from transformers import AutoTokenizer

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_utils import get_tokenizer, remove_prefix_space


# Token ids produced by Transformers 4.56.1 with add_prefix_space=False.
# LLM Studio encodes conversation components separately, so none may acquire a
# word-boundary token merely because it was encoded as a separate component.
EXPECTED_TOKEN_IDS = {
    "h2oai/h2o-danube2-1.8b-base": {
        "<|prompt|>": [28789, 28766, 14350, 447, 28766, 28767],
        "a": [28708],
        "Hello world": [16230, 1526],
        " leading space": [5374, 2764],
        "Multi\nline\ttext": [10889, 13, 1081, 12, 772],
    },
    "h2oai/llama2-0b-unit-test": {
        "<|prompt|>": [29966, 29989, 14032, 415, 29989, 29958],
        "a": [29874],
        "Hello world": [10994, 3186],
        " leading space": [8236, 2913],
        "Multi\nline\ttext": [15329, 13, 1220, 12, 726],
    },
    "t5-small": {
        "<|prompt|>": [2, 9175, 1409, 1167, 17, 9175, 3155],
        "a": [9],
        "Hello world": [566, 7126, 296],
        " leading space": [1374, 628],
        "Multi\nline\ttext": [31922, 689, 1499],
    },
    "EleutherAI/pythia-70m": {
        "<|prompt|>": [29, 93, 43274, 49651],
        "a": [66],
        "Hello world": [12092, 1533],
        " leading space": [4283, 2317],
        "Multi\nline\ttext": [22495, 187, 1282, 186, 1156],
    },
}

NO_PREFIX_SPACE_BACKBONES = {"EleutherAI/pythia-70m"}


def make_cfg(llm_backbone: str, tokenizer_kwargs: str | None = None):
    tokenizer_cfg = ConfigNLPCausalLMTokenizer(max_length=64)
    if tokenizer_kwargs is not None:
        tokenizer_cfg.tokenizer_kwargs = tokenizer_kwargs
    cfg = ConfigProblemBase(tokenizer=tokenizer_cfg)
    cfg.llm_backbone = llm_backbone
    return cfg


@pytest.fixture(scope="module", params=sorted(EXPECTED_TOKEN_IDS))
def tokenizer_case(request):
    llm_backbone = request.param
    return llm_backbone, get_tokenizer(make_cfg(llm_backbone))


def test_get_tokenizer_preserves_transformers_4_token_ids(tokenizer_case):
    llm_backbone, tokenizer = tokenizer_case

    for value, expected in EXPECTED_TOKEN_IDS[llm_backbone].items():
        actual = tokenizer(value, add_special_tokens=False)["input_ids"]
        assert actual == expected, f"unexpected token ids for {value!r}"


def test_concatenated_chunks_do_not_gain_spaces(tokenizer_case):
    _, tokenizer = tokenizer_case

    def encode(value):
        return tokenizer(value, add_special_tokens=False)["input_ids"]

    chunked = encode("Prompt:") + encode("hello") + encode("Answer:") + encode("world")
    assert tokenizer.decode(chunked) == "Prompt:helloAnswer:world"


def test_exported_tokenizer_matches_training_tokenization(tokenizer_case, tmp_path):
    _, tokenizer = tokenizer_case
    trained_ids = tokenizer("Hello world", add_special_tokens=False)["input_ids"]

    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)

    assert (
        reloaded("Hello world", add_special_tokens=False)["input_ids"]
        == trained_ids
    )


@pytest.mark.parametrize("llm_backbone", sorted(EXPECTED_TOKEN_IDS))
def test_prefix_space_is_only_removed_when_disabled(llm_backbone):
    tokenizer = get_tokenizer(
        make_cfg(llm_backbone, '{"use_fast": true}')
    )

    differs = any(
        tokenizer(value, add_special_tokens=False)["input_ids"] != expected
        for value, expected in EXPECTED_TOKEN_IDS[llm_backbone].items()
    )
    assert differs is (llm_backbone not in NO_PREFIX_SPACE_BACKBONES)

    changed = remove_prefix_space(tokenizer)
    assert changed is (llm_backbone not in NO_PREFIX_SPACE_BACKBONES)
    assert remove_prefix_space(tokenizer) is False
