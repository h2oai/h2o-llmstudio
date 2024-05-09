import os

import pytest
from jinja2.exceptions import TemplateError

from llm_studio.app_utils.hugging_face_utils import get_chat_template
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.config_utils import load_config_yaml


def test_chat_template_no_system_prompt():

    test_directory = os.path.abspath(os.path.dirname(__file__))
    cfg_path = os.path.join(test_directory, "../test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)
    cfg.dataset.text_prompt_start = "<|prompt|>"
    cfg.dataset.text_answer_separator = "<|answer|>"
    cfg.dataset.add_eos_token_to_prompt = True
    cfg.dataset.add_eos_token_to_answer = True

    tokenizer = get_tokenizer(cfg)
    tokenizer.eos_token = '</s>'
    tokenizer.chat_template = get_chat_template(cfg)

    chat = [
        {"role": "user", "content": "[user prompt]"},
        {"role": "assistant", "content": "[assistant response]"},
        {"role": "user", "content": "[user prompt2]"},
    ]

    input = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = "<|prompt|>[user prompt]</s><|answer|>[assistant response]</s><|prompt|>[user prompt2]</s><|answer|>"  # noqa
    assert input == expected

    # raise error test
    for chat in [
        [
            {"role": "system", "content": "[system prompt]"},
            {"role": "user", "content": "[user prompt]"},
            {"role": "assistant", "content": "[assistant response]"},
            {"role": "user", "content": "[user prompt2]"},
        ],
        [
            {"role": "user", "content": "[user prompt]"},
            {"role": "assistant", "content": "[assistant response]"},
            {"role": "user", "content": "[user prompt2]"},
            {"role": "system", "content": "[system prompt]"},
        ],
    ]:
        with pytest.raises(TemplateError) as e:
            input = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        expected = "System role not supported"
        assert expected in str(e.value)

    # raise error test
    for chat in [
        [
            {"role": "user", "content": "[user prompt]"},
            {"role": "user", "content": "[user prompt2]"},
            {"role": "assistant", "content": "[assistant response]"},
        ],
        [
            {"role": "assistant", "content": "[assistant response]"},
            {"role": "user", "content": "[user prompt]"},
            {"role": "assistant", "content": "[assistant response]"},
        ],
        [
            {"role": "assistant", "content": "[assistant response]"},
            {"role": "assistant", "content": "[user prompt]"},
            {"role": "user", "content": "[assistant response]"},
        ],
    ]:
        with pytest.raises(TemplateError) as e:
            input = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        expected = "Conversation roles must alternate user/assistant/user/assistant/..."
        assert expected in str(e.value)


def test_chat_template_with_system_prompt():

    test_directory = os.path.abspath(os.path.dirname(__file__))
    cfg_path = os.path.join(test_directory, "../test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)
    cfg.dataset.text_prompt_start = "<|prompt|>"
    cfg.dataset.text_answer_separator = "<|answer|>"
    cfg.dataset.text_system_start = "<|system|>"
    cfg.dataset.add_eos_token_to_prompt = True
    cfg.dataset.add_eos_token_to_answer = True
    cfg.dataset.add_eos_token_to_system = True
    cfg.dataset.system_column = "system"

    tokenizer = get_tokenizer(cfg)
    tokenizer.eos_token = '</s>'
    tokenizer.chat_template = get_chat_template(cfg)

    chat = [
        {"role": "system", "content": "[system prompt]"},
        {"role": "user", "content": "[user prompt]"},
        {"role": "assistant", "content": "[assistant response]"},
        {"role": "user", "content": "[user prompt2]"},
    ]

    input = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = "<|system|>[system prompt]</s><|prompt|>[user prompt]</s><|answer|>[assistant response]</s><|prompt|>[user prompt2]</s><|answer|>"  # noqa
    assert input == expected

    # works w/o system prompt as well
    chat = [
        {"role": "user", "content": "[user prompt]"},
        {"role": "assistant", "content": "[assistant response]"},
        {"role": "user", "content": "[user prompt2]"},
    ]

    input = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = "<|prompt|>[user prompt]</s><|answer|>[assistant response]</s><|prompt|>[user prompt2]</s><|answer|>"  # noqa
    assert input == expected

    # raise error test
    for chat in [
        [
            {"role": "user", "content": "[user prompt]"},
            {"role": "system", "content": "[system prompt]"},
            {"role": "user", "content": "[user prompt2]"},
            {"role": "assistant", "content": "[assistant response]"},
        ],
        [
            {"role": "system", "content": "[system prompt]"},
            {"role": "user", "content": "[user prompt]"},
            {"role": "user", "content": "[user prompt2]"},
            {"role": "assistant", "content": "[assistant response]"},
        ],
        [
            {"role": "user", "content": "[user prompt]"},
            {"role": "user", "content": "[user prompt2]"},
            {"role": "assistant", "content": "[assistant response]"},
        ],
        [
            {"role": "user", "content": "[user prompt]"},
            {"role": "assistant", "content": "[assistant response]"},
            {"role": "assistant", "content": "[assistant response2]"},
        ],
    ]:
        with pytest.raises(TemplateError) as e:
            input = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        expected = "Conversation roles must alternate system(optional)/user/assistant/user/assistant/..."  # noqa
        assert expected in str(e.value)


def test_chat_template_no_eos_token():

    test_directory = os.path.abspath(os.path.dirname(__file__))
    cfg_path = os.path.join(test_directory, "../test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)
    cfg.dataset.text_prompt_start = "<|prompt|>"
    cfg.dataset.text_answer_separator = "<|answer|>"
    cfg.dataset.text_system_start = "<|system|>"
    cfg.dataset.add_eos_token_to_system = False
    cfg.dataset.add_eos_token_to_prompt = False
    cfg.dataset.add_eos_token_to_answer = False
    cfg.dataset.system_column = "system"

    tokenizer = get_tokenizer(cfg)
    tokenizer.chat_template = get_chat_template(cfg)

    chat = [
        {"role": "system", "content": "[system prompt]"},
        {"role": "user", "content": "[user prompt]"},
        {"role": "assistant", "content": "[assistant response]"},
        {"role": "user", "content": "[user prompt2]"},
    ]

    input = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = "<|system|>[system prompt]<|prompt|>[user prompt]<|answer|>[assistant response]<|prompt|>[user prompt2]<|answer|>" # noqa
    assert input == expected


def test_chat_template_no_special_token():

    test_directory = os.path.abspath(os.path.dirname(__file__))
    cfg_path = os.path.join(test_directory, "../test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)
    cfg.dataset.system_column = "system"
    cfg.dataset.text_system_start = ""
    cfg.dataset.text_prompt_start = ""
    cfg.dataset.text_answer_separator = ""
    cfg.dataset.add_eos_token_to_system = False
    cfg.dataset.add_eos_token_to_prompt = False
    cfg.dataset.add_eos_token_to_answer = False

    tokenizer = get_tokenizer(cfg)
    tokenizer.chat_template = get_chat_template(cfg)

    chat = [
        {"role": "system", "content": "[system prompt]"},
        {"role": "user", "content": "[user prompt]"},
        {"role": "assistant", "content": "[assistant response]"},
        {"role": "user", "content": "[user prompt2]"},
    ]

    input = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = expected = "[system prompt][user prompt][assistant response][user prompt2]"
    assert input == expected
