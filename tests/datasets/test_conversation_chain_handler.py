from unittest.mock import MagicMock

import pandas as pd
import pytest

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    ConversationChainHandler,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id4", "x1", "x2", "x3", "x4"],
            "parent_id": ["None", "id1", "id2", "id3", "None", "x1", "x2", "x3"],
            "answer": [
                "answer1",
                "answer2",
                "answer3",
                "answer4",
                "a1",
                "a2",
                "a3",
                "a4",
            ],
            "system": [
                "system1",
                "system2",
                "system3",
                "system4",
                "s1",
                "s2",
                "s3",
                "s4",
            ],
            "prompt": [
                "prompt1",
                "prompt2",
                "prompt3",
                "prompt4",
                "p1",
                "p2",
                "p3",
                "p4",
            ],
        }
    )


@pytest.fixture
def df_short():
    return pd.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id4"],
            "parent_id": ["None", "id1", "id2", "id3"],
            "answer": ["answer1", "answer2", "answer3", "answer4"],
            "system": ["system1", "system2", "system3", "system4"],
            "prompt": ["prompt1", "prompt2", "prompt3", "prompt4"],
        }
    )


@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.dataset.parent_id_column = "parent_id"
    cfg.dataset.system_column = "system"
    cfg.dataset.prompt_column = "prompt"
    cfg.dataset.answer_column = "answer"
    cfg.dataset.limit_chained_samples = True
    return cfg


def test_conversation_chain_handler(cfg, df):
    handler = ConversationChainHandler(df, cfg)

    assert len(handler) == 2, len(handler)

    data = handler[0]
    assert data == {
        "prompts": ["prompt1", "prompt2", "prompt3", "prompt4"],
        "answers": ["answer1", "answer2", "answer3", "answer4"],
        "systems": ["system1", "system2", "system3", "system4"],
    }

    data = handler[1]
    assert data == {
        "prompts": ["p1", "p2", "p3", "p4"],
        "answers": ["a1", "a2", "a3", "a4"],
        "systems": ["s1", "s2", "s3", "s4"],
    }


def test_chained_samples_disabled(df_short, cfg):
    cfg.dataset.limit_chained_samples = False
    cfg.dataset.parent_id_column = "None"

    handler = ConversationChainHandler(df_short, cfg)
    assert len(handler) == 4
    for i in range(4):
        assert handler[i] == {
            "prompts": [f"prompt{i+1}"],
            "answers": [f"answer{i+1}"],
            "systems": [f"system{i+1}"],
        }


def test_incomplete_chained_samples(cfg, df_short):
    cfg.dataset.limit_chained_samples = False

    handler = ConversationChainHandler(df_short, cfg)
    assert handler.conversation_ids_lists == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    assert len(handler) == 4
    for i in range(4):
        assert handler[i] == {
            "prompts": [f"prompt{j+1}" for j in range(i + 1)],
            "answers": [f"answer{j+1}" for j in range(i + 1)],
            "systems": [f"system{j+1}" for j in range(i + 1)],
        }


def test_get_conversation_ids():
    # test the get_conversation_ids method - normal case
    conv_ids = ConversationChainHandler.get_conversation_ids(
        {"id2": "id1", "id3": "id2", "id4": "id3"}, "id4"
    )
    assert conv_ids == ["id1", "id2", "id3", "id4"]

    # test the get_conversation_ids method - circular case, should raise ValueError
    with pytest.raises(ValueError):
        ConversationChainHandler.get_conversation_ids(
            {"id1": "id4", "id2": "id1", "id3": "id2", "id4": "id3"}, "id4"
        )


@pytest.fixture
def df_with_nan():
    # mapping is
    # a1 -> " " -> -inf -> 1234567890 -> "1234567890" -> "x1" -> 1 -> 2 -> 3 -> 4
    # a2
    # a3
    # a4
    return pd.DataFrame(
        {
            "id": [
                "a1",
                " ",
                "-inf",
                1234567890,
                "1234567890",
                "x1",
                1,
                2,
                3.0,
                4.0,
                "a2",
                "a3",
                "a4",
            ],
            "parent_id": [
                " ",  # valid
                "-inf",  # valid
                1234567890,  # valid
                "1234567890",  # valid, different type
                "x1",  # valid
                1.0,  # valid, needs to map to the int value
                2.0,  # valid, needs to map to the int value
                3,  # valid, needs to map to the float value
                4,  # valid, needs to map to the float value
                float("nan"),  # should be ignored
                "None",  # should be ignored
                None,  # should be ignored
                float("inf"),  # should be ignored
            ],
            "answer": [f"answer{i+1}" for i in range(13)],
            "system": [f"system{i+1}" for i in range(13)],
            "prompt": [f"prompt{i+1}" for i in range(13)],
        }
    )


def test_conversation_chain_handles_nan_parent_ids(df_with_nan, cfg):
    handler = ConversationChainHandler(df_with_nan, cfg)
    assert handler.conversation_ids_lists == [
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [10],
        [11],
        [12],
    ]
    assert len(handler) == 4
    assert handler[0] == {
        "prompts": [
            "prompt10",
            "prompt9",
            "prompt8",
            "prompt7",
            "prompt6",
            "prompt5",
            "prompt4",
            "prompt3",
            "prompt2",
            "prompt1",
        ],
        "answers": [
            "answer10",
            "answer9",
            "answer8",
            "answer7",
            "answer6",
            "answer5",
            "answer4",
            "answer3",
            "answer2",
            "answer1",
        ],
        "systems": [
            "system10",
            "system9",
            "system8",
            "system7",
            "system6",
            "system5",
            "system4",
            "system3",
            "system2",
            "system1",
        ],
    }
    assert handler[1] == {
        "prompts": ["prompt11"],
        "answers": ["answer11"],
        "systems": ["system11"],
    }
    assert handler[2] == {
        "prompts": ["prompt12"],
        "answers": ["answer12"],
        "systems": ["system12"],
    }
    assert handler[3] == {
        "prompts": ["prompt13"],
        "answers": ["answer13"],
        "systems": ["system13"],
    }
