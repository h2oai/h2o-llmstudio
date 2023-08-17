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
    cfg.dataset.parent_id_column = "parent_id"
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
