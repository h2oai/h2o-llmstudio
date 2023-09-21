import time
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from llm_studio.app_utils.utils import prepare_default_dataset
from llm_studio.src.datasets.conversation_chain_handler import (
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
            "prompts": [f"prompt{i + 1}"],
            "answers": [f"answer{i + 1}"],
            "systems": [f"system{i + 1}"],
        }


def test_incomplete_chained_samples(cfg, df_short):
    cfg.dataset.limit_chained_samples = False

    handler = ConversationChainHandler(df_short, cfg)
    assert handler.conversation_chain_ids == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    assert len(handler) == 4
    for i in range(4):
        assert handler[i] == {
            "prompts": [f"prompt{j + 1}" for j in range(i + 1)],
            "answers": [f"answer{j + 1}" for j in range(i + 1)],
            "systems": [f"system{j + 1}" for j in range(i + 1)],
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
            "answer": [f"answer{i + 1}" for i in range(13)],
            "system": [f"system{i + 1}" for i in range(13)],
            "prompt": [f"prompt{i + 1}" for i in range(13)],
        }
    )


def test_conversation_chain_handles_nan_parent_ids(df_with_nan, cfg):
    handler = ConversationChainHandler(df_with_nan, cfg)
    assert handler.conversation_chain_ids == [
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


def test_conversation_chain_handler_filters_parent_ids(df_with_nan, cfg):
    for i in range(len(df_with_nan)):
        df_with_nan_1 = df_with_nan.copy()
        df_with_nan_1.loc[i, "parent_id"] = "MISSING"

        handler_1 = ConversationChainHandler(df_with_nan_1, cfg)
        df_with_nan_2 = df_with_nan.copy()
        df_with_nan_2.loc[i, "parent_id"] = "None"

        handler_2 = ConversationChainHandler(df_with_nan_2, cfg)
        assert handler_1.conversation_chain_ids == handler_2.conversation_chain_ids


@pytest.mark.skip("slow test due to downloading oasst")
def test_oasst_conversation_chain_handler(tmp_path):
    """
    Test conversation chain handler on default OASST dataset.
    """

    df = prepare_default_dataset(tmp_path)
    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = "instruction"
    cfg.dataset.answer_column = "output"
    cfg.dataset.parent_id_column = "parent_id"
    cfg.dataset.system_column = "None"

    for limit_chained_samples in [False, True]:
        cfg.dataset.limit_chained_samples = limit_chained_samples
        conversation_chain_handler = ConversationChainHandler(df, cfg=cfg)
        for conversation_chain_ids in conversation_chain_handler.conversation_chain_ids:
            parent_idx = conversation_chain_ids[0]
            for idx in conversation_chain_ids[1:]:
                df.loc[idx, "parent_id"] = parent_idx
                parent_idx = idx

    # explicit example of a chained conversation
    cfg.dataset.limit_chained_samples = True
    conversation_chain_handler = ConversationChainHandler(df, cfg=cfg)

    prompts = [
        "I am building a mechanical keyboard from scratch. I already have the working hardware and am in the process of configuring the firmware. However i find that the qwertz layout gives me wrist pain. I will use the keyboard for writing in english, german and french, and for coding mainly.\nWhat keyboard layout would be best suited for me?",  # noqa: E501
        "what is the advantage of BvoFrak and Bépo, over Azerty that is more common in France.",  # noqa: E501
        "Is there any information on how long it takes a typical person to adopt the BvoFrak or Bépo layouts? I worry that the time investment may be a concern, or that I may  make more mistakes when programming during the transition period.",  # noqa: E501
    ]
    answers = [
        "Generally, it seems that Neo layout may be what you are looking for.\nHere are some keyboard layouts for various use cases: \nFor French: BvoFrak, Bépo\nFor German: Neo, AdNW\nFor English: DHIATENSOR, Dvorak,\nFor programing: Programmer Dvorak, Evolved, Capewell, QGMLWY, Arensito\nNote that while layout may contribute to wrist pain, other factors like the angle of the keyboard, key spacing (vertical, horizontal, and depth), more may also be contributing factors. Learning a new layout takes time and effort, and may make it more difficult to type on other keyboards.",  # noqa: E501
        "BvoFrak and Bépo are alternative French keyboard layouts that aim to improve on the traditional Azerty layout. Some of the claimed advantages include:\n\n- BvoFrak aims to minimize finger movement and optimize typing efficiency. It groups letters by finger positions rather than alphabetic order.\n- Bépo aims to allow for faster typing of common French letters and letter combinations. It also groups letters by finger positions and makes some other rearrangements to commonly used keys.\n- Both BvoFrak and Bépo proponents argue that Azerty is inefficient and optimized for typewriters, not modern keyboards and typing. However, Azerty is deeply entrenched and the alternative layouts have not achieved widespread adoption.\n\nSo the potential advantages are increased typing efficiency and comfort, but the alternative layouts have to overcome the momentum of the established Azerty layout. Whether the claimed advantages hold up in practice for most typists is unclear without much broader adoption of the alternative layouts.",  # noqa: E501
        "The time it takes to learn a new keyboard layout can vary depending on several factors, including the individual's previous typing experience and the amount of time they can devote to practice. However, on average, it can take between 2-4 weeks to become proficient in a new keyboard layout.\n\nDuring the transition period, it's natural to make more mistakes when typing, especially when programming or using special characters. It's important to be patient with yourself and allow yourself time to adjust to the new layout. With practice, the new layout will become more natural and comfortable to use, and the speed and accuracy of your typing will improve.\n\nUltimately, the investment of time and effort required to learn a new keyboard layout is a personal decision, and it's important to consider the potential benefits and drawbacks before making a decision. While there may be some initial challenges, many people find that the improved ergonomics and efficiency of a new keyboard layout make the investment well worth it in the long run.",  # noqa: E501
    ]
    systems = ["", "", ""]
    sample = conversation_chain_handler[
        842
    ]  # 842 == first sample with 3 round conversation
    assert sample["prompts"] == prompts
    assert sample["answers"] == answers
    assert sample["systems"] == systems


@pytest.mark.skip("slow test due to downloading oasst")
def test_oasst_conversation_chain_handler_is_fast(tmp_path):
    df_oasst = prepare_default_dataset(tmp_path)
    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = "instruction"
    cfg.dataset.answer_column = "output"
    cfg.dataset.parent_id_column = "parent_id"
    cfg.dataset.system_column = "None"
    cfg.dataset.limit_chained_samples = True
    dfs = []
    for i in range(50):
        df = df_oasst.copy()
        df["parent_id"] = df["parent_id"].apply(
            lambda x: x + str(i) if x is not None else x
        )
        df["id"] = df["id"].apply(lambda x: x + str(i))
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)

    assert len(df) > 400_000

    t_0 = time.time()
    conversation_chain_handler = ConversationChainHandler(df, cfg)
    _ = [conversation for conversation in conversation_chain_handler]
    t_1 = time.time()
    assert t_1 - t_0 < 10  # shouldn't tkae longer than ~5 seconds
