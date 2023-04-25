from unittest import mock

import numpy as np

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


def test_clean_output():
    output = {
        "predicted_text": np.array(
            [
                "This is a test",
                "This is a test <stop> This is a test",
                "This is a test <stop2> This is a test",
                "This is a test <stop3> <stop> This is a test",
                "<stop2> <stop> This is a test",
                "This is a test <stop>",
            ]
        )
    }

    cfg = mock.MagicMock()
    cfg.tokenizer._stop_words = ["<stop>", "<stop2>", "<stop3>"]

    predicted_text_clean = CustomDataset.clean_output(
        output=output, prompts=None, cfg=cfg
    )["predicted_text"]
    assert predicted_text_clean == [
        "This is a test",
        "This is a test",
        "This is a test",
        "This is a test",
        "",
        "This is a test",
    ]
