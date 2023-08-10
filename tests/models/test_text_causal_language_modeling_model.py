import torch

from llm_studio.src.models.text_base_model import TokenStoppingCriteria


def test_token_stopping_criteria():
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([0, 1, 2, 8]), prompt_input_ids_len=4
    )

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ]
    ).long()

    # prompt input len is 4, so generated ids of last sample of the batch are
    # [9, 10, 11, 12, 13, 14], do not trigger stopping criteria
    assert not token_stopping_criteria(input_ids=input_ids, scores=None)

    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([6]), prompt_input_ids_len=0
    )

    # first item reads [ 0,  1,  2,  3,  4,  5], so do not trigger stopping criteria
    assert not token_stopping_criteria(input_ids=input_ids[:, :6], scores=None)
    assert token_stopping_criteria(input_ids=input_ids[:, :7], scores=None)

    # Test stopping criteria with compound tokens
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([[6, 7]]), prompt_input_ids_len=0
    )

    assert not token_stopping_criteria(input_ids=input_ids[:, :6], scores=None)
    assert not token_stopping_criteria(input_ids=input_ids[:, :7], scores=None)
    assert token_stopping_criteria(input_ids=input_ids[:, :8], scores=None)

    # Test stopping criteria with stop word ids being longer than generated text
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]),
        prompt_input_ids_len=0,
    )

    assert not token_stopping_criteria(input_ids=input_ids, scores=None)
