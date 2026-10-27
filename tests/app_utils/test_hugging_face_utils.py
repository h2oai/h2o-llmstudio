from unittest.mock import MagicMock, create_autospec, patch

import transformers

from llm_studio.app_utils.hugging_face_utils import publish_model_to_hugging_face


@patch("llm_studio.app_utils.hugging_face_utils.get_model_card")
@patch("llm_studio.app_utils.hugging_face_utils.huggingface_hub")
@patch("llm_studio.app_utils.hugging_face_utils.check_disk_space")
@patch("llm_studio.app_utils.hugging_face_utils.load_cfg_model_tokenizer")
def test_publish_model_to_hugging_face(
    mock_load, mock_check_disk_space, mock_hub, mock_model_card, tmp_path
):
    cfg = MagicMock()
    cfg.problem_type = "text_causal_classification_modeling"
    cfg.output_directory = str(tmp_path)
    model = MagicMock()
    model.backbone = create_autospec(transformers.PreTrainedModel, instance=True)
    mock_load.return_value = (cfg, model, MagicMock())

    publish_model_to_hugging_face(
        path_to_experiment=str(tmp_path),
        model_name="test-model",
        user_id="user",
        api_key="key",
        device="cpu",
    )

    model.backbone.push_to_hub.assert_called_once_with(
        repo_id="user/test-model", private=True, commit_message="Upload model"
    )
    assert (tmp_path / "hf.yaml").is_file()
