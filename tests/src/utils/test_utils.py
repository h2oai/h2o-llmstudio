import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigDPODataset,
    ConfigProblemBase,
)
from llm_studio.src.utils.utils import (
    PatchedAttribute,
    add_file_to_zip,
    create_symlinks_in_parent_folder,
    kill_child_processes,
)


@patch("psutil.Process")
def test_kill_child_processes(mock_process):
    mock_process.return_value.status.return_value = "running"
    mock_child_1 = MagicMock()
    mock_child_2 = MagicMock()
    mock_process.return_value.children.return_value = [mock_child_1, mock_child_2]

    assert kill_child_processes(1234)
    mock_child_1.kill.assert_called_once()
    mock_child_2.kill.assert_called_once()


def test_add_file_to_zip():
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"Test content")
        temp_file.flush()

        with tempfile.NamedTemporaryFile(suffix=".zip") as temp_zip:
            with zipfile.ZipFile(temp_zip.name, "w") as zf:
                add_file_to_zip(zf, temp_file.name)

            with zipfile.ZipFile(temp_zip.name, "r") as zf:
                assert os.path.basename(temp_file.name) in zf.namelist()


def test_patched_attribute():
    cfg = ConfigProblemBase(
        dataset=ConfigDPODataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
            parent_id_column="None",
        )
    )
    with PatchedAttribute(cfg.dataset, "answer_column", "chosen_response"):
        assert cfg.dataset.answer_column == "chosen_response"

    with PatchedAttribute(cfg.dataset, "answer_column", "new_answer_column"):
        assert cfg.dataset.answer_column == "new_answer_column"

    assert cfg.dataset.answer_column == "answer_column"

    with PatchedAttribute(cfg.dataset, "new_property", "new_value"):
        assert cfg.dataset.new_property == "new_value"  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        cfg.dataset.new_property  # type: ignore[attr-defined]


def test_create_symlinks_in_parent_folder():
    with tempfile.TemporaryDirectory() as temp_dir:
        sub_dir = os.path.join(temp_dir, "sub")
        os.mkdir(sub_dir)

        # Create some files in the subdirectory
        for i in range(3):
            with open(os.path.join(sub_dir, f"file{i}.txt"), "w") as f:
                f.write(f"Content {i}")

        create_symlinks_in_parent_folder(sub_dir)

        # Check if symlinks were created in the parent directory
        for i in range(3):
            symlink_path = os.path.join(temp_dir, f"file{i}.txt")
            assert os.path.islink(symlink_path)
            assert os.readlink(symlink_path) == os.path.join(sub_dir, f"file{i}.txt")
