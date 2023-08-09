from llm_studio.src.utils.export_utils import get_size_str


def test_get_size_atomic_units():
    assert get_size_str(1, input_unit="B") == "1 B"
    assert get_size_str(1024, input_unit="B", output_unit="KB") == "1.0 KB"
    assert get_size_str(1048576, input_unit="B", output_unit="MB") == "1.0 MB"
    assert get_size_str(1073741824, input_unit="B", output_unit="GB") == "1.0 GB"
    assert get_size_str(1099511627776, input_unit="B", output_unit="TB") == "1.0 TB"

    assert get_size_str(1024**5) == "1024.0 TB"


def test_get_size_str_dynamic():
    assert get_size_str(1500, input_unit="B", output_unit="dynamic") == "1.46 KB"
    assert (
        get_size_str(1500, sig_figs=3, input_unit="B", output_unit="dynamic")
        == "1.465 KB"
    )
