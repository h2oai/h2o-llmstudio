from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigProblemBase as CausalClassificationConfigProblemBase,
)
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigProblemBase as CausalConfigProblemBase,
)
from llm_studio.python_configs.text_sequence_to_sequence_modeling_config import (
    ConfigProblemBase as Seq2SeqConfigProblemBase,
)
from llm_studio.src.utils.config_utils import (
    NON_GENERATION_PROBLEM_TYPES,
    convert_cfg_base_to_nested_dictionary,
)


def test_from_dict():
    for cfg_class in [
        CausalConfigProblemBase,
        Seq2SeqConfigProblemBase,
        CausalClassificationConfigProblemBase,
    ]:
        cfg = cfg_class()
        cfg_dict = convert_cfg_base_to_nested_dictionary(cfg)
        cfg2 = cfg_class.from_dict(cfg_dict)  # type: ignore
        cfg_dict_2 = convert_cfg_base_to_nested_dictionary(cfg2)
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    assert cfg_dict_2[k][k2] == v2
            assert cfg_dict_2[k] == v


def test_classification_config_is_in_non_generating_problem_types():
    cfg = CausalClassificationConfigProblemBase()
    assert cfg.problem_type in NON_GENERATION_PROBLEM_TYPES
