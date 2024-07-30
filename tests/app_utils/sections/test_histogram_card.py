import random

import numpy as np

from llm_studio.app_utils.sections.histogram_card import compute_quantile_df


def test_quantiles_are_computed_correctly() -> None:
    for _ in range(5):
        data: list[int] = np.random.randint(0, 1000, 100_000).tolist()
        a = round(random.uniform(0.05, 0.95), 2)
        b = round(random.uniform(a, 0.95), 2)
        a, b = min(a, b), max(a, b)

        df_quantile = compute_quantile_df(data, a, b)
        first = df_quantile[
            df_quantile["data_type"] == f"first {int(a * 100)}% quantile"
        ]
        last = df_quantile[
            df_quantile["data_type"] == f"last {100 - int(b * 100)}% quantile"
        ]
        sorted_data = sorted(data)
        # use -1 and +1 to account for rounding errors
        expected_first_quantile_range = sorted_data[
            int(len(sorted_data) * a) - 1 : int(len(sorted_data) * a) + 1
        ]
        expected_last_quantile_range = sorted_data[
            -int(len(sorted_data) * (1 - b)) - 1 : -int(len(sorted_data) * (1 - b)) + 1
        ]

        assert first["length"].values[-1] in expected_first_quantile_range
        assert last["length"].values[0] in expected_last_quantile_range
