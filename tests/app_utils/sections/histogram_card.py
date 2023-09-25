import pandas as pd

from llm_studio.app_utils.sections.histogram_card import compute_quantiles


def test_compute_histogram_df():
    # Define inputs & expected output
    x = [1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3
    a = 0.25
    b = 0.75
    expected_output = pd.DataFrame(
        {
            "length": [1, 2, 3, 4, 5],
            "count": [4, 3, 3, 3, 3],
            "data_type": [
                "first 25% quantile",
                "25%-75% quantile",
                "25%-75% quantile",
                "last 25% quantile",
                "last 25% quantile",
            ],
        }
    )

    # Call function & compute output
    df_quantile = compute_quantiles(x, a, b)

    assert df_quantile["count"].sum() == len(x)
    assert df_quantile["data_type"].nunique() == 3

    assert df_quantile.loc[df_quantile["data_type"] == "first 25% quantile", "count"].sum() == 7
    assert df_quantile.loc[df_quantile["data_type"] == "25%-75% quantile", "count"].sum() == 3
    assert df_quantile.loc[df_quantile["data_type"] == "last 25% quantile", "count"].sum() == 6

    x_2 = [[number] * count for number, count in zip(df_quantile["length"], df_quantile["count"])]
    x_2 = [item for sublist in x_2 for item in sublist]
    assert sorted(x) == x_2