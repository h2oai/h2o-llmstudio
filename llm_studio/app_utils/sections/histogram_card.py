from typing import List

import pandas as pd
from h2o_wave import data, ui


def histogram_card(
    x,
    a=0.1,
    b=0.9,
    x_axis_description="text_length",
    histogram_box="first",
    title="Text Length (split by whitespace)",
):
    assert " " not in x_axis_description, (
        "x_axis_description in histogram card must not contain spaces, "
        "as the card would not be rendered."
    )
    df_quantile = compute_quantile_df(x, a, b)
    df_quantile = df_quantile.rename(columns={"length": x_axis_description})
    card = ui.plot_card(
        box=histogram_box,
        title=title,
        data=data(
            fields=df_quantile.columns.tolist(),
            rows=df_quantile.values.tolist(),
            pack=True,
        ),
        plot=ui.plot(
            marks=[
                ui.mark(
                    type="area",
                    x=f"={x_axis_description}",
                    x_title=f"Total samples: {len(x)}",
                    y="=count",
                    y_title="Count",
                    color="=data_type",
                    shape="circle",
                )
            ]
        ),
    )
    return card


def compute_quantile_df(x: List[int], a: float, b: float):
    """
    Compute the quantiles based on the input list x.

    Returns a dataframe with the following columns:
    - length: length of the text
    - count: number of texts with this length
    - data_type: quantile type
     (first (a * 100)% quantile, (a * 100)%-(100 * (1 - b))% quantile,
      last (100 * (1 - b))% quantile)

     Note that quantiles are overlapping on the edges.
    """
    if not x:
        raise ValueError("Input list x is empty")

    if not 0.05 <= a <= b <= 0.95:
        raise ValueError(
            "Values of a and b must be in [0.05, 0.95] "
            "and a should be less than or equal to b"
        )

    x_axis_description = "length"
    df = pd.DataFrame(x, columns=[x_axis_description])
    df["count"] = 1
    df_quantile = (
        df.groupby([x_axis_description])
        .sum()
        .reset_index()
        .sort_values(by=x_axis_description)[[x_axis_description, "count"]]
    )
    sorted_data = sorted(x)
    first_quantile = sorted_data[int(len(sorted_data) * a)]
    last_quantile = sorted_data[-int(len(sorted_data) * (1 - b))]

    df_first = df_quantile.loc[df_quantile[x_axis_description] <= first_quantile].copy()
    df_first["data_type"] = f"first {int(a * 100)}% quantile"
    df_last = df_quantile.loc[df_quantile[x_axis_description] >= last_quantile].copy()
    df_last["data_type"] = f"last {100 - int(b * 100)}% quantile"
    df_quantile["data_type"] = f"{int(a * 100)}%-{int(b * 100)}% quantile"
    middle_quantile_min = max(0, len(df_first) - 1)
    middle_quantile_max = (
        min(len(df_quantile), (len(df_quantile) - len(df_last) - 1)) + 1
    )
    df_quantile = pd.concat(
        [
            df_first,
            df_quantile.loc[middle_quantile_min:middle_quantile_max],
            df_last,
        ]
    )
    return df_quantile
