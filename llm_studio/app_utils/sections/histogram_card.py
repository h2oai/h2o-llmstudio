import numpy as np
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
    df = pd.DataFrame(x, columns=[x_axis_description])

    df["count"] = 1
    df_agg = (
        df.groupby([x_axis_description])
        .sum()
        .reset_index()
        .sort_values(by=x_axis_description)[[x_axis_description, "count"]]
    )
    df_agg["count"] = df_agg["count"]
    first_quantile = np.quantile(df_agg[x_axis_description], a)
    last_quantile = np.quantile(df_agg[x_axis_description], b)

    df_first = df_agg.loc[df_agg[x_axis_description] <= first_quantile].copy()

    df_first["data_type"] = f"first {int(a * 100)}% quantile"

    df_last = df_agg.loc[df_agg[x_axis_description] >= last_quantile].copy()
    df_last["data_type"] = f"last {100 - int(b * 100)}% quantile"

    df_agg["data_type"] = f"{int(a * 100)}%-{int(b * 100)}% quantile"

    # need to have an overlap between (df_first, df_agg) and (df_agg, df_last)
    # otherwise graphics gets unfilled gaps
    df_agg = pd.concat(
        [
            df_first,
            df_agg.loc[
                max(0, len(df_first) - 1) : min(
                    len(df_agg), (len(df_agg) - len(df_last))
                )
            ],
            df_last,
        ]
    )

    card = ui.plot_card(
        box=histogram_box,
        title=title,
        data=data(
            fields=df_agg.columns.tolist(), rows=df_agg.values.tolist(), pack=True
        ),
        plot=ui.plot(
            marks=[
                ui.mark(
                    type="area",
                    x=f"={x_axis_description}",
                    x_title=f"Total samples: {len(df)}",
                    y="=count",
                    y_title="Count",
                    color="=data_type",
                    shape="circle",
                )
            ]
        ),
    )
    return card
