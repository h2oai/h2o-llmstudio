import numpy as np
import pandas as pd
from h2o_wave import data, ui


class HistogramCard:
    def __init__(
        self,
        column="text_length",
        histogram_box="first",
        title="Text Length (split by whitespace)",
        text_description="",
        unit="words",
        unit_singular=None,
        a=0.1,
        b=0.9,
    ):
        self.column = column
        self.histogram_box = histogram_box
        self.title = title
        self.text_description = text_description
        self.unit = unit
        self.unit_singular = unit_singular or unit[:-1]
        self.text_length_proposal = None
        self.a = a
        self.b = b

    def __call__(self, q, df):
        df = df.copy()

        df["count"] = 1
        df_agg = (
            df.groupby([self.column])
            .sum()
            .reset_index()
            .sort_values(by=self.column)[[self.column, "count"]]
        )
        df_agg["count"] = df_agg["count"]
        first_quantile = np.quantile(df_agg[self.column], self.a)
        last_quantile = np.quantile(df_agg[self.column], self.b)

        df_first = df_agg.loc[df_agg[self.column] <= first_quantile].copy()

        df_first["data_type"] = f"first {int(self.a * 100)}% quantile"

        df_last = df_agg.loc[df_agg[self.column] >= last_quantile].copy()
        df_last["data_type"] = f"last {100 - int(self.b * 100)}% quantile"

        df_agg["data_type"] = f"{int(self.a * 100)}%-{int(self.b * 100)}% quantile"

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
            box=self.histogram_box,
            title=self.title,
            data=data(
                fields=df_agg.columns.tolist(), rows=df_agg.values.tolist(), pack=True
            ),
            plot=ui.plot(
                marks=[
                    ui.mark(
                        type="area",
                        x=f"={self.column}",
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
