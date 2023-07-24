import html
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import numpy as np
from bokeh.embed import file_html
from bokeh.embed.standalone import ModelLike
from bokeh.resources import Resources
from bokeh.themes import Theme
from bs4 import BeautifulSoup

BOKEH_THEME = Theme(
    json={
        "attrs": {
            "Figure": {
                "background_fill_alpha": 0.0,
                "outline_line_color": None,
                "border_fill_color": None,
            },
            "Title": {"text_color": "#CDDC39"},
            "Legend": {"background_fill_alpha": 0.0},
        }
    }
)

BOKEH_TEMPLATE = """
{% block postamble %}
<style>
/* Make the Tab and Panel Glyphs work with Wave themes. */
.bk-root .bk-headers-wrapper {
    border-bottom: 1px solid var(--themePrimary) !important;
}

.bk-root .bk-tabs-header .bk-tab {
    color: var(--text) !important;
}

.bk-root .bk-tabs-header .bk-tab:hover {
    background-color:  var(--neutralLighter);
}

.bk-root .bk-tabs-header .bk-tab.bk-active {
    background-color: var(--card) !important;
    border-color: var(--themePrimary);
}

/* Show only the first tooltip if multiple glyphs are on top of each other */
/* see https://stackoverflow.com/a/64544102. */
div.bk-tooltip.bk-right>div.bk>div:not(:first-child) {
    display:none !important;
}
div.bk-tooltip.bk-left>div.bk>div:not(:first-child) {
    display:none !important;
}
</style>
{% endblock %}
"""


def to_html(model: ModelLike) -> str:
    """
    Converts the given Bokeh Model to HTML.

    Args:
        model: The model to convert.

    Returns:
        String containing HTML.
    """

    markup = file_html(model, Resources(), theme=BOKEH_THEME, template=BOKEH_TEMPLATE)

    # Wave uses Reacts `dangerouslySetInnerHTML` to insert HTML:
    # https://github.com/h2oai/wave/blob/f64987f3f211dd3e46cc880644eba31a7df52cb9/ui/src/markup.tsx#L64-L66
    #
    # But `dangerouslySetInnerHTML` does not run Javascript code by itself.
    # However, we can use an XSS attack to execute code:
    # https://reactjs.org/docs/dom-elements.html#dangerouslysetinnerhtml.
    #
    # So below we extract the script tags from the generated HTML, and insert
    # the Javascript code back using the `onerror` XSS exploit:
    # https://owasp.org/www-community/xss-filter-evasion-cheatsheet
    #
    # If there are issues with this implementation, there might be a cleaner way of
    # solving this using Waves `ui.inline_script`.

    soup = BeautifulSoup(markup, "html.parser")

    scripts = [
        s.extract().string
        for s in soup.select("script")
        if s["type"] == "text/javascript"
    ]

    scripts = "\n".join(
        f'<img src onerror="{html.escape(script)}"\\>'
        for script in scripts
        if script is not None
    )

    return f"""
        <div>
            {soup}
            {scripts}
        </div>
    """


def get_line_separator_html():
    return (
        "<div style='height: 1px; width: 100%; margin: 1em 0; "
        "background-color: white; background-color: var(--text);'></div>"
    )


def text_to_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


@dataclass
class PlotData:
    """
    Data to plot.

    Args:
        data: the data to plot:
            - a base64 encoded PNG if `encoding` is `png`.
            - a string in HTML if `encoding` is `html`.
        encoding: the encoding of the data, one of `png` or `html`.
    """

    data: str
    encoding: str


def format_to_html(color, word, opacity):
    return '<mark style="background-color: {color}; opacity:{opacity}; \
    line-height:1.75"><font color="white"> {word}\
                            </font></mark>'.format(
        color=color, word=html.escape(word), opacity=opacity
    )


def decode_bytes(chunks: List[bytes]):
    """Decodes bytes to string

    Args:
        chunks: byte chunks

    Returns:
        list of decoded strings
    """
    decoded_tokens = []
    buffer = b""

    for chunk in chunks:
        combined = buffer + chunk
        try:
            # Try to decode the combined bytes
            decoded_tokens.append(combined.decode("utf-8"))
            # If successful, clear the buffer
            buffer = b""
        except UnicodeDecodeError:
            # If decoding failed, keep the current chunk in the buffer
            # and attempt to combine it with the next chunk
            buffer = chunk

    # Attempt to decode any remaining bytes in the buffer
    try:
        decoded_tokens.append(buffer.decode("utf-8"))
    except UnicodeDecodeError:
        pass

    return decoded_tokens


def color_code_tokenized_text(tokenized_text_list: List[Union[str, bytes]], tokenizer):
    """Color code tokenized text.

    Args:
        tokenized_text_list: list of tokenized text
        tokenizer: tokenizer

    Returns:
        HTML string with color coded tokens
    """

    if isinstance(tokenized_text_list[0], bytes):
        tokenized_text_list = decode_bytes(tokenized_text_list)  # type: ignore

    html_text = ""
    for token in tokenized_text_list:
        if token == tokenizer.sep_token:
            html_text += format_to_html("blue", token, 1.0)
        elif token == tokenizer.cls_token:
            html_text += format_to_html("green", token, 1.0)
        elif token == tokenizer.pad_token:
            html_text += format_to_html("black", token, 0.7)
        else:
            html_text += format_to_html("black", token, 1.0)
    return html_text


def get_best_and_worst_sample_idxs(
    cfg, metrics: np.ndarray, n_plots: int
) -> Tuple[np.ndarray, np.ndarray]:
    sorted_metrics_idx = np.argsort(metrics)
    if is_lower_score_better(cfg):
        best_sample_idxs = sorted_metrics_idx[:n_plots]
        worst_samples_idxs = sorted_metrics_idx[-n_plots:][::-1]
    else:
        worst_samples_idxs = sorted_metrics_idx[:n_plots]
        best_sample_idxs = sorted_metrics_idx[-n_plots:][::-1]
    return best_sample_idxs, worst_samples_idxs


def is_lower_score_better(cfg: Any) -> bool:
    """Returns if lower loss (potentially metric) score is better or not.

    Args:
        cfg: config
    Returns:
        sort order
    """
    _, optimize, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)
    return optimize != "max"
