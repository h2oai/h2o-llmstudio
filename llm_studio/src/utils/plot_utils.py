import html
import re
from dataclasses import dataclass
from typing import List, Union

PLOT_ENCODINGS = ["image", "html", "df"]


@dataclass
class PlotData:
    """
    Data to plot.

    Args:
        data: the data to plot:
            - a base64 encoded PNG if `encoding` is `png`.
            - a string in HTML if `encoding` is `html`.
            - a path to a parquet file if `encoding` is `df`.
        encoding: the encoding of the data, one of PLOT_ENCODINGS.
    """

    data: str
    encoding: str

    def __post_init__(self):
        assert self.encoding in PLOT_ENCODINGS, f"Unknown plot encoding {self.encoding}"


def get_line_separator_html():
    return (
        "<div style='height: 1px; width: 100%; margin: 1em 0; "
        "background-color: white; background-color: var(--text);'></div>"
    )


def text_to_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


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


def format_for_markdown_visualization(text: str) -> str:
    """
    Convert newlines to <br /> tags, except for those inside code blocks.
    This is needed because the markdown_table_cell_type() function does not
    convert newlines to <br /> tags, so we have to do it ourselves.

    This function is rather simple and may fail on text that uses `
    in some other context than marking code cells or uses ` within
    the code itself (as this function).
    """
    code_block_regex = r"(```.*?```|``.*?``)"
    parts = re.split(code_block_regex, text, flags=re.DOTALL)
    for i in range(len(parts)):
        # Only substitute for text outside matched code blocks
        if "`" not in parts[i]:
            parts[i] = parts[i].replace("\n", "<br />").strip()
    text = "".join(parts)

    # Restore newlines around code blocks, needed for correct rendering
    for x in ["```", "``", "`"]:
        text = text.replace(f"<br />{x}", f"\n{x}")
        text = text.replace(f"{x}<br />", f"{x}\n")
    return text
