import glob
import re
from dataclasses import dataclass
from typing import Dict

CLEANR = re.compile("<[^<]+?>")
tooltip_files = glob.glob("tooltips/**/*.mdx", recursive=True)


def read_tooltip_file(path: str) -> str:
    """
    Reads all lines of a text file.

    Args:
        filename: path to the file

    Returns:
        str: the text of the file
    """

    with open(path) as f:
        lines = f.readlines()
    return "".join(lines)


def cleanhtml(raw_html: str) -> str:
    """
    Removes html tags from a string.

    Args:
        raw_html: the string to clean

    Returns:
        str: the cleaned string
    """

    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def clean_docusaurus_tags(text: str) -> str:
    """
    Removes docusaurus tags from a string.

    Args:
        text: the string to clean

    Returns:
        str: the cleaned string
    """

    text = text.replace(":::info note", "")
    text = text.replace(":::info Note", "")
    text = text.replace(":::tip tip", "")
    text = text.replace(":::", "")
    return text


def clean_md_links(text: str) -> str:
    """
    Removes markdown links from a string.

    Args:
        text: the string to clean

    Returns:
        str: the cleaned string
    """

    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


@dataclass
class Tooltip:
    """
    A tooltip.

    Returns:
        str: the text of the tooltip
    """

    name: str
    text: str

    def __repr__(self):
        return f"{self.name}: {self.text}"


class Tooltips:
    """
    A collection of tooltips.

    During initialization, all tooltips are read from the tooltip files.

    Usage:
        tooltips = Tooltips()
        a tooltip can be accessed by its name:
        tooltips["name"] returns the tooltip with the name "name"
    """

    def __init__(self):
        self.tooltips: Dict[str, Tooltip] = {}
        for filename in tooltip_files:
            name = filename.split("/")[-1].split(".")[0]
            name = name.replace("-", "_")
            name = name[1:]  # remove leading underscore
            section = filename.split("/")[1]
            text = read_tooltip_file(filename)
            text = cleanhtml(text)
            text = clean_docusaurus_tags(text)
            text = clean_md_links(text)
            if name in self.tooltips.keys():
                raise ValueError
            self.add_tooltip(Tooltip(f"{section}_{name}", text))

    def add_tooltip(self, tooltip):
        self.tooltips[tooltip.name] = tooltip

    def __getitem__(self, name: str) -> str:
        try:
            text = self.tooltips[name].text
        except KeyError:
            text = None
        return text

    def __len__(self):
        return len(self.tooltips)

    def __repr__(self):
        return f"{self.tooltips}"

    def get(self, name: str, default=None):
        if name in self.tooltips.keys():
            return self.tooltips[name].text
        else:
            return default


tooltips = Tooltips()
