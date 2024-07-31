import glob
import re
from dataclasses import dataclass
from typing import Dict

tooltip_files = glob.glob("documentation/docs/tooltips/**/*.mdx", recursive=True)


def read_tooltip_file(path: str) -> str:
    """
    Reads all lines of a text file and returns its content as a single string.

    Args:
        path (str): The path to the file to be read.

    Returns:
        str: The entire content of the file as a single string.

    Raises:
        FileNotFoundError: If the specified file is not found.
        IOError: If there's an error reading the file.
    """
    with open(path) as f:
        return f.read()


def cleanhtml(raw_html: str) -> str:
    """
    Removes HTML tags from a string.

    Args:
        raw_html (str): The string containing HTML tags to be removed.

    Returns:
        str: The input string with all HTML tags removed.
    """
    cleantext = re.sub(re.compile("<[^<]+?>"), "", raw_html)
    return cleantext


def clean_docusaurus_tags(text: str) -> str:
    """
    Removes Docusaurus tags from a string.

    Args:
        text (str): The string containing Docusaurus tags to be removed.

    Returns:
        str: The input string with Docusaurus tags removed.
    """
    text = text.replace(":::info note", "")
    text = text.replace(":::info Note", "")
    text = text.replace(":::tip tip", "")
    text = text.replace(":::", "")
    return text.strip()


def clean_md_links(text: str) -> str:
    """
    Removes Markdown links from a string, keeping only the link text.

    Args:
        text (str): The string containing Markdown links to be cleaned.

    Returns:
        str: The input string with Markdown links replaced by their text content.
    """
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


@dataclass
class Tooltip:
    """
    Represents a single tooltip with a name and associated text.

    Attributes:
        name (str): A name for the tooltip.
        text (str): The content of the tooltip.
    """

    name: str
    text: str

    def __repr__(self):
        return f"{self.name}: {self.text}"


class Tooltips:
    """
    A collection of tooltips that can be accessed by their names.

    During initialization, all tooltips are read from the specified tooltip files.

    Attributes:
        tooltips (Dict[str, Tooltip]): A dictionary mapping tooltip names to Tooltip\
        objects.

    Methods:
        add_tooltip(tooltip: Tooltip): Adds a new tooltip to the collection.
        __getitem__(name: str) -> Optional[str]: Retrieves the text of a tooltip by its\
            name.
        __len__() -> int: Returns the number of tooltips in the collection.
        __repr__() -> str: Returns a string representation of the tooltips collection.
        get(name: str, default=None) -> Optional[str]: Retrieves the text of a tooltip\
            by its name, with an optional default value.
    """

    def __init__(self, tooltip_files: list[str] = tooltip_files):
        """
        Initializes the Tooltips collection by reading and processing tooltip files.

        Args:
            tooltip_files (List[str]): A list of file paths to tooltip files.

        Raises:
            ValueError: If a tooltip file name does not start with an underscore.
            ValueError: If a duplicate tooltip name is encountered.
        """
        self.tooltips: Dict[str, Tooltip] = {}
        for filename in tooltip_files:
            name = filename.split("/")[-1].split(".")[0]
            name = name.replace("-", "_")

            if name.startswith("_"):
                name = name[1:]  # remove leading underscore
            else:
                raise ValueError("Tooltip file names must start with an underscore.")

            # documentation/docs/tooltips/SECTION/_TOOLTIPNAME.mdx
            section = filename.split("/")[3]

            tooltip_name = f"{section}_{name}"
            if tooltip_name in self.tooltips.keys():
                raise ValueError("Tooltip names must be unique.")

            text = read_tooltip_file(filename)
            text = cleanhtml(text)
            text = clean_docusaurus_tags(text)
            text = clean_md_links(text)

            self.add_tooltip(Tooltip(tooltip_name, text))

    def add_tooltip(self, tooltip: Tooltip):
        """
        Adds a new tooltip to the collection.

        Args:
            tooltip (Tooltip): The tooltip object to be added.
        """
        self.tooltips[tooltip.name] = tooltip

    def __getitem__(self, name: str) -> None | str:
        """
        Retrieves the text of a tooltip by its name.

        Args:
            name (str): The name of the tooltip to retrieve.

        Returns:
            Optional[str]: The text of the tooltip if found, None otherwise.
        """
        try:
            text = self.tooltips[name].text
        except KeyError:
            text = None
        return text

    def __len__(self) -> int:
        return len(self.tooltips)

    def __repr__(self):
        return f"{self.tooltips}"

    def get(self, name: str, default=None):
        """
        Retrieves the text of a tooltip by its name, with an optional default value.

        Args:
            name (str): The name of the tooltip to retrieve.
            default (Optional[str]): The default value to return if the tooltip is not \
                found.

        Returns:
            Optional[str]: The text of the tooltip if found, or the default value \
                otherwise.
        """
        if name in self.tooltips.keys():
            return self.tooltips[name].text
        else:
            return default


tooltips = Tooltips()
