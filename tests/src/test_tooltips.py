import unittest
from unittest.mock import mock_open, patch

from llm_studio.src.tooltips import (
    Tooltip,
    Tooltips,
    clean_docusaurus_tags,
    clean_md_links,
    cleanhtml,
    read_tooltip_file,
)


class TestTooltipFunctions(unittest.TestCase):

    def test_read_tooltip_file_mock_file(self):
        mock_content = "This is a test file content\n\nwith multiple lines."
        with patch("builtins.open", mock_open(read_data=mock_content)):
            result = read_tooltip_file("dummy/path.mdx")
        self.assertEqual(result, mock_content)

    def test_read_tooltip_file_real_file(self):
        expected_content = "Adds EOS token at end of answer."
        result = read_tooltip_file(
            "documentation/docs/tooltips/experiments/_add-eos-token-to-answer.mdx"
        )
        self.assertEqual(result, expected_content)

    def test_read_tooltip_file_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            read_tooltip_file("nonexistent/path.mdx")

    def test_cleanhtml(self):
        html = "<p>This is <b>bold</b> text</p><br><script>Script</script>"
        expected = "This is bold textScript"
        self.assertEqual(cleanhtml(html), expected)

    def test_clean_docusaurus_tags_note(self):
        text = ":::info note Some note :::"
        expected = "Some note"
        self.assertEqual(clean_docusaurus_tags(text), expected)
        text = ":::info Note Some note :::"
        expected = "Some note"
        self.assertEqual(clean_docusaurus_tags(text), expected)

    def test_clean_docusaurus_tags_tip(self):
        text = ":::tip tip Some tip :::"
        expected = "Some tip"
        self.assertEqual(clean_docusaurus_tags(text), expected)

    def test_clean_md_links(self):
        md_text = "This is a [link](https://example.com) in text"
        expected = "This is a link in text"
        self.assertEqual(clean_md_links(md_text), expected)


class TestTooltip(unittest.TestCase):

    def test_tooltip_creation(self):
        tooltip = Tooltip("test", "This is a test tooltip")
        self.assertEqual(tooltip.name, "test")
        self.assertEqual(tooltip.text, "This is a test tooltip")

    def test_tooltip_repr(self):
        tooltip = Tooltip("test", "This is a test tooltip")
        self.assertEqual(repr(tooltip), "test: This is a test tooltip")


class TestTooltips(unittest.TestCase):

    @patch("llm_studio.src.tooltips.read_tooltip_file")
    def setUp(self, mock_read):
        mock_files = [
            "documentation/docs/tooltips/section1/_file1.mdx",
            "documentation/docs/tooltips/section1/_file2.mdx",
            "documentation/docs/tooltips/section2/_file1.mdx",
            "documentation/docs/tooltips/section2/_file2.mdx",
        ]
        mock_read.side_effect = ["Content 1", "Content 2", "Content 3", "Content 4"]
        self.tooltips = Tooltips(tooltip_files=mock_files)

    @patch("llm_studio.src.tooltips.read_tooltip_file")
    def test_tooltips_no_underscore(self, mock_read):
        mock_files = [
            "documentation/docs/tooltips/section1/_file1.mdx",
            "documentation/docs/tooltips/section2/file2.mdx",
        ]
        mock_read.side_effect = ["Content 1", "Content 2"]
        with self.assertRaises(ValueError):
            Tooltips(tooltip_files=mock_files)

    def test_tooltips_initialization(self):
        self.assertEqual(len(self.tooltips), 4)
        self.assertIn("section1_file1", self.tooltips.tooltips)
        self.assertIn("section1_file2", self.tooltips.tooltips)
        self.assertIn("section2_file1", self.tooltips.tooltips)
        self.assertIn("section2_file2", self.tooltips.tooltips)

    def test_add_tooltip(self):
        length_before = len(self.tooltips)
        new_tooltip = Tooltip("new", "New tooltip")
        self.tooltips.add_tooltip(new_tooltip)
        self.assertEqual(len(self.tooltips), length_before + 1)
        self.assertEqual(self.tooltips["new"], "New tooltip")

    def test_getitem(self):
        self.assertEqual(self.tooltips["section1_file1"], "Content 1")
        self.assertEqual(self.tooltips["section1_file2"], "Content 2")
        self.assertEqual(self.tooltips["section2_file1"], "Content 3")
        self.assertEqual(self.tooltips["section2_file2"], "Content 4")

        self.assertIsNone(self.tooltips["nonexistent"])

    def test_len(self):
        self.assertEqual(len(self.tooltips), 4)

    def test_repr(self):
        repr_string = repr(self.tooltips)
        self.assertIn("section1_file1", repr_string)
        self.assertIn("section1_file2", repr_string)
        self.assertIn("section2_file1", repr_string)
        self.assertIn("section2_file2", repr_string)

    def test_get(self):
        self.assertEqual(self.tooltips.get("section1_file1"), "Content 1")
        self.assertEqual(self.tooltips.get("section1_file1", "default"), "Content 1")
        self.assertEqual(self.tooltips.get("nonexistent", "default"), "default")

    @patch("llm_studio.src.tooltips.read_tooltip_file")
    def test_duplicate_tooltip_name(self, mock_read):
        with self.assertRaises(ValueError):
            mock_files = [
                "documentation/docs/tooltips/section1/_file1.mdx",
                "documentation/docs/tooltips/section1/_file2.mdx",
                "documentation/docs/tooltips/section1/_file1.mdx",
            ]
            mock_read.side_effect = ["Content 1", "Content 2", "Content 3"]
            Tooltips(tooltip_files=mock_files)


if __name__ == "__main__":
    unittest.main()
