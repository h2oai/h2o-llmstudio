module.exports = {
  defaultSidebar: [
    "index",
    {
      "Get started": [
        "get-started/what-is-h2o-llm-studio",
        "get-started/set-up-llm-studio",
        "get-started/llm-studio-flow",
        "get-started/core-features",
        "get-started/videos",
      ],
    },
    "concepts",
    {
      type: "category",
      label: "Guide",
      items: [
        {
          type: "category",
          label: "Datasets",
          items: [
            "guide/data-connectors-format",
            "guide/import-dataset",
            "guide/view-dataset",
          ],
        },
        {
          type: "category",
          label: "Experiments",
          items: [
            "guide/create-experiment",
          ],
        },
      ],
    },
    "tutorial",
    "release-notes",
    "faqs",
  ],
};

