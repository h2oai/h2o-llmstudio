module.exports = {
  defaultSidebar: [
    "index",
    {
      "Get started": [
        "get-started/what-is-h2o-llm-studio",
        "get-started/set-up-llm-studio",
        "get-started/application-name-flow",
        "get-started/core-features",
        "get-started/videos",
      ],
    },
    {
      type: "category",
      label: "Guide",
      items: [
        {
          type: "category",
          label: "Datasets",
          items: [
            "user-guide/data-connectors-format",
            "user-guide/import-dataset",
            "user-guide/view-dataset",
          ],
        },
        {
          type: "category",
          label: "Experiments",
          items: [
            "tutorials/experiments/tutorial-1b",
            "tutorials/experiments/tutorial-2b",
            "tutorials/experiments/tutorial-3b",
          ],
        },
        {
          type: "category",
          label: "Predictions",
          items: [
            "tutorials/predictions/tutorial-1c",
            "tutorials/predictions/tutorial-2c",
            "tutorials/predictions/tutorial-3c",
          ],
        },
      ],
    },
    "concepts",
    "key-terms",
    "release-notes",
    "faqs",
  ],
};

