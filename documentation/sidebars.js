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
            "user-guide/experiments/experiment-settings",
            "user-guide/experiments/view-an-experiment",
            "user-guide/experiments/compare-experiments",
            "user-guide/experiments/create-an-experiment",
            "user-guide/experiments/monitoring-experiments",
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

