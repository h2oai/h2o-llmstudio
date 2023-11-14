module.exports = {
  defaultSidebar: [
    "index",
    {
      "Get started": [
        "get-started/what-is-h2o-llm-studio",
        "get-started/set-up-llm-studio",
        "get-started/llm-studio-performance",
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
            "guide/datasets/data-connectors-format",
            "guide/datasets/import-dataset",
            "guide/datasets/view-dataset",
            "guide/datasets/merge-datasets",
          ],
        },
        {
          type: "category",
          label: "Experiments",
          items: [
            "guide/experiments/experiment-settings",
            "guide/experiments/create-an-experiment",
            "guide/experiments/view-an-experiment",
            "guide/experiments/compare-experiments",
            "guide/experiments/export-trained-model",
            "guide/experiments/import-to-h2ogpt"
          ],
        },
      ],
    },
    "faqs",
  ],
};

