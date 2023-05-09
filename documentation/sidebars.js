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
      label: "Tutorials",
      link: { type: "doc", id: "tutorials/tutorials-overview" },
      items: [
        {
          type: "category",
          label: "Datasets",
          items: [
            "tutorials/datasets/tutorial-1a",
            "tutorials/datasets/tutorial-2a",
            "tutorials/datasets/tutorial-3a",
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
    {
      "User guide": ["user-guide/page-1"],
    },
    "key-terms",
    "release-notes",
    "faqs",
  ],
};

