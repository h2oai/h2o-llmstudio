---
description: The flow of creating and fine-tuning large language models using H2O LLM Studio.
---
# Model flow

The flow of creating and fine-tuning large language models using H2O LLM Studio can be summarized in the following sequential steps:

- [Step 1: Import a dataset](#step-1-import-a-dataset)
- [Step 2: Create an experiment](#step-2-create-an-experiment)
- [Step 3: Monitor an experiment](#step-3-monitor-an-experiment)
- [Step 4: Compare experiments](#step-4-compare-experiments)
- [Step 5: Export a model to Hugging Face Hub](#step-5-export-a-model-to-hugging-face-hub)

## Step 1: Import a dataset

As the first step in the experiment flow, prep your data and import your dataset to H2O LLM Studio. 

- To learn about supported data connectors and data format, see [Supported data connectors and format](../guide/datasets/data-connectors-format).
- To learn about how to import a dataset to H2O LLM Studio, see [Import a dataset](../guide/datasets/import-dataset).
- To learn about reviewing and editing a dataset, see [View and manage dataset](../guide/datasets/view-dataset.md).

## Step 2: Create an experiment

As the second step in the experiment flow, create an experiment using the imported dataset. H2O LLM Studio offers several hyperparameter settings that you can adjust for your experiment model. To ensure that your training process is effective, you may need to specify the [hyperparameters](../concepts#parameters-and-hyperparameters) like learning rate, batch size, and the number of epochs. H2O LLM Studio provides an overview of all the parameters you’ll need to specify for your experiment.

- To learn about creating a new experiment, see [Create an experiment](../guide/experiments/create-an-experiment.md).
- To learn about the settings available for creating an experiment, see [Experiment settings](../guide/experiments/experiment-settings.md).

## Step 3: Monitor an experiment

As the third step in the experiment flow, monitor the launched experiment. H2O LLM Studio allows you to inspect your experiment (model) during and after model training. Simple interactive graphs in H2O LLM Studio allow you to understand the impact of selected hyperparameter values during and after model training. You can then adjust the [hyperparameters](../concepts#parameters-and-hyperparameters) to further optimize model performance. 

To learn about viewing and monitoring an experiment, see [View and manage experiments](../guide/experiments/view-an-experiment.md).

## Step 4: Compare experiments

The H2O LLM studio provides a useful feature  that allows comparing various experiments and analyzing how different model parameters affect model performance. This feature is a powerful tool for fine-tuning your machine-learning models and ensuring they meet your desired performance metrics.

To learn about comparing multiple experiments, see [Compare experiments](../guide/experiments/compare-experiments.md).

## Step 5: Export a model to Hugging Face Hub

As the final step in the experiment flow, you can export the fine-tuned model to Hugging Face with a single click.

To learn about exporting a trained model to Hugging Face Hub, see, [Export trained model to Hugging Face](../guide/experiments/export-trained-model.md).

