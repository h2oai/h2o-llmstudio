import Icon from "@material-ui/core/Icon";


# Create an experiment for Causal Classification Modeling

## Overview

This tutorial will guide you through the process of setting up and conducting an experiment for causal classification modeling using LLM Studio. It covers how to import datasets, configure key experiment settings, and understand the required parameters for effective model training and evaluation. By following these steps, you will learn how to design experiments that can identify causal relationships in classification tasks.

## Objectives

1. Learn how to import datasets using various connectors in LLM Studio.
2. Set up an experiment for Causal Classification Modeling with the correct parameters.
3. Understand how to configure the model, including selecting the correct problem type, number of classes, and adjusting learning rates.
4. Understand how to apply the appropriate loss function, especially for multiclass classification problems.

## Prerequisites

1. Access to the latest version of LLM Studio.
2. Basic Understanding of classification and causal models.

## Step 1: Import dataset

For this tutorial, let's utilize the imdb movie review dataset on Hugging Face. The dataset contains 25,000 movie reviews for training, each labeled as either positive or negative. Let's import the dataset.

1. Click on **Import dataset**.
2. Select **Hugging Face** as the data source from the **Source** dropdown. 
3. In the **Hugging Face dataset** field, enter `stanfordnlp/imdb`.
4. In the **Split** field, enter `train`.
5. Click **Continue**.

## Step 2: Configure dataset

In this step, we will review the dataset configuration page and adjust the dataset settings for our experiment.

1. In the **Dataset name** field, leave the dataset name at its default value.
2. In the **Problem type** dropdown, select **Causal classification modeling**.
3. In the **Train dataframe** dropdown, leave the default train dataframe as `imdb_train.pq`.
4. In the **Validation dataframe** dropdown, leave the default value as `None`. 
5. In the **Prompt column** dropdown, select **Text**.
6. In the **Answer column** dropdown, select **Label**.
7. Click **Continue**.
8. In the **Sample data visualization** page, click **Continue** if the input data and labels appear correctly.

## Step 3: Create a new experiment

After importing the dataset, it's time to start a new experiment for causal classification modeling.

1. Click **New experiment** from the <Icon>more_vert</Icon> Kebab menu next to the `imdb_train` dataset on the **View datasets** page.
2. In **General settings**, enter `IMDB causal classification` in the **Experiment name** text box.
3. In **Dataset settings**, set the **Data Sample** to 0.1.
4. In **Dataset settings**, set the **Num classes** to 1.
5. In **Training settings**, select the **BinaryCrossEntrophyLoss** from the **Loss function** dropdown.
6. In **Prediction settings**, select **LogLoss** from the **Metric** dropdown.
7. Leave the other configurations at their default values.
8. Click **Run experiment**. 

## Summary

In this tutorial, we walked through the process of setting up a causal classification experiment using LLM Studio. You learned how to import the IMDb dataset from Hugging Face, configure the dataset and experiment settings, and create a new experiment. With these steps, you're now ready to explore different datasets and experiment with various configurations for causal classification problems in LLM Studio.

:::info note
The Num classes field should be set to the total number of classes in the answer column of the dataset.

- For IMDb, we are dealing with binary classification, so set this to 2.

- This column should contain integer values starting from zero, with each number representing a distinct class label.

- Important: If your dataset contains multiple target columns, LLM Studio supports multilabel problems:
    - Multi-class classification: Single column with the class label.
    - Binary classification: Single column with a binary integer label.
    - Multilabel classification: Multiple columns, each containing a binary integer label (0 or 1).
    - Regression: Target column should contain float values.

For a multi-class problem, it’s essential to choose the right loss function.

- Set the loss function to cross-entropy since we are working on a multi-class problem.

Fine-tuning the learning rate is crucial for optimal model performance. In LLM Studio, you can set differential learning rates for different layers.

- By default, LLM Studio applies Differential Learning Rate Layers, with the learning rate for the classification head being 10 times smaller than the rest of the model.
- If needed, you can specify a custom learning rate for the head, but in this tutorial, we will keep the default settings.
:::