---
description: The H2O LLM studio provides a useful feature to compare experiments which allow comparing multiple experiments and analyzing how different model parameters affect model performance. 
---
# Compare experiments

The H2O LLM studio provides a useful feature to compare experiments which allow comparing multiple experiments and analyzing how different model parameters affect model performance. 

Follow the relevant steps below to compare experiments in H2O LLM Studio.

1. On the H2O LLM Studio left-navigation pane, click **View experiments**.
2. Click **Compare experiments**.
3. Select the experiments you want to compare.
4. Click **Compare experiments**.

    ![compare experiments](compare-experiments.png)

    The **Charts** tab visually represents the comparison of train/validation loss, metrics, and learning rate of selected experiments. The **Config** tab compares the configuration settings of selected experiments.  

:::info note
In addition, H2O LLM Studio also integrates with [Neptune](https://neptune.ai/), a powerful experiment tracking platform. By enabling Neptune logging when starting an experiment, you can easily track and visualize all aspects of your experiment in real time. This includes model performance, hyperparameter tuning, and other relevant metrics.
:::