---
description: Using H2O LLM Studio, you can compare experiments and analyze how different model parameters affect model performance. 
---
# Compare experiments

Using H2O LLM Studio, you can compare experiments and analyze how different model parameters affect model performance. 

Follow the relevant steps below to compare experiments in H2O LLM Studio.

1. On the H2O LLM Studio left-navigation pane, click **View experiments**.
2. Click **Compare experiments**.
3. Select the experiments you want to compare.
4. Click **Compare experiments**.

    ![compare experiments](compare-experiments.png)

    The **Charts** tab visually represents the comparison of train/validation loss, metrics, and learning rate of selected experiments. The **Config** tab compares the configuration settings of selected experiments.  

:::info note
In addition, H2O LLM Studio also integrates with [Neptune](https://neptune.ai/) and [W&B](https://wandb.ai/), two powerful experiment tracking platforms. By enabling Neptune or W&B logging when starting an experiment, you can easily track and visualize all aspects of your experiment in real time. This includes model performance, hyperparameter tuning, and other relevant metrics.
:::