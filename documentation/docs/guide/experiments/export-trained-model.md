# Export trained model to Hugging Face

If you’re ready to share your trained model with a broader community, H2O LLM Studio allows you to export the fine-tuned model to [Hugging Face](https://huggingface.co/) with a single click.

:::info note
Before exporting your model to the Hugging Face Hub, you need to have an API key with the write access. To obtain an API token with write access, you can follow the [instructions provided by Hugging Face](https://huggingface.co/docs/hub/security-tokens), which involve creating an account, logging in, and generating an access token with the appropriate permission.
:::

To export a trained model to Hugging Face Hub:

1. On the H2O LLM Studio left-navigation pane, click **View experiments**. You will see the experiments table with a list of all the experiments you have launched so far. 

2. Click the name of the experiment that you want to export the model.

3. Click **Push checkpoint to huggingface**.

4. Enter the **Account name** on Hugging Face that you want to push the model. Leaving it empty will push it to the default user account.

5. Enter the **Huggingface API** Key with the write access.

6. Click **Export**.

    ![export model to hugging face](export-model-to-huggingface.png)