import Icon from "@material-ui/core/Icon";

# Create an experiment

Follow the relevant steps below to create an experiment in H2O LLM Studio.

1. On the H2O LLM Studio left-navigation pane, click **Create experiment**. Alternatively, you can click **New experiment** on the <Icon>more_vert</Icon> Kebab menu of the [View datasets](../datasets/view-dataset.md) page.

2. Select the **Dataset** you want to use to fine-tune an LLM model.

3. Select the **Problem type**.

4. Provide a meaningful **Experiment name**.

5. Define the parameters. The most important parameters are:
    - **LLM Backbone**: This parameter determines the LLM architecture to use. It is the foundation model that you continue training. H2O LLM Studio has a predefined list of recommended types of foundation models, but you can also use [Hugging Face models](https://huggingface.co/models).
    - **Mask Prompt Labels**: This option controls whether to mask the prompt labels during training and only train on the loss of the answer.
    - Hyperparameters such as **Learning rate**, **Batch size**, and number of epochs determine the training process. You can refer to the tooltips that are shown next to each hyperparameter in the GUI to learn more about them.
    - **Evaluate Before Training**: This option lets you evaluate the model before training, which can help you judge the quality of the LLM backbone before fine-tuning. 

    H2O LLM Studio provides several metric options for evaluating the performance of your model. In addition to the BLEU score, H2O LLM Studio also offers the GPT3.5 and GPT4 metrics that utilize the OpenAI API to determine whether the predicted answer is more favorable than the ground truth answer. To use these metrics, you can either export your OpenAI API key as an environment variable before starting LLM Studio, or you can specify it in the **Settings** menu within the UI.

    :::info note
    H2O LLM Studio provides an overview of all the parameters you need to specify for your experiment. The default settings are suitable when you first start an experiment. To learn more about the parameters, see [Experiment settings](experiment-settings.md).
    :::

6. Click **Run experiment**.

    ![run-experiment](run-experiment.png)

## Run an experiment on the OASST data via CLI

The steps below provide an example of how to to run an experiment on [OASST](https://huggingface.co/OpenAssistant) data via the command line interface (CLI).

1. Get the training dataset (`train_full.csv`), [OpenAssistant Conversations Dataset OASST1](https://www.kaggle.com/code/philippsinger/openassistant-conversations-dataset-oasst1?scriptVersionId=126228752) and place it into the `examples/data_oasst1` folder; or download it directly using the [Kaggle API](https://www.kaggle.com/docs/api) command given below.

 ```bash
 kaggle kernels output philippsinger/openassistant-conversations-dataset-oasst1 -p examples/data_oasst1/
 ```

2. Go into the interactive shell or open a new terminal window. Install the dependencies first, if you have not installed them already. 

 ```bash
 make setup  # installs all dependencies
 make shell
 ```

3. Run the following command to run the experiment. 

 ```bash
 python train.py -C examples/cfg_example_oasst1.py
 ```

After the experiment is completed, you can find all output artifacts in the `examples/output_oasst1` folder.
You can then use the `prompt.py` script to chat with your model.

```bash
python prompt.py -e examples/output_oasst1
```