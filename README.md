<p align="center"><img src="app_utils/static/llm-studio-logo-light.png#gh-dark-mode-only"></p>
<p align="center"><img src="app_utils/static/llm-studio-logo.png#gh-light-mode-only"></p>

<h3 align="center">
    <p>Welcome to H2O LLM Studio, a framework and no-code GUI designed for<br />
    fine-tuning state-of-the-art large language models (LLMs).
</p>
</h3>

<a href="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"><img width="50%" alt="home" src="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"></a><a href="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"><img width="50%" alt="logs" src="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"></a>


## With H2O LLM Studio, you can

- easily and effectively fine-tune LLMs **without the need for any coding experience**.
- use a **graphic user interface (GUI)** specially designed for large language models.
- finetune any LLM using a large variety of hyperparameters.
- use recent finetuning techniques such as [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) and 8-bit model training with a low memory footprint.
- use advanced evaluation metrics to judge generated answers by the model.
- track and compare your model performance visually. In addition, [Neptune](https://neptune.ai/) integration can be used.
- chat with your model and get instant feedback on your model performance.
- easily export your model to the [Hugging Face Hub](https://huggingface.co/) and share it with the community.


## Quickstart

We offer several ways of getting started quickly.

Using CLI for fine-tuning LLMs:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/philippsinger/h2o-llm-studio-cli/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-OYccyTvmfa3r7cAquw8sioFFPJcn4R9?usp=sharing)


## What's New

- [PR 12](https://github.com/h2oai/h2o-llmstudio/pull/12). Experiment configurations are now stored in yaml format,
allowing for more flexibility in the configuration while making it much easier to be backward compatible. Old experiment configurations that are stored in pickle format will be converted to yaml format automatically.
- [PR 40](https://github.com/h2oai/h2o-llmstudio/pull/40) Added functionality for supporting nested conversations in data. A new `parent_id_column` can be selected for datasets to support tree-like structures in your conversational data. Additional `augmentation` settings have been added for this feature.

Please note that due to current rapid development we cannot guarantee full backwards compatibility of new functionality. 
We thus recommend to pin the version of the framework to the one you used for your experiments. 
For resetting, please delete/backup your `data` and `output` folders.

## Setup
H2O LLM Studio requires a machine with Ubuntu 16.04+ and at least one recent Nvidia GPU with Nvidia drivers version >= 470.57.02. For larger models, we recommend at least 24GB of GPU memory.


### Recommended Install
The recommended way to install H2O LLM Studio is using pipenv with Python 3.10.
To install Python 3.10 on Ubuntu 16.04+, execute the following commands:

#### System installs (Python 3.10)
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo apt-get install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```
#### Create virtual environment (pipenv)
The following command will create a virtual environment using pipenv and will install the dependencies using pipenv:
```bash
make setup
```

### Using requirements.txt
If you wish to use conda or another virtual environment, you can also install the dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```


## Run H2O LLM Studio GUI

You can start H2O LLM Studio using the following command:
```bash
make wave
```
This command will start the [H2O wave](https://github.com/h2oai/wave) server and app.
Navigate to http://localhost:10101/ (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models!

If you are running H2O LLM Studio outside of Pipenv, you need to start wave as follows:
```bash
H2O_WAVE_MAX_REQUEST_SIZE=25MB \
H2O_WAVE_NO_LOG=True \
H2O_WAVE_PRIVATE_DIR="/download/@output/download" \
wave run app
```


## Run H2O LLM Studio with command line interface (CLI)
You can also use H2O LLM Studio with the command line interface (CLI) and specify the configuration file that contains all the experiment parameters. 
To finetune using H2O LLM Studio with CLI, activate the pipenv environment by running `make shell`, and then use the following command:
```bash
python train.py -C {path_to_config_file}
```

To run on multiple GPUs in DDP mode, run the following command:
```bash
bash distributed_train.sh {NR_OF_GPUS} -C {path_to_config_file}
```

By default, the framework will run on the first `k` GPUs. If you want to specify specific GPUs to run on, use the `CUDA_VISIBLE_DEVICES` environment variable before the command.

To start an interactive chat with your trained model, use the following command:
```bash
python prompt.py -e {experiment_name}
```
where `experiment_name` is the output folder of the experiment you want to chat with (see configuration).
The interactive chat will also work with model that were finetuned using the UI.

## Data Format
H2O LLM studio expects a csv file with at least two columns, one being the instruct column, the other 
being the answer that the model should generate. You can also provide an extra validation dataframe using the same format or use an automatic train/validation split to evaluate the model performance. 


During an experiment you can adapt the data representation with the following settings 

- **Prompt Column:** The column in the dataset containing the user prompt.
- **Answer Column:** The column in the dataset containing the expected output.
- **Parent Id Column:** An optional column specifying the parent id to be used for chained conversations. The value of this column needs to match an additional column with the name `id`. If provided, the prompt will be concatenated after preceeding parent rows.

### Example data:
We provide an example dataset (converted dataset from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1))
that can be downloaded [here](https://www.kaggle.com/code/philippsinger/openassistant-conversations-dataset-oasst1?scriptVersionId=127047926). It is recommended to use `train_full.csv` for training. This dataset is also downloaded and prepared by default when first starting the GUI. Multiple dataframes can be uploaded into a single dataset by uploading a `.zip` archive.

## Training your model

With H2O LLM Studio, training your large language model is easy and intuitive. 
First, upload your dataset and then start training your model.

### Starting an experiment
H2O LLM Studio provides various parameters to set for a given experiment, with some of the most important being:

- **LLM Backbone**: This parameter determines the LLM architecture to use.
- **Mask Prompt Labels**: This option controls whether to mask the prompt labels during training and only train on the loss of the answer.
- **Hyperparameters** such as learning rate, batch size, and number of epochs determine the training process.
An overview of all parameters is given in the [parameter description](docs/parameters.md).
- **Evaluate Before Training** This option lets you evaluate the model before training, which can help you judge the quality of the LLM backbone before fine-tuning.

We provide several metric options for evaluating the performance of your model.
In addition to the BLEU score, we offer the GPT3.5 and GPT4 metrics that utilize the OpenAI API to determine whether 
the predicted answer is more favorable than the ground truth answer. 
To use these metrics, you can either export your OpenAI API key as an environment variable before starting LLM Studio,
or you can specify it in the Settings Menu within the UI.

### Monitoring the experiment
During the experiment, you can monitor the training progress and model performance in several ways:

- The **Charts** tab displays train/validation loss, metrics, and learning rate.
- The **Train Data Insights** tab shows you the first batch of the model to verify that the input data representation is correct.
- The **Validation Prediction Insights** tab displays model predictions for random/best/worst validation samples. This tab is available after the first validation run.
- **Logs** and **Config** show the logs and the configuration of the experiment.
- **Chat** tab lets you chat with your model and get instant feedback on its performance. This tab becomes available after the training is completed.

### Push to Hugging Face 🤗
If you want to publish your model, you can export it with a single click to the [Hugging Face Hub](https://huggingface.co/)
and share it with the community. To be able to push your model to the Hub, you need to have an API token with write access.
You can also click the **Download model** button to download the model locally.
To use a converted model, you can use the following code snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "path_to_downloaded_model"  # either local folder or huggingface model name

# Important: The prompt needs to be in the same format the model was trained with.
# You can find an example prompt in the experiment logs.
prompt = "<|prompt|>How are you?<|endoftext|><|answer|>"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda().eval()

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
# generate configuration can be modified to your needs
tokens = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.3,
    repetition_penalty=1.2,
    num_beams=2
)[0]
tokens = tokens[inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(answer)
```

### Compare experiments
In the **View Experiments** view, you can compare your experiments and see how different model parameters affect the model performance.
In addition, you can track your experiments with [Neptune](https://neptune.ai/) by enabling neptune logging when starting an experiment.

## Example: Run on OASST data via CLI

As an example, you can run an experiment on the OASST data via CLI.

First, get the training dataset (`train_full.csv`) [here](https://www.kaggle.com/code/philippsinger/openassistant-conversations-dataset-oasst1?scriptVersionId=126228752) and place it into the `examples/data_oasst1` folder; or download it directly via [Kaggle API](https://www.kaggle.com/docs/api) command:
```bash
kaggle kernels output philippsinger/openassistant-conversations-dataset-oasst1 -p examples/data_oasst1/
```

Then, go into the interactive shell. If not already done earlier, install the dependencies first:
```bash
make setup  # installs all dependencies
make shell
```

You can now run the experiment via:
```bash
python train.py -C examples/cfg_example_oasst1.py
```

After the experiment finishes, you can find all output artifacts in the `examples/output_oasst1` folder.
You can then use the `prompt.py` script to chat with your model:
```bash
python prompt.py -e examples/output_oasst1
```

## Model checkpoints

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).

## Model checkpoints

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/) and our [H2OGPT](https://github.com/h2oai/h2ogpt) repository.


## License
H2O LLM Studio is licensed under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
