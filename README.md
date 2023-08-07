<p align="center"><img src="app_utils/static/llm-studio-logo-light.png#gh-dark-mode-only"></p>
<p align="center"><img src="app_utils/static/llm-studio-logo.png#gh-light-mode-only"></p>

<h3 align="center">
    <p>Welcome to H2O LLM Studio, a framework and no-code GUI designed for<br />
    fine-tuning state-of-the-art large language models (LLMs).
</p>
</h3>

<a href="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"><img width="50%" alt="home" src="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"></a><a href="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"><img width="50%" alt="logs" src="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"></a>

## Jump to

- [With H2O LLM Studio, you can](#with-h2o-llm-studio-you-can)
- [Quickstart](#quickstart)
- [What's New](#whats-new)
- [Setup](#setup)
  - [Recommended Install](#recommended-install)
  - [Using requirements.txt](#using-requirementstxt)
- [Run H2O LLM Studio GUI](#run-h2o-llm-studio-gui)
- [Run H2O LLM Studio GUI using Docker from a nightly build](#run-h2o-llm-studio-gui-using-docker-from-a-nightly-build)
- [Run H2O LLM Studio GUI by building your own Docker image](#run-h2o-llm-studio-gui-by-building-your-own-docker-image)
- [Run H2O LLM Studio with command line interface (CLI)](#run-h2o-llm-studio-with-command-line-interface-cli)
- [Data format and example data](#data-format-and-example-data)
- [Training your model](#training-your-model)
- [Example: Run on OASST data via CLI](#example-run-on-oasst-data-via-cli)
- [Model checkpoints](#model-checkpoints)
- [Documentation](#documentation)
- [License](#license)

## With H2O LLM Studio, you can

- easily and effectively fine-tune LLMs **without the need for any coding experience**.
- use a **graphic user interface (GUI)** specially designed for large language models.
- finetune any LLM using a large variety of hyperparameters.
- use recent finetuning techniques such as [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) and 8-bit model training with a low memory footprint.
- use Reinforcement Learning (RL) to finetune your model (experimental)
- use advanced evaluation metrics to judge generated answers by the model.
- track and compare your model performance visually. In addition, [Neptune](https://neptune.ai/) integration can be used.
- chat with your model and get instant feedback on your model performance.
- easily export your model to the [Hugging Face Hub](https://huggingface.co/) and share it with the community.

## Quickstart

For questions, discussing, or just hanging out, come and join our [Discord](https://discord.gg/WKhYMWcVbq)!

We offer several ways of getting started quickly.

Using CLI for fine-tuning LLMs:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/philippsinger/h2o-llm-studio-cli/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-OYccyTvmfa3r7cAquw8sioFFPJcn4R9?usp=sharing)

## What's New

- [PR 328](https://github.com/h2oai/h2o-llmstudio/pull/328) RLHF is now a separate problem type. Note that starting a new RLHF experiment from an old experiment that used RLHF is no longer supported. To continue from a previous experiment, please start a new experiment and enter the settings from the previous experiment manually. 
- [PR 308](https://github.com/h2oai/h2o-llmstudio/pull/308) Sequence to sequence models have been added as a new problem type.
- [PR 152](https://github.com/h2oai/h2o-llmstudio/pull/152) Add RLHF functionality for fine-tuning LLMs.
- [PR 132](https://github.com/h2oai/h2o-llmstudio/pull/131) Add 4bit training that allows training of larger LLM backbones with less GPU memory. See [here](https://huggingface.co/blog/4bit-transformers-bitsandbytes) for a comprehensive summary of this method.
- [PR 40](https://github.com/h2oai/h2o-llmstudio/pull/40) Added functionality for supporting nested conversations in data. A new `parent_id_column` can be selected for datasets to support tree-like structures in your conversational data. Additional `augmentation` settings have been added for this feature.

Please note that due to current rapid development we cannot guarantee full backwards compatibility of new functionality. We thus recommend to pin the version of the framework to the one you used for your experiments. For resetting, please delete/backup your `data` and `output` folders.

## Setup

H2O LLM Studio requires a machine with Ubuntu 16.04+ and at least one recent Nvidia GPU with Nvidia drivers version >= 470.57.02. For larger models, we recommend at least 24GB of GPU memory.

For more information about installation prerequisites, see the [Set up H2O LLM Studio](https://docs.h2o.ai/h2o-llmstudio/get-started/set-up-llm-studio#prerequisites) guide in the documentation. 

### Recommended Install

The recommended way to install H2O LLM Studio is using pipenv with Python 3.10. To install Python 3.10 on Ubuntu 16.04+, execute the following commands:

#### System installs (Python 3.10)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo apt-get install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```

#### Installing NVIDIA Drivers (if required)

If deploying on a 'bare metal' machine running Ubuntu, one may need to install the required Nvidia drivers and CUDA. The following commands show how to retrieve the latest drivers for a machine running Ubuntu 20.04 as an example. One can update the following based on their OS.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.3-470.82.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.3-470.82.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get -y update
sudo apt-get -y install cuda
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
make llmstudio
```

This command will start the [H2O wave](https://github.com/h2oai/wave) server and app.
Navigate to <http://localhost:10101/> (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models!

If you are running H2O LLM Studio with a custom environment other than Pipenv, you need to start the app as follows:

```bash
H2O_WAVE_MAX_REQUEST_SIZE=25MB \
H2O_WAVE_NO_LOG=true \
H2O_WAVE_PRIVATE_DIR="/download/@output/download" \
wave run app
```

## Run H2O LLM Studio GUI using Docker from a nightly build

Install Docker first by following instructions from [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
H2O LLM Studio images are stored in the h2oai GCR vorvan container repository.

```bash
mkdir -p `pwd`/data
mkdir -p `pwd`/output
docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -u `id -u`:`id -g` \
    -p 10101:10101 \
    -v `pwd`/data:/workspace/data \
    -v `pwd`/output:/workspace/output \
    -v ~/.cache:/home/llmstudio/.cache \
    gcr.io/vorvan/h2oai/h2o-llmstudio:nightly
```

Navigate to <http://localhost:10101/> (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models!

(Note other helpful docker commands are `docker ps` and `docker kill`.)

## Run H2O LLM Studio GUI by building your own Docker image

```bash
docker build -t h2o-llmstudio .
docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -u `id -u`:`id -g` \
    -p 10101:10101 \
    -v `pwd`/data:/workspace/data \
    -v `pwd`/output:/workspace/output \
    -v ~/.cache:/home/llmstudio/.cache \
    h2o-llmstudio
```

## Run H2O LLM Studio with command line interface (CLI)

You can also use H2O LLM Studio with the command line interface (CLI) and specify the configuration file that contains all the experiment parameters. To finetune using H2O LLM Studio with CLI, activate the pipenv environment by running `make shell`, and then use the following command:

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

## Data format and example data

For details on the data format required when importing your data or example data that you can use to try out H2O LLM Studio, see [Data format](https://docs.h2o.ai/h2o-llmstudio/guide/datasets/data-connectors-format#data-format) in the H2O LLM Studio documentation. 

## Training your model

With H2O LLM Studio, training your large language model is easy and intuitive. First, upload your dataset and then start training your model. Start by [creating an experiment](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/create-an-experiment). You can then [monitor and manage your experiment](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/view-an-experiment), [compare experiments](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/compare-experiments), or [push the model to Hugging Face](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/export-trained-model) to share it with the community. 

## Example: Run on OASST data via CLI

As an example, you can run an experiment on the OASST data via CLI. For instructions, see [Run an experiment on the OASST data]([https://docs.h2o.ai/h2o-llmstudio/guide/experiments/create-an-experiment#run-an-experiment-on-the-oasst-data-via-cli]) guide in the H2O LLM Studio documentation. 

## Model checkpoints

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/) and our [H2OGPT](https://github.com/h2oai/h2ogpt) repository.

## Documentation

Detailed documentation and frequently asked questions (FAQs) for H2O LLM Studio can be found at <https://docs.h2o.ai/h2o-llmstudio/>. If you wish to contribute to the docs, navigate to the `/documentation` folder of this repo and refer to the [README.md](documentation/README.md) for more information.

## License

H2O LLM Studio is licensed under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
