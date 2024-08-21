<p align="center"><img src="llm_studio/app_utils/static/llm-studio-logo-light.png#gh-dark-mode-only"></p>
<p align="center"><img src="llm_studio/app_utils/static/llm-studio-logo.png#gh-light-mode-only"></p>

<h3 align="center">
    <p>Welcome to H2O LLM Studio, a framework and no-code GUI designed for<br />
    fine-tuning state-of-the-art large language models (LLMs).
</p>
</h3>

<a href="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"><img width="50%" alt="home" src="https://user-images.githubusercontent.com/1069138/233859311-32aa1f8c-4d68-47ac-8cd9-9313171ff9f9.png"></a><a href="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"><img width="50%" alt="logs" src="https://user-images.githubusercontent.com/1069138/233859315-e6928aa7-28d2-420b-8366-bc7323c368ca.png"></a>

[![Build Docker Image - Nightly](https://github.com/h2oai/h2o-llmstudio/actions/workflows/build-and-push-nightly.yml/badge.svg)](https://github.com/h2oai/h2o-llmstudio/actions/workflows/build-and-push-nightly.yml)

## Jump to

- [With H2O LLM Studio, you can](#with-h2o-llm-studio-you-can)
- [Quickstart](#quickstart)
- [What's New](#whats-new)
- [Setup](#setup)
  - [Recommended Install](#recommended-install)
  - [Virtual Environments](#virtual-environments)
- [Run H2O LLM Studio GUI](#run-h2o-llm-studio-gui)
- [Run H2O LLM Studio GUI using Docker from a nightly build](#run-h2o-llm-studio-gui-using-docker-from-a-nightly-build)
- [Run H2O LLM Studio GUI by building your own Docker image](#run-h2o-llm-studio-gui-by-building-your-own-docker-image)
- [Run H2O LLM Studio with command line interface (CLI)](#run-h2o-llm-studio-with-command-line-interface-cli)
- [Troubleshooting](#troubleshooting)
- [Data format and example data](#data-format-and-example-data)
- [Training your model](#training-your-model)
- [Example: Run on OASST data via CLI](#example-run-on-oasst-data-via-cli)
- [Model checkpoints](#model-checkpoints)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## With H2O LLM Studio, you can

- easily and effectively fine-tune LLMs **without the need for any coding experience**.
- use a **graphic user interface (GUI)** specially designed for large language models.
- finetune any LLM using a large variety of hyperparameters.
- use recent finetuning techniques such as [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) and 8-bit model training with a low memory footprint.
- use Reinforcement Learning (RL) to finetune your model (experimental)
- use advanced evaluation metrics to judge generated answers by the model.
- track and compare your model performance visually. In addition, [Neptune](https://neptune.ai/) and [W&B](https://wandb.ai/) integration can be used.
- chat with your model and get instant feedback on your model performance.
- easily export your model to the [Hugging Face Hub](https://huggingface.co/) and share it with the community.

## Quickstart

For questions, discussing, or just hanging out, come and join our [Discord](https://discord.gg/WKhYMWcVbq)!

Use cloud-based runpod.io instance to run the H2O LLM Studio GUI.

[![open_in_runpod](https://github.com/user-attachments/assets/0dffd945-0be0-4ef0-85cd-4e6f260d4e6c)](https://www.runpod.io/console/deploy?template=3oh3easrlu)

Using CLI for fine-tuning LLMs:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ilu000/h2o-llm-studio-cli/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1soqfJjwDJwjjH-VzZYO_pUeLx5xY4N1K?usp=sharing)

## What's New

- [PR 788](https://github.com/h2oai/h2o-llmstudio/pull/788) New problem type for Causal Regression Modeling allows to train single target regression data using LLMs.
- [PR 747](https://github.com/h2oai/h2o-llmstudio/pull/747) Fully removed RLHF in favor of DPO/IPO/KTO optimization.
- [PR 741](https://github.com/h2oai/h2o-llmstudio/pull/741) Removing separate max length settings for prompt and answer in favor of a single `max_length` settings better resembling `chat_template` functionality from `transformers`.
- [PR 592](https://github.com/h2oai/h2o-llmstudio/pull/599) Added `KTOPairLoss` for DPO modeling allowing to train models with simple preference data. Data currently needs to be manually prepared by randomly matching positive and negative examples as pairs.
- [PR 592](https://github.com/h2oai/h2o-llmstudio/pull/592) Starting to deprecate RLHF in favor of DPO/IPO optimization. Training is disabled, but old experiments are still viewable. RLHF will be fully removed in a future release.
- [PR 530](https://github.com/h2oai/h2o-llmstudio/pull/530) Introduced a new problem type for DPO/IPO optimization. This optimization technique can be used as an alternative to RLHF.
- [PR 288](https://github.com/h2oai/h2o-llmstudio/pull/288) Introduced Deepspeed for sharded training allowing to train larger models on machines with multiple GPUs. Requires NVLink. This feature replaces FSDP and offers more flexibility. Deepspeed requires a system installation of cudatoolkit and we recommend using version 12.1. See [Recommended Install](#recommended-install).
- [PR 449](https://github.com/h2oai/h2o-llmstudio/pull/449) New problem type for Causal Classification Modeling allows to train binary and multiclass models using LLMs.
- [PR 364](https://github.com/h2oai/h2o-llmstudio/pull/364) User secrets are now handled more securely and flexible. Support for handling secrets using the 'keyring' library was added. User settings are tried to be migrated automatically.

Please note that due to current rapid development we cannot guarantee full backwards compatibility of new functionality. We thus recommend to pin the version of the framework to the one you used for your experiments. For resetting, please delete/backup your `data` and `output` folders.

## Setup

H2O LLM Studio requires a machine with Ubuntu 16.04+ and at least one recent Nvidia GPU with Nvidia drivers version >= 470.57.02. For larger models, we recommend at least 24GB of GPU memory.

For more information about installation prerequisites, see the [Set up H2O LLM Studio](https://docs.h2o.ai/h2o-llmstudio/get-started/set-up-llm-studio#prerequisites) guide in the documentation.

For a performance comparison of different GPUs, see the [H2O LLM Studio performance](https://h2oai.github.io/h2o-llmstudio/get-started/llm-studio-performance) guide in the documentation.

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
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

alternatively, one can install cudatoolkits in a cuda environment:

```bash
conda create -n llmstudio python=3.10
conda activate llmstudio
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

### Virtual environments

We offer various ways of setting up the necessary python environment.

#### Pipenv virtual environment

The following command will create a virtual environment using pipenv and will install the dependencies using pipenv:

```bash
make setup
```

If you are having troubles installing the flash_attn package, consider running

```bash
make setup-no-flash
```

instead. This will install the dependencies without the flash_attn package. Note that this will disable the use of Flash Attention 2 and model training will be slower and consume more memory.

#### Nightly Conda virtual environment

You can also setup a conda virtual environment that can also deviate from the recommended setup. The ```Makefile``` contains a command ```setup-conda-nightly``` that installs a fresh conda environment with CUDA 12.4 and current nightly PyTorch.

#### Using requirements.txt

If you wish to use another virtual environment, you can also install the dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
pip install flash-attn==2.6.1 --no-build-isolation  # optional for Flash Attention 2
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

If you are using the [nightly conda environment](#nightly-conda-virtual-environment), you can run ```make llmstudio-conda```.

## Run H2O LLM Studio GUI using Docker from a nightly build

Install Docker first by following instructions from [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Make sure to have `nvidia-container-toolkit` installed on your machine as outlined in the instructions.

H2O LLM Studio images are stored in the h2oai GCR vorvan container repository.

```bash
mkdir -p `pwd`/llmstudio_mnt

# make sure to pull latest image if you still have a prior version cached
docker pull gcr.io/vorvan/h2oai/h2o-llmstudio:nightly

# run the container
docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -it \
    -u `id -u`:`id -g` \
    -p 10101:10101 \
    -v `pwd`/llmstudio_mnt:/home/llmstudio/mount \
    -v ~/.cache:/home/llmstudio/.cache \
    gcr.io/vorvan/h2oai/h2o-llmstudio:nightly
```

Navigate to <http://localhost:10101/> (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models!

(Note other helpful docker commands are `docker ps` and `docker kill`.)

## Run H2O LLM Studio GUI by building your own Docker image

```bash
docker build -t h2o-llmstudio .

mkdir -p `pwd`/llmstudio_mnt

docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -it \
    -u `id -u`:`id -g` \
    -p 10101:10101 \
    -v `pwd`/llmstudio_mnt:/home/llmstudio/mount \
    -v ~/.cache:/home/llmstudio/.cache \
    h2o-llmstudio
```

Alternatively, you can run H2O LLM Studio GUI by using our self-hosted Docker image available [here](https://console.cloud.google.com/gcr/images/vorvan/global/h2oai/h2o-llmstudio).

## Run H2O LLM Studio with command line interface (CLI)

You can also use H2O LLM Studio with the command line interface (CLI) and specify the configuration .yaml file that contains all the experiment parameters. To finetune using H2O LLM Studio with CLI, activate the pipenv environment by running `make shell`, and then use the following command:

```bash
python train.py -Y {path_to_config_yaml_file}
```

To run on multiple GPUs in DDP mode, run the following command:

```bash
bash distributed_train.sh {NR_OF_GPUS} -Y {path_to_config_yaml_file}
```

By default, the framework will run on the first `k` GPUs. If you want to specify specific GPUs to run on, use the `CUDA_VISIBLE_DEVICES` environment variable before the command.

To start an interactive chat with your trained model, use the following command:

```bash
python prompt.py -e {experiment_name}
```

where `experiment_name` is the output folder of the experiment you want to chat with (see configuration).
The interactive chat will also work with model that were finetuned using the UI.

To publish the model to Hugging Face, use the following command:

```bash
make shell 

python publish_to_hugging_face.py -p {path_to_experiment} -d {device} -a {api_key} -u {user_id} -m {model_name} -s {safe_serialization}
```

`path_to_experiment` is the output folder of the experiment.
`device` is the target device for running the model, either 'cpu' or 'cuda:0'. Default is 'cuda:0'.
`api_key` is the Hugging Face API Key. If user logged in, it can be omitted.
`user_id` is the Hugging Face user ID. If user logged in, it can be omitted.
`model_name` is the name of the model to be published on Hugging Face. It can be omitted.
`safe_serialization` is a flag indicating whether safe serialization should be used. Default is True.

## Troubleshooting

If running on cloud based machines such as runpod, you may need to set the following environment variable to allow the H2O Wave server to accept connections from the proxy:

```bash
H2O_WAVE_ALLOWED_ORIGINS="*"
```

If you are experiencing timeouts when running the H2O Wave server remotely, you can increase the timeout by setting the following environment variables:

```bash
H2O_WAVE_APP_CONNECT_TIMEOUT="15"
H2O_WAVE_APP_WRITE_TIMEOUT="15"
H2O_WAVE_APP_READ_TIMEOUT="15"
H2O_WAVE_APP_POOL_TIMEOUT="15"
```

All default to 5 (seconds). Increase them if you are experiencing timeouts. Use -1 to disable the timeout.

## Data format and example data

For details on the data format required when importing your data or example data that you can use to try out H2O LLM Studio, see [Data format](https://docs.h2o.ai/h2o-llmstudio/guide/datasets/data-connectors-format#data-format) in the H2O LLM Studio documentation.

## Training your model

With H2O LLM Studio, training your large language model is easy and intuitive. First, upload your dataset and then start training your model. Start by [creating an experiment](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/create-an-experiment). You can then [monitor and manage your experiment](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/view-an-experiment), [compare experiments](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/compare-experiments), or [push the model to Hugging Face](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/export-trained-model) to share it with the community.

## Example: Run on OASST data via CLI

As an example, you can run an experiment on the OASST data via CLI. For instructions, see [Run an experiment on the OASST data](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/create-an-experiment#run-an-experiment-on-the-oasst-data-via-cli) guide in the H2O LLM Studio documentation.

## Model checkpoints

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/) and our [H2OGPT](https://github.com/h2oai/h2ogpt) repository.

## Documentation

Detailed documentation and frequently asked questions (FAQs) for H2O LLM Studio can be found at <https://docs.h2o.ai/h2o-llmstudio/>. If you wish to contribute to the docs, navigate to the `/documentation` folder of this repo and refer to the [README.md](documentation/README.md) for more information.

## Contributing

We are happy to accept contributions to the H2O LLM Studio project. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## License

H2O LLM Studio is licensed under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
