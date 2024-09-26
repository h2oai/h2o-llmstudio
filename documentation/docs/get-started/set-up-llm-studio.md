---
description: Learn how to set up LLM Studio.
---
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

# Set up H2O LLM Studio

This page guides you through setting up and installing H2O LLM Studio on your local system. 

First, download the H2O LLM Studio package from the [H2O LLM Studio Github repository](https://github.com/h2oai/h2o-llmstudio). You can use `git clone` or navigate to the [releases page](https://github.com/h2oai/h2o-llmstudio/releases) and download the `.zip` file found within the **Assets** of the relevant release. 

## Prerequisites

H2O LLM Studio requires the following minimum requirements:

- A machine with Ubuntu 16.04+ with atleast one recent Nvidia GPU
- Have at least 128GB+ of system RAM. Larger models and complex tasks may require 256GB+ or more.
- Nvidia drivers v470.57.02 or a later version
- Access to the following URLs:
  - developer.download.nvidia.com
  - pypi.org
  - huggingface.co
  - download.pytorch.org
  - cdn-lfs.huggingface.co

:::info Notes
- Atleast 24GB of GPU memory is recommended for larger models.
- For more information on performance benchmarks based on the hardware setup, see [H2O LLM Studio performance](llm-studio-performance.md).
- The required URLs are accessible by default when you start a GCP instance, however, if you have network rules or custom firewalls in place, it is recommended to confirm that the URLs are accessible before running `make setup`.
:::

## Installation

:::note Installation methods

<Tabs className="unique-tabs">
  <TabItem
    value="recommended-install"
    label="Linux/Ubuntu installation (recommended)"
    default
  >
    <p>
      The recommended way to install H2O LLM Studio is using pipenv with Python
      3.10. To install Python 3.10 on Ubuntu 16.04+, execute the following
      commands.
    </p>
    <p>
      <b>System installs (Python 3.10)</b>
    </p>
    <pre>
      <code>
        sudo add-apt-repository ppa:deadsnakes/ppa <br></br>
        sudo apt install python3.10 <br></br>
        sudo apt-get install python3.10-distutils <br></br>
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
      </code>
    </pre>
    <p>
      <b>Install NVIDIA drivers (if required)</b>
      <br></br>
      If you are deploying on a 'bare metal' machine running Ubuntu, you may need
      to install the required Nvidia drivers and CUDA. The following commands show
      how to retrieve the latest drivers for a machine running Ubuntu 20.04 as an
      example. You can update the following based on your respective operating system.
    </p>
    <pre>
      <code>
        wget
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin{" "}
        <br></br>
        sudo mv cuda-ubuntu2004.pin
        /etc/apt/preferences.d/cuda-repository-pin-600 <br></br>
        wget
        https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.3-470.82.01-1_amd64.deb{" "}
        <br></br>
        sudo dpkg -i
        cuda-repo-ubuntu2004-11-4-local_11.4.3-470.82.01-1_amd64.deb <br></br>
        sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub <br></br>
        sudo apt-get -y update <br></br>
        sudo apt-get -y install cuda
      </code>
    </pre>
    <p>
      <b>Create virtual environment (pipenv) </b>
      <br></br>
      The following command creates a virtual environment using pipenv and will install
      the dependencies using pipenv.
      <pre>
        <code>make setup</code>
      </pre>
    </p>
  </TabItem>
  <TabItem value="using-requirements" label="Using requirements.txt">
    <p>
      If you wish to use conda or another virtual environment, you can also
      install the dependencies using the <code>requirements.txt</code>{" "}
      file.{" "}
    </p>
    <pre>
      <code>pip install -r requirements.txt</code>
    </pre>
  </TabItem>
  <TabItem value="wsl2-install" label="Windows installation" default>
    <p>
      Follow the steps below to install H2O LLM Studio on a Windows machine
      using Windows Subsystem for Linux{" "}
      <a href="https://learn.microsoft.com/en-us/windows/wsl/">WSL2</a>
    </p>
    <p>
      1. Download the{" "}
      <a href="https://www.nvidia.com/download/index.aspx">
        latest nvidia driver
      </a>{" "}
      for Windows.{" "}
    </p>
    <p>
      2. Open PowerShell or a Windows Command Prompt window in administrator
      mode.{" "}
    </p>
    <p>
      3. Run the following command to confirm that the driver is installed
      properly and see the driver version.
      <pre>
        <code>nvidia-smi</code>
      </pre>
    </p>
    <p>
      4. Run the following command to install WSL2.
      <pre>
        <code>wsl --install</code>
      </pre>
    </p>
    <p>5. Launch the WSL2 Ubuntu installation. </p>
    <p>
      6. Install the{" "}
      <a href="https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0">
        WSL2 Nvidia Cuda Drivers
      </a>
      .
      <pre>
        <code>
          wget
          https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin{" "}
          <br></br>
          sudo mv cuda-ubuntu2004.pin
          /etc/apt/preferences.d/cuda-repository-pin-600 <br></br>
          wget
          https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb{" "}
          <br></br>
          sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb <br></br>
          sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg
          /usr/share/keyrings/ <br></br>
          sudo apt-get update <br></br>
          sudo apt-get -y install cuda
        </code>
      </pre>
    </p>
    <p>
      7. Set up the required python system installs (Python 3.10).
      <pre>
        <code>
          sudo add-apt-repository ppa:deadsnakes/ppa <br></br>
          sudo apt install python3.10 <br></br>
          sudo apt-get install python3.10-distutils <br></br>
          curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
        </code>
      </pre>
    </p>
    <p>
      8. Create the virtual environment.
      <pre>
        <code>
          sudo apt install -y python3.10-venv<br></br>
          python3 -m venv llmstudio<br></br>
          source llmstudio/bin/activate<br></br>
        </code>
      </pre>
    </p>
    <p>
      9.Clone the H2O LLM Studio repository locally.
      <pre>
        <code>
          git clone https://github.com/h2oai/h2o-llmstudio.git<br></br>
          cd h2o-llmstudio
        </code>
      </pre>
    </p>
    <p>
      10. Install H2O LLM Studio using the `requirements.txt`.
      <pre>
        <code>pip install -r requirements.txt</code>
      </pre>
    </p>
    <p>
      11. Run the H2O LLM Studio application.
      <pre>
        <code>
          H2O_WAVE_MAX_REQUEST_SIZE=25MB \ <br></br>
          H2O_WAVE_NO_LOG=True \ <br></br>
          H2O_WAVE_PRIVATE_DIR="/download/@output/download" \ <br></br>
          wave run llm_studio.app
        </code>
      </pre>
    </p>
    <p>
      This will start the H2O Wave server and the H2O LLM Studio app. Navigate
      to <a>http://localhost:10101/</a> (we recommend using Chrome) to access
      H2O LLM Studio and start fine-tuning your models.
    </p>
  </TabItem>
</Tabs>
:::

## Install custom package

If required, you can install additional Python packages into your environment. This can be done using pip after activating your virtual environment via `make shell`. For example, to install flash-attention, you would use the following commands:

```bash
make shell
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

Alternatively, you can also directly install the custom package by running the following command.

```bash
pipenv install package_name
```

## Run H2O LLM Studio

There are several ways to run H2O LLM Studio depending on your requirements.

1. [Run H2O LLM Studio GUI](#run-h2o-llm-studio-gui)
2. [Run using Docker from a nightly build](#run-using-docker-from-a-nightly-build)
3. [Run by building your own Docker image](#run-by-building-your-own-docker-image)
4. [Run with the CLI (command-line interface)](#run-with-command-line-interface-cli)

### Run H2O LLM Studio GUI

Run the following command to start the H2O LLM Studio.

```sh
make llmstudio
```

This will start the H2O Wave server and the H2O LLM Studio app. Navigate to [http://localhost:10101/](http://localhost:10101/) (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models.

![home-screen](llm-studio-home-screen.png)

If you are running H2O LLM Studio with a custom environment other than Pipenv, start the app as follows:

```sh
H2O_WAVE_MAX_REQUEST_SIZE=25MB \
H2O_WAVE_NO_LOG=True \
H2O_WAVE_PRIVATE_DIR="/download/@output/download" \
wave run llm_studio.app
```

### Run using Docker from a nightly build

First, install Docker by following the instructions from the [NVIDIA Container Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). H2O LLM Studio images are stored in the `h2oai GCR vorvan` container repository.

```sh
mkdir -p `pwd`/llmstudio_mnt
docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -it \
    -p 10101:10101 \
    -v `pwd`/llmstudio_mnt:/home/llmstudio/mount \
    -v ~/.cache:/home/llmstudio/.cache \
    gcr.io/vorvan/h2oai/h2o-llmstudio:nightly
```

Navigate to [http://localhost:10101/](http://localhost:10101/) (we recommend using Chrome) to access H2O LLM Studio and start fine-tuning your models.

:::info
Other helpful docker commands are `docker ps` and `docker kill`.
:::

### Run by building your own Docker image

```sh
docker build -t h2o-llmstudio .
mkdir -p `pwd`/llmstudio_mnt
docker run \
    --runtime=nvidia \
    --shm-size=64g \
    --init \
    --rm \
    -it \
    -p 10101:10101 \
    -v `pwd`/llmstudio_mnt:/home/llmstudio/mount \
    -v ~/.cache:/home/llmstudio/.cache \
    h2o-llmstudio
```

### Run with command line interface (CLI)

You can also use H2O LLM Studio with the command line interface (CLI) and specify the configuration .yaml file that contains all the experiment parameters. To finetune using H2O LLM Studio with CLI, activate the pipenv environment by running `make shell`.

To specify the path to the configuration file that contains the experiment parameters, run:

```sh
python llm_studio/train.py -Y {path_to_config_yaml_file}
```

To run on multiple GPUs in DDP mode, run:

```sh
bash distributed_train.sh {NR_OF_GPUS} -Y {path_to_config_yaml_file}
```

:::info
By default, the framework will run on the first `k` GPUs. If you want to specify specific GPUs to run on, use the `CUDA_VISIBLE_DEVICES` environment variable before the command.
:::

To start an interactive chat with your trained model, run:

```sh
python llm_studio/prompt.py -e {experiment_name}
```

`experiment_name` is the output folder of the experiment you want to chat with. The interactive chat will also work with models that were fine-tuned using the GUI.
