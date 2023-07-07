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
- [Example Data](#example-data)
- [Training your model](#training-your-model)
- [Documentation](#documentation)
- [Model checkpoints](#model-checkpoints)
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

- [PR 152](https://github.com/h2oai/h2o-llmstudio/pull/152) Add RLHF functionality for fine-tuning LLMs.
- [PR 132](https://github.com/h2oai/h2o-llmstudio/pull/131) Add 4bit training that allows training of larger LLM backbones with less GPU memory. See [here](https://huggingface.co/blog/4bit-transformers-bitsandbytes) for a comprehensive summary of this method.
- [PR 40](https://github.com/h2oai/h2o-llmstudio/pull/40) Added functionality for supporting nested conversations in data. A new `parent_id_column` can be selected for datasets to support tree-like structures in your conversational data. Additional `augmentation` settings have been added for this feature.

Please note that due to current rapid development we cannot guarantee full backwards compatibility of new functionality. We thus recommend to pin the version of the framework to the one you used for your experiments. For resetting, please delete/backup your `data` and `output` folders.

## Example data

We provide an example dataset (converted dataset from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1))
that can be downloaded [here](https://www.kaggle.com/code/philippsinger/openassistant-conversations-dataset-oasst1?scriptVersionId=127047926). It is recommended to use `train_full.csv` for training. This dataset is also downloaded and prepared by default when first starting the GUI. Multiple dataframes can be uploaded into a single dataset by uploading a `.zip` archive.

## Training your model

With H2O LLM Studio, training your large language model is easy and intuitive. First, upload your dataset and then start training your model.

## Documentation

Detailed documentation and FAQs for H2O LLM Studio can be found at <https://docs.h2o.ai/h2o-llm-studio/>. 
If you wish to contribute to the docs, navigate to the `/documentation` folder of this repo and refer to the [README.md](documentation/README.md) for more information. 

## Model checkpoints

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/) and our [H2OGPT](https://github.com/h2oai/h2ogpt) repository.

## License

H2O LLM Studio is licensed under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
