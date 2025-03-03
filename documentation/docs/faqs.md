---
description: Learn about frequently asked questions. 
---
<!-- import Icon from "@material-ui/core/Icon"; -->

# FAQs

The sections below provide answers to frequently asked questions. If you have additional questions, please send them to [cloud-feedback@h2o.ai](mailto:cloud-feedback@h2o.ai).

---

### What are the general recommendations for using H2O LLM Studio?

The recommendation is to always start with the default settings. From there, the parameters that tend to have the largest impact are: 
- the LLM backbone
- the number of epochs
- the learning rate
- the LoRA settings 

:::info
For more information on experiment settings, see [Experiment Settings](guide/experiments/experiment-settings). 
:::

The parameters that have the largest impact on the amount of GPU memory being used are the [backbone dtype](guide/experiments/experiment-settings#backbone-dtype) and the [max length](guide/experiments/experiment-settings#max-length) (the length of the input sequence being used during model training). 

:::info
For more information, see [this FAQ about GPU out-of-memory issues](#i-encounter-gpu-out-of-memory-issues-what-can-i-change-to-be-able-to-train-large-models). 
:::

While these parameters will change the behavior of the fine-tuned model, the change that will be most impactful is the actual data used for fine tuning. Having clean data and enough samples (i.e., atleast 1000 records) is imperative.

---

### Is the tool multi-user or single user? 

While it is possible for multiple users to use the same instance, the tool was created for a single user at a time. 

----

### How can human feedback be applied in LLM Studio?

In order to apply human feedback to H2O LLM Studio, there is a problem type called DPO (Direct Preference Optimization), which is specifically used for learning human feedback. For these types of use cases, there would be a selected answer and a rejected answer column to train a reward model. This is a more stable form of the traditional RLHF. For more information, see [this paper about DPO](https://arxiv.org/abs/2305.18290) by Stanford University.

----

### How does H2O LLM Studio evaluate the fine-tuned model? 

The valuation options are [BLEU](concepts#bleu), [Perplexity](concepts#perplexity), and an AI Judge. For more information about the traditional NLP similarity metrics, see [BLEU](concepts#bleu) and [Perplexity](concepts#perplexity) explained on the concepts page. You can also opt to use an AI judge by having an LLM model (ChatGPT or a local LLM) judge the performance of the response. This [sample prompt](https://github.com/h2oai/h2o-llmstudio/blob/main/prompts/general.txt) is an example of a prompt that is used to have the LLM evaluate the response.

----

### Can I use a different AI Judge than ChatGPT? 

Yes. For instructions on how to use a local LLM to evaluate the fine-tuned model, see [Evaluate model using an AI judge](guide/experiments/evaluate-model-using-llm). 

---

### How much data is generally required to fine-tune a model?

There is no clear answer. As a rule of thumb, 1000 to 50000 samples of conversational data should be enough. Quality and diversity is very important. Make sure to try training on a subsample of data using the "sample" parameter to see how big the impact of the dataset size is. Recent studies suggest that less data is needed for larger foundation models.

---

### Are there any recommendations for which backbone to use? Are some backbones better for certain types of tasks?

The majority of the LLM backbones are trained on a very similar corpus of data. The main difference is the size of the model and the number of parameters. Usually, the larger the model, the better they are. The larger models also take longer to train. It is recommended to start with the smallest model and then increase the size if the performance is not satisfactory. If you are looking to train for tasks that are not directly question answering in English, it is also a good idea to look for specialized LLM backbones.

---

### What if my data is not in question-and-answer form and I just have documents? How can I fine-tune the LLM model?

To train a chatbot style model, you need to convert your data into a question and answer format.

If you really want to continue pretraining on your own data without teaching a question-answering style, prepare a dataset with all your data in a single column Dataframe. Make sure that the length of the text in each row is not too long. In the experiment setup, remove all additional tokens (e.g. `<|prompt|>`, `<|answer|>`, for Text Prompt Start and Text Answer Start respectively) and disable **Add Eos Token To Prompt** and **Add Eos Token To Answer**. Deselect everything in the Prompt Column.

There are also other enterprise solutions from H2O.ai that may help you convert your data into a Q&A format. For more information, see [H2O.ai's Generative AI page](https://h2o.ai/) and this blogpost about [H2O LLM DataStudio: Streamlining Data Curation and Data Preparation for LLMs related tasks](https://h2o.ai/blog/2023/streamlining-data-preparation-for-fine-tuning-of-large-language-models/).

---


### Can the adapter be downloaded after fine-tuning so that the adapter can be combined with the backbone LLM for deployment?

H2O LLM Studio provides the option to download only the LoRA adapter when a model was trained with LoRA. Once the experiment has finished running, click the **Download adapter** button to download the lora adapter_weights separately from a fine-tuned model. 

---

###  I encounter GPU out-of-memory issues. What can I change to be able to train large models?

There are various parameters that can be tuned while keeping a specific LLM backbone fixed. It is advised to choose 4bit/8bit precision as a backbone dtype to be able to train models >=7B on a consumer type GPU. [LORA](concepts#lora-low-rank-adaptation) should be enabled. Besides that there are the usual parameters such as batch size and maximum sequence length that can be decreased to save GPU memory (please ensure that your prompt+answer text is not truncated too much by checking the train data insights).

---

### When does the model stop the fine-tuning process?

The number of epochs are set by the user.

---

### What is the maximum dataset size that an LLM Studio instance can handle?

The total dataset size is basically unlimited / only bound by disk space as all training is done in batches. There is no specific rule of thumb for maximum batch size - this depends strongly on backbone, context size, use of flash attention 2.0, use of gradient checkpointing, etc.
We suggest using a batch size that just fills the RAM for maximum efficiency. While testing for maximum memory consumption, set padding quantile to `0`. Make sure to set it back to `1` when you have found a good setting for the batch size to save on runtime.

----

### Where does H2O LLM Studio store its data?

By default, H2O LLM Studio stores its data in two folders located in the root directory in the app. The folders are named `data` and `output`. Here is the breakdown of the data storage structure:
- `data/dbs`: This folder contains the user database used within the app.
- `data/user`: This folder is where uploaded datasets from the user are stored.
- `output/user`: All experiments conducted in H2O LLM Studio are stored in this folder. For each experiment, a separate folder is created within the `output/user` directory, which contains all the relevant data associated with that particular experiment.
- `output/download`: Utility folder that is used to store data the user downloads within the app. 

It is possible to change the default working directory of H2O LLM Studio by setting the `H2O_LLM_STUDIO_WORKDIR` environment variable. By default, the working directory is set to the root directory of the app.

----

### How can I update H2O LLM Studio?

To update H2O LLM Studio, you have two options:

1. Using the latest main branch: Execute the commands `git checkout main` and `git pull` to obtain the latest updates from the main branch.
2. Using the latest release tag: Execute the commands `git pull` and `git checkout v0.0.3` (replace 'v0.0.3' with the desired version number) to switch to the latest release branch.

The update process does not remove or erase any existing data folders or experiment records. This means that all your old data, including the user database, uploaded datasets, and experiment results, will still be available to you within the updated version of H2O LLM Studio.

Before updating, it is recommended to run the `git rev-parse --short HEAD` command and save the commit hash. 
This will allow you to revert to your existing version if needed. 

---

### Once I have the [LoRA](guide/experiments/experiment-settings.md#lora), what is the recommended way of utilizing it with the base model?

You can also export the LoRA weights. You may add them to the files to be exported [here](https://github.com/h2oai/h2o-llmstudio/blob/main/llm_studio/app_utils/sections/experiment.py#L1552). Before exporting, the LoRA weights are merged back into the original LLM backbone weights to make downstream tasks easier. You do not need to have PEFT, or anything else for your deployment.

---

### How to use H2O LLM Studio in Windows? 

Use WSL 2 on Windows 

---

### How can I easily fine-tune a large language model (LLM) using the command-line interface (CLI) of H2O LLM Studio when I have limited GPU memory?

If you have limited GPU memory but still want to fine-tune a large language model using H2O LLM Studio's CLI, there are alternative methods you can use to get started quickly.

- [Using Kaggle kernels](https://www.kaggle.com/code/ilu000/h2o-llm-studio-cli/) 
- [Using Google Colab](https://colab.research.google.com/drive/1soqfJjwDJwjjH-VzZYO_pUeLx5xY4N1K?usp=sharing)

---

### Can I run a validation metric on a model post-training, optionally on a different validation dataset?

Yes.

1. After you have finished creating an experiment, click on the <Icon>more_vert</Icon> Kebab menu of the relevant experiment and select **New Experiment**. 

2. Enable the **Use previous experiments weight** setting found at the top of the screen. 
   This will now load the previous weights, and you can now change eval dataset, metric, and anything else as you see fit. To only do evaluation without any retraining, set the **Epochs** to 0.

----

### What are the hardware/infrastructure sizing recommendations for H2O LLM Studio?

When it comes to hardware requirements, it is important to note that the primary demand centers around the GPU and its associated VRAM. In terms of CPUs, most modern choices should suffice as NLP tasks typically do not heavily stress CPU performance. As for RAM, it's advisable to have a minimum of 128GB, with a stronger recommendation of 256GB or more, particularly when dealing with substantial model weights that must be accommodated in the CPU RAM.

----

### I am seeing an OS error during the H2O LLM Studio training session. What should I do? 

If you recieve the following error, it is most likely because of network issues either with your own connection or on the Hugging Face Hub side. 

```title="Error"
OSError: Consistency check failed: file should be of size 4999819336 but has size 
14099570832 ((…)ve/main/ model-00002-of-00003.safetensors). 
```

In most cases, rerunning the experiment will solve it as the download of the model weights will be re-initiated.

---