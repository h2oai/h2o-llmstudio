import Icon from "@material-ui/core/Icon";

# FAQs

The sections below provide answers to frequently asked questions. If you have additional questions, please send them to <cloud-feedback@h2o.ai>.

---

### How much data is generally required to fine-tune a model?

There is no clear answer. As a rule of thumb, 1000 to 50000 samples of conversational data should be enough. Quality and diversity is very important. Make sure to try training on a subsample of data using the "sample" parameter to see how big the impact of the dataset size is. Recent studies suggest that less data is needed for larger foundation models.

---

### Are there any recommendations for which backbone to use? Are some backbones better for certain types of tasks?

The majority of the LLM backbones are trained on a very similar corpus of data. The main difference is the size of the model and the number of parameters. Usually, the larger the model, the better they are. The larger models also take longer to train. It is recommended to start with the smallest model and then increase the size if the performance is not satisfactory. If you are looking to train for tasks that are not directly question answering in English, it is also a good idea to look for specialized LLM backbones.

---

### What if my data is not in question-and-answer form and I just have documents? How can I fine-tune the LLM model?

To train a chatbot style model, you need to convert your data into a question and answer format.

---

###  I encounter GPU out-of-memory issues. What can I change to be able to train large models?

There are various parameters that can be tuned while keeping a specific LLM backbone fixed. It is advised to choose 4bit/8bit precision as a backbone dtype to be able to train models >=7B on a consumer type GPU. [LORA](concepts#lora-low-rank-adaptation) should be enabled. Besides that there are the usual parameters such as batch size and maximum sequence length that can be decreased to save GPU memory (please ensure that your prompt+answer text is not truncated too much by checking the train data insights).

---

### When does the model stop the fine-tuning process?

The number of epochs are set by the user.

---

### How many records are recommended for fine-tuning?

An order of 100K records is recommended for fine-tuning.

---

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

You can also export the LoRA weights. You may add them to the files to be exported [here](https://github.com/h2oai/h2o-llmstudio/blob/main/app_utils/sections/experiment.py#L1552). Before exporting, the LoRA weights are merged back into the original LLM backbone weights to make downstream tasks easier. You do not need to have PEFT, or anything else for your deployment.

---

### How to use H2O LLM Studio in Windows? 

Use WSL 2 on Windows 

---

### How can I easily fine-tune a large language model (LLM) using the command-line interface (CLI) of H2O LLM Studio when I have limited GPU memory?

If you have limited GPU memory but still want to fine-tune a large language model using H2O LLM Studio's CLI, there are alternative methods you can use to get started quickly.

- [Using Kaggle kernels](https://www.kaggle.com/code/philippsinger/h2o-llm-studio-cli/) 
- [Using Google Colab](https://colab.research.google.com/drive/1-OYccyTvmfa3r7cAquw8sioFFPJcn4R9?usp=sharing)

---

### Can I run a validation metric on a model post-training, optionally on a different validation dataset?

Yes.

1. After you have finished creating an experiment, click on the <Icon>more_vert</Icon> Kebab menu of the relevant experiment and select **New Experiment**. 

2. Enable the **Use previous experiments weight** setting found at the top of the screen. 
   This will now load the previous weights, and you can now change eval dataset, metric, and anything else as you see fit. To only do evaluation without any retraining, set the **Epochs** to 0.

----






