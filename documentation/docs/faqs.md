# FAQs

The sections below provide answers to frequently asked questions. If you have additional questions, please send them to <cloud-feedback@h2o.ai>.

---

### How much data is generally required to fine-tune a model?

There is no clear answer. As a rule of thumb, 1000 to 50000 samples of conversational data should be enough. Quality and diversity is very important. Make sure to try training on a subsample of data using the "sample" parameter to see how big the impact of the dataset size is. Recent studies suggest that less data is needed for larger foundation models.

---

### Are there any recommendations for which backbone to use? Are some backbones better for certain types of tasks?

The majority of the LLM backbones are trained on a very similar corpus of data. The main difference is the size of the model and the number of parameters. Usually, the larger the model, the better they are. The larger models also take longer to train. We recommend starting with the smallest model and then increasing the size if the performance is not satisfactory. If you are looking to train for tasks that are not directly question answering in English, it is also a good idea to look for specialized LLM backbones.

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

H2O LLM Studio stores its data in two folders located in the root directory. The folders are named `data` and `output`. Here is the breakdown of the data storage structure:
- `data/dbs`: This folder contains the user database used within the app.
- `data/user`: This folder is where uploaded datasets from the user are stored.
- `output/user`: All experiments conducted in H2O LLM Studio are stored in this folder. For each experiment, a separate folder is created within the `output/user` directory, which contains all the relevant data associated with that particular experiment.
- `output/download`: Utility folder that is used to store data the user downloads within the app. 

It is possible to change the default folders `data` and `output` in [app_utils/config.py](app_utils/config.py) 
(change `data_folder` and `output_folder`).

----

### How can I update H2O LLM Studio?

To update H2O LLM Studio, you have two options:

1. Using the latest main branch: Execute the commands `git checkout main` and `git pull` to obtain the latest updates from the main branch.
2. Using the latest release tag: Execute the commands `git pull` and `git checkout v0.0.3` (replace 'v0.0.3' with the desired version number) to switch to the latest release branch.

The update process does not remove or erase any existing data folders or experiment records.
This means that all your old data, including the user database, uploaded datasets, and experiment results, 
will still be available to you within the updated version of H2O LLM Studio.

Before updating, we recommend running the command `git rev-parse --short HEAD` and saving the commit hash. 
This will allow you to revert to your existing version if needed. 









