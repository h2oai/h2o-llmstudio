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









