## Parameters 

LLM Studio allows to tune a variety of parameters and enables fast iterations to be able to explore different hyperparameters easily.
The default settings are chosen with care and should give a good baseline.

The **LLM Backbone** option is the most important setting as it sets the pretrained model weights.

It is possible to tune the following parameters:

### **Dataset Parameters**
- **Text Prompt Start:** Optional text to prepend to each prompt. A new token will be added to the tokenizer and to the stop conditions.
- **Text Answer Separator:** Optional text to append to each prompt / prepend to each answer. A new token will be added to the tokenizer and to the stop conditions.
- **Add Eos Token to Prompt:** Adds EOS token at end of prompt.
- **Add Eos Token to Answer:** Adds EOS token at end of answer.
- **Mask Prompt Labels:** Whether to mask the prompt labels during training and only train on the loss of the answer.

### **Tokenizer Parameters**
- **Max Length Prompt:** The maximum sequence length of the prompt to use during training.
- **Max Length Answer:** The maximum sequence length of the answer to use during training.
- **Max Length:** The maximum sequence length of both prompt and answer to use during training.
- **Padding Quantile:** Truncates batches to the maximum sequence length based on specified quantile; setting to 0 disables this functionality.

### **Augmentation Parameters**
- **Token Mask Probability:** The probability of masking each token during training.

### **Architecture Parameters**
- **Backbone Dtype:** The datatype of the weights in the LLM backbone.
- **Gradient Checkpointing:** Whether to use gradient checkpointing during training.
- **Force Embedding Gradients:** Whether to force the computation of gradients for the input embeddings during training. Useful for LORA.
- **Intermediate Dropout:** The probability of applying dropout to the intermediate layers of the model during training.

### **Training Parameters**
- **Optimizer:** The optimizer to use during training.
- **Learning Rate:** The learning rate to use during training.
- **Batch Size:** The batch size to use during training.
- **Epochs:** The number of epochs to train the model for.
- **Schedule:** The learning rate schedule to use during training.
- **Warmup Epochs:** The number of warmup epochs to use during training, associated with the scheduler.
- **Weight Decay:** The weight decay coefficient to use during training.
- **Gradient Clip:** The maximum gradient norm to use during training. Useful to mitigate exploding gradients.
- **Grad Accumulation:** The number of gradient accumulation steps to use during training.

### **Lora Parameters**
- **Lora:** Whether to use low rank approximations (LoRA) during training.
- **Lora R:** The dimension of the matrix decomposition used in LoRA.
- **Lora Alpha:** The alpha value.
- **Lora Dropout:** The probability of applying dropout to the LoRA weights during training.
- **Lora Target Modules:** The modules in the model to apply the LoRA approximation to.

### **Inference Parameters**
- **Save Best Checkpoint:** Whether to save the best checkpoint based on the validation metric during training.
- **Evaluation Epochs:** The number of epochs between model evaluations during training. Can be set to a higher value for faster runtimes. 
- **Evaluate Before Training:** Whether to evaluate the model before training. Useful to assess the model quality before finetuning.
- **Train Validation Data:** Whether to concatenate training and validation data during training.

### **Prediction Parameters**
- **Metric:** The metric to use during evaluation.
- **Min Length Inference:** The minimum sequence length to use during inference.
- **Max Length Inference:** The maximum sequence length to use during inference.
- **Batch Size Inference:** Can be chosen independent of training batch size.
- **Generate Parameters:** Typical generate parameters including e.g., num beams, temperature, repetition penalty.
- **Stop Tokens:** Will stop generation at occurrence of these additional tokens; multiple tokens should be split by comma `,`.

### **Environment Parameters**
- **Gpus:** Will train on selected GPUs.
- **Mixed Precision:** Enables mixed precision.
- **Compile model:** Compiles the model with Torch. Experimental!
- **FSDP:** Wraps the model with native Torch FSDP. Experimental!
- **Logger:** Choose your favorite logger.