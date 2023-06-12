# Concepts

H2O LLM Studio is based on a few key concepts and uses several key terms across its documentation. Each, in turn, is explained within the sections below.

## LLM

A Large Language Model (LLM) is a type of AI model that uses deep learning techniques and uses massive datasets to analyze and generate human-like language. For example, many AI chatbots or AI search engines are powered by LLMs.  

Generally speaking, LLMs can be characterized by the following parameters: 
- size of the training dataset
- cost of training (computational power)
- size of the model (parameters)
- performance after training (or how well the model is able to respond to a particular question)

## Parameters and hyperparameters

In the context of an LLM, parameters and hyperparameters are a crucial part of determinining the model's performance and overall behaviour. 

- **Parameters:** The internal variables of the model that are learned during the training process. In the case of an LLM, parameters typically include the weights and biases associated with the neural network layers. The values of parameters directly influence the model's predictions and the quality of generated text.

- **Hyperparameters:** The configuration choices that are set before training the model and are not learned directly from the data (e.g., number of epochs, batch size etc.). These choices impact the learning process and influence the model's overall behavior. Hyperparameters need to be tuned and optimized to achieve the best performance. H2O LLM Studio GUI shows tooltips next to each hyperparameter to explain what each hyperparameter is for. You can also see the following references for more details about hyperparameters in H2O LLM Studio.
    - Dataset settings
    - [Experiment settings](./guide/experiments/experiment-settings)


## LLM Backbone

LLM Backbone is a key hyperparamter that determines the model's architecture. This option is the most important setting when it comes to experiment creation, as it sets the pretrained model weights. For more information about LLM Backbone, see [Experiment settings](guide/experiments/experiment-settings.md#llm-backbone).


## Generative AI

Generative AI refers to AI models that can generate new content, such as images, videos, or text, that did not exist before. These models learn from large datasets and use this knowledge to create new content that is similar in style or content to the original dataset.


## Fine-tuning

Fine-tuning refers to the process of taking a pre-trained language model and further training it on a specific task or domain to improve its performance on that task. It is an important technique used to adapt LLMs to specific tasks and domains. 

## LoRA (Low-Rank Adaptation)

Low-Rank Adapation (LoRa) involves modifying the pre-trained model by adjusting its weights and biases to better fit the new task. This adaptation is done in a way that preserves the pre-trained weights from the original dataset while also adjusting for the new task's specific requirements. This method of training or fine-turning models consumes less memory. By using low rank adaptation, the pre-trained model can be quickly adapted to new tasks, without requiring a large amount of new training data.

## 8-bit model training with a low memory footprint

8-bit model training with a low memory footprint refers to a fine-tuning technique that reduces the memory requirements for training neural networks by using 8-bit integers instead of 32-bit floating-point numbers. This approach can significantly reduce the amount of memory needed to store the model's parameters and can make it possible to train larger models on hardware with limited memory capacity.