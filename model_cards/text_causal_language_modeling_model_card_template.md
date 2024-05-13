---
language:
- en
library_name: transformers
inference: false
thumbnail: https://h2o.ai/etc.clientlibs/h2o/clientlibs/clientlib-site/resources/images/favicon.ico
tags:
- gpt
- llm
- large language model
- h2o-llmstudio
---
# Model Card
## Summary

This model was trained using [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio).
- Base model: [{{base_model}}](https://huggingface.co/{{base_model}})


## Usage

To use the model with the `transformers` library on a machine with GPUs, first make sure you have the `transformers` library installed.

```bash
pip install transformers=={{transformers_version}}
```

Also make sure you are providing your huggingface token to the pipeline if the model is lying in a private repo.

- Either leave `token=True` in the `pipeline` and login to hugginface_hub by running

```python
import huggingface_hub
huggingface_hub.login(<ACCESS_TOKEN>)
```

- Or directly pass your <ACCESS_TOKEN> to `token` in the `pipeline`

```python
from transformers import pipeline

generate_text = pipeline(
    model="{{repo_id}}",
    torch_dtype="auto",
    trust_remote_code=True,
    use_fast={{use_fast}},
    device_map={"": "cuda:0"},
    token=True,
)

# generate configuration can be modified to your needs
# generate_text.model.generation_config.min_new_tokens = {{min_new_tokens}}
# generate_text.model.generation_config.max_new_tokens = {{max_new_tokens}}
# generate_text.model.generation_config.do_sample = {{do_sample}}
# generate_text.model.generation_config.num_beams = {{num_beams}}
# generate_text.model.generation_config.temperature = float({{temperature}})
# generate_text.model.generation_config.repetition_penalty = float({{repetition_penalty}})

messages = {{sample_messages}}

res = generate_text(
    messages,
    renormalize_logits=True
)
print(res[0]["generated_text"][-1]['content'])
```

You can print a sample prompt after applying chat template to see how it is feed to the tokenizer:

```python
print(generate_text.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
))
```

You may also construct the pipeline from the loaded model and tokenizer yourself and consider the preprocessing steps:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{{repo_id}}"  # either local folder or huggingface model name
# Important: The prompt needs to be in the same format the model was trained with.
# You can find an example prompt in the experiment logs.
messages = {{sample_messages}}

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast={{use_fast}},
    trust_remote_code={{trust_remote_code}},
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
    trust_remote_code={{trust_remote_code}},
)
model.cuda().eval()

# generate configuration can be modified to your needs
# model.generation_config.min_new_tokens = {{min_new_tokens}}
# model.generation_config.max_new_tokens = {{max_new_tokens}}
# model.generation_config.do_sample = {{do_sample}}
# model.generation_config.num_beams = {{num_beams}}
# model.generation_config.temperature = float({{temperature}})
# model.generation_config.repetition_penalty = float({{repetition_penalty}})

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

tokens = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    renormalize_logits=True
)[0]

tokens = tokens[inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(answer)
```

## Quantization and sharding

You can load the models using quantization by specifying ```load_in_8bit=True``` or ```load_in_4bit=True```. Also, sharding on multiple GPUs is possible by setting ```device_map=auto```.

## Model Architecture

```
{{model_architecture}}
```

## Model Configuration

This model was trained using H2O LLM Studio and with the configuration in [cfg.yaml](cfg.yaml). Visit [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) to learn how to train your own large language models.


## Disclaimer

Please read this disclaimer carefully before using the large language model provided in this repository. Your use of the model signifies your agreement to the following terms and conditions.

- Biases and Offensiveness: The large language model is trained on a diverse range of internet text data, which may contain biased, racist, offensive, or otherwise inappropriate content. By using this model, you acknowledge and accept that the generated content may sometimes exhibit biases or produce content that is offensive or inappropriate. The developers of this repository do not endorse, support, or promote any such content or viewpoints.
- Limitations: The large language model is an AI-based tool and not a human. It may produce incorrect, nonsensical, or irrelevant responses. It is the user's responsibility to critically evaluate the generated content and use it at their discretion.
- Use at Your Own Risk: Users of this large language model must assume full responsibility for any consequences that may arise from their use of the tool. The developers and contributors of this repository shall not be held liable for any damages, losses, or harm resulting from the use or misuse of the provided model.
- Ethical Considerations: Users are encouraged to use the large language model responsibly and ethically. By using this model, you agree not to use it for purposes that promote hate speech, discrimination, harassment, or any form of illegal or harmful activities.
- Reporting Issues: If you encounter any biased, offensive, or otherwise inappropriate content generated by the large language model, please report it to the repository maintainers through the provided channels. Your feedback will help improve the model and mitigate potential issues.
- Changes to this Disclaimer: The developers of this repository reserve the right to modify or update this disclaimer at any time without prior notice. It is the user's responsibility to periodically review the disclaimer to stay informed about any changes.

By using the large language model provided in this repository, you agree to accept and comply with the terms and conditions outlined in this disclaimer. If you do not agree with any part of this disclaimer, you should refrain from using the model and any content generated by it.
