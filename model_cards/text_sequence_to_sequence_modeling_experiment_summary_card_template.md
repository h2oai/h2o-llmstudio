### Usage with HF transformers

To use the model with the <b>`transformers`</b> library on a machine with GPUs:
- First, push the model to a huggingface repo by clicking the <b>Push checkpoint to huggingface</b> button below
- Make sure you have the <b>`transformers`</b> library installed in the machine's environment

```bash
pip install transformers=={{transformers_version}}
```
- Make sure to be logged in to your huggingface account if accessing a private repo
- Then, you can use the following code snippet:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "{{repo_id}}"  # either local folder or huggingface model name
# Important: The prompt needs to be in the same format the model was trained with.
# You can find an example prompt in the experiment logs.
prompt = "{{text_prompt_start}}How are you?{{end_of_sentence}}{{text_answer_separator}}"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast={{use_fast}},
    trust_remote_code={{trust_remote_code}},
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
    trust_remote_code={{trust_remote_code}},
)
model.cuda().eval()
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

# generate configuration can be modified to your needs
tokens = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    min_new_tokens={{min_new_tokens}},
    max_new_tokens={{max_new_tokens}},
    do_sample={{do_sample}},
    num_beams={{num_beams}},
    temperature=float({{temperature}}),
    repetition_penalty=float({{repetition_penalty}}),
    renormalize_logits=True
)[0]

answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(answer)
```
