### Usage with HF transformers

To use the model with the <b>`transformers`</b> library on a machine with GPUs:
- First, push the model to a huggingface repo by clicking the <b>Push checkpoint to huggingface</b> button below
- Make sure you have the <b>`transformers`</b> library installed in the machine's environment

```bash
pip install transformers=={{transformers_version}}
```

Also make sure you are providing your huggingface token if the model is lying in a private repo.
    - You can login to hugginface_hub by running
        ```python
        import huggingface_hub
        huggingface_hub.login(<ACCESS_TOKEN>)
        ```

You will also need to download the classification head, either manually, or by running the following code:

```python
from huggingface_hub import hf_hub_download

model_name = "{{repo_id}}"  # either local folder or Hugging Face model name
hf_hub_download(repo_id=model_name, filename="classification_head.pth", local_dir="./")
```

You can make classification predictions by following the example below:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "{{repo_id}}"  # either local folder or Hugging Face model name
# Important: The prompt needs to be in the same format the model was trained with.
# You can find an example prompt in the experiment logs.
prompt = "{{text_prompt_start}}How are you?{{end_of_sentence}}{{text_answer_separator}}"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code={{trust_remote_code}},
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
    trust_remote_code={{trust_remote_code}},
).cuda().eval()

head_weights = torch.load("classification_head.pth", map_location="cuda")
# settings can be arbitrary here as we overwrite with saved weights
head = torch.nn.Linear(1, 1, bias=False).to("cuda")
head.weight.data = head_weights

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

out = model(**inputs).logits

logits = head(out[:,-1])

print(logits)
```
