### Usage with HF transformers

To use the model with the <b>`transformers`</b> library on a machine with GPUs:
- First, push the model to a huggingface repo by clicking the <b>Push checkpoint to huggingface</b> button below
- Make sure you have the <b>`transformers`</b>, <b>`accelerate`</b> and <b>`torch`</b> libraries installed in the machine's environment

```bash
pip install transformers=={{transformers_version}}
pip install einops=={{einops_version}}
pip install accelerate=={{accelerate_version}}
pip install torch=={{torch_version}}
```
- Pass model path from the huggingface repo to the following pipeline
```python
import torch
from transformers import pipeline

generate_text = pipeline(
    model="{{repo_id}}",
    torch_dtype="auto",
    trust_remote_code=True,
    use_fast={{use_fast}},
    device_map={"": "cuda:0"},
)

res = generate_text(
    "Why is drinking water so healthy?",
    min_new_tokens={{min_new_tokens}},
    max_new_tokens={{max_new_tokens}},
    do_sample={{do_sample}},
    num_beams={{num_beams}},
    temperature=float({{temperature}}),
    repetition_penalty=float({{repetition_penalty}}),
    renormalize_logits=True
)
print(res[0]["generated_text"])
```
