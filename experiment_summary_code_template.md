### Usage with HF transformers

To use the model with the <b>`transformers`</b> library on a machine with GPUs:
- First, push the model to a huggingface repo by clicking the <b>Push checkpoint to huggingface</b> button below
- Make sure you have the <b>`transformers`</b> library installed in the machine's environment

```bash
pip install transformers=={{transformers_version}}
```
- Pass model path from the huggingface repo to the following pipeline
- Also make sure you are providing your huggingface token to the pipeline if the model is lying in a private repo.
   - Either leave <b>token=True</b> in the <b>pipeline</b> and login to hugginface_hub by running
   ```python
   import huggingface_hub
   huggingface_hub.login(<ACCES_TOKEN>)
   ```
   - Or directly pass your <ACCES_TOKEN> to <b>token</b> in the <b>pipeline</b>
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
