---
description: Setting up and runnning H2O LLM Studio requires the following minimal prerequisites. This page lists out the speed and performance metrics of H2O LLM Studio based on different hardware setups.
---
# H2O LLM Studio performance

Setting up and runnning H2O LLM Studio requires the following minimal [prerequisites](set-up-llm-studio.md#prerequisites). This page lists out the speed and performance metrics of H2O LLM Studio based on different hardware setups.

The following metrics were measured. 

- **Hardware setup:** The type and number of computing devices used to train the model.
- **LLM backbone:** The underlying architecture of the language model. For more information, see [LLM backbone](concepts.md#llm-backbone).
- **Quantization:** A technique used to reduce the size and memory requirements of the model. For more information, see [Quantization](concepts.md#quantization).
- **Train**: The amount of time it took to train the model in hours and minutes.
- **Validation:** The amount of time it took to validate the mode in hours and minutes. 

| Hardware setup | LLM backbone | Quantization | Train (hh:mm:ss)| Validation (hh:mm:ss) |
|---|---|---|---|---|
| 8xA10G | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 11:35 | 3:32 |
| 4xA10G | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 21:13 | 06:35 |
| 2xA10G | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 37:04 | 12:21 |
| 1xA10G | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 1:25:29 | 15:50 |
| 8xA10G | h2oai/h2ogpt-4096-llama2-7b | nf4 | 14:26 | 06:13 |
| 4xA10G | h2oai/h2ogpt-4096-llama2-7b | nf4 | 26:55 | 11:59 |
| 2xA10G | h2oai/h2ogpt-4096-llama2-7b | nf4 | 48:24 | 23:37 |
| 1xA10G | h2oai/h2ogpt-4096-llama2-7b | nf4 | 1:26:59 | 42:17 |
| 8xA10G | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | OOM | OOM |
| 4xA10G | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | OOM | OOM |
| 2xA10G | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | OOM | OOM |
| 1xA10G | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | OOM | OOM |
| 8xA10G | h2oai/h2ogpt-4096-llama2-13b | nf4 | 25:07 | 10:58 |
| 4xA10G | h2oai/h2ogpt-4096-llama2-13b | nf4 | 48:43 | 21:25 |
| 2xA10G | h2oai/h2ogpt-4096-llama2-13b | nf4 | 1:30:45 | 42:06 |
| 1xA10G | h2oai/h2ogpt-4096-llama2-13b | nf4 | 2:44:36 | 1:14:20 |
| 8xA10G | h2oai/h2ogpt-4096-llama2-70b | nf4 | OOM | OOM |
| 4xA10G | h2oai/h2ogpt-4096-llama2-70b | nf4 | OOM | OOM |
| 2xA10G | h2oai/h2ogpt-4096-llama2-70b | nf4 | OOM | OOM |
| 1xA10G | h2oai/h2ogpt-4096-llama2-70b | nf4 | OOM | OOM |
|---|---|---|---|---|
| 4xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 7:04 | 3:55 |
| 2xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 13:14 | 7:23 |
| 1xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | bfloat16 | 23:36 | 13:25 |
| 4xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | nf4 | 9:44 | 6:30 |
| 2xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | nf4 | 18:34 | 12:16 |
| 1xA100 80GB | h2oai/h2ogpt-4096-llama2-7b | nf4 | 34:06 | 21:51 |
| 4xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | 11:46 | 5:56 |
| 2xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | 21:54 | 11:17 |
| 1xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | bfloat16 | 39:10 | 18:55 |
| 4xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | nf4 | 16:51 | 10:35 |
| 2xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | nf4 | 32:05 | 21:00 |
| 1xA100 80GB | h2oai/h2ogpt-4096-llama2-13b | nf4 | 59:11 | 36:53 |
| 4xA100 80GB | h2oai/h2ogpt-4096-llama2-70b | nf4 | 1:13:33 | 46:02 |
| 2xA100 80GB | h2oai/h2ogpt-4096-llama2-70b | nf4 | 2:20:44 | 1:33:42 |
| 1xA100 80GB | h2oai/h2ogpt-4096-llama2-70b | nf4 | 4:23:57 | 2:44:51 |

:::info
The runtimes were gathered using the default parameters. 

<details>
<summary>Expand to see the default parameters </summary>

```
architecture:
    backbone_dtype: int4
    force_embedding_gradients: false
    gradient_checkpointing: true
    intermediate_dropout: 0.0
    pretrained: true
    pretrained_weights: ''
augmentation:
    random_parent_probability: 0.0
    skip_parent_probability: 0.0
    token_mask_probability: 0.0
dataset:
    add_eos_token_to_answer: true
    add_eos_token_to_prompt: true
    add_eos_token_to_system: true
    answer_column: output
    chatbot_author: H2O.ai
    chatbot_name: h2oGPT
    data_sample: 1.0
    data_sample_choice:
    - Train
    - Validation
    limit_chained_samples: false
    mask_prompt_labels: true
    parent_id_column: None
    personalize: false
    prompt_column:
    - instruction
    system_column: None
    text_answer_separator: <|answer|>
    text_prompt_start: <|prompt|>
    text_system_start: <|system|>
    train_dataframe: /data/user/oasst/train_full.pq
    validation_dataframe: None
    validation_size: 0.01
    validation_strategy: automatic
environment:
    compile_model: false
    find_unused_parameters: false
    gpus:
    - '0'
    - '1'
    - '2'
    - '3'
    - '4'
    - '5'
    - '6'
    - '7'
    huggingface_branch: main
    mixed_precision: true
    number_of_workers: 8
    seed: -1
    trust_remote_code: true
    use_fsdp: false
experiment_name: default-8-a10g
llm_backbone: h2oai/h2ogpt-4096-llama2-7b
logging:
    logger: None
    neptune_project: ''
output_directory: /output/...
prediction:
    batch_size_inference: 0
    do_sample: false
    max_length_inference: 256
    metric: BLEU
    metric_gpt_model: gpt-3.5-turbo-0301
    metric_gpt_template: general
    min_length_inference: 2
    num_beams: 1
    num_history: 4
    repetition_penalty: 1.2
    stop_tokens: ''
    temperature: 0.3
    top_k: 0
    top_p: 1.0
problem_type: text_causal_language_modeling
tokenizer:
    add_prompt_answer_tokens: false
    max_length: 512
    max_length_answer: 256
    max_length_prompt: 256
    padding_quantile: 1.0
    use_fast: true
training:
    batch_size: 2
    differential_learning_rate: 1.0e-05
    differential_learning_rate_layers: []
    drop_last_batch: true
    epochs: 1
    evaluate_before_training: false
    evaluation_epochs: 1.0
    grad_accumulation: 1
    gradient_clip: 0.0
    learning_rate: 0.0001
    lora: true
    lora_alpha: 16
    lora_dropout: 0.05
    lora_r: 4
    lora_target_modules: ''
    loss_function: TokenAveragedCrossEntropy
    optimizer: AdamW
    save_checkpoint = "last"
    schedule: Cosine
    train_validation_data: false
    warmup_epochs: 0.0
    weight_decay: 0.0
```
</details>
:::