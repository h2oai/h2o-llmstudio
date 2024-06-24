import os

from llm_studio.src.utils.config_utils import load_config_yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse

import numpy as np
import torch

from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.modeling_utils import load_checkpoint, set_generation_config


def parse_param(cfg, prompt):
    prompt = prompt.replace("--", "")
    parts = prompt.split(" ")
    args = [" ".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
    for arg in args:
        splitted_arg = arg.split(" ")
        setattr(
            cfg.prediction,
            splitted_arg[0],
            type(getattr(cfg.prediction, splitted_arg[0]))(splitted_arg[1]),
        )
        print(
            f"Permanently changed {splitted_arg[0]} to",
            getattr(cfg.prediction, splitted_arg[0]),
        )
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample prompting.")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment output folder",
    )
    parser.add_argument(
        "-d", "--device", type=str, required=False, default="cuda:0", help="Device"
    )

    args, unknown = parser.parse_known_args()
    DEVICE = args.device

    cfg = load_config_yaml(os.path.join(args.experiment, "cfg.yaml"))

    cfg.training.epochs = 0

    cfg.environment._device = DEVICE
    cfg.environment._local_rank = 0

    cfg.tokenizer.padding_quantile = 0

    cfg.environment.mixed_precision = True
    cfg.architecture.gradient_checkpointing = False
    cfg.architecture.pretrained = False

    cfg.prediction.max_length_inference = 256

    if cfg.dataset.text_prompt_start == "":
        cfg.dataset.text_prompt_start = "\n"

    # cfg.prediction.min_length_inference = 2
    # cfg.prediction.max_length_inference = 256
    # cfg.prediction.repetition_penalty = 1.5
    # cfg.prediction.temperature = 0.3
    # cfg.prediction.num_beams = 2
    # cfg.prediction.do_sample = False
    # cfg.prediction.top_p = 0.9
    # cfg.prediction.top_k = 40

    tokenizer = get_tokenizer(cfg)

    print("Loading model weights...")

    with torch.device(DEVICE):
        model = cfg.architecture.model_class(cfg)
        cfg.architecture.pretrained_weights = os.path.join(
            args.experiment, "checkpoint.pth"
        )
        load_checkpoint(cfg, model, strict=True)

    model = model.to(DEVICE).eval()
    model.backbone.use_cache = True
    model.backbone = set_generation_config(model.backbone, cfg.prediction)

    print()
    print("=============")
    print(
        "You can change inference parameters on the fly by typing --param value, "
        "such as --num_beams 4. You can also chain them such as --num_beams 4 "
        "--top_k 30."
    )
    print()

    while True:
        prompt = input("Please enter some prompt (type 'exit' to stop): ")

        try:
            if prompt.lower() == "exit":
                break

            if prompt.lower().startswith("--"):
                cfg = parse_param(cfg, prompt)
                model.backbone = set_generation_config(model.backbone, cfg.prediction)
                continue

            prompt = cfg.dataset.dataset_class.parse_prompt(cfg, prompt)

            print(prompt)

            inputs = cfg.dataset.dataset_class.encode(
                tokenizer, prompt, cfg.tokenizer.max_length, "left"
            )
            inputs["prompt_input_ids"] = inputs.pop("input_ids").unsqueeze(0).to(DEVICE)
            inputs["prompt_attention_mask"] = (
                inputs.pop("attention_mask").unsqueeze(0).to(DEVICE)
            )

            output = {}
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output["predicted_answer_ids"] = (
                        model.generate(inputs, cfg).detach().cpu()
                    )

            predicted_text = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output["predicted_answer_ids"]
            ]
            output["predicted_text"] = np.array(predicted_text)

            output = cfg.dataset.dataset_class.clean_output(output, cfg)

            output = output["predicted_text"][0]

            print(output)
            print()
        except Exception as e:
            print("Error: {}".format(e))
            print("Something went wrong, please try again.")
