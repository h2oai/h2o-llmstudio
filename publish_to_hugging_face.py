import argparse
import logging
import sys

from app_utils import hugging_face_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-e",
        "--experiment_name",
        required=True,
        help="experiment name",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-d",
        "--device",
        required=False,
        help="'cpu' or 'cuda:0', if the GPU device id is 0",
        default="cuda:0",
    )

    parser.add_argument(
        "-a",
        "--api_key",
        required=False,
        help="Hugging Face API Key",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-u",
        "--user_id",
        required=False,
        help="Hugging Face User ID",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        help="Hugging Face Model Name",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-s",
        "--safe_serialization",
        required=False,
        help="A flag indicating whether safe serialization should be used.",
        default=True,
    )

    parser_args, unknown = parser.parse_known_args(sys.argv)

    experiment_name = parser_args.experiment_name
    device = parser_args.device
    safe_serialization = parser_args.safe_serialization

    api_key = parser_args.api_key if parser_args.api_key is not None else ""
    user_id = parser_args.user_id if parser_args.user_id is not None else ""
    model_name = parser_args.model_name if parser_args.model_name is not None else ""


    try:
        hugging_face_utils.publish_model_to_hugging_face(
            experiment_name=experiment_name,
            device=device,
            api_key=api_key,
            user_id=user_id,
            model_name=model_name,
            safe_serialization=safe_serialization
        )
    except Exception:
        logging.error("Exception occurred during the run:", exc_info=True)
