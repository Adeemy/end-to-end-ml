"""Submit the training stage (src/training/train.py) as an Azure ML job.

Run on the submit side (a 3.11/3.12 env with the `azure` extra + .env workspace
credentials):

    python src/azureml/submit_train.py --config_yaml_path ./config/training-config.yml [--wait]

Assumes the data splits are available to the job (produced by a prior split_data
job with feature_backend=azureml, or passed as a job input). See
docs/azure-ml-refactor-plan.md.
"""

import argparse

from src.azureml.jobs import submit_command_job
from src.training.schemas import Config, build_training_config
from src.utils.config_loader import load_config


def main(config_yaml_path: str, wait: bool = False):
    """Submits the training job."""
    azureml_config = load_config(
        Config, build_training_config, config_yaml_path
    ).azureml
    return submit_command_job(
        config_yaml_path=config_yaml_path,
        command_str=(
            "python src/training/train.py "
            "--config_yaml_path config/training-config.yml"
        ),
        experiment_name=azureml_config.train_experiment_name,
        display_name="end-to-end-ml-training",
        wait=wait,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit training to Azure ML.")
    parser.add_argument(
        "--config_yaml_path", type=str, default="./config/training-config.yml"
    )
    parser.add_argument("--wait", action="store_true", help="Stream until complete.")
    args = parser.parse_args()
    main(args.config_yaml_path, wait=args.wait)
