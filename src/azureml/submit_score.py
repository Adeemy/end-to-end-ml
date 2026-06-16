"""Submit batch scoring (src/inference/predict.py) as an Azure ML job.

The cloud equivalent of `make predict_batch`: runs the existing batch predictor on
the cluster against an input data asset and writes predictions to an output. The
champion model is loaded from the workspace registry by predict.py (unchanged).
Run on the submit side (a 3.11/3.12 env with the `azure` extra + .env workspace
credentials):

    python src/azureml/submit_score.py \
        --config_yaml_path ./config/training-config.yml \
        --input_data azureml://datastores/<store>/paths/inference.parquet [--wait]

`--input_data` accepts any azure-ai-ml Input path (a registered data asset URI, an
azureml:// datastore path, or a local file that is uploaded with the job).
"""

import argparse

from src.azureml.client import get_ml_client
from src.azureml.compute import ensure_compute_cluster
from src.azureml.environment import ensure_environment
from src.azureml.jobs import REPO_ROOT
from src.training.schemas import Config, build_training_config
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(config_yaml_path: str, input_data: str, wait: bool = False):
    """Submits the batch-scoring job.

    Args:
        config_yaml_path: Path to the training config (for the azureml settings).
        input_data: Input data path/URI for the rows to score.
        wait: If True, stream the job until completion.

    Returns:
        The submitted job.
    """
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes

    azureml_config = load_config(
        Config, build_training_config, config_yaml_path
    ).azureml

    ml_client = get_ml_client()
    cluster = ensure_compute_cluster(ml_client, azureml_config)
    environment = ensure_environment(ml_client, azureml_config)

    job = command(
        code=str(REPO_ROOT),
        command=(
            "python src/inference/predict.py "
            "--config_yaml_path config/training-config.yml "
            "--input_file ${{inputs.input_data}} "
            "--output_file ${{outputs.predictions}}/batch_predictions.parquet"
        ),
        inputs={"input_data": Input(type=AssetTypes.URI_FILE, path=input_data)},
        outputs={"predictions": Output(type=AssetTypes.URI_FOLDER)},
        environment=f"{environment.name}:{environment.version}",
        compute=cluster.name,
        experiment_name=azureml_config.batch_score_experiment_name,
        display_name="end-to-end-ml-batch-scoring",
        environment_variables={
            "AZURE_SUBSCRIPTION_ID": ml_client.subscription_id,
            "AZURE_RESOURCE_GROUP": ml_client.resource_group_name,
            "AZURE_WORKSPACE_NAME": ml_client.workspace_name,
        },
    )

    submitted = ml_client.jobs.create_or_update(job)
    logger.info(
        "Submitted batch-scoring job '%s': %s",
        submitted.name,
        getattr(submitted, "studio_url", "(no url)"),
    )
    if wait:
        ml_client.jobs.stream(submitted.name)
    return submitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit batch scoring to Azure ML.")
    parser.add_argument(
        "--config_yaml_path", type=str, default="./config/training-config.yml"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Input data path/URI (data asset, azureml:// datastore path, or local file).",
    )
    parser.add_argument("--wait", action="store_true", help="Stream until complete.")
    args = parser.parse_args()
    main(args.config_yaml_path, args.input_data, wait=args.wait)
