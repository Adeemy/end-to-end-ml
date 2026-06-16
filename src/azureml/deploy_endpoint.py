"""Deploy the champion model as an Azure ML managed online endpoint.

Serves the registered champion (a calibrated pipeline that includes
preprocessing) behind a managed endpoint, the cloud analog of the local FastAPI
service. Run on the submit side:

    python src/azureml/deploy_endpoint.py --config_yaml_path ./config/training-config.yml

The scoring script (score.py) takes raw feature rows and the pipeline applies the
same preprocessing as training. For low-latency real-time use, score.py can
instead fetch online features from the feature store by entity id (see the note
in score.py and docs/azure-ml-refactor-plan.md).
"""

import argparse
from pathlib import Path

from src.azureml.client import get_ml_client
from src.training.schemas import Config, build_training_config
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
_SCORE_DIR = str(Path(__file__).parent)


def deploy(
    config_yaml_path: str,
    endpoint_name: str = "end-to-end-ml-champion",
    instance_type: str = "Standard_DS3_v2",
):
    """Creates/updates the endpoint and a 100%-traffic champion deployment."""
    from azure.ai.ml.entities import (
        CodeConfiguration,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
    )

    config = load_config(Config, build_training_config, config_yaml_path)
    champion_name = config.modelregistry.champion_model_name

    ml_client = get_ml_client()
    model = ml_client.models.get(name=champion_name, label="latest")
    environment = ml_client.environments.get(
        name=config.azureml.environment_name, label="latest"
    )

    logger.info("Creating/updating endpoint '%s'.", endpoint_name)
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    ).result()

    deployment = ManagedOnlineDeployment(
        name="champion",
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=_SCORE_DIR, scoring_script="score.py"
        ),
        instance_type=instance_type,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {"champion": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Champion deployed to endpoint '%s' (100%% traffic).", endpoint_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy champion to an online endpoint.")
    parser.add_argument(
        "--config_yaml_path", type=str, default="./config/training-config.yml"
    )
    parser.add_argument("--endpoint_name", type=str, default="end-to-end-ml-champion")
    args = parser.parse_args()
    deploy(args.config_yaml_path, endpoint_name=args.endpoint_name)
