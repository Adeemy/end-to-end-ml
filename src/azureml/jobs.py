"""Submit Azure ML command jobs that run this project's entrypoints.

Mirrors orlando's submit-script pattern: the submit side (run locally or in CI)
ensures compute + environment and submits a job whose entrypoint is the existing
train.py / evaluate.py running on the cluster. On Azure ML compute, azureml-mlflow
auto-points MLflow at the workspace, so the entrypoints' existing MLflow logging
writes into the workspace with no change (experiment.py uses os.environ.setdefault
for the tracking URI, so it does not override the Azure-provided one).
"""

from src.azureml.client import get_ml_client
from src.azureml.compute import ensure_compute_cluster
from src.azureml.environment import ensure_environment
from src.training.schemas import Config, build_training_config
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.path import PARENT_DIR

logger = get_logger(__name__)

# Repo root: src/utils -> src -> repo root. The whole repo is uploaded as the
# job `code` so `import src.*` resolves on the cluster.
REPO_ROOT = PARENT_DIR.parent


def submit_command_job(
    config_yaml_path: str,
    command_str: str,
    experiment_name: str,
    display_name: str,
    wait: bool = False,
):
    """Submits a command job to Azure ML.

    Args:
        config_yaml_path: Path to the training config (for the azureml settings).
        command_str: Command the job runs on the cluster.
        experiment_name: Azure ML experiment to log the run under.
        display_name: Friendly run name.
        wait: If True, stream the job until completion.

    Returns:
        The submitted job.
    """
    from azure.ai.ml import command

    config = load_config(Config, build_training_config, config_yaml_path)
    azureml_config = config.azureml

    ml_client = get_ml_client()
    cluster = ensure_compute_cluster(ml_client, azureml_config)
    environment = ensure_environment(ml_client, azureml_config)

    job = command(
        code=str(REPO_ROOT),
        command=command_str,
        environment=f"{environment.name}:{environment.version}",
        compute=cluster.name,
        experiment_name=experiment_name,
        display_name=display_name,
        # The cluster's managed identity supplies the credential; pass the
        # workspace ids so the feature-store client resolves at run time.
        # PYTHONPATH="." puts the uploaded repo root (the job's working dir) on
        # sys.path so `import src.*` resolves (the project isn't pip-installed in
        # the image; only its dependencies are).
        environment_variables={
            "AZURE_SUBSCRIPTION_ID": ml_client.subscription_id,
            "AZURE_RESOURCE_GROUP": ml_client.resource_group_name,
            "AZURE_WORKSPACE_NAME": ml_client.workspace_name,
            "PYTHONPATH": ".",
        },
    )

    submitted = ml_client.jobs.create_or_update(job)
    logger.info(
        "Submitted job '%s' to experiment '%s': %s",
        submitted.name,
        experiment_name,
        getattr(submitted, "studio_url", "(no url)"),
    )
    if wait:
        ml_client.jobs.stream(submitted.name)
    return submitted
