"""Azure ML environment registration.

Registers the training/evaluation Environment from the Dockerfile + conda spec in
this directory, versioned by a content hash so it only rebuilds when those files
change (mirrors orlando's hash-based rebuild). The image runs Python 3.11 (the v2
SDK + managed feature store do not target 3.14); the job uploads the project
source as the job ``code``, so the conda spec installs the runtime dependencies
directly rather than ``pip install -e .`` (which would hit requires-python>=3.14).
"""

import hashlib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENV_DIR = Path(__file__).parent
CONDA_FILE = _ENV_DIR / "conda.yml"
DOCKERFILE = _ENV_DIR / "Dockerfile"


def _content_version(*paths: Path) -> str:
    """Returns a short stable hash of the given files (the environment version)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def ensure_environment(ml_client, azureml_config):
    """Returns the registered Environment, building it if this version is new.

    Args:
        ml_client: azure-ai-ml MLClient.
        azureml_config: AzureMLConfig (environment name).

    Returns:
        The registered Environment asset (use ``f"{name}:{version}"`` in jobs).
    """
    from azure.ai.ml.entities import BuildContext, Environment

    name = azureml_config.environment_name
    version = _content_version(CONDA_FILE, DOCKERFILE)

    try:
        existing = ml_client.environments.get(name=name, version=version)
        logger.info("Reusing environment %s:%s", name, version)
        return existing
    except Exception:  # pylint: disable=broad-except
        logger.info("Building environment %s:%s", name, version)

    environment = Environment(
        name=name,
        version=version,
        build=BuildContext(path=str(_ENV_DIR)),  # builds the Dockerfile in this dir
        description="end-to-end-ml training and evaluation environment",
    )
    return ml_client.environments.create_or_update(environment)
