"""Azure ML workspace client and authentication.

Builds an ``azure-ai-ml`` ``MLClient`` for the workspace and resolves the
workspace's MLflow tracking URI. Workspace identifiers and the Service Principal
come from the environment (a ``.env`` for local submission, or the cluster's
managed identity at run time), never from the committed config. This mirrors the
auth/workspace pattern in orlando_cash_forecast_ts, ported to the v2 SDK.

Required environment variables (local submit side):
    AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME

Service Principal (optional; falls back to DefaultAzureCredential, which also
covers the cluster's managed identity):
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET

Key Vault (optional; used to resolve the SP secret instead of AZURE_CLIENT_SECRET):
    AZURE_KEY_VAULT_URL, AZURE_CLIENT_SECRET_NAME
"""

import os
from typing import Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


def _require_env(name: str) -> str:
    """Returns a required environment variable or raises a clear error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"Environment variable '{name}' is required for Azure ML access. "
            "Set it in your .env (local) or as a workspace/cluster variable."
        )
    return value


def _resolve_sp_secret() -> Optional[str]:
    """Resolves the Service Principal secret from the environment or Key Vault.

    Returns None when no Service Principal is configured, signalling the caller
    to fall back to ``DefaultAzureCredential`` (managed identity / az login).
    """
    secret = os.environ.get("AZURE_CLIENT_SECRET")
    if secret:
        return secret

    vault_url = os.environ.get("AZURE_KEY_VAULT_URL")
    secret_name = os.environ.get("AZURE_CLIENT_SECRET_NAME")
    if vault_url and secret_name:
        # Lazy imports: only needed on the Key Vault path.
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
        return client.get_secret(secret_name).value

    return None


def get_credential():
    """Builds an Azure credential.

    Uses a Service Principal (``ClientSecretCredential``) when tenant/client id
    and a resolvable secret are present; otherwise ``DefaultAzureCredential``
    (covers ``az login`` locally and the cluster's managed identity).
    """
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    secret = _resolve_sp_secret()

    if tenant_id and client_id and secret:
        from azure.identity import ClientSecretCredential

        logger.info("Authenticating to Azure with a Service Principal.")
        return ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=secret
        )

    from azure.identity import DefaultAzureCredential

    logger.info("Authenticating to Azure with DefaultAzureCredential.")
    return DefaultAzureCredential()


def get_ml_client():
    """Returns an ``MLClient`` for the configured workspace.

    Returns:
        azure.ai.ml.MLClient: client bound to the subscription/resource
        group/workspace from the environment.
    """
    from azure.ai.ml import MLClient

    return MLClient(
        credential=get_credential(),
        subscription_id=_require_env("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=_require_env("AZURE_RESOURCE_GROUP"),
        workspace_name=_require_env("AZURE_WORKSPACE_NAME"),
    )


def workspace_mlflow_uri(ml_client=None) -> str:
    """Returns the workspace's MLflow tracking URI.

    The cluster entrypoints call ``mlflow.set_tracking_uri(...)`` with this so the
    existing MLflow logging in train.py/evaluate.py writes into the workspace.

    Args:
        ml_client: Optional pre-built MLClient; one is created if omitted.
    """
    client = ml_client or get_ml_client()
    workspace = client.workspaces.get(client.workspace_name)
    return workspace.mlflow_tracking_uri
