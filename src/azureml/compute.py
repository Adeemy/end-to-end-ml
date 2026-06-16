"""Azure ML compute cluster management.

Create-or-get an ``AmlCompute`` cluster for the training/evaluation jobs, mirroring
orlando_cash_forecast_ts (config-driven size/scaling + a system-assigned identity
so the cluster can read the Key Vault secret at run time).
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_compute_cluster(ml_client, azureml_config):
    """Returns the configured compute cluster, creating it if absent.

    Args:
        ml_client: azure-ai-ml MLClient.
        azureml_config: AzureMLConfig (cluster name/size/scaling).

    Returns:
        The AmlCompute compute target.
    """
    from azure.ai.ml.entities import AmlCompute, IdentityConfiguration
    from azure.core.exceptions import ResourceNotFoundError

    name = azureml_config.compute_cluster_name
    try:
        cluster = ml_client.compute.get(name)
        logger.info("Reusing compute cluster '%s'.", name)
        return cluster
    except ResourceNotFoundError:
        logger.info("Creating compute cluster '%s'.", name)

    cluster = AmlCompute(
        name=name,
        type="amlcompute",
        size=azureml_config.compute_vm_size,
        min_instances=azureml_config.compute_min_nodes,
        max_instances=azureml_config.compute_max_nodes,
        idle_time_before_scale_down=azureml_config.compute_idle_seconds_before_scaledown,
        # System-assigned identity lets the cluster fetch the SP secret / data
        # from Key Vault and storage at run time.
        identity=IdentityConfiguration(type="SystemAssigned"),
    )
    return ml_client.compute.begin_create_or_update(cluster).result()
