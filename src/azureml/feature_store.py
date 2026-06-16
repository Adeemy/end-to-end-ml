"""Azure ML managed feature store integration (replaces local Feast).

``get_training_features`` is the direct analog of split_data's Feast
``get_historical_features``: it point-in-time joins the registered feature set
onto an observation/entity frame and returns the features + target, so the rest
of split_data is unchanged. ``setup_feature_store`` registers and materializes the
feature set from the UCI-derived prepared data.

The exact azureml-featurestore call signatures vary by SDK version; these follow
the documented v2 API and must be validated against the installed version and a
real feature store (see docs/azure-ml-refactor-plan.md).
"""

from pathlib import PosixPath

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _feature_store_client(azureml_config):
    """Builds a FeatureStoreClient for the configured feature store."""
    from azureml.featurestore import FeatureStoreClient

    from src.azureml.client import _require_env, get_credential

    return FeatureStoreClient(
        credential=get_credential(),
        subscription_id=_require_env("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=_require_env("AZURE_RESOURCE_GROUP"),
        name=azureml_config.feature_store_name,
    )


def get_training_features(
    data_config,
    files_config,
    data_dir: PosixPath,
    azureml_config,
) -> pd.DataFrame:
    """Retrieves features for the observation (target) frame from the feature store.

    Args:
        data_config: TrainFeaturesConfig (entity/timestamp column names).
        files_config: TrainFilesConfig (the target/observation parquet name).
        data_dir: Local data directory holding the observation frame.
        azureml_config: AzureMLConfig (feature store + feature set names).

    Returns:
        pd.DataFrame: the observation frame joined with the feature set's features,
        the same shape split_data's Feast path returns.
    """
    from azureml.featurestore import get_offline_features

    observation = pd.read_parquet(
        data_dir / files_config.preprocessed_dataset_target_file_name
    )

    fs_client = _feature_store_client(azureml_config)
    feature_set = fs_client.feature_sets.get(
        name=azureml_config.feature_set_name,
        version=azureml_config.feature_set_version,
    )

    logger.info(
        "Retrieving offline features from '%s:%s' for %d observations.",
        azureml_config.feature_set_name,
        azureml_config.feature_set_version,
        len(observation),
    )
    training_df = get_offline_features(
        features=feature_set.features,
        observation_data=observation,
        timestamp_column=data_config.event_timestamp_col_name,
    )
    # get_offline_features may return a Spark/lazy frame depending on the SDK;
    # normalize to pandas for the downstream split logic.
    return training_df.to_pandas() if hasattr(training_df, "to_pandas") else training_df


def setup_feature_store(azureml_config, feature_set_spec_path: str) -> None:
    """Registers and materializes the feature set from a feature-set spec.

    One-time (or on data refresh) setup, intended to be called from prep_data or a
    dedicated setup job after the UCI-derived prepared data is written to the
    feature store's offline source.

    Args:
        azureml_config: AzureMLConfig.
        feature_set_spec_path: Path to the FeatureSetSpec YAML.
    """
    from azure.ai.ml.entities import FeatureSet, FeatureSetSpecification

    from src.azureml.client import get_ml_client

    ml_client = get_ml_client()
    feature_set = FeatureSet(
        name=azureml_config.feature_set_name,
        version=azureml_config.feature_set_version,
        specification=FeatureSetSpecification(path=feature_set_spec_path),
        entities=[f"azureml:{azureml_config.feature_set_name}_entity:1"],
        stage="Development",
    )
    registered = ml_client.feature_sets.begin_create_or_update(feature_set).result()
    logger.info(
        "Registered feature set %s:%s; submitting materialization.",
        registered.name,
        registered.version,
    )
    ml_client.feature_sets.begin_backfill_materialization(
        name=registered.name, version=registered.version
    ).result()
