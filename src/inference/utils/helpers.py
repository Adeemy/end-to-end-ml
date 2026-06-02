"""
Inference utility functions and model loading strategies.

This module provides strategy pattern implementations for loading models from
different sources (registry vs local files) and utilities for batch prediction
processing. Includes fallback mechanisms for robust model loading.

Classes:
    ModelLoadingStrategy: Abstract base for model loading strategies
    RegistryModelLoader: Loads models from experiment tracking registries (MLflow/Comet)
    LocalModelLoader: Loads models from local filesystem
    ModelLoadingContext: Context manager with fallback between strategies

Functions:
    extract_model_config: Extracts model configuration from config files
    predict_from_file: Processes batch predictions from parquet files
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.inference.utils.model import ModelLoaderManager
from src.utils.path import ARTIFACTS_DIR

# Defaults used when a model has no persisted serving metadata sidecar.
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_ENCODED_POS_CLASS_LABEL = 1


def load_serving_metadata(model_name: str) -> dict:
    """Loads the serving metadata sidecar (threshold + positive class) for a model.

    Args:
        model_name: Name of the model whose metadata to load.

    Returns:
        dict: ``{"decision_threshold": float, "encoded_pos_class_label": int}``.
            Falls back to defaults if the sidecar is missing or unreadable.
    """
    metadata_path = ARTIFACTS_DIR / f"{model_name}_metadata.json"
    metadata = {
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "encoded_pos_class_label": DEFAULT_ENCODED_POS_CLASS_LABEL,
    }
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata.update(json.load(metadata_file))
    return metadata


def positive_class_predictions(
    model: object, data_df: pd.DataFrame, model_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Computes positive-class probabilities and labels using persisted metadata.

    The positive-class column is located via ``model.classes_`` (not a hardcoded
    index) and the decision threshold comes from the model's serving metadata, so
    inference matches the operating point chosen at registration.

    Args:
        model: Loaded model/pipeline exposing ``predict_proba`` and ``classes_``.
        data_df: Input features.
        model_name: Name of the model (used to locate metadata).

    Returns:
        tuple: (positive-class probabilities, integer labels, threshold applied).
    """
    metadata = load_serving_metadata(model_name)
    threshold = float(metadata["decision_threshold"])
    pos_label = metadata["encoded_pos_class_label"]

    proba = model.predict_proba(data_df)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_col = classes.index(pos_label) if pos_label in classes else 1
    pos_proba = np.asarray(proba)[:, pos_col]
    pred_class = (pos_proba >= threshold).astype(int)
    return pos_proba, pred_class, threshold


class ModelLoadingStrategy:
    """Abstract base for model loading strategies."""

    def load_model(
        self,
        model_name: str,
        config_params: dict,
        comet_api_key: str = None,
        logger: logging.Logger = None,
    ) -> object:
        """Load model using the specific strategy."""
        raise NotImplementedError


class RegistryModelLoader(ModelLoadingStrategy):
    """Load model from experiment tracking registry (MLflow/Comet)."""

    def load_model(
        self,
        model_name: str,
        config_params: dict,
        comet_api_key: str = None,
        logger: logging.Logger = None,
    ) -> object:
        """Load model from registry of the specified tracker type in config file.

        Args:
            model_name: Name of the model to load.
            config_params: Configuration parameters including tracker type and workspace.
            comet_api_key: API key for Comet tracker (required if tracker_type is 'comet').
            logger: Logger object.

        Returns:
            Loaded model object.

        Raises:
            ValueError: If comet_api_key is not provided for Comet tracker.
            RuntimeError: If model loading fails.
        """

        # Validate comet_api_key for Comet tracker
        if config_params["tracker_type"].lower() == "comet" and not comet_api_key:
            raise ValueError("comet_api_key must be provided when using Comet tracker")

        # Initialize ModelLoaderManager with the provided Comet API key if any
        load_model = ModelLoaderManager(comet_api_key=comet_api_key)

        model = load_model.load_model(
            tracker_type=config_params["tracker_type"],
            model_name=model_name,
            artifacts_path=ARTIFACTS_DIR,
            workspace_name=config_params["workspace_name"],
        )

        if logger:
            logger.info(
                "Model loaded from %s registry: %s",
                config_params["tracker_type"],
                model_name,
            )
        return model


class LocalModelLoader(ModelLoadingStrategy):
    """Load model from local artifacts directory."""

    def load_model(
        self,
        model_name: str,
        config_params: dict,
        comet_api_key: str = None,
        logger: logging.Logger = None,
    ) -> object:
        """Load model from local path."""
        local_model_path = ARTIFACTS_DIR / f"{model_name}.pkl"

        if not local_model_path.exists():
            raise FileNotFoundError(f"Model not found at {local_model_path}")

        model = joblib.load(str(local_model_path))
        if logger:
            logger.info("Model loaded from local fallback path: %s", local_model_path)
        return model


class ModelLoadingContext:
    """Context for model loading with fallback strategy."""

    def __init__(
        self,
        primary_strategy: ModelLoadingStrategy,
        fallback_strategy: ModelLoadingStrategy,
    ):
        self.primary_strategy = primary_strategy
        self.fallback_strategy = fallback_strategy

    def load_model(
        self,
        model_name: str,
        config_params: dict,
        comet_api_key: str = None,
        logger: logging.Logger = None,
    ) -> object:
        """Load model using primary strategy with fallback."""
        try:
            return self.primary_strategy.load_model(
                model_name, config_params, comet_api_key, logger
            )
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
            if logger:
                logger.warning(
                    "Primary model loading failed, trying fallback: %s", str(e)
                )
            return self.fallback_strategy.load_model(
                model_name, config_params, comet_api_key, logger
            )


def extract_model_config(config_yaml_path: str) -> dict:
    """Extract model configuration parameters from config file.
    Args:
        config_yaml_path: Path to the config yaml file.

    Returns:
        Dictionary with configuration parameters.
    """
    # Extract config without API key dependency
    load_model = ModelLoaderManager(comet_api_key=None)

    (
        tracker_type,
        workspace_name,
        champ_model_name,
        *_,
    ) = load_model.get_config_params(config_yaml_abs_path=config_yaml_path)

    return {
        "tracker_type": tracker_type,
        "workspace_name": workspace_name,
        "model_name": champ_model_name,
    }


def predict_from_file(
    config_yaml_path: str,
    input_file_path: str,
    output_file_path: str = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Generate predictions for data in a parquet file.

    Args:
        config_yaml_path (str): Path to the config yaml file.
        input_file_path (str): Path to input parquet file.
        output_file_path (str, optional): Path to save predictions. If None, prints to console.
        logger (logging.Logger): Logger object.

    Returns:
        pd.DataFrame: DataFrame with input data and predictions.
    """
    # Load data from parquet file
    try:
        data_df = pd.read_parquet(input_file_path)
        logger.info("Loaded %d records from %s", len(data_df), input_file_path)
    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        logger.error("Failed to load parquet file %s: %s", input_file_path, e)
        raise RuntimeError(f"Failed to load parquet file: {e}") from e

    # Extract config and get API key
    config_params = extract_model_config(config_yaml_path)
    comet_api_key = os.environ.get("COMET_API_KEY")

    # Create loading strategies with fallback
    primary_strategy = RegistryModelLoader()
    fallback_strategy = LocalModelLoader()
    loader_context = ModelLoadingContext(primary_strategy, fallback_strategy)

    # Load model
    model = loader_context.load_model(
        model_name=config_params["model_name"],
        config_params=config_params,
        comet_api_key=comet_api_key,
        logger=logger,
    )

    # Generate predictions using the persisted positive-class index and threshold
    try:
        pos_proba, pred_class, threshold = positive_class_predictions(
            model, data_df, config_params["model_name"]
        )
        data_df["predicted_probability"] = pos_proba
        data_df["prediction"] = pred_class
        logger.info(
            "Generated predictions for %d records (threshold=%.3f)",
            len(data_df),
            threshold,
        )
    except (ValueError, AttributeError, KeyError) as e:
        logger.error("Prediction failed: %s", e)
        raise RuntimeError(f"Prediction failed: {e}") from e

    # Handle results
    if output_file_path:
        data_df.to_parquet(output_file_path, index=False)
        logger.info("Predictions saved to %s", output_file_path)
    else:
        print("\\nPrediction Results (showing first 10 rows):")
        print("=" * 60)
        print(
            data_df[["predicted_probability", "prediction"]]
            .head(10)
            .to_string(index=False)
        )
        print(f"\\nTotal records processed: {len(data_df)}")

    return data_df
