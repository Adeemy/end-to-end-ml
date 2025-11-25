"""
Model selection utilities - selects best performing model from candidates.

This module queries experiment tracking backends to retrieve recent experiments and
their validation metrics. It automatically discovers experiments based on naming
patterns and selects the best performing model. Supports both MLflow (default) and
Comet ML backends.
"""

from pathlib import PosixPath

import comet_ml
import pandas as pd

from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ModelSelector:
    """Selects the best performing model from trained candidates.

    Single Responsibility: Model selection based on validation metrics.
    Open/Closed: Extensible for different selection strategies.
    """

    def __init__(
        self,
        project_name: str,
        workspace_name: str,
        comparison_metric: str,
    ):
        """Initializes the ModelSelector.

        Args:
            project_name: Comet project name.
            workspace_name: Comet workspace name.
            comparison_metric: Metric name for comparison (e.g., 'valid_f1_score').
        """
        self.project_name = project_name
        self.workspace_name = workspace_name
        self.comparison_metric = comparison_metric

        # Follow the working pattern: login first
        comet_ml.login()

    def get_recent_experiments(self, max_experiments: int = None) -> pd.DataFrame:
        """Get recent experiments directly from Comet ML.

        Args:
            max_experiments: Maximum number of recent experiments to retrieve.
                           If None, uses default of 50.

        Returns:
            DataFrame with columns [model_name, experiment_key] extracted from
            recent experiments based on experiment name patterns.
        """
        try:
            # Follow your working pattern
            comet_ml.login()
            api = comet_ml.API()

            # Get recent experiments from Comet ML
            max_experiments = int(max_experiments or 50)  # Default to 50 if None
            experiments = api.get_experiments(
                workspace=self.workspace_name,
                project_name=self.project_name,
                sort_by="startTime",
                sort_order="desc",
                page_size=max_experiments,
            )

            experiment_data = []
            # Extract model names from experiment names using common patterns
            for exp in experiments:
                exp_name = exp.name if hasattr(exp, "name") else exp.get_name()
                exp_key = exp.key if hasattr(exp, "key") else exp.get_key()

                # Extract model type from experiment name (e.g., "lightgbm_2024-11-22" -> "lightgbm")
                if exp_name:
                    # Common model name patterns
                    model_patterns = [
                        "logistic_regression",
                        "logisticregression",
                        "lr",
                        "random_forest",
                        "randomforest",
                        "rf",
                        "lightgbm",
                        "lgbm",
                        "xgboost",
                        "xgb",
                        "voting_ensemble",
                    ]

                    model_name = None
                    exp_name_lower = exp_name.lower()

                    for pattern in model_patterns:
                        if pattern in exp_name_lower:
                            if "logistic" in pattern:
                                model_name = "logistic-regression"
                            elif "random" in pattern or pattern == "rf":
                                model_name = "random-forest"
                            elif "lightgbm" in pattern or pattern == "lgbm":
                                model_name = "lightgbm"
                            elif "xgboost" in pattern or pattern == "xgb":
                                model_name = "xgboost"
                            elif "voting" in pattern:
                                model_name = "voting-ensemble"
                            break

                    if model_name:
                        experiment_data.append([model_name, exp_key])

            if not experiment_data:
                logger.warning(
                    "No experiments found with recognizable model name patterns"
                )
                return pd.DataFrame(columns=["0", "1"])

            # Create DataFrame with same column structure as CSV file
            df = pd.DataFrame(experiment_data, columns=["0", "1"])
            logger.info("Found %d recent experiments from Comet ML", len(df))
            return df

        except Exception as e:  # pylint: disable=W0718
            logger.error("Failed to retrieve experiments from Comet ML: %s", e)
            return pd.DataFrame(columns=["0", "1"])

    def select_best_model(
        self,
        experiment_keys: pd.DataFrame = None,
        max_experiments: int = None,
    ) -> tuple[str, str]:
        """Selects best model based on validation performance.

        First attempts to use provided experiment_keys DataFrame (from CSV or training).
        If not provided or empty, queries ML workspace directly for recent experiments.

        Args:
            experiment_keys: Optional DataFrame with columns [model_name, experiment_key].
                           If None, will query ML workspace directly.
            max_experiments: Maximum number of recent experiments to consider.

        Returns:
            Tuple of (best_model_name, best_model_experiment_key).

        Raises:
            ValueError: If no successful experiments found or no valid metrics.
        """
        # Use provided experiment_keys or query ML workspace directly
        max_experiments = int(max_experiments or 50)
        if experiment_keys is None or experiment_keys.shape[0] == 0:
            logger.info("No experiment_keys provided, querying ML workspace directly")
            experiment_keys = self.get_recent_experiments(max_experiments)

        if experiment_keys.shape[0] == 0:
            raise ValueError(
                "No experiments found. Please run training first: 'make train'"
            )

        # Get metrics for each experiment from Comet ML
        exp_scores = {}
        valid_experiments = {}  # Track experiments with available models

        for _, row in experiment_keys.iterrows():
            model_name = row["0"]  # First column is model name
            exp_key = row["1"]  # Second column is experiment key

            try:
                # Follow your working pattern
                comet_ml.login()
                api = comet_ml.API()

                experiment = api.get_experiment(
                    workspace=self.workspace_name,
                    project_name=self.project_name,
                    experiment=exp_key,
                )

                # Check if experiment has model files available
                assets = experiment.get_asset_list()
                model_assets = [
                    asset
                    for asset in assets
                    if asset["fileName"].endswith(".pkl")
                    and model_name in asset["fileName"]
                ]

                has_model = len(model_assets) > 0

                metric_value = experiment.get_metrics(self.comparison_metric)
                if metric_value:
                    # Get the latest value of the metric and convert to float
                    score = float(metric_value[-1]["metricValue"])
                    exp_scores[model_name] = score
                    valid_experiments[model_name] = {
                        "exp_key": exp_key,
                        "score": score,
                        "has_model": has_model,
                    }

                    logger.debug(
                        "Model %s: %s = %s (model available: %s)",
                        model_name,
                        self.comparison_metric,
                        score,
                        has_model,
                    )
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(
                    "Failed to get metrics for %s (exp: %s): %s", model_name, exp_key, e
                )
                continue
            except Exception as e:  # pylint: disable=W0718
                logger.warning("Error checking experiment %s: %s", exp_key, e)
                continue

        if not exp_scores:
            raise ValueError(
                f"No valid scores found for metric '{self.comparison_metric}' "
                "across experiments."
            )

        # Prioritize experiments with available models
        experiments_with_models = {
            name: info for name, info in valid_experiments.items() if info["has_model"]
        }

        if experiments_with_models:
            # Select best from experiments that have model files
            best_model_name = max(
                experiments_with_models,
                key=lambda x: experiments_with_models[x]["score"],
            )
            best_model_exp_key = experiments_with_models[best_model_name]["exp_key"]
            logger.info(
                "Selected best model with available artifacts for evaluation: %s with %s = %.4f (from %d experiments)",
                best_model_name,
                self.comparison_metric,
                experiments_with_models[best_model_name]["score"],
                len(experiments_with_models),
            )
        else:
            # Fallback to best metric score (will likely fail during download)
            best_model_name = max(exp_scores, key=exp_scores.get)
            best_model_exp_key = experiment_keys.loc[
                experiment_keys["0"] == best_model_name, "1"
            ].iloc[0]
            logger.warning(
                "No experiments have model artifacts available for evaluation. "
                "Selected best model by metric only: %s with %s = %.4f (from %d experiments)",
                best_model_name,
                self.comparison_metric,
                float(exp_scores[best_model_name]),
                len(exp_scores),
            )

        return best_model_name, best_model_exp_key
