"""
Model selection utilities - selects best performing model from candidates.
"""

from pathlib import PosixPath

import pandas as pd
from comet_ml import API

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
        comet_api_key: str,
    ):
        """Initializes the ModelSelector.

        Args:
            project_name: Comet project name.
            workspace_name: Comet workspace name.
            comparison_metric: Metric name for comparison (e.g., 'valid_f1_score').
            comet_api_key: Comet ML API key.
        """
        self.project_name = project_name
        self.workspace_name = workspace_name
        self.comparison_metric = comparison_metric
        self.api = API(api_key=comet_api_key)

    def select_best_model(
        self,
        experiment_keys: pd.DataFrame,
    ) -> tuple[str, str]:
        """Selects best model based on validation performance.

        Args:
            experiment_keys: DataFrame with columns [model_name, experiment_key].

        Returns:
            Tuple of (best_model_name, best_model_experiment_key).

        Raises:
            ValueError: If no successful experiments found or no valid metrics.
        """
        if experiment_keys.shape[0] == 0:
            raise ValueError(
                "No successful experiments found. Please check the experiment logs."
            )

        # Get metrics for each experiment from Comet ML
        exp_scores = {}
        for _, row in experiment_keys.iterrows():
            model_name = row["0"]  # First column is model name
            exp_key = row["1"]  # Second column is experiment key

            try:
                experiment = self.api.get_experiment(
                    workspace=self.workspace_name,
                    project_name=self.project_name,
                    experiment=exp_key,
                )
                metric_value = experiment.get_metrics(self.comparison_metric)
                if metric_value:
                    # Get the latest value of the metric
                    exp_scores[model_name] = metric_value[-1]["metricValue"]
                    logger.debug(
                        "Model %s: %s = %s",
                        model_name,
                        self.comparison_metric,
                        exp_scores[model_name],
                    )
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(
                    "Failed to get metrics for %s (exp: %s): %s", model_name, exp_key, e
                )
                continue

        if not exp_scores:
            raise ValueError(
                f"No valid scores found for metric '{self.comparison_metric}' "
                "across experiments."
            )

        # Select the best performer (highest score)
        best_model_name = max(exp_scores, key=exp_scores.get)

        # Get experiment key for the best model
        best_model_exp_key = experiment_keys.loc[
            experiment_keys["0"] == best_model_name, "1"
        ].iloc[0]

        logger.info(
            "Selected best model: %s with %s = %.4f",
            best_model_name,
            self.comparison_metric,
            float(exp_scores[best_model_name]),
        )

        return best_model_name, best_model_exp_key
