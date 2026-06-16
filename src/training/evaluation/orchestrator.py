"""
Test set evaluation orchestration - evaluates models on held-out test data.
The selected model is then calibrated using validation data and registered as the champion
if it meets deployment criteria based on its performance on the test set.
"""

import os
from datetime import datetime
from pathlib import PosixPath
from typing import Any, Callable, Optional

# Load comet_ml early to avoid issues with sklearn auto-logging
import comet_ml  # pylint: disable=unused-import
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.evaluation import metrics
from src.training.evaluation.champion import ModelChampionManager
from src.training.evaluation.evaluator import create_model_evaluator
from src.training.evaluation.selector import ModelSelector
from src.training.tracking.experiment import (
    get_tracker_base_config,
    get_tracker_credentials,
)
from src.training.tracking.experiment_tracker import (
    CometExperimentTracker,
    ExperimentTracker,
    MLflowExperimentTracker,
)
from src.utils.logger import get_logger

module_name: str = PosixPath(__file__).stem
logger = get_logger(module_name)


class TrackerRegistry:
    """Registry for experiment tracker factories. It allows for easy addition
    of new tracker types without modifying existing code. It follows the factory
    pattern to create tracker instances based on registered factory functions. If
    new tracker types are needed, they can be registered without changing the core
    registry implementation.
    """

    def __init__(self):
        """Initialize the registry with an empty tracker dictionary."""
        self._trackers = {}

    def register(
        self, name: str, factory_func: Callable[..., ExperimentTracker]
    ) -> None:
        """Register a tracker factory function.

        Args:
            name: Name of the tracker type (e.g., 'comet', 'mlflow').
            factory_func: Factory function that creates the tracker instance.
        """
        self._trackers[name.lower()] = factory_func

    def create_tracker(self, tracker_type: str, **kwargs) -> ExperimentTracker:
        """Create a tracker instance using registered factory.

        Args:
            tracker_type: Type of tracker to create.
            **kwargs: Arguments to pass to the factory function.

        Returns:
            ExperimentTracker instance.

        Raises:
            ValueError: If tracker type is not registered.
        """
        factory_func = self._trackers.get(tracker_type.lower())
        if not factory_func:
            available_types = ", ".join(self._trackers.keys())
            raise ValueError(
                f"Unsupported tracker type: {tracker_type}. "
                f"Available types: {available_types}"
            )
        return factory_func(**kwargs)


def _create_comet_tracker(
    experiment_instance=None, **kwargs  # pylint: disable=unused-argument
) -> CometExperimentTracker:
    """Factory function for creating Comet tracker. If no experiment instance is provided,
    a dummy disabled experiment is created.

    Args:
        experiment_instance: Pre-initialized Comet experiment instance (optional).
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        CometExperimentTracker instance.
    """
    if experiment_instance is not None:
        return CometExperimentTracker(experiment=experiment_instance)
    else:
        # Create a dummy experiment for now - will be set later via set_experiment
        from comet_ml import Experiment

        dummy_experiment = Experiment(disabled=True)
        return CometExperimentTracker(experiment=dummy_experiment)


def _create_mlflow_tracker(
    run_id=None, **kwargs  # pylint: disable=unused-argument
) -> MLflowExperimentTracker:
    """Factory function for creating MLflow tracker. If no run_id is provided,
    a new run will be started.

    Args:
        run_id: Optional MLflow run ID.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        MLflowExperimentTracker instance.
    """
    return MLflowExperimentTracker(run_id=run_id)


# Create and configure the default tracker registry
_default_tracker_registry = TrackerRegistry()
_default_tracker_registry.register("comet", _create_comet_tracker)
_default_tracker_registry.register("mlflow", _create_mlflow_tracker)


def create_evaluation_orchestrator(
    tracker_type: str,
    train_features: pd.DataFrame,
    train_class: np.ndarray,
    test_features: pd.DataFrame,
    test_class: np.ndarray,
    artifacts_path: str,
    fbeta_score_beta: float = 1.0,
    voting_ensemble_name: Optional[str] = None,
    decision_threshold: float = 0.5,
    tune_decision_threshold: bool = False,
    encoded_pos_class_label: int = 1,
    task_type: str = "binary",
    cv_folds: int = 1,
    model_preference: Optional[list] = None,
    random_seed: Optional[int] = None,
    experiment_instance: Optional[Any] = None,
    **tracker_kwargs,
) -> "TestSetEvaluationOrchestrator":
    """Factory function to create TestSetEvaluationOrchestrator with appropriate tracker.

    Args:
        tracker_type: Type of tracker to use (e.g., 'comet', 'mlflow').
        train_features: Training features.
        train_class: Training class labels.
        test_features: Test features.
        test_class: Test class labels.
        artifacts_path: Path to model artifacts.
        fbeta_score_beta: Beta value for fbeta score.
        voting_ensemble_name: Name of voting ensemble model (if exists).
        decision_threshold: Default operating threshold persisted with the champion.
        tune_decision_threshold: If True, tune the threshold on the calibration set.
        encoded_pos_class_label: Encoded label of the positive class.
        task_type: ML task ("binary", "multiclass", "regression").
        cv_folds: When > 1, champion selection uses a bootstrap SE + 1-SE rule.
        model_preference: Model-name order (simplest first) for 1-SE tie-breaking.
        random_seed: Seed for the selection bootstrap.
        experiment_instance: Pre-initialized experiment instance (for Comet).
        **tracker_kwargs: Additional arguments for tracker initialization.

    Returns:
        TestSetEvaluationOrchestrator instance with appropriate tracker.

    Raises:
        ValueError: If unsupported tracker_type is provided.
    """

    # Create tracker using registry pattern
    tracker = _default_tracker_registry.create_tracker(
        tracker_type=tracker_type,
        experiment_instance=experiment_instance,
        **tracker_kwargs,
    )

    return TestSetEvaluationOrchestrator(
        tracker=tracker,
        train_features=train_features,
        train_class=train_class,
        test_features=test_features,
        test_class=test_class,
        artifacts_path=artifacts_path,
        fbeta_score_beta=fbeta_score_beta,
        voting_ensemble_name=voting_ensemble_name,
        decision_threshold=decision_threshold,
        tune_decision_threshold=tune_decision_threshold,
        encoded_pos_class_label=encoded_pos_class_label,
        task_type=task_type,
        cv_folds=cv_folds,
        model_preference=model_preference,
        random_seed=random_seed,
    )


class TestSetEvaluationOrchestrator:
    """Orchestrates model evaluation on test set and champion model registration.

    Single Responsibility: Coordinate test evaluation and champion selection.
    Dependency Inversion: Depends on abstractions (ModelSelector, ModelEvaluator, ExperimentTracker).
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        test_features: pd.DataFrame,
        test_class: np.ndarray,
        artifacts_path: str,
        fbeta_score_beta: float = 1.0,
        voting_ensemble_name: Optional[str] = None,
        decision_threshold: float = 0.5,
        tune_decision_threshold: bool = False,
        encoded_pos_class_label: int = 1,
        task_type: str = "binary",
        cv_folds: int = 1,
        model_preference: Optional[list] = None,
        random_seed: Optional[int] = None,
    ):
        """Initializes the TestSetEvaluationOrchestrator.

        Args:
            tracker: Experiment tracker instance for logging.
            train_features: Training features.
            train_class: Training class labels.
            test_features: Test features.
            test_class: Test class labels.
            artifacts_path: Path to model artifacts.
            fbeta_score_beta: Beta value for fbeta score.
            voting_ensemble_name: Name of voting ensemble model (if exists).
            decision_threshold: Default operating threshold persisted with the champion.
            tune_decision_threshold: If True, pick the threshold maximizing F-beta
                on the calibration set instead of using ``decision_threshold``.
            encoded_pos_class_label: Encoded label of the positive class.
            task_type: ML task ("binary", "multiclass", "regression"); drives the
                evaluator and metric resolution.
            cv_folds: When > 1, champion selection uses a bootstrap standard error
                of the selection metric and applies a 1-SE rule.
            model_preference: Optional model-name order (simplest/cheapest first)
                used to break ties under the 1-SE rule.
            random_seed: Seed for the selection bootstrap (reproducibility).
        """
        self.tracker = tracker
        self.train_features = train_features
        self.train_class = train_class
        self.test_features = test_features
        self.test_class = test_class
        self.artifacts_path = artifacts_path
        self.fbeta_score_beta = fbeta_score_beta
        self.voting_ensemble_name = voting_ensemble_name
        self.decision_threshold = decision_threshold
        self.tune_decision_threshold = tune_decision_threshold
        self.encoded_pos_class_label = encoded_pos_class_label
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.model_preference = model_preference
        self.random_seed = random_seed

    def evaluate_on_test_set(
        self,
        model_pipeline: "Pipeline",
        model_name: str,
        decision_threshold: float = 0.5,
    ) -> dict:
        """Evaluates a model on the held-out test set.

        The test metrics must reflect the artifact that is actually deployed, so
        callers pass the *calibrated* pipeline and its persisted operating
        threshold here (binary). The threshold is ignored for multi-class
        (argmax) and regression.

        Args:
            model_pipeline: Fitted model pipeline to evaluate (calibrated champion).
            model_name: Name of the model being evaluated.
            decision_threshold: Operating threshold applied for binary tasks so
                reported metrics match the deployed operating point.

        Returns:
            Dictionary of test metrics.
        """

        is_voting_ensemble = (
            model_name == self.voting_ensemble_name
            if self.voting_ensemble_name
            else False
        )

        evaluator = create_model_evaluator(
            tracker=self.tracker,
            pipeline=model_pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=self.test_features,
            valid_class=self.test_class,
            fbeta_score_beta=self.fbeta_score_beta,
            encoded_pos_class_label=self.encoded_pos_class_label,
            is_voting_ensemble=is_voting_ensemble,
            task_type=self.task_type,
        )

        # Create class encoder for confusion matrix logging (classification only).
        # Fit on combined train and test class labels to ensure all labels are known.
        class_encoder = None
        if self.task_type in metrics.CLASSIFICATION_TASKS:
            class_encoder = LabelEncoder()
            all_class_labels = np.concatenate([self.train_class, self.test_class])
            class_encoder.fit(all_class_labels)

        # Evaluate at the deployed operating threshold so test metrics match the
        # served model rather than a default 0.5 cut.
        test_scores = evaluator.evaluate_test_set_only(
            class_encoder=class_encoder,
            pos_class_label_thresh=decision_threshold,
        )

        test_metrics = evaluator.convert_metrics_from_df_to_dict(
            scores=test_scores, prefix="test_"
        )

        logger.info(
            "Evaluated %s on test set (threshold=%.3f). Test metrics: %s",
            model_name,
            decision_threshold,
            test_metrics,
        )

        return test_metrics

    def calibrate_and_resolve_threshold(
        self,
        model_pipeline: "Pipeline",
        model_name: str,
        calibration_features: pd.DataFrame,
        calibration_class: np.ndarray,
        champion_manager: ModelChampionManager,
    ) -> tuple["Pipeline", float]:
        """Calibrates the champion and resolves its serving threshold.

        Runs BEFORE test evaluation so the test set is scored on the exact
        artifact that gets deployed (calibrated pipeline at its operating
        threshold). Calibration and threshold tuning use the dedicated
        calibration split, disjoint from the selection (validation) data. For
        non-classification tasks calibration is skipped.

        Args:
            model_pipeline: Selected (uncalibrated) champion pipeline.
            model_name: Name of the champion model.
            calibration_features: Calibration features (held-out from train/valid).
            calibration_class: Calibration class labels.
            champion_manager: ModelChampionManager instance.

        Returns:
            tuple: (deployable_pipeline, decision_threshold).
        """
        if self.task_type not in metrics.CLASSIFICATION_TASKS:
            # Regression: no probability calibration or threshold.
            return model_pipeline, self.decision_threshold

        calibrated_pipeline = champion_manager.calibrate_pipeline(
            valid_features=calibration_features,
            valid_class=calibration_class,
            fitted_pipeline=model_pipeline,
        )
        logger.info("Calibrated champion model: %s", model_name)

        decision_threshold = self._resolve_decision_threshold(
            calibrated_pipeline, calibration_features, calibration_class
        )
        return calibrated_pipeline, decision_threshold

    def register_champion(
        self,
        calibrated_pipeline: "Pipeline",
        model_name: str,
        decision_threshold: float,
        champion_manager: ModelChampionManager,
    ) -> None:
        """Persists serving metadata and registers the (already-calibrated) champion.

        Called only after the calibrated, thresholded model passes the deployment
        gate, so we never register a model that failed the gate.

        Args:
            calibrated_pipeline: The deployable (calibrated) champion pipeline.
            model_name: Name of the champion model.
            decision_threshold: Operating threshold to persist for serving.
            champion_manager: ModelChampionManager instance.
        """
        # Persist the operating point next to the model so inference uses the
        # same threshold instead of a hardcoded 0.5.
        metadata_path = champion_manager.save_model_metadata(
            local_path=self.artifacts_path,
            decision_threshold=decision_threshold,
            encoded_pos_class_label=self.encoded_pos_class_label,
        )
        logger.info(
            "Champion decision threshold %.3f saved to %s",
            decision_threshold,
            metadata_path,
        )

        # Set tracker and register
        champion_manager.tracker = self.tracker
        champion_manager.log_and_register_champ_model(
            local_path=self.artifacts_path,
            pipeline=calibrated_pipeline,
        )

        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        logger.info(
            "Registered champion model: %s in evaluation experiment: %s",
            model_name,
            evaluation_exp_name,
        )

        # Save the CALIBRATED champion locally so the local artifact matches the
        # registered one. (Dumping the un-calibrated pipeline here would overwrite
        # the calibrated pickle written by log_and_register_champ_model.)
        champion_model_path = (
            f"{self.artifacts_path}/{champion_manager.champ_model_name}.pkl"
        )
        joblib.dump(calibrated_pipeline, champion_model_path)
        logger.info("Saved champion model to: %s", champion_model_path)

    def _resolve_decision_threshold(
        self,
        calibrated_pipeline: "Pipeline",
        calibration_features: pd.DataFrame,
        calibration_class: np.ndarray,
    ) -> float:
        """Returns the operating threshold to persist with the champion.

        Tunes the threshold on the calibration set (maximizing F-beta) when
        ``tune_decision_threshold`` is enabled; otherwise returns the configured
        default. Only meaningful for binary models — multiclass keeps the default.

        Args:
            calibrated_pipeline: The calibrated champion pipeline.
            calibration_features: Calibration features.
            calibration_class: Calibration class labels.

        Returns:
            float: decision threshold in (0, 1).
        """

        # Only binary classification has a single operating threshold to tune.
        if self.task_type not in metrics.CLASSIFICATION_TASKS:
            return self.decision_threshold

        classes = list(calibrated_pipeline.classes_)
        is_binary = len(classes) == 2
        if not self.tune_decision_threshold or not is_binary:
            return self.decision_threshold

        pos_col = classes.index(self.encoded_pos_class_label)
        pos_class_proba = calibrated_pipeline.predict_proba(calibration_features)[
            :, pos_col
        ]
        return ModelChampionManager.tune_decision_threshold(
            valid_class=calibration_class,
            pos_class_proba=pos_class_proba,
            fbeta_beta=self.fbeta_score_beta,
        )

    def _create_standalone_evaluation_kwargs(
        self, experiment_kwargs: dict, experiment_name: str
    ) -> dict:
        """Create tracker-agnostic kwargs for standalone evaluation experiments.

        Args:
            experiment_kwargs: Original experiment configuration.
            experiment_name: Name for the evaluation experiment.

        Returns:
            Dictionary of experiment kwargs for the specific tracker.
        """
        tracker_type = experiment_kwargs.get("experiment_tracker_type", "comet")

        # Start with base configuration
        kwargs = {
            "experiment_name": experiment_name,
        }

        # Add tracker-specific base configuration using registry
        try:
            base_config = get_tracker_base_config(tracker_type, experiment_kwargs)
            kwargs.update(base_config)
        except ValueError:
            logger.warning(
                "Unknown tracker type: %s, using generic fallback", tracker_type
            )
            # Generic fallback - include non-tracker-type keys
            kwargs.update(
                {
                    k: v
                    for k, v in experiment_kwargs.items()
                    if k not in ["experiment_tracker_type"]
                }
            )

        # Add credentials using the credential provider
        try:
            credentials = get_tracker_credentials(tracker_type)
            kwargs.update(credentials)
        except ValueError:
            logger.warning(
                "Could not get credentials for tracker type: %s", tracker_type
            )

        return kwargs

    def _resolve_candidate_models(
        self,
        experiment_keys: Optional[pd.DataFrame],
        model_selector: ModelSelector,
        max_eval_experiments: int,
    ) -> list:
        """Resolves the candidate (model_name, experiment_key) pairs to compare.

        Uses the experiment keys passed from training when available; otherwise
        falls back to the tracker only to *enumerate* the best candidate (the
        selection metric is still recomputed in-process by the caller).

        Args:
            experiment_keys: Optional DataFrame of [model_name, experiment_key].
            model_selector: ModelSelector used for the standalone discovery fallback.
            max_eval_experiments: Max recent experiments to consider in the fallback.

        Returns:
            list: candidate (model_name, experiment_key) tuples.
        """
        if experiment_keys is not None and not experiment_keys.empty:
            return [
                (str(row[0]), str(row[1]))
                for row in experiment_keys.itertuples(index=False)
            ]

        # Standalone fallback: let the selector enumerate the best candidate.
        best_model_name, best_experiment_key = model_selector.select_best_model(
            experiment_keys=experiment_keys, max_experiments=max_eval_experiments
        )
        return [(best_model_name, best_experiment_key)]

    def _load_candidate_pipeline(
        self, model_name: str, experiment_key: Optional[str]
    ) -> Optional["Pipeline"]:
        """Loads a candidate pipeline from local artifacts, else the tracker.

        Args:
            model_name: Registered model name.
            experiment_key: Experiment key for tracker download (may be None).

        Returns:
            Optional[Pipeline]: the loaded pipeline, or None if it cannot be loaded.
        """
        model_path = f"{self.artifacts_path}/{model_name}.pkl"
        if os.path.exists(model_path):
            try:
                pipeline = joblib.load(model_path)
                logger.info("Loaded model from local path: %s", model_path)
                return pipeline
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Could not load model from local path %s: %s", model_path, exc
                )
                return None

        if not experiment_key:
            logger.warning("No local artifact or experiment key for %s", model_name)
            return None

        try:
            return self._download_model_from_comet(
                experiment_key=experiment_key,
                model_name=model_name,
                save_path=model_path,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Could not load model %s from workspace: %s", model_name, exc
            )
            return None

    def _make_evaluator(
        self,
        pipeline: "Pipeline",
        model_name: str,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
    ):
        """Builds a task-appropriate evaluator for scoring a candidate."""
        return create_model_evaluator(
            tracker=self.tracker,
            pipeline=pipeline,
            train_features=self.train_features,
            train_class=self.train_class,
            valid_features=valid_features,
            valid_class=valid_class,
            fbeta_score_beta=self.fbeta_score_beta,
            encoded_pos_class_label=self.encoded_pos_class_label,
            is_voting_ensemble=(model_name == self.voting_ensemble_name),
            task_type=self.task_type,
        )

    def _predict_for_metrics(
        self, pipeline: "Pipeline", valid_features: pd.DataFrame
    ) -> tuple:
        """Returns (pred_class, proba_for_metrics) for a candidate pipeline.

        proba_for_metrics is the positive-class 1-D array (binary), the full
        probability matrix (multi-class), or None (regression / no predict_proba).
        """
        pred_class = pipeline.predict(valid_features)
        if self.task_type == metrics.REGRESSION or not hasattr(
            pipeline, "predict_proba"
        ):
            return pred_class, None
        proba = np.asarray(pipeline.predict_proba(valid_features))
        classes = list(pipeline.classes_)
        if len(classes) == 2:
            pos_col = (
                classes.index(self.encoded_pos_class_label)
                if self.encoded_pos_class_label in classes
                else 1
            )
            return pred_class, proba[:, pos_col]
        return pred_class, proba

    @staticmethod
    def _metric_from_predictions(
        evaluator,
        y_true: np.ndarray,
        pred_class: np.ndarray,
        pred_proba: Optional[np.ndarray],
        comparison_metric_name: str,
    ) -> float:
        """Extracts the comparison metric from precomputed predictions."""
        scores = evaluator.calc_perf_metrics(
            true_class=y_true, pred_class=pred_class, pred_proba=pred_proba
        )
        row = scores.loc[scores["Metric"] == comparison_metric_name, "Score"]
        return float(row.iloc[0]) if not row.empty else float("-inf")

    def _score_pipeline_on_valid(
        self,
        pipeline: "Pipeline",
        model_name: str,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        comparison_metric_name: str,
    ) -> float:
        """Computes the comparison metric for a pipeline on the validation set.

        Args:
            pipeline: Candidate fitted pipeline.
            model_name: Name of the candidate (used to flag the voting ensemble).
            valid_features: Validation features.
            valid_class: Validation class labels.
            comparison_metric_name: Metric name as produced by calc_perf_metrics.

        Returns:
            float: the metric value, or -inf if it cannot be computed.
        """
        evaluator = self._make_evaluator(
            pipeline, model_name, valid_features, valid_class
        )
        pred_class, pred_proba = self._predict_for_metrics(pipeline, valid_features)
        return self._metric_from_predictions(
            evaluator, valid_class, pred_class, pred_proba, comparison_metric_name
        )

    def _bootstrap_metric_se(
        self,
        pipeline: "Pipeline",
        model_name: str,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        comparison_metric_name: str,
        n_boot: int = 200,
    ) -> tuple:
        """Returns (point_estimate, bootstrap_SE) of the comparison metric.

        Predictions are computed once on the fixed (already-trained) candidate;
        the validation set is then resampled with replacement to estimate the
        metric's sampling variability without refitting.

        Args:
            pipeline: Candidate fitted pipeline.
            model_name: Candidate name (voting-ensemble flag).
            valid_features: Validation features.
            valid_class: Validation labels.
            comparison_metric_name: Metric row name to bootstrap.
            n_boot: Number of bootstrap resamples.

        Returns:
            tuple: (point_estimate, standard_error).
        """
        evaluator = self._make_evaluator(
            pipeline, model_name, valid_features, valid_class
        )
        pred_class, pred_proba = self._predict_for_metrics(pipeline, valid_features)
        y_true = np.asarray(valid_class)
        pred_arr = np.asarray(pred_class)
        point = self._metric_from_predictions(
            evaluator, y_true, pred_arr, pred_proba, comparison_metric_name
        )

        n_samples = len(y_true)
        if n_samples == 0:
            return point, 0.0
        rng = np.random.default_rng(
            self.random_seed if self.random_seed is not None else 0
        )
        boot_scores = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_samples, n_samples)
            proba_b = None if pred_proba is None else pred_proba[idx]
            value = self._metric_from_predictions(
                evaluator, y_true[idx], pred_arr[idx], proba_b, comparison_metric_name
            )
            if np.isfinite(value):
                boot_scores.append(value)
        se = float(np.std(boot_scores)) if boot_scores else 0.0
        return float(point), se

    def _select_champion_in_process(
        self,
        candidates: list,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        comparison_metric_name: str,
    ) -> tuple:
        """Selects the champion by recomputing the comparison metric on validation.

        Loads each candidate pipeline and scores it in-process on the validation
        set, so champion selection does not depend on metrics read back from the
        experiment tracker. When CV is enabled (``cv_folds > 1``) each candidate
        also gets a bootstrap standard error and a 1-SE rule is applied: among
        candidates whose mean is within one SE of the best, the most preferred
        (simplest/cheapest per ``model_preference``) is chosen, guarding against
        selecting on noise.

        Args:
            candidates: List of (model_name, experiment_key) tuples.
            valid_features: Validation features.
            valid_class: Validation class labels.
            comparison_metric_name: Metric to rank on (e.g. 'f_0.5_score', 'roc_auc').

        Returns:
            tuple: (champion_model_name, champion_pipeline_or_None).
        """
        use_se = bool(self.cv_folds and self.cv_folds > 1)
        scored = []  # list of (model_name, pipeline, mean, se)
        for model_name, experiment_key in candidates:
            pipeline = self._load_candidate_pipeline(model_name, experiment_key)
            if pipeline is None:
                continue
            if use_se:
                mean_score, se_score = self._bootstrap_metric_se(
                    pipeline,
                    model_name,
                    valid_features,
                    valid_class,
                    comparison_metric_name,
                )
            else:
                mean_score = self._score_pipeline_on_valid(
                    pipeline,
                    model_name,
                    valid_features,
                    valid_class,
                    comparison_metric_name,
                )
                se_score = 0.0
            logger.info(
                "Candidate %s scored %.4f (+/- %.4f) on %s (valid)",
                model_name,
                mean_score,
                se_score,
                comparison_metric_name,
            )
            scored.append((model_name, pipeline, mean_score, se_score))

        if not scored:
            return (candidates[0][0] if candidates else "unknown"), None

        best_name, best_pipeline, best_mean, best_se = max(
            scored, key=lambda item: item[2]
        )

        # 1-SE rule (only when variance is available): among candidates within
        # one SE of the best mean, prefer the simplest/cheapest model.
        if use_se and best_se > 0 and self.model_preference:
            within_one_se = [item for item in scored if item[2] >= best_mean - best_se]
            if len(within_one_se) > 1:

                def _preference_rank(name: str) -> int:
                    return (
                        self.model_preference.index(name)
                        if name in self.model_preference
                        else len(self.model_preference)
                    )

                chosen = min(within_one_se, key=lambda item: _preference_rank(item[0]))
                if chosen[0] != best_name:
                    logger.info(
                        "1-SE rule: preferring %s over %s (within %.4f of best mean %.4f)",
                        chosen[0],
                        best_name,
                        best_se,
                        best_mean,
                    )
                best_name, best_pipeline = chosen[0], chosen[1]

        return best_name, best_pipeline

    def run_evaluation_workflow(
        self,
        model_selector: ModelSelector,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        calibration_features: pd.DataFrame,
        calibration_class: np.ndarray,
        champion_manager: ModelChampionManager,
        comparison_metric_name: str,
        deployment_threshold: float,
        experiment_keys: Optional[pd.DataFrame] = None,
        max_eval_experiments: int = 50,
        **experiment_kwargs,
    ) -> tuple[str, dict]:
        """Runs complete evaluation workflow: select, evaluate, calibrate, register.

        Selection uses ``valid`` (recomputed in-process), while calibration and
        threshold tuning use the dedicated, disjoint ``calibration`` split.

        Args:
            model_selector: ModelSelector instance.
            valid_features: Validation features for in-process model selection.
            valid_class: Validation class labels for in-process model selection.
            calibration_features: Calibration features (calibration + threshold).
            calibration_class: Calibration class labels (calibration + threshold).
            champion_manager: ModelChampionManager instance.
            comparison_metric_name: Metric name for deployment decision.
            deployment_threshold: Minimum score required for deployment.
            experiment_keys: Optional DataFrame with model names and experiment keys.
                           If None, ModelSelector will query the tracking backend directly.
            max_eval_experiments: Maximum number of recent experiments to consider.
            **experiment_kwargs: Additional arguments for experiment setup (e.g., api_key, experiment_key).

        Returns:
            Tuple of (champion_model_name, test_metrics).

        Raises:
            ValueError: If test score is below deployment threshold.
        """
        # Select the champion IN-PROCESS: recompute the comparison metric on the
        # validation set for each candidate pipeline, instead of trusting the
        # metric values read back from the experiment tracker.
        candidates = self._resolve_candidate_models(
            experiment_keys, model_selector, max_eval_experiments
        )
        best_model_name, model_pipeline = self._select_champion_in_process(
            candidates=candidates,
            valid_features=valid_features,
            valid_class=valid_class,
            comparison_metric_name=comparison_metric_name,
        )
        # Recover the champion's experiment key (used below only for tracker
        # parent/child experiment linking, not for selection).
        best_experiment_key = dict(candidates).get(best_model_name)

        if model_pipeline is None:
            logger.warning("No model pipeline available for evaluation")
            return best_model_name, {}

        # Handle experiment creation based on tracker type and available experiment ID
        if hasattr(self.tracker, "set_experiment"):
            tracker_type = experiment_kwargs.get("experiment_tracker_type", "comet")

            # Check if we're running from training pipeline (with experiment_keys) or standalone
            if experiment_keys is not None and best_experiment_key:
                # Running from training pipeline - create child/linked experiment
                evaluation_kwargs = {
                    "is_child_experiment": True,
                }

                # Add tracker-specific parent/child linking
                if tracker_type == "comet":
                    evaluation_kwargs["experiment_key"] = best_experiment_key
                elif tracker_type == "mlflow":
                    evaluation_kwargs["parent_run_id"] = best_experiment_key
                # Additional trackers can be added here without modifying existing code

                # Add credentials using the credential provider
                try:
                    credentials = get_tracker_credentials(tracker_type)
                    evaluation_kwargs.update(credentials)
                except ValueError:
                    logger.warning(
                        "Could not get credentials for tracker type: %s", tracker_type
                    )

                self.tracker.set_experiment(**evaluation_kwargs)
                logger.info(
                    "Creating child/linked evaluation experiment for model: %s (parent: %s)",
                    best_model_name,
                    best_experiment_key,
                )
            else:
                # Running standalone - create independent evaluation experiment
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                evaluation_experiment_name = (
                    f"eval_{best_model_name.replace('-', '_')}_{timestamp}"
                )
                evaluation_kwargs = self._create_standalone_evaluation_kwargs(
                    experiment_kwargs, evaluation_experiment_name
                )

                self.tracker.set_experiment(**evaluation_kwargs)
                logger.info(
                    "Creating standalone evaluation experiment: %s",
                    evaluation_experiment_name,
                )

        # Calibrate and resolve the operating threshold on the dedicated
        # calibration split BEFORE test evaluation, so the test set is scored on
        # the exact artifact that gets deployed (calibrated pipeline at its
        # serving threshold) rather than an uncalibrated model at 0.5.
        deployable_pipeline, decision_threshold = self.calibrate_and_resolve_threshold(
            model_pipeline=model_pipeline,
            model_name=best_model_name,
            calibration_features=calibration_features,
            calibration_class=calibration_class,
            champion_manager=champion_manager,
        )

        # Evaluate the deployable model on the held-out test set at its threshold.
        test_metrics = self.evaluate_on_test_set(
            model_pipeline=deployable_pipeline,
            model_name=best_model_name,
            decision_threshold=decision_threshold,
        )

        # Log test metrics with evaluation experiment context
        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        logger.info(
            "Evaluated %s on test set in experiment: %s. Test metrics: %s",
            best_model_name,
            evaluation_exp_name,
            test_metrics,
        )
        self.tracker.log_metrics(test_metrics)

        # Check deployment threshold against the calibrated/thresholded test score.
        metric_key = f"test_{comparison_metric_name}"
        test_score = test_metrics.get(metric_key)

        if test_score is None:
            available_metrics = ", ".join(test_metrics.keys())
            raise ValueError(
                f"Metric '{metric_key}' not found in test metrics. "
                f"Available metrics: {available_metrics}"
            )

        test_score = float(test_score)
        try:
            evaluation_exp_name = (
                self.tracker.experiment.get_name()
                if hasattr(self.tracker, "experiment")
                else "unknown"
            )
        except AttributeError:
            evaluation_exp_name = "unknown"
        if test_score < deployment_threshold:
            logger.error(
                "Deployment check failed in experiment %s: Best model score (%.4f) is below threshold (%.4f). Model not deployed.",
                evaluation_exp_name,
                test_score,
                deployment_threshold,
            )
            raise ValueError(
                f"Best model score ({test_score:.4f}) is below deployment "
                f"threshold ({deployment_threshold:.4f}). Model not deployed."
            )

        logger.info(
            "Deployment check passed in experiment %s: Test score (%s: %.4f) meets threshold (%.4f)",
            evaluation_exp_name,
            comparison_metric_name,
            test_score,
            deployment_threshold,
        )

        # Register the calibrated champion only after it clears the gate.
        self.register_champion(
            calibrated_pipeline=deployable_pipeline,
            model_name=best_model_name,
            decision_threshold=decision_threshold,
            champion_manager=champion_manager,
        )

        # End experiment if tracker supports it
        if hasattr(self.tracker, "end_experiment"):
            self.tracker.end_experiment()

        return best_model_name, test_metrics

    def _download_model_from_comet(
        self,
        experiment_key: str,
        model_name: str,
        save_path: str,
    ) -> "Pipeline":
        """Downloads model from Comet ML experiment.

        Args:
            experiment_key: Comet ML experiment key.
            model_name: Name of the model to download.
            save_path: Local path to save the downloaded model.

        Returns:
            Loaded model pipeline.
        """
        # Login and get API
        comet_ml.login()
        api = comet_ml.API()

        # Get experiment
        experiment = api.get_experiment_by_key(experiment_key)

        # Download model assets
        assets = experiment.get_asset_list()
        model_assets = [
            asset
            for asset in assets
            if asset["fileName"].endswith(".pkl") and model_name in asset["fileName"]
        ]

        if not model_assets:
            raise FileNotFoundError(
                f"No model file found for {model_name} in experiment {experiment_key}. "
                f"Available assets: {len(assets)} total. "
                f"Model files (.pkl): {len([a for a in assets if a['fileName'].endswith('.pkl')])}"
            )

        # Download the model file
        model_asset = model_assets[0]  # Take the first matching model
        asset_id = model_asset["assetId"]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download and save
        experiment.get_asset(asset_id, save_path)
        logger.info("Downloaded model from Comet ML to: %s", save_path)

        # Load and return the model
        return joblib.load(save_path)
