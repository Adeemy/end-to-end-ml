"""
Backward-compatible wrappers for training workflows.

This module maintains the original ModelTrainer and VotingEnsembleCreator classes
for backward compatibility while delegating to the new focused orchestrator classes.
"""

from pathlib import PosixPath
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.training.utils.config import SupportedModelsConfig
from src.training.utils.ensemble import EnsembleOrchestrator
from src.training.utils.trainer import TrainingOrchestrator
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ModelTrainer:
    """Backward-compatible wrapper for TrainingOrchestrator.

    Delegates all work to TrainingOrchestrator while maintaining the original interface.
    """

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        train_features_preprocessed: pd.DataFrame,
        valid_features_preprocessed: pd.DataFrame,
        n_features: int,
        class_encoder: LabelEncoder,
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        artifacts_path: str,
        supported_models: SupportedModelsConfig,
        num_feature_names: Optional[list] = None,
        cat_feature_names: Optional[list] = None,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
    ):
        """Initializes ModelTrainer and its underlying orchestrator."""
        self.orchestrator = TrainingOrchestrator(
            train_features=train_features,
            train_class=train_class,
            valid_features=valid_features,
            valid_class=valid_class,
            train_features_preprocessed=train_features_preprocessed,
            valid_features_preprocessed=valid_features_preprocessed,
            n_features=n_features,
            class_encoder=class_encoder,
            preprocessor_step=preprocessor_step,
            selector_step=selector_step,
            artifacts_path=artifacts_path,
            supported_models=supported_models,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
            fbeta_score_beta=fbeta_score_beta,
            encoded_pos_class_label=encoded_pos_class_label,
        )

    def submit_train_exp(
        self,
        comet_api_key: str,
        project_name: str,
        comet_exp_name: str,
        model: Callable,
        search_space_params: dict,
        registered_model_name: str,
        max_search_iters: int = 100,
        optimize_in_parallel: bool = False,
        n_parallel_jobs: int = 4,
        model_opt_timeout_secs: int = 600,
        is_voting_ensemble: bool = False,
        ece_nbins: int = 5,
    ) -> tuple[Optional[Pipeline], Experiment]:
        """Delegates to TrainingOrchestrator.run_training_experiment()."""
        return self.orchestrator.run_training_experiment(
            comet_api_key=comet_api_key,
            project_name=project_name,
            experiment_name=comet_exp_name,
            model=model,
            search_space_params=search_space_params,
            registered_model_name=registered_model_name,
            max_search_iters=max_search_iters,
            optimize_in_parallel=optimize_in_parallel,
            n_parallel_jobs=n_parallel_jobs,
            model_opt_timeout_secs=model_opt_timeout_secs,
            is_voting_ensemble=is_voting_ensemble,
            ece_nbins=ece_nbins,
        )


class VotingEnsembleCreator:
    """Backward-compatible wrapper for EnsembleOrchestrator.

    Delegates all work to EnsembleOrchestrator while maintaining the original interface.
    """

    def __init__(
        self,
        comet_api_key: str,
        project_name: str,
        comet_exp_name: str,
        train_features: pd.DataFrame,
        valid_features: pd.DataFrame,
        train_class: np.ndarray,
        valid_class: np.ndarray,
        class_encoder: LabelEncoder,
        artifacts_path: str,
        supported_models: SupportedModelsConfig,
        lr_calib_pipeline: Optional[Pipeline] = None,
        rf_calib_pipeline: Optional[Pipeline] = None,
        lgbm_calib_pipeline: Optional[Pipeline] = None,
        xgb_calib_pipeline: Optional[Pipeline] = None,
        voting_rule: Literal["hard", "soft"] = "soft",
        encoded_pos_class_label: int = 1,
        fbeta_score_beta: float = 1.0,
        registered_model_name: Optional[str] = None,
        ece_nbins: int = 5,
    ):
        """Initializes VotingEnsembleCreator and its underlying orchestrator."""
        self.comet_api_key = comet_api_key
        self.project_name = project_name
        self.comet_exp_name = comet_exp_name
        self.registered_model_name = registered_model_name or "VotingEnsemble"
        self.ece_nbins = ece_nbins

        self.orchestrator = EnsembleOrchestrator(
            train_features=train_features,
            valid_features=valid_features,
            train_class=train_class,
            valid_class=valid_class,
            class_encoder=class_encoder,
            artifacts_path=artifacts_path,
            supported_models=supported_models,
            lr_calib_pipeline=lr_calib_pipeline,
            rf_calib_pipeline=rf_calib_pipeline,
            lgbm_calib_pipeline=lgbm_calib_pipeline,
            xgb_calib_pipeline=xgb_calib_pipeline,
            voting_rule=voting_rule,
            encoded_pos_class_label=encoded_pos_class_label,
            fbeta_score_beta=fbeta_score_beta,
        )

    def create_voting_ensemble(self) -> Union[Pipeline, Experiment]:
        """Delegates to EnsembleOrchestrator.create_voting_ensemble()."""
        return self.orchestrator.create_voting_ensemble(
            comet_api_key=self.comet_api_key,
            project_name=self.project_name,
            experiment_name=self.comet_exp_name,
            registered_model_name=self.registered_model_name,
            ece_nbins=self.ece_nbins,
        )
