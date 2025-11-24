# Training Module

Automated machine learning training with hyperparameter optimization and experiment tracking.

## Quick Commands

```bash
make split_data    # Create train/validation/test splits
make train         # Train models with hyperparameter optimization
make evaluate      # Evaluate models and select and calibrates champion before registering it ML workspace
```

## Pipeline Overview

1. **Data Splitting** → Load from feature store, create stratified splits
2. **Training** → Hyperparameter optimization + model training with Optuna
3. **Evaluation** → Test evaluation + champion model registration

## Supported Models

- **Logistic Regression**: Linear baseline with regularization
- **Random Forest**: Ensemble method with feature importance
- **LightGBM**: Gradient boosting with categorical feature support
- **XGBoost**: High-performance gradient boosting
- **Voting Ensemble**: Combination of models

## Key Features

- **Automated Hyperparameter Tuning**: Optuna-based optimization
- **Experiment Tracking**: Complete logging with Comet ML
- **Model Evaluation**: F-beta score optimization with configurable beta
- **Champion Selection**: Automatic best model selection and registration
- **Reproducibility**: Deterministic training with seed management

## Core Scripts

- **`split_data.py`**: Creates stratified train/validation/test splits from feature store
- **`train.py`**: Runs hyperparameter optimization and model training
- **`evaluate.py`**: Evaluates trained models and registers champion

#### `TrainingOrchestrator` (trainer.py)
Orchestrates the complete training workflow.
- Coordinates model optimization and evaluation
- Manages experiment lifecycle
- Handles model registration

#### `ModelOptimizer` (optimizer.py)
Handles hyperparameter optimization using Optuna.
- Defines search spaces for each model type
- Runs Bayesian optimization
- Logs optimization progress to Comet ML

### Experiment Tracking (`utils/tracking/`)
Sets up experiment tracking system. Currently, it uses comet for experiemnt tracking
but it can be extended to other ML experiment trackers.

#### `CometExperimentManager` (experiment.py)
Manages Comet ML experiment lifecycle.
- Creates and configures experiments
- Logs parameters, metrics, and artifacts
- Handles model registration

#### `CometExperimentTracker` (experiment_tracker.py)
Provides interface for logging to Comet ML.
- Metric and parameter logging
- Figure and confusion matrix logging
- Model artifact management

### Evaluation (`utils/evaluation/`)

#### `ModelSelector` (selector.py)
Discovers and selects best models from experiments.
- Queries Comet ML for recent experiments
- Extracts model performance metrics
- Selects champion based on validation scores

#### `EvaluationOrchestrator` (orchestrator.py)
Manages model evaluation workflow.
- Loads and evaluates trained models
- Generates comprehensive evaluation metrics
- Handles champion model registration

## Configuration

Training behavior is controlled through `src/config/training-config.yml`:

```yaml
train:
  # Model configuration
  supported_models: ["LGBMClassifier", "LogisticRegression", ...]

  # Optimization settings
  max_search_iters: 100
  model_opt_timeout_secs: 180

  # Experiment tracking
  experiment_tracker: "comet_ml"
  project_name: "end-to-end-ml"
  workspace_name: "your-workspace"

  # Evaluation metrics
  comparison_metric: "fbeta_score"
  fbeta_score_beta_val: 0.5
  deployment_score_thresh: 0.7
```

## Usage Examples

### Basic Training
```bash
# Recommended: Use makefile
make train

# Or run directly with Python
python ./src/training/train.py --config_yaml_path ./src/config/training-config.yml
```

### Programmatic Training
```python
import os
from pathlib import Path
from src.training.train import main
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR, ARTIFACTS_DIR

# Set up environment
os.environ["COMET_API_KEY"] = "your_api_key"
logger = get_console_logger("training")

# Run training pipeline
experiment_keys = main(
    config_yaml_path="./src/config/training-config.yml",
    api_key=os.environ["COMET_API_KEY"],
    data_dir=DATA_DIR,
    artifacts_dir=ARTIFACTS_DIR,
    logger=logger,
    run_evaluation=False  # Set to True to run evaluation after training
)

print(f"Completed {len(experiment_keys)} experiments")
```

### Custom Model Training
```python
from src.training.utils.core.trainer import TrainingOrchestrator
from src.training.utils.tracking.experiment import CometExperimentManager

# Initialize orchestrator
trainer = TrainingOrchestrator(
    experiment_manager=CometExperimentManager(),
    train_features=train_features,
    train_class=train_class,
    valid_features=valid_features,
    valid_class=valid_class,
    # ... other parameters
)

# Optimize specific model
from lightgbm import LGBMClassifier
study, optimizer = trainer.optimize_model(
    tracker=tracker,
    model=LGBMClassifier,
    search_space_params=lgbm_search_space,
    registered_model_name="lightgbm_model"
)
```

### Model Evaluation
```python
from src.training.utils.evaluation.selector import ModelSelector

# Select best model
selector = ModelSelector(
    project_name="end-to-end-ml",
    workspace_name="your-workspace",
    comparison_metric="validation_score"
)

best_model_name, best_experiment_key = selector.select_best_model()
```

## Experiment Tracking

### Comet ML Integration
All experiments are automatically logged to Comet ML with:
- **Parameters**: Model hyperparameters and configuration
- **Metrics**: Training and validation scores over time
- **Artifacts**: Trained models, plots, confusion matrices
- **Code**: Source code and git information
- **System Metrics**: CPU, memory, GPU utilization

### Experiment Discovery
The evaluation pipeline automatically discovers experiments by:
1. Querying Comet ML API for recent experiments
2. Extracting model names from experiment names
3. Retrieving validation metrics for comparison
4. Selecting best performing model

## Model Support

### Supported Algorithms
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble of decision trees
- **LightGBM**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **Voting Ensemble**: Combination of top models

### Hyperparameter Search Spaces
Each model has optimized search spaces defined in the configuration:
- Bayesian optimization using Optuna
- Model-specific parameter ranges
- Early stopping for efficiency

## Best Practices

### Training
1. **Reproducibility**: Set random seeds in configuration
2. **Cross-Validation**: Use stratified splits for imbalanced data
3. **Early Stopping**: Configure timeouts for long-running optimizations
4. **Resource Management**: Monitor GPU/CPU usage during training

### Experiment Tracking
1. **Naming Convention**: Use descriptive experiment names with timestamps
2. **Metadata**: Log all relevant parameters and configuration
3. **Artifacts**: Save models, plots, and evaluation results
4. **Organization**: Use consistent project and workspace structure

### Evaluation
1. **Holdout Test Set**: Never use test data during training or validation
2. **Metric Selection**: Choose metrics appropriate for the problem at hand
3. **Threshold Setting**: Set realistic deployment thresholds
4. **Monitoring**: Track model performance over time

## Troubleshooting

### Common Issues

1. **Experiment Not Found**: Ensure Comet ML authentication is configured
2. **Memory Errors**: Reduce batch size or use gradient checkpointing
3. **Optimization Timeout**: Increase timeout or reduce search iterations
4. **Model Loading Errors**: Check artifact paths and file permissions

### Debugging Tips

- Check Comet ML dashboard for experiment logs
- Monitor training logs for convergence issues
- Validate data preprocessing pipeline
- Use debug mode for detailed optimization logs

### Performance Optimization

- Use parallel optimization for faster hyperparameter search
- Enable GPU acceleration for XGBoost/LightGBM
- Optimize data loading with proper batch sizes
- Consider distributed training for large datasets
