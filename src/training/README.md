# Training Module

This module handles the complete machine learning training workflow including hyperparameter optimization, model training, experiment tracking, and model evaluation.

## Overview

The training module provides:
- Automated hyperparameter optimization using Optuna
- Multi-model training (Logistic Regression, Random Forest, LightGBM, XGBoost)
- Experiment tracking with Comet ML
- Model evaluation and selection
- Champion model registration and deployment

## Workflow

### 1. Data Splitting
```bash
python src/training/split_data.py
```

**Purpose**: Creates train/validation/test splits from feature store data.
- Loads processed data from feature store
- Creates stratified splits maintaining class balance
- Saves training, validation, and testing data splits as parquet files for training pipeline

### 2. Model Training
```bash
make train
# OR
python src/training/train.py --config_yaml_path ./src/config/training-config.yml
```

**Purpose**: Trains multiple models with hyperparameter optimization.
- Loads train/validation splits
- Optimizes hyperparameters using Optuna
- Trains models: Logistic Regression, Random Forest, LightGBM, XGBoost
- Logs experiments to Comet ML
- Saves trained models as pickle files
- Generates study results and artifacts

**Process Flow**:
1. Load and preprocess data using `TrainingDataPrep`
2. Create data transformation pipeline (scaling, encoding, feature selection)
3. For each model type:
   - Initialize Comet ML experiment
   - Run hyperparameter optimization (Optuna)
   - Train model with best parameters
   - Log metrics, parameters, and artifacts
   - Save model to artifacts directory

### 3. Model Evaluation
```bash
make evaluate
# OR
python src/training/evaluate.py --config_yaml_path ./src/config/training-config.yml
```

**Purpose**: Evaluates trained models and selects champion.
- Queries Comet ML for recent experiments
- Selects best model based on validation metrics
- Evaluates champion model on test set
- Registers champion model for deployment

**Process Flow**:
1. Query Comet ML for recent experiments
2. Compare models using validation scores
3. Load best performing model
4. Evaluate on test set with comprehensive metrics
5. Check deployment threshold
6. Register as champion model if threshold met

### 4. Ensemble Training (Optional)
The training pipeline also supports ensemble methods:
- Voting ensemble of top-performing models
- Automatic ensemble creation and evaluation
- Integration with single model workflow

## Directory Structure

```
training/
├── README.md                 # This file
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation script
├── split_data.py            # Data splitting script
├── artifacts/               # Model artifacts and results
│   ├── *.pkl               # Trained model files
│   ├── experiment_keys.csv # Experiment tracking
│   └── study_*.csv         # Hyperparameter optimization results
└── utils/                  # Training utilities
    ├── config/             # Configuration management
    ├── core/              # Core training components
    ├── evaluation/        # Model evaluation utilities
    └── tracking/          # Experiment tracking
```

## Key Components

### Core Training (`utils/core/`)

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
```python
from src.training.train import main as train_main

# Run complete training pipeline
experiment_keys = train_main(
    config_yaml_path="./src/config/training-config.yml"
)
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
2. **Metric Selection**: Choose metrics appropriate for your problem
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
