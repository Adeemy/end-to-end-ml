# Utils Module

This module provides shared utilities and common functionality used across the entire ML pipeline. It includes configuration management, logging, path utilities, and other cross-cutting concerns.

## Overview

The utils module provides:
- Configuration loading and management
- Centralized logging configuration
- Path and file utilities
- Common data structures and helpers
- Shared constants and enums

## Directory Structure

```
utils/
├── README.md          # This file
├── __init__.py        # Module initialization
├── config_loader.py   # Configuration loading utilities
├── logger.py          # Logging configuration and setup
└── path.py            # Path constants and utilities
```

## Key Components

### Configuration Management (`config_loader.py`)

Provides a centralized system for loading and validating configuration files using dataclasses and type hints.

**Key Features**:
- Type-safe configuration loading
- YAML configuration file support
- Environment variable override support
- Configuration validation and error handling
- Builder pattern for complex configurations

**Usage Example**:
```python
from src.utils.config_loader import load_config
from src.training.utils.config.config import Config, build_training_config

# Load training configuration
config = load_config(
    config_class=Config,
    builder_func=build_training_config,
    config_path="./src/config/training-config.yml"
)

# Access configuration values
print(config.train_params.max_search_iters)
print(config.data.class_col_name)
```

**Configuration Structure**:
```python
@dataclass
class Config:
    train_params: TrainParams
    data: DataConfig
    files: FilesConfig
    modelregistry: ModelRegistryConfig
```

### Logging System (`logger.py`)

Provides standardized logging configuration across the entire application with support for different log levels, formats, and outputs.

**Key Features**:
- Console and file logging support
- Structured log formatting
- Log level configuration
- Module-specific loggers
- Performance-optimized logging

**Usage Example**:
```python
from src.utils.logger import get_console_logger

# Get module-specific logger
module_name = Path(__file__).stem
logger = get_console_logger(module_name)

# Use logger
logger.info("Processing started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

**Logger Configuration**:
```python
# Default console logger with formatted output
logger = get_console_logger("module_name")

# Example output:
# 2025-11-23 10:00:00,123 - module_name - INFO - Processing started
```

### Path Management (`path.py`)

Defines centralized path constants and utilities for consistent file and directory management across the project.

**Key Features**:
- Centralized path definitions
- Cross-platform path handling
- Environment-aware path resolution
- Path validation utilities

**Path Constants**:
```python
from src.utils.path import (
    PROJECT_ROOT,
    DATA_DIR,
    ARTIFACTS_DIR,
    CONFIG_DIR,
    LOGS_DIR
)

# Use in your code
train_file = DATA_DIR / "train_set.parquet"
model_file = ARTIFACTS_DIR / "lightgbm.pkl"
config_file = CONFIG_DIR / "training-config.yml"
```

**Defined Paths**:
- `PROJECT_ROOT`: Root directory of the project
- `DATA_DIR`: Directory for data files (train/validation/test sets)
- `ARTIFACTS_DIR`: Directory for model artifacts and outputs
- `CONFIG_DIR`: Directory for configuration files
- `LOGS_DIR`: Directory for log files

## Configuration System

### Configuration Loading Pattern

The utils module implements a robust configuration loading system using dataclasses:

```python
from dataclasses import dataclass
from typing import List, Optional
from src.utils.config_loader import load_config

@dataclass
class MyConfig:
    param1: str
    param2: int
    param3: List[str]
    optional_param: Optional[float] = None

def build_my_config(params: dict) -> MyConfig:
    return MyConfig(
        param1=params.get("param1", "default"),
        param2=params.get("param2", 100),
        param3=params.get("param3", []),
        optional_param=params.get("optional_param")
    )

# Load configuration
config = load_config(
    config_class=MyConfig,
    builder_func=build_my_config,
    config_path="config.yml"
)
```

### YAML Configuration Format

Configuration files use YAML format for readability and maintainability:

```yaml
# training-config.yml
train:
  max_search_iters: 100
  model_opt_timeout_secs: 180
  supported_models:
    - "LGBMClassifier"
    - "RandomForestClassifier"
    - "LogisticRegression"
    - "XGBClassifier"

data:
  class_col_name: "Diabetes_binary"
  num_col_names:
    - "BMI"
    - "PhysHlth"
  cat_col_names:
    - "Age"
    - "HighBP"
    - "HighChol"

files:
  train_set_file_name: "train_set.parquet"
  valid_set_file_name: "valid_set.parquet"
  test_set_file_name: "test_set.parquet"
```

## Logging Best Practices

### Module-Level Logging
```python
from pathlib import PosixPath
from src.utils.logger import get_console_logger

# Standard pattern for module logging
module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)

def process_data():
    logger.info("Starting data processing")
    try:
        # Processing logic
        logger.debug("Processing step completed")
    except Exception as e:
        logger.error("Processing failed: %s", e, exc_info=True)
        raise
    finally:
        logger.info("Data processing finished")
```

### Structured Logging
```python
# Log with additional context
logger.info(
    "Model training completed",
    extra={
        "model_type": "LightGBM",
        "training_time": 120.5,
        "accuracy": 0.89
    }
)

# Performance logging
import time
start_time = time.time()
# ... operation ...
logger.info(
    "Operation completed in %.2f seconds",
    time.time() - start_time
)
```

### Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened but the program continues
- **ERROR**: A serious problem occurred that prevented a function from working
- **CRITICAL**: A very serious error that might stop the program

## Path Utilities

### Working with Paths
```python
from src.utils.path import DATA_DIR, ARTIFACTS_DIR
import pandas as pd

# Load data files
train_data = pd.read_parquet(DATA_DIR / "train_set.parquet")
valid_data = pd.read_parquet(DATA_DIR / "valid_set.parquet")

# Save model artifacts
import joblib
joblib.dump(model, ARTIFACTS_DIR / "my_model.pkl")

# Create subdirectories
experiment_dir = ARTIFACTS_DIR / "experiment_001"
experiment_dir.mkdir(parents=True, exist_ok=True)
```

### Environment-Specific Paths
```python
import os
from pathlib import Path

# Override default paths with environment variables
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
```

## Common Utilities

### Error Handling
```python
from src.utils.logger import get_console_logger

logger = get_console_logger(__name__)

def safe_operation(func, *args, **kwargs):
    """Wrapper for safe operation execution with logging."""
    try:
        result = func(*args, **kwargs)
        logger.info("Operation %s completed successfully", func.__name__)
        return result
    except Exception as e:
        logger.error(
            "Operation %s failed: %s",
            func.__name__,
            str(e),
            exc_info=True
        )
        raise
```

### Configuration Validation
```python
from typing import Union, List

def validate_config_value(
    value: Union[str, int, float, List],
    expected_type: type,
    name: str
) -> None:
    """Validate configuration value type and constraints."""
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Configuration '{name}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    if isinstance(value, (int, float)) and value <= 0:
        raise ValueError(f"Configuration '{name}' must be positive")
```

### File Operations
```python
from pathlib import Path
import json
import yaml

def save_json(data: dict, file_path: Path) -> None:
    """Save dictionary as JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_yaml(file_path: Path) -> dict:
    """Load YAML file as dictionary."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
```

## Integration Examples

### Training Pipeline Integration
```python
from src.utils.config_loader import load_config
from src.utils.logger import get_console_logger
from src.utils.path import DATA_DIR, ARTIFACTS_DIR
from src.training.utils.config.config import Config, build_training_config

# Setup
logger = get_console_logger("training")
config = load_config(Config, build_training_config, "config.yml")

# Use throughout training
logger.info("Starting training with config: %s", config.train_params)
train_data_path = DATA_DIR / config.files.train_set_file_name
model_save_path = ARTIFACTS_DIR / "model.pkl"
```

### Feature Store Integration
```python
from src.utils.path import DATA_DIR
from src.utils.logger import get_console_logger

logger = get_console_logger("feature_store")

def process_features():
    logger.info("Processing features")

    # Load raw data
    raw_data_path = DATA_DIR / "raw_data.parquet"

    # Process and save
    processed_data_path = DATA_DIR / "processed_features.parquet"

    logger.info("Features saved to %s", processed_data_path)
```

### Inference Integration
```python
from src.utils.path import ARTIFACTS_DIR
from src.utils.logger import get_console_logger
import joblib

logger = get_console_logger("inference")

def load_model():
    model_path = ARTIFACTS_DIR / "champion_model.pkl"
    logger.info("Loading model from %s", model_path)

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model
```

## Best Practices

### Configuration Management
1. **Type Safety**: Use dataclasses with type hints for configuration
2. **Validation**: Validate configuration values at load time
3. **Defaults**: Provide sensible defaults for optional parameters
4. **Documentation**: Document all configuration options
5. **Environment Overrides**: Support environment variable overrides for deployment

### Logging Guidelines
1. **Consistent Format**: Use the same logging format across modules
2. **Appropriate Levels**: Choose correct log levels for different messages
3. **Context Information**: Include relevant context in log messages
4. **Performance**: Use lazy evaluation for expensive log operations
5. **Security**: Never log sensitive information like API keys

### Path Management
1. **Centralization**: Define all paths in one place
2. **Cross-Platform**: Use Path objects for cross-platform compatibility
3. **Validation**: Check path existence and permissions
4. **Documentation**: Document the purpose of each directory
5. **Environment Awareness**: Support different environments (dev, staging, prod)

## Troubleshooting

### Common Issues
1. **Configuration Not Found**: Check file paths and working directory
2. **Import Errors**: Ensure proper Python path configuration
3. **Permission Errors**: Check file and directory permissions
4. **Type Errors**: Validate configuration types and formats

### Debugging Tips
- Enable debug logging to see detailed execution flow
- Use absolute paths when relative paths fail
- Validate configuration files with YAML linters
- Check environment variables and their values
- Use logging to trace configuration loading process
