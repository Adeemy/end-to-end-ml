"""
Test functions for training configuration class in src/training/utils/config.py
"""

import pytest

from src.training.utils.config import Config
from src.utils.path import PARENT_DIR


@pytest.fixture
def valid_params():
    return {
        "description": "Test description",
        "data": {
            "split_rand_seed": 42,
            "split_type": "random",
            "train_test_split_curoff_date": "none",
            "train_valid_split_curoff_date": "none",
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "params": {
                "fbeta_score_beta_val": 0.5,
                "comparison_metric": "f1",
                "voting_rule": "soft",
            }
        },
        "modelregistry": {},
    }


def test_config_init():
    """Tests if Config class init method returns the correct
    config yaml file (config file exists)."""

    config_path = f"{str(PARENT_DIR)}/config/training-config.yml"
    config = Config(config_path)
    assert config.config_path == config_path


def test_config_init_file_not_found():
    """Tests if Config class init method raises FileNotFoundError if
    config file doesn't exist."""

    incorrect_config_path = f"{str(PARENT_DIR)}/config/nonexistent_config.yml"
    with pytest.raises(FileNotFoundError):
        Config(incorrect_config_path)


def test_missing_description_raises_value_error(valid_params):
    """Tests if missing description raises a ValueError."""

    del valid_params["description"]
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert str(excinfo.value) == "description is not included in config file"


def test_missing_data_raises_value_error(valid_params):
    """Tests if missing data raises a ValueError."""

    del valid_params["data"]
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert str(excinfo.value) == "data is not included in config file"


def test_missing_train_raises_value_error(valid_params):
    """Tests if missing train raises a ValueError."""
    del valid_params["train"]
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert str(excinfo.value) == "train is not included in config file"


def test_missing_modelregistry_raises_value_error(valid_params):
    """Tests if missing modelregistry raises a ValueError."""

    del valid_params["modelregistry"]
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert str(excinfo.value) == "modelregistry is not included in config file"


def test_invalid_split_rand_seed_type_raises_value_error(valid_params):
    """Tests if an invalid split rand seed type raises a ValueError."""

    valid_params["data"]["split_rand_seed"] = "not_an_int"
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert "split_rand_seed must be integer type." in str(excinfo.value)


def test_invalid_split_type_raises_value_error(valid_params):
    """Tests if an invalid split type raises a ValueError."""

    valid_params["data"]["split_type"] = "not_valid"
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert "split_type must be either 'random' or 'time'." in str(excinfo.value)


def test_negative_fbeta_score_beta_val_raises_value_error(valid_params):
    """Tests if a negative fbeta_score_beta_val raises a ValueError."""

    valid_params["train"]["fbeta_score_beta_val"] = -1
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert "fbeta_score_beta_val must be a positive float. Got -1.0" in str(
        excinfo.value
    )


def test_invalid_comparison_metric_raises_value_error(valid_params):
    """Tests if an invalid comparison metric raises a ValueError."""

    valid_params["train"]["comparison_metric"] = "not_supported"
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert "Supported metrics are" in str(excinfo.value)


def test_invalid_voting_rule_raises_value_error(valid_params):
    """Tests if an invalid voting rule raises a ValueError."""

    valid_params["train"]["voting_rule"] = "not_valid"
    with pytest.raises(ValueError) as excinfo:
        Config._check_params(valid_params)
    assert (
        f"Voting rule in Voting Ensemble must be 'soft' or 'hard'. Got {excinfo.value}!"
    )


def test_date_format_and_cutoff_dates_with_time_split_type(valid_params):
    """Tests if the cutoff dates are valid when split type is 'time'."""

    valid_params["data"]["train_test_split_curoff_date"] = "none"
    valid_params["data"]["train_valid_split_curoff_date"] = "2023-01-01"
    valid_params["data"]["split_type"] = "time"
    date_format = valid_params["data"]["split_date_col_format"]
    with pytest.raises(ValueError):
        Config._check_params(valid_params)
    assert f"train_test_split_curoff_date and train_valid_split_curoff_date must be a date (format {date_format}) or None if split type is 'random'."
