"""
Test functions for training configuration class in src.training.utils.config.config.py
"""

import pytest

from src.training.utils.config.config import Config
from src.utils.path import PARENT_DIR


@pytest.fixture(scope="module")
def config_fixture():
    """Fixture to provide a valid Config instance and its path."""
    config_path = f"{str(PARENT_DIR)}/config/training-config.yml"
    config = Config(config_path)
    return config, config_path


def test_config_init(config_fixture):  # pylint: disable=redefined-outer-name
    """Tests if Config class init method correctly initializes the config path."""
    config, config_path = config_fixture
    assert config.config_path == config_path


def test_config_init_file_not_found():
    """Tests if Config class init method raises FileNotFoundError when the config file doesn't exist."""
    incorrect_config_path = f"{str(PARENT_DIR)}/config/nonexistent_config.yml"
    with pytest.raises(FileNotFoundError):
        Config(incorrect_config_path)


def test_check_params_valid(config_fixture):  # pylint: disable=redefined-outer-name
    """Tests if check_params method passes with valid parameters."""

    valid_config, _ = config_fixture

    # This should not raise any exceptions
    valid_config.params = {
        "description": "Test description",
        "data": {
            "split_rand_seed": 42,
            "split_type": "random",
            "train_test_split_curoff_date": "none",
            "train_valid_split_curoff_date": "none",
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "fbeta_score_beta_val": 0.5,
            "comparison_metric": "f1",
            "voting_rule": "soft",
        },
        "modelregistry": {},
    }
    valid_config.check_params()


def test_check_params_missing_required_param(
    config_fixture,  # pylint: disable=redefined-outer-name
):
    """Tests if check_params method raises AssertionError when a
    required parameter is missing.
    """

    invalid_config, _ = config_fixture

    # Save invalid params in params attribute (missing 'description' key)
    invalid_config.params = {
        "data": {
            "split_rand_seed": 42,
            "split_type": "random",
            "train_test_split_curoff_date": "none",
            "train_valid_split_curoff_date": "none",
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "fbeta_score_beta_val": 0.5,
            "comparison_metric": "f1",
            "voting_rule": "soft",
        },
        "modelregistry": {},
    }

    with pytest.raises(KeyError):
        invalid_config.check_params()


def test_check_params_invalid_split_rand_seed_type(
    config_fixture,  # pylint: disable=redefined-outer-name
):
    """Tests if check_params method raises ValueError when a split_rand_seed is of invalid type
    and when cutoff dates are of invalid format.
    """

    invalid_config, _ = config_fixture

    # Save invalid params in params attribute (invalid type for 'split_rand_seed')
    invalid_config.params = {
        "description": "Test description",
        "data": {
            "split_rand_seed": "42r",  # Invalid type, should be an integer
            "split_type": "random",
            "train_test_split_curoff_date": "none",
            "train_valid_split_curoff_date": "20-10-2019",  # Invalid date format
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "fbeta_score_beta_val": 0.5,
            "comparison_metric": "f1",
            "voting_rule": "soft",
        },
        "modelregistry": {},
    }

    with pytest.raises(ValueError):
        invalid_config.check_params()


def test_check_params_invalid_split_type(
    config_fixture,  # pylint: disable=redefined-outer-name
):
    """Tests if check_params method raises ValueError when a
    when split_rand_seed is of invalid type and split_type is invalid.
    """
    invalid_config, _ = config_fixture

    # Save invalid params in params attribute
    invalid_config.params = {
        "description": "Test description",
        "data": {
            "split_rand_seed": "42",  # Invalid type, should be an integer
            "split_type": "randoms",
            "train_test_split_curoff_date": "none",  # Invalid type, should be "none"
            "train_valid_split_curoff_date": "none",  # Invalid date format
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "fbeta_score_beta_val": 0.5,
            "comparison_metric": "f1",
            "voting_rule": "soft",
        },
        "modelregistry": {},
    }

    with pytest.raises(KeyError):
        invalid_config.check_params()


def test_check_params_invalid_split_date_missing(
    config_fixture,  # pylint: disable=redefined-outer-name
):
    """Tests if check_params method raises ValueError when
    when split_type is 'time' but cutoff dates are invalid.
    """

    invalid_config, _ = config_fixture

    # Save invalid params in params attribute
    invalid_config.params = {
        "description": "Test description",
        "data": {
            "split_rand_seed": "42",
            "split_type": "time",
            "train_test_split_curoff_date": "none",  # Invalid type, should NOT be "none"
            "train_valid_split_curoff_date": "none",  # Invalid type, should NOT be "none"
            "split_date_col_format": "%Y-%m-%d",
        },
        "train": {
            "fbeta_score_beta_val": 0.5,
            "comparison_metric": "f1",
            "voting_rule": "soft",
        },
        "modelregistry": {},
    }

    with pytest.raises(KeyError):
        invalid_config.check_params()
