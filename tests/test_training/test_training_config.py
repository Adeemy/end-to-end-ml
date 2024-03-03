# import pytest

# from src.training.utils.config import Config
# from src.utils.path import PARENT_DIR


# def test_config_init():
#     """Tests if Config class init method returns the correct
#     config yaml file (config file exists)."""

#     config_path = f"{str(PARENT_DIR)}/config/training-config.yml"
#     config = Config(config_path)
#     assert config.config_path == config_path


# def test_config_init_file_not_found():
#     """Tests if Config class init method raises FileNotFoundError if
#     config file doesn't exist."""

#     incorrect_config_path = f"{str(PARENT_DIR)}/config/nonexistent_config.yml"
#     with pytest.raises(FileNotFoundError):
#         Config(incorrect_config_path)


# def test_check_params_valid():
#     """Tests if _check_params method passes with valid parameters."""

#     params = {
#         "description": "Test description",
#         "data": {
#             "params": {
#                 "split_rand_seed": 42,
#                 "split_type": "random",
#                 "train_test_split_curoff_date": "none",
#                 "train_valid_split_curoff_date": "none",
#                 "split_date_col_format": "%Y-%m-%d",
#             }
#         },
#         "train": {
#             "params": {
#                 "fbeta_score_beta_val": 0.5,
#                 "comparison_metric": "f1",
#                 "voting_rule": "soft",
#             }
#         },
#         "modelregistry": {},
#     }

#     Config._check_params(params)


# def test_check_params_missing_required_param():
#     """Tests if _check_params method raises AssertionError when a required parameter is missing."""

#     params = {
#         "data": {
#             "params": {
#                 "split_rand_seed": 42,
#                 "split_type": "random",
#                 "train_test_split_curoff_date": "none",
#                 "train_valid_split_curoff_date": "none",
#                 "split_date_col_format": "%Y-%m-%d",
#             }
#         },
#         "train": {
#             "params": {
#                 "fbeta_score_beta_val": 0.5,
#                 "comparison_metric": "f1",
#                 "voting_rule": "soft",
#             }
#         },
#         "modelregistry": {},
#     }

#     with pytest.raises(AssertionError):
#         Config._check_params(params)


# def test_check_params_invalid_split_rand_seed_type():
#     """Tests if _check_params method raises ValueError when a parameter has an invalid type."""

#     params = {
#         "description": "Test description",
#         "data": {
#             "params": {
#                 "split_rand_seed": "42r",  # Invalid type, should be an integer
#                 "split_type": "random",
#                 "train_test_split_curoff_date": "none",
#                 "train_valid_split_curoff_date": "20-10-2019",  # Invalid date format
#                 "split_date_col_format": "%Y-%m-%d",
#             }
#         },
#         "train": {
#             "params": {
#                 "fbeta_score_beta_val": 0.5,
#                 "comparison_metric": "f1",
#                 "voting_rule": "soft",
#             }
#         },
#         "modelregistry": {},
#     }

#     with pytest.raises(ValueError):
#         Config._check_params(params)


# def test_check_params_invalid_split_type():
#     """Tests if _check_params method raises ValueError when a parameter has an invalid type."""

#     params = {
#         "description": "Test description",
#         "data": {
#             "params": {
#                 "split_rand_seed": "42",  # Invalid type, should be an integer
#                 "split_type": "randoms",
#                 "train_test_split_curoff_date": "none",  # Invalid type, should be "none"
#                 "train_valid_split_curoff_date": "none",  # Invalid date format
#                 "split_date_col_format": "%Y-%m-%d",
#             }
#         },
#         "train": {
#             "params": {
#                 "fbeta_score_beta_val": 0.5,
#                 "comparison_metric": "f1",
#                 "voting_rule": "soft",
#             }
#         },
#         "modelregistry": {},
#     }

#     with pytest.raises(ValueError):
#         Config._check_params(params)


# def test_check_params_invalid_split_date_missing():
#     """Tests if _check_params method raises ValueError when a parameter has an invalid type."""

#     params = {
#         "description": "Test description",
#         "data": {
#             "params": {
#                 "split_rand_seed": "42",
#                 "split_type": "time",
#                 "train_test_split_curoff_date": "none",  # Invalid type, should NOT be "none"
#                 "train_valid_split_curoff_date": "none",  # Invalid type, should NOT be "none"
#                 "split_date_col_format": "%Y-%m-%d",
#             }
#         },
#         "train": {
#             "params": {
#                 "fbeta_score_beta_val": 0.5,
#                 "comparison_metric": "f1",
#                 "voting_rule": "soft",
#             }
#         },
#         "modelregistry": {},
#     }

#     with pytest.raises(ValueError):
#         Config._check_params(params)
