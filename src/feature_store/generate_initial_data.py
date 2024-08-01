"""
Imports original dataset from UCI data repository and stores it local path.
Also, 5% of the original dataset is reserved as inference set, which simulates
production data that will be scored by the deployed model in inference pipeline.

The raw dataset was released by the CDC and it was imported from the following
UCI repo: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.

Note: this script is only used in the beginning of this project just to generate
data for the project and it isn't part of feature or inference pipelines.
"""

import argparse
import logging
import logging.config
import os

from azureml.core import Dataset, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv

from src.feature_store.utils.config import Config
from src.feature_store.utils.helpers import import_uci_data, split_uci_data
from src.utils.logger import get_console_logger

# import sys
# sys.path.append("/workspaces/end-to-end-ml/")


load_dotenv()


def main(
    config_yaml_path: str,
    logger: logging.Logger,
) -> None:
    """Imports raw dataset from UCI data repository and creates raw and
    inference sets and registers them in Azure ML workspace. The raw data
    is used for model development and inference set is reserved to simulate
    production data.

    Args:
        config_yaml_path (str): path to the config yaml file.
        logger (logging.Logger): logger object.
    """

    logger.info(
        "Directory of data perprocessing and transformation config file: %s",
        config_yaml_path,
    )

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)
    inference_set_ratio = float(config.params["data"]["inference_set_ratio"])

    raw_dataset_name = config.params["azure_datasets"]["raw_dataset_name"]
    inference_set_name = config.params["azure_datasets"]["inference_set_name"]

    raw_dataset_desc = config.params["azure_datasets"]["raw_dataset_desc"]
    inference_set_desc = config.params["azure_datasets"]["inference_set_desc"]

    raw_dataset_tags = config.params["azure_datasets"]["raw_dataset_tags"]
    inference_set_tags = config.params["azure_datasets"]["inference_set_tags"]

    # Connect to the training workspace
    sp_authentication = ServicePrincipalAuthentication(
        tenant_id=os.environ["TENANT_ID"],
        service_principal_id=os.environ["APP_REGISTRATION_ID"],
        service_principal_password=os.environ["SP_PWD"],
    )
    ws = Workspace(
        os.environ["SUBSCRIPTION_ID"],
        os.environ["RESOURCE_GROUP_NAME"],
        os.environ["AML_WORKSPACE_NAME"],
        auth=sp_authentication,
    )

    #################################
    # Import raw dataset, split it and register it in Azure ML workspace
    raw_dataset = import_uci_data(config_yaml_path=config_yaml_path)
    raw_dataset, inference_set = split_uci_data(
        raw_dataset=raw_dataset, config_yaml_path=config_yaml_path
    )

    Dataset.Tabular.register_pandas_dataframe(
        dataframe=raw_dataset,
        target=ws.get_default_datastore(),
        name=raw_dataset_name,
        description=raw_dataset_desc,
        tags=raw_dataset_tags,
        show_progress=True,
    )

    Dataset.Tabular.register_pandas_dataframe(
        dataframe=inference_set,
        target=ws.get_default_datastore(),
        name=inference_set_name,
        description=inference_set_desc,
        tags=inference_set_tags,
        show_progress=True,
    )

    logger.info("Inference and raw dataset (for model development) generated.")
    logger.info(
        "Ratio of raw dataset out of original dataset: " + "%.1f%% (%d rows).",
        100 * (1 - inference_set_ratio),
        raw_dataset.shape[0],
    )
    logger.info(
        "Ratio of inference set out of original dataset: " + "%.1f%% (%d rows).",
        100 * inference_set_ratio,
        inference_set.shape[0],
    )


###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./config.yml",
        help="Path to the configuration yaml file.",
    )
    parser.add_argument(
        "--logger_path",
        type=str,
        default="./logger.conf",
        help="Path to the logger configuration file.",
    )

    args = parser.parse_args()

    # Get the logger objects by name
    console_logger = get_console_logger("gen_initial_data_logger")

    console_logger.info("Generating Raw Dataset Starts ...")

    main(config_yaml_path=args.config_yaml_path, logger=console_logger)
