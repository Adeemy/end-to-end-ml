"""
This script includes functions to generate EDA plots.
"""

from datetime import datetime
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from numpy import ravel
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


##########################################################
class ModelEvaluator:
    """A class to evaluate models."""

    def __init__(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        fbeta_score_beta: float = 0.5,
    ) -> None:
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.fbeta_score_beta = fbeta_score_beta

    def calculate_performance_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
    ) -> pd.DataFrame:
        """
        Calculates different performance metrics for binary classification models.

        Args:
            true_class (list): true class label.
            pred_class (list): predicted class label not probability.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        # Calculate metrics
        results = [
            ("accuracy", accuracy_score(true_class, pred_class)),
            (
                "precision",
                precision_score(
                    true_class,
                    pred_class,
                ),
            ),
            ("recall", recall_score(true_class, pred_class)),
            ("f1", f1_score(true_class, pred_class)),
            (
                f"f_{self.fbeta_score_beta}_score",
                fbeta_score(
                    true_class,
                    pred_class,
                    beta=self.fbeta_score_beta,
                ),
            ),
            (
                "roc_auc",
                roc_auc_score(true_class, pred_class, average=None),
            ),
        ]

        performance_metrics = pd.DataFrame(results, columns=["Metric", "Score"])

        return performance_metrics

    def plot_feature_importance(
        self,
        feature_importance_scores: np.ndarray,
        feature_names: list,
        figure_obj,
        n_top_features: int = 30,
        font_size: int = 10,
        fig_size: tuple = (8, 18),
    ) -> None:
        """
        Plots top feature importance with their encoded names. It requires
        an empty figure object (figure_obj) to add plot to it and return
        plot as a figure object that can be logged.
        """

        feat_importances = pd.Series(feature_importance_scores, index=feature_names)
        feat_importances = feat_importances.nlargest(n_top_features, keep="all")
        feat_importances.sort_values(ascending=True, inplace=True)

        feat_importances.plot(
            kind="barh", fontsize=font_size, legend=None, figsize=fig_size
        )
        plt.title(f"Top {n_top_features} important features")
        plt.show()

        return figure_obj

    def extract_feature_importance(
        self,
        pipeline: Pipeline,
        num_feature_names: list = None,
        cat_feature_names: list = None,
        n_top_features: int = 30,
        figure_size: tuple = (24, 36),
        font_size: float = 10.0,
    ) -> None:
        """Extracts feature importance and returns figure object and
        column names from fitted pipeline."""

        # Catch any error raised in this function to prevent experiment
        # from registering model as it's not worth failing experiment for
        # an error in this function.
        try:
            # Get feature names
            if num_feature_names is None and cat_feature_names is not None:
                col_names = list(
                    pipeline.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .named_steps["onehot_encoder"]
                    .get_feature_names_out(cat_feature_names)
                )

            elif num_feature_names is not None and cat_feature_names is None:
                col_names = num_feature_names

            elif num_feature_names is not None and cat_feature_names is not None:
                col_names = num_feature_names + list(
                    pipeline.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .named_steps["onehot_encoder"]
                    .get_feature_names_out(cat_feature_names)
                )

            else:
                raise ValueError(
                    f"{num_feature_names} and/or {cat_feature_names} must be provided."
                )

            # Extract transformed feature names
            col_names = [
                i
                for (i, v) in zip(
                    col_names,
                    list(pipeline.named_steps["selector"].get_support()),
                )
                if v
            ]

            print(
                f"No. of features including encoded categorical features: {len(col_names)}"
            )

            # Note: there is no feature_importances_ attribute for LogisticRegression, hence,
            # this if statement is needed.
            classifier_name = pipeline.named_steps["classifier"].__class__.__name__
            if classifier_name == "LogisticRegression":
                feature_importance_scores = pipeline.named_steps["classifier"].coef_[0]

            if classifier_name not in [
                "LogisticRegression",
                "VotingClassifier",
            ]:
                feature_importance_scores = pipeline.named_steps[
                    "classifier"
                ].feature_importances_

            if classifier_name not in ["VotingClassifier"]:
                # Log feature importance figure
                feature_importance_fig = plt.figure(figsize=figure_size)
                feature_importance_fig = self.plot_feature_importance(
                    feature_importance_scores=feature_importance_scores,
                    feature_names=col_names,
                    figure_obj=feature_importance_fig,
                    n_top_features=n_top_features,
                    font_size=font_size,
                    fig_size=figure_size,
                )

        except Exception as e:  # pylint: disable=W0718
            print(f"Feature importance extraction error --> {e}")
            feature_importance_fig, col_names = None, None

    def plot_confusion_matrix(
        self,
        y_decoded: np.ndarray,
        pred_y_decoded: np.ndarray,
        decoded_class_labels: list,
        fig_dir: str,
        normalize_conf_mat: Literal["true", None] = None,
        confusion_matrix_fig_name: str = "confusion_matrix.png",
        cm_title: str = "Confusion Matrix",
    ) -> None:
        """Plots confusion matrix: normalized and un-normalized."""
        confusion_mat_norm = confusion_matrix(
            y_true=y_decoded,
            y_pred=pred_y_decoded,
            labels=decoded_class_labels,
            normalize=normalize_conf_mat,
        )
        confusion_mat_norm = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat_norm,
            display_labels=decoded_class_labels,
        )
        _, ax = plt.subplots(figsize=(11, 11))
        confusion_mat_norm.plot(ax=ax, xticks_rotation=75)
        confusion_mat_norm.ax_.set_title(cm_title)
        confusion_mat_norm.figure_.savefig(fig_dir + confusion_matrix_fig_name)

    def evaluate_model_perf(
        self,
        class_encoder: LabelEncoder,
        fig_dir: str,
    ) -> Union[pd.DataFrame, pd.DataFrame, Pipeline, list]:
        """Evaluates a model on both training and testing set."""

        # Generate class labels for testing set
        pred_y_train = self.pipeline.predict(self.X_train)
        pred_test_probs = self.pipeline.predict_proba(self.X_test)
        pred_y_test = np.where(pred_test_probs[:, 1] > 0.5, 1, 0)

        # Extract expressive class names for confusion matrix
        y_train_decoded = class_encoder.inverse_transform(self.y_train)
        pred_y_train_decoded = class_encoder.inverse_transform(pred_y_train)
        y_test_decoded = class_encoder.inverse_transform(self.y_test)
        pred_y_test_decoded = class_encoder.inverse_transform(pred_y_test)

        # Calculate performance metrics
        training_scores = self.calculate_performance_metrics(
            true_class=self.y_train,
            pred_class=pred_y_train,
        )
        testing_scores = self.calculate_performance_metrics(
            true_class=self.y_test,
            pred_class=pred_y_test,
        )

        # Extract expressive class label names
        # Note: should be included in registered model tags.
        decoded_class_labels = list(
            class_encoder.inverse_transform(self.pipeline.classes_)
        )

        self.plot_confusion_matrix(
            y_decoded=y_train_decoded,
            pred_y_decoded=pred_y_train_decoded,
            decoded_class_labels=decoded_class_labels,
            normalize_conf_mat="true",
            confusion_matrix_fig_name="train_set_confusion_mat_norm.png",
            cm_title="train_set_confusion_mat_norm",
            fig_dir=fig_dir,
        )

        self.plot_confusion_matrix(
            y_decoded=y_train_decoded,
            pred_y_decoded=pred_y_train_decoded,
            decoded_class_labels=decoded_class_labels,
            normalize_conf_mat=None,
            confusion_matrix_fig_name="train_set_confusion_mat.png",
            cm_title="train_set_confusion_mat",
            fig_dir=fig_dir,
        )

        self.plot_confusion_matrix(
            y_decoded=y_test_decoded,
            pred_y_decoded=pred_y_test_decoded,
            decoded_class_labels=decoded_class_labels,
            normalize_conf_mat="true",
            confusion_matrix_fig_name="test_set_confusion_mat_norm.png",
            cm_title="test_set_confusion_mat_norm",
            fig_dir=fig_dir,
        )

        self.plot_confusion_matrix(
            y_decoded=y_test_decoded,
            pred_y_decoded=pred_y_test_decoded,
            decoded_class_labels=decoded_class_labels,
            normalize_conf_mat=None,
            confusion_matrix_fig_name="test_set_confusion_mat.png",
            cm_title="test_set_confusion_mat",
            fig_dir=fig_dir,
        )

        return (
            training_scores,
            testing_scores,
            decoded_class_labels,
        )


##########################################################
def specify_data_types(
    input_data: pd.DataFrame,
    date_cols_names: list = None,
    datetime_cols_names: list = None,
    numerical_cols_names: list = None,
    categorical_cols_names: list = None,
) -> pd.DataFrame:
    """Enforces the specified data types of input dataset columns with
    their proper missing value indicator. If date, datetime, and numerical
    columns are not provided, all columns will be converted to categorical
    data type (categorical type).
    """

    if date_cols_names is None:
        date_cols_names = []

    if datetime_cols_names is None:
        datetime_cols_names = []

    if numerical_cols_names is None:
        numerical_cols_names = []

    if categorical_cols_names is None:
        categorical_cols_names = []

    # Categorical variables are all veriables that are not numerical or date
    dataset = input_data.copy()
    input_data_vars_names = dataset.columns.tolist()
    non_cat_col_names = date_cols_names + datetime_cols_names + numerical_cols_names

    # Replace common missing values representations with with np.nan
    dataset = dataset.replace(
        {
            "": np.nan,
            "<NA>": np.nan,
            "null": np.nan,
            "?": np.nan,
            None: np.nan,
            "N/A": np.nan,
            "NAN": np.nan,
            "nan": np.nan,
            pd.NA: np.nan,
        }
    )

    # Identify categorical variables if not provided
    if len(categorical_cols_names) == 0:
        categorical_cols_names = [
            col for col in input_data_vars_names if col not in non_cat_col_names
        ]

    # Cast date columns
    if len(date_cols_names) > 0:
        dataset[date_cols_names] = dataset[date_cols_names].apply(
            pd.to_datetime, format="%Y-%d-%m", errors="coerce"
        )

        dataset.loc[:, date_cols_names] = dataset.loc[:, date_cols_names].replace(
            {np.nan: pd.NaT}
        )

        print(f"Date columns:\n{date_cols_names}\n\n")

    # Cast datetime columns
    if len(datetime_cols_names) > 0:
        dataset[datetime_cols_names] = dataset[datetime_cols_names].apply(
            pd.to_datetime, format="%Y-%d-%m %H:%M:%S", errors="coerce"
        )

        dataset.loc[:, datetime_cols_names] = dataset.loc[
            :, datetime_cols_names
        ].replace({np.nan: pd.NaT})

        print(f"Datetime columns:\n{datetime_cols_names}\n\n")

    # Cast numerical as float type
    if len(numerical_cols_names) > 0:
        dataset[numerical_cols_names] = dataset[numerical_cols_names].astype("float32")

        print(f"Numerical columns:\n{numerical_cols_names}\n\n")

    # Cast categorical columns to object stype
    # Note: replacing NaNs after casting to object will convert columns to object data type.
    if len(categorical_cols_names) > 0:
        print(
            "The following (categorical) columns will be converted to 'object' type.\n",
            categorical_cols_names,
        )

        dataset[categorical_cols_names] = dataset[categorical_cols_names].astype(
            "string"
        )

    return dataset


##########################################################
def check_class_distributions(
    input_data: pd.DataFrame, class_col_name: str
) -> Union[pd.Series, pd.Series]:
    """
    This funtion checks class distributions (counts and percentages).
    """

    # Calculate class labels counts and percentages
    dataset = input_data.copy()
    class_labels_counts = dataset[class_col_name].value_counts()
    class_labels_proportions = round(100 * class_labels_counts / dataset.shape[0], 2)

    # Print class proportions as dictionaries
    class_counts = class_labels_counts.to_dict()
    class_proportions = class_labels_proportions.to_dict()

    print("\nDataset class counts:\n")
    print(class_counts, "\n")
    print(class_proportions, "\n")

    return class_labels_counts, class_labels_proportions


##########################################################
def check_datasets_overlap(
    first_dataset: pd.DataFrame, second_dataset: pd.DataFrame, primary_key_name: list
) -> None:
    """
    Checks if there is overlapping between two sets (e.g., training and testing sets)
    based on a shared primary key. It returns a message indicating whether there are
    samples that exist in both sets or not.

    Args:
        first_dataset (dataframe): it can be either training or testing set.
        second_dataset (dataframe): it can be either training or testing set.
        primary_key_name (str or list): name of the shared primary key column(s).
    """

    left_dataset = first_dataset.copy()
    right_dataset = second_dataset.copy()

    if left_dataset.shape[0] == 0 or right_dataset.shape[0] == 0:
        raise ValueError("\nEither dataset has a sample size of zero!\n")

    # Set primary key as index to speed up join process
    left_dataset.set_index(primary_key_name, inplace=True)
    right_dataset.set_index(primary_key_name, inplace=True)

    # Join datasets (inner join) on primary key to get overlapping samples
    overlap_samples = left_dataset.join(
        right_dataset, how="inner", lsuffix="_left", rsuffix="_right"
    )

    if len(overlap_samples) > 0:
        raise ValueError(
            f"\n{len(overlap_samples)} overlapping samples between the two datasets.\n"
        )
    else:
        return print("\nNo overlapping samples between the two datasets.\n")


##########################################################
def plot_missing_values(input_data: pd.DataFrame, fig_size: tuple = (24, 24)) -> None:
    """Visualizes missing values"""

    # Calculate missing values
    dataset = input_data.copy()
    missing_values_counts = dataset.isna().sum()
    missing_values_percentages = round(
        100 * missing_values_counts / dataset.shape[0], 2
    )

    if missing_values_counts.sum() != 0:
        missing_values_percentages = missing_values_percentages.drop(
            missing_values_percentages[missing_values_percentages == 0].index
        ).sort_values(ascending=True)
        missing_data = pd.DataFrame(
            {"Missing Values Ratio %": missing_values_percentages}
        )
        missing_data.plot(kind="barh", fontsize=24, legend=None, figsize=fig_size)
        plt.title("Missing Values (%)", fontsize=26)
        plt.show()
    else:
        print("No missing values found")


##########################################################
def plot_two_dataset_size_by_date(
    first_dataset: pd.DataFrame,
    second_dataset: pd.DataFrame,
    date_col_name: str,
    vline_date: datetime.date,
    first_dataset_name: str = "First Dataset",
    second_dataset_name: str = "Second Dataset",
    font_size: float = 12,
    fig_width: int = 700,
    fig_height: int = 500,
) -> None:
    """
    Plots counts of date column by date for two input dataset. It can be used
    to plot training and testing sets by date column used to split dataset in
    order to determine if there is overlap (possible data leakage) or not. Order
    of input datasets in function inputs do not matter.

    Args:
    first_dataset (pd.DataFrame): first dataset.
    second_dataset (pd.DataFrame): second dataset.
    date_col_name (str): name of the date column, e.g., name of date column used
                         to split data into training and testing sets.
    vline_date (datetime.date): date of the begining of second dataset like cut-off
                                date used to split data into training and testing sets.
    first_dataset_name (str): name of the first dataset.
    second_dataset_name (str): name of the second dataset.
    font_size (int): font size for the plot (default=12).
    fig_width, fig_height (int, int): figure width and height (default= 700, 500).

    Returns:
        None
    """

    first_dataset_count = (
        first_dataset.groupby(date_col_name).size().reset_index(name="Count")
    )
    second_dataset_count = (
        second_dataset.groupby(date_col_name).size().reset_index(name="Count")
    )
    fig = px.line(
        first_dataset_count,
        x=date_col_name,
        y="Count",
        labels={"Count": "Count"},
        title="Count by Date",
    )
    fig.add_scatter(
        x=first_dataset_count[date_col_name],
        y=first_dataset_count["Count"],
        mode="lines",
        name=first_dataset_name,
        line_color="blue",
    )
    fig.add_scatter(
        x=second_dataset_count[date_col_name],
        y=second_dataset_count["Count"],
        mode="lines",
        name=second_dataset_name,
        line_color="red",
    )
    fig.update_layout(font=dict(size=font_size), width=fig_width, height=fig_height)
    if vline_date is not None:
        fig.add_vline(x=vline_date, line_dash="dash")
    fig.show()


##########################################################
def calculate_percentage_per_time_period(
    input_data: pd.DataFrame,
    date_col_name: str,
    class_col_name: str,
    freq: Literal["Month", "Week", "Day"] = "month",
) -> pd.DataFrame:
    """
    Calculates the percentage of a specific value per time period (month/week/day) in a DataFrame.

    Args:
    input_data (pd.DataFrame): Input DataFrame
    date_col_name (str): Name of the date column
    class_col_name (str): Name of the value column
    freq (str): Frequency of aggregation ('M' for month, 'W' for week, 'D' for day)

    Returns:
    result (pd.DataFrame): Resulting DataFrame with percentage per time period
    """

    # Convert dataframe to polars for better speed
    dataset_pl = pl.from_pandas(input_data)

    # Convert date column to datetime and extract time period
    if freq == "Month":
        dataset_pl = dataset_pl.with_columns(
            pl.col(date_col_name).dt.strftime("%Y-%m").alias(freq)
        )
    elif freq == "Week":
        dataset_pl = dataset_pl.with_columns(
            pl.col(date_col_name).dt.strftime("%Y-%W").alias(freq)
        )
    elif freq == "Day":
        dataset_pl = dataset_pl.with_columns(
            pl.col(date_col_name).dt.strftime("%Y-%m-%d").alias(freq)
        )

    # Group by time period and calculate percentage
    dataset_pl = (
        dataset_pl.groupby([class_col_name, freq], maintain_order=True).agg(
            [pl.count().alias("Count"), pl.first(date_col_name)]
        )
    ).select(
        [
            pl.all().exclude("Count"),
            (100 * pl.col("Count") / pl.sum("Count").over(freq)).alias(
                "Class Proportions"
            ),
        ]
    )

    dataset = dataset_pl.to_pandas()

    # Sort date column to plot x-axis properly
    dataset.sort_values(by=date_col_name, ascending=True, inplace=True)

    return dataset


##########################################################
def plot_cont_var_histograms(
    input_data: pd.DataFrame,
    cont_var_names: list,
    no_of_bins: int = 100,
    x_axis_label_size: int = 8,
    y_axis_label_size: int = 8,
    figure_size: tuple = (16, 20),
    lower_percentile: float = 0.0,
    upper_percentile: float = 1.0,
) -> None:
    """Plots histograms of all specified continuous variables in one figure. It
    gives the option to remove outlier by limitting values to desired lower
    and upper percentiles to make distribution more readable. Default is
    not excluding outliers.
    """

    # Plot histograms in one figure
    dataset = input_data[cont_var_names].copy()
    dataset = dataset.apply(
        lambda x: x[
            (x >= x.quantile(lower_percentile)) & (x < x.quantile(upper_percentile))
        ],
        axis=0,
    )
    dataset.hist(
        figsize=figure_size,
        bins=no_of_bins,
        xlabelsize=x_axis_label_size,
        ylabelsize=y_axis_label_size,
    )


##########################################################
def remove_vars_with_nans(
    input_data: pd.DataFrame,
    vars_to_exclude_from_removal: list = None,
    threshold_val_for_exclusion: float = 0.3,
) -> pd.DataFrame:
    """Removes variables with missing values above threshold value [0, 1]."""

    # Specify vars to include in nans removal
    dataset = input_data.copy()
    vars_with_nans_to_include_in_removal = dataset.columns.difference(
        vars_to_exclude_from_removal
    ).to_list()

    # Remove variables with missing values higher than nans_removal_threshold_val
    missing_values_counts = dataset[vars_with_nans_to_include_in_removal].isna().sum()
    missing_values_percentages = (
        missing_values_counts / dataset.shape[0]
    ) > threshold_val_for_exclusion
    missing_values_percentages = missing_values_counts[
        missing_values_percentages > threshold_val_for_exclusion
    ].index.to_list()
    dataset.drop(missing_values_percentages, axis=1, inplace=True)

    return dataset


##########################################################
def plot_cont_var_dist_by_class(
    input_data: pd.DataFrame,
    num_col_names: list,
    class_col_name: str,
    figure_size: tuple = (8, 4),
    box_max_length: float = 1.5,
) -> None:
    """Plots the distributions (boxplots) of numerical variables versus class."""

    # Plot boxplots with respect to class label
    dataset = input_data.copy()
    for i, col_name in enumerate(num_col_names):
        # Check the distribution of numerical variables vs. class
        plt.figure(figsize=figure_size)
        ax = sns.boxplot(
            x=class_col_name,
            y=col_name,
            data=dataset,
            whis=box_max_length,
            palette="Set2",
        )
        plt.setp(ax.artists, alpha=0.5, linewidth=2, edgecolor="k")
        plt.xticks(rotation=0)
        plt.tight_layout(pad=1)


##########################################################
def plot_cont_vars_density(
    input_data: pd.DataFrame,
    cont_vars_names: list,
    figure_size: tuple = (12, 60),
    smooth_function_grid_size: int = 500,
    lower_percentile: float = 0.0,
    upper_percentile: float = 1,
) -> None:
    """Plots density of continuous variables with respect to class. It
    gives the option to remove outlier by limitting values to desired
    lower and upper percentiles to make overlap betwen distribution more
    visible if any. Default is not excluding outliers.
    """

    # Copy continuous variables
    dataset = input_data.copy()
    cont_vars = dataset[cont_vars_names].copy()

    if len(cont_vars_names) > 1:
        fig, axes = plt.subplots(
            nrows=int(np.ceil(len(cont_vars_names) / 2)),
            ncols=2,
            figsize=figure_size,
            sharex=False,
            sharey=False,
        )

        axes = axes.ravel()  # array to 1D

        # Create a list of dataframe columns to use
        cols = cont_vars[cont_vars_names].columns

        for col, ax in zip(cols, axes):
            # Extract column values to set upper and lowr x-axis limits
            cont_col_values = cont_vars[col]

            # Plot
            sns.kdeplot(
                data=cont_vars[[col]],
                x=col,
                fill=True,
                ax=ax,
                gridsize=smooth_function_grid_size,
            )
            ax.set(title=f"Distribution of: {col}", xlabel=None)
            ax.set_xlim(
                cont_col_values.quantile(lower_percentile),
                cont_col_values.quantile(upper_percentile),
            )
    else:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=figure_size, sharex=False, sharey=False
        )

        # Plot
        col = cont_vars_names[0]
        cont_col_values = cont_vars[col]

        sns.kdeplot(
            data=cont_vars[[col]],
            x=col,
            fill=True,
            ax=ax,
            gridsize=smooth_function_grid_size,
        )
        ax.set(title=f"Distribution of: {col}", xlabel=None)
        ax.set_xlim(
            cont_col_values.quantile(lower_percentile),
            cont_col_values.quantile(upper_percentile),
        )

    fig.tight_layout()


##########################################################
def plot_cont_vars_density_by_class(
    input_data: pd.DataFrame,
    cont_vars_names: list,
    class_col_name: str,
    figure_size: tuple = (12, 60),
    smooth_function_grid_size: int = 500,
    lower_percentile: float = 0.0,
    upper_percentile: float = 1,
) -> None:
    """Plots density of continuous variables with respect to class. It
    gives the option to remove outlier by limitting values to desired
    lower and upper percentiles to make overlap betwen distribution more
    visible if any. Default is not excluding outliers.
    """

    # Copy continuous variables and class
    dataset = input_data.copy()
    cont_vars_with_class = dataset[cont_vars_names + [class_col_name]].copy()
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(cont_vars_names) / 2)),
        ncols=2,
        figsize=figure_size,
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()  # array to 1D

    # Create a list of dataframe columns to use
    cols = cont_vars_with_class[cont_vars_names].columns

    for col, ax in zip(cols, axes):
        # Extract column values to set upper and lowr x-axis limits
        cont_col_values = cont_vars_with_class[col]

        # Plot
        sns.kdeplot(
            data=cont_vars_with_class[[col, class_col_name]],
            x=col,
            hue=class_col_name,
            fill=True,
            ax=ax,
            gridsize=smooth_function_grid_size,
        )
        ax.set(title=f"Distribution of: {col}", xlabel=None)
        ax.set_xlim(
            cont_col_values.quantile(lower_percentile),
            cont_col_values.quantile(upper_percentile),
        )

    fig.tight_layout()


##########################################################
def plot_cat_vars_dist(
    input_data: pd.DataFrame,
    cat_vars_names: list,
    top_cat_count: int = 10,
    x_axis_label_size: int = 12,
    y_axis_label_size: int = 12,
    figure_size: tuple = (12, 60),
    x_axis_rot: int = 45,
) -> None:
    """Plots distributions of categorical variables."""

    dataset = input_data.copy()
    fig, _ = plt.subplots(len(cat_vars_names), 1, figsize=figure_size)

    for i, ax in enumerate(fig.axes):
        if i < len(dataset[cat_vars_names].columns):
            categories_order = (
                dataset[cat_vars_names[i]]
                .value_counts()
                .sort_values(ascending=False)
                .iloc[:top_cat_count]
                .index
            )
            sns.countplot(
                x=dataset[cat_vars_names].columns[i],
                alpha=0.7,
                data=dataset[cat_vars_names],
                ax=ax,
                order=categories_order,
            )
            ax.tick_params(axis="x", labelsize=(x_axis_label_size - 2))
            ax.tick_params(axis="y", labelsize=(y_axis_label_size - 2))
            ax.set(xlabel="", ylabel=dataset[cat_vars_names].columns[i] + " Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_axis_rot, ha="right")

        for p in ax.patches:
            percentage = f"{100 * p.get_height() / dataset.shape[0]:.1f}%"
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()

            # Percentage alone is enough but you can add {y} before ({percentage}) to show count if desired
            ax.annotate(
                f"({percentage})",
                (x, y),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

    fig.tight_layout()


##########################################################
def plot_non_missing_values_count_over_time(
    input_data: pd.DataFrame,
    var_names: list,
    date_col_name: str,
    figure_size: tuple = (12, 100),
) -> None:
    """Plots continuous variables values count over time to see if certain
    columns started populated on specific date."""

    # Correlation matrix for continuous variables
    dataset = input_data[[date_col_name] + var_names].copy()

    fig, axes = plt.subplots(
        nrows=len(var_names), ncols=1, figsize=figure_size, sharex=False, sharey=False
    )
    axes = axes.ravel()  # array to 1D

    # Create a list of dataframe columns to use
    cols = dataset[var_names].columns

    for col, ax in zip(cols, axes):
        if col != date_col_name:
            data = pd.DataFrame(
                dataset.groupby([date_col_name])[col].count()
            ).reset_index()
            sns.lineplot(data=data[[date_col_name, col]], x=date_col_name, y=col, ax=ax)
            ax.set(
                title=f"Value Count Over Time ({date_col_name}) for column: {col}",
                xlabel=None,
            )
            ax.set_ylim(0, data[col].max() + (data[col].max() / 10))

        fig.tight_layout()


##########################################################
def calculate_cat_vars_cardinality(
    input_data: pd.DataFrame, cat_var_names: list
) -> pd.DataFrame:
    """Calculates the cardinality of each categorical feature."""

    dataset = input_data[cat_var_names].copy()
    cat_vars_cardinality = pd.DataFrame(
        columns=("Categorical Variable Name", "Unique Values count")
    )

    for i, col_name in enumerate(cat_var_names):
        feature = col_name
        cat_vars_cardinality.loc[i, "Categorical Feature Name"] = dataset[
            [feature]
        ].columns[0]
        cat_vars_cardinality.loc[i, "Unique Values count"] = dataset[feature].nunique()

    cat_vars_cardinality.sort_values(
        by=["Unique Values count"], ascending=False, inplace=True
    )

    return cat_vars_cardinality


##########################################################
def plot_cont_vars_corr_heatmap(
    input_data: pd.DataFrame,
    cont_var_names: list,
    min_pos_corr_threshold: float = 0.5,
    min_neg_corr_threshold: float = -0.4,
    figure_size: tuple = (12, 10),
) -> None:
    """Plots heatmap for correlation matrix for continuous variables."""

    # Correlation matrix for continuous variables
    dataset = input_data.copy()
    continuous_vars_corr = dataset[cont_var_names].corr()
    plt.figure(figsize=figure_size)

    # Highlight only correlations between min_pos_corr_threshold and min_neg_corr_threshold
    sns.heatmap(
        continuous_vars_corr[
            (continuous_vars_corr >= min_pos_corr_threshold)
            | (continuous_vars_corr <= min_neg_corr_threshold)
        ],
        cmap="viridis",
        vmax=1.0,
        vmin=-1.0,
        linewidths=0.1,
        annot=True,
        annot_kws={"size": 8},
        square=True,
    )


##########################################################
# Extract feature importance
def prepare_train_set(
    model,
    training_set: pd.DataFrame,
    train_class_col_name: str,
    continuous_features: list,
    cont_features_simple_imputer: str = "median",
    cat_features_simple_imputer: str = "constant",
    cat_features_one_hot_encoder_handle_unknown: str = "infrequent_if_exist",
    features_selection_threshold: float = 0.05,
    cat_features_missing_vals_replacement=np.nan,
    high_cardinality_cat_features: list = [],
    output_hashed_cat_features_count: int = 200,
) -> pd.DataFrame:
    """Prepares training set for feature importance."""

    # Extract categorical features by excluding continuous features names
    categorical_features = [
        col
        for col in list(training_set.columns)
        if col not in continuous_features + [train_class_col_name]
    ]
    train_features_set = training_set[
        continuous_features + categorical_features + [train_class_col_name]
    ].copy()

    # Encode class labels
    class_encoder = LabelEncoder()
    binary_class_col = train_features_set.pop(train_class_col_name)
    train_class = class_encoder.fit_transform(ravel(binary_class_col))

    # Combine transformers parameters into a dict to log them in parent run
    transformer_params = {
        "Continuous Features Simple Imputer": cont_features_simple_imputer,
        "Categorical Features Simple Imputer": cat_features_simple_imputer,
        "Categorical Features Missing Values Replacement": cat_features_missing_vals_replacement,
        "Categorical Features OneHot Encoder Unknown Handling": cat_features_one_hot_encoder_handle_unknown,
        "Features Selection Threshold Value": features_selection_threshold,
    }

    print("Transformer configuration:", transformer_params)

    # Define transformers and create pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cont_features_simple_imputer)),
            ("scaler", StandardScaler()),
        ]
    )

    if len(high_cardinality_cat_features) > 0:
        # Extract low cardinality features for one-hot encoding
        low_cardinality_cat_features = [
            col
            for col in categorical_features
            if col not in high_cardinality_cat_features
        ]

        # Convert high cardinality categorical features to string data type as it's one of the required data types
        train_features_set.loc[
            :, high_cardinality_cat_features
        ] = train_features_set.loc[:, high_cardinality_cat_features].astype("object")

        low_cardinality_categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=cat_features_simple_imputer, fill_value=np.nan
                    ),
                ),  # Replace missing values with np.NaNs
                (
                    "onehot_encoder",
                    OneHotEncoder(
                        handle_unknown=cat_features_one_hot_encoder_handle_unknown,
                        categories="auto",
                        drop="first",
                        sparse=False,
                    ),
                ),
            ]
        )

        high_cardinality_categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=cat_features_simple_imputer, fill_value=np.nan
                    ),
                ),  # Replace missing values with np.NaNs
                (
                    "hash_encoder",
                    FeatureHasher(
                        n_features=output_hashed_cat_features_count, input_type="string"
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, continuous_features),
                (
                    "low_cardinality_categorical",
                    low_cardinality_categorical_transformer,
                    low_cardinality_cat_features,
                ),
                (
                    "high_cardinality_categorical",
                    high_cardinality_categorical_transformer,
                    high_cardinality_cat_features,
                ),
            ]
        )

        print("\nContinuous features:\n", continuous_features)
        print("\nLow cardinality categorical features:\n", low_cardinality_cat_features)
        print(
            "\nHigh cardinality categorical features:\n", high_cardinality_cat_features
        )

    if len(high_cardinality_cat_features) == 0:
        # Extract low cardinality features for one-hot encoding
        low_cardinality_cat_features = categorical_features

        low_cardinality_categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=cat_features_simple_imputer, fill_value=np.nan
                    ),
                ),  # Replace missing values with np.NaNs
                (
                    "onehot_encoder",
                    OneHotEncoder(
                        handle_unknown=cat_features_one_hot_encoder_handle_unknown,
                        categories="auto",
                        drop="first",
                        sparse=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, continuous_features),
                (
                    "low_cardinality_categorical",
                    low_cardinality_categorical_transformer,
                    low_cardinality_cat_features,
                ),
            ]
        )

        print("\nContinuous features:\n", continuous_features)
        print("\nCategorical features:\n", categorical_features)

    selector = VarianceThreshold(threshold=features_selection_threshold)

    data_transform_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("selector", selector)]
    )

    # Extract the transformed training set to be used for hyperparameters optimization search
    train_set_transformed_features = data_transform_pipeline.fit_transform(
        train_features_set
    )

    # Fit pipeline
    fitted_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("classifier", model),
        ]
    )
    fitted_model = fitted_pipeline.fit(train_features_set, train_class)

    # Convert fitted data to pandas
    train_set_transformed_features = pd.DataFrame(train_set_transformed_features)

    # Add features names to ouput transformed dataframe
    col_names = continuous_features + list(
        fitted_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot_encoder"]
        .get_feature_names_out(low_cardinality_cat_features)
    )
    col_names = [i for (i, v) in zip(col_names, list(selector.get_support())) if v]
    transformed_hashed_features_count = len(
        list(pd.DataFrame(train_set_transformed_features).columns)
    ) - len(col_names)
    col_names = col_names + [
        f"Hashed Feature {i}" for i in range(1, transformed_hashed_features_count + 1)
    ]  # Adds names of output hashed features
    train_set_transformed_features.columns = col_names

    return train_set_transformed_features, fitted_model


##########################################################
def convert_class_prob_to_label(
    value,
    class_threshold=0.5,
    class_label_above_threshold="Yes",
    class_label_below_threshold="No",
):
    """
    Converts class probability to No/Yes label based on threshold value.

    Args:
        value (float): class probability
        class_threshold (float): threshold value to map class probability to No/Yes label

    Returns:
        pred_class_label (string): class label ("No" if value < class_threshold, otherwise "Yes")
    """

    if value < class_threshold:
        pred_class_label = class_label_below_threshold
    else:
        pred_class_label = class_label_above_threshold

    return predicted_class_label


##########################################################
def calculate_performance_metrics(true_class, predicted_class):
    """
    Calculates different performance metrics for classification models.

    input:
        true_class (list): true class label.
        predicted_class (list): predicted class label not probability.
    output:
        performance_metrics (dataframe): two-column dataframe with metric name and score.
    """

    # Compute performance metrics on the testing set using model trained with unbalanced training set
    try:
        calculated_accuracy = accuracy_score(true_class, predicted_class)
    except Exception as e:
        calculated_accuracy = None
        print(f"Accuracy calculation error --> {e}")

    try:
        calculated_percision = precision_score(true_class, predicted_class)
    except Exception as e:
        calculated_percision = None
        print(f"Precision calculation error --> {e}")

    try:
        calculated_recall = recall_score(true_class, predicted_class)
    except Exception as e:
        calculated_recall = None
        print(f"Recall (sensitivity) calculation error --> {e}")

    try:
        calculated_f1 = f1_score(true_class, predicted_class)
    except Exception as e:
        calculated_f1 = None
        print(f"F1 score calculation error --> {e}")

    try:
        calculated_auc = roc_auc_score(true_class, predicted_class)
    except Exception as e:
        calculated_auc = None
        print(f"AUC calculation error --> {e}")

    performance_metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC Score"],
            "Score": [
                calculated_accuracy,
                calculated_percision,
                calculated_recall,
                calculated_f1,
                calculated_auc,
            ],
        }
    )

    return performance_metrics


##########################################################
def calculate_accuracy_by_class(
    true_class: np.ndarray,
    pred_class: np.ndarray,
    class_labels: list,
    normalize_conf_mat: str = None,
) -> Union[dict, pd.DataFrame]:
    """
    Returns accuracies per class label and confusion
    matrix as dataframe. If 'normalize_conf_mat' argument
    is notset to None, it will normalize confusion matrix
    over the true (rows), predicted (columns) conditions
    or all the population. If None, confusion matrix will
    not be normalized (per sklearn package documentation).
    """

    confusion_mat = confusion_matrix(
        y_true=true_class, y_pred=pred_class, normalize=normalize_conf_mat
    )
    class_accuracies = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
    acc_per_class_label = {
        label: acc for label, acc in zip(class_labels, class_accuracies)
    }
    confusion_mat = pd.DataFrame(
        confusion_mat, index=class_labels, columns=class_labels
    )

    return acc_per_class_label, confusion_mat


##########################################################
def plot_feature_importance(
    feature_importance_scores: np.array,
    feature_names: list,
    top_features_count: int = 30,
    font_size: int = 10,
    fig_size: tuple = (8, 18),
    figure_title: str = None,
) -> None:
    """Plots top feature importance with their encoded names."""

    feat_importances = pd.Series(feature_importance_scores, index=feature_names)
    feat_importances = feat_importances.nlargest(top_features_count, keep="all")
    feat_importances.sort_values(ascending=True, inplace=True)
    feat_importances.plot(
        kind="barh", fontsize=font_size, legend=None, figsize=fig_size
    )
    if figure_title is None:
        plt.title(f"Top {top_features_count} important features")
    if figure_title is not None:
        plt.title(figure_title)
    plt.show()
