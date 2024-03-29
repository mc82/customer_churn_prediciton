"""Implements test and fixtures of of the churn library module
__author__ = "Maik Goetze"
__date__='2023-02-05'
"""
import logging
import pathlib
import pickle
import random
import string
from os import makedirs, path
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier

from churn_library import (Barplot, ChurnPredictionFactory, Directory,
                           Distplot, FeatureImportancePlot, Heatmap, Histogram,
                           LogisticRegression, ModelSummary, RandomForest,
                           RocCurve, ShapPlot, get_logger)


N_Y_TRAIN = 300
N_Y_TEST = 100

N_FEATURES = 10
N_SAMPLES = 100
NUMBER_OF_TEST_SAMPLES = 100

ARTIFACT_MODEL_DIR = "test/artifacts/data/model/"
BANK_DATA_PATH = "test/artifacts/bank_data.csv"


logger = get_logger(log_level=logging.DEBUG,
                    log_file_name='churn_library_test.log')


@pytest.fixture
def bank_data() -> DataFrame:
    """Fixture to deliver bank data

    Returns:
        DataFrame: Bank data from CSV file
    """
    return pd.read_csv(BANK_DATA_PATH)


@pytest.fixture
def X_train() -> pd.DataFrame:
    """Set of dependent data to train a classifier

    Returns:
        DataFrame: Random values
    """
    return pd.DataFrame(
        {
            "a": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES),
            "b": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES)
        }
    )


@pytest.fixture
def X_test(X_train: DataFrame) -> DataFrame:
    """Replicates training data

    Args:
        X_train (DataFrame): Data coming from another fixture

    Returns:
        DataFrame: Random values
    """
    return X_train


@pytest.fixture
def y_train() -> DataFrame:
    """Crates y variable with random values

    Returns:
        DataFrame: Random int values to train / verify classifier
    """
    return pd.DataFrame({"y": np.random.randint(
        low=0, high=2, size=NUMBER_OF_TEST_SAMPLES)})


@pytest.fixture
def y_test(y_train: DataFrame) -> DataFrame:
    """Replicates the data of another fixture

    Args:
        y_train (DataFrame): 1-dim data

    Returns:
        DataFrame: Random int values to train / verify classifier
    """
    return y_train


@pytest.fixture
def logistic_regression(tmpdir) -> LogisticRegression:
    """Creates object of classifier

    Args:
        tmpdir: Tmp directory to init classifier

    Returns:
        LogisticRegression: Initialized estimator
    """
    return LogisticRegression(model_dir=tmpdir)


@pytest.fixture
def logistic_regression_trained(
        logistic_regression: LogisticRegression,
        X_train: DataFrame,
        y_train: DataFrame
) -> LogisticRegression:
    """Fixture to create a trained classifier

    Args:
        logistic_regression (LogisticRegression): Classifier to train
        X_train (DataFrame):  Dependent data set to fit the model
        y_train (DataFrame):  Independent data set to fit the model

    Returns:
        LogisticRegression: Fitted classifier with random data
    """

    if Path(logistic_regression._model_path).is_file():
        logistic_regression.load()
    else:
        logistic_regression.fit(X_train, y_train)
        logistic_regression.save()
    return logistic_regression


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_fit_logistic_regression(
    logistic_regression: LogisticRegression,
    X_train: DataFrame,
    y_train: DataFrame
):
    """Test fitting the model with random data

    Args:
        logistic_regression (LogisticRegression): _description_
        X_train (DataFrame): Dependent data set
        y_train (DataFrame): Independent data set
    """

    try:
        logistic_regression.fit(X=X_train, y=y_train)
        assert logistic_regression._model is not None
        logger.info("Testing test_fit_logistic_regression: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_fit_logistic_regression: Estimator not fitted")


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_load_logistic_regression() -> None:
    """Test loading a pickled model
    """
    try:
        logistic_regression = LogisticRegression(model_dir=ARTIFACT_MODEL_DIR)
        logistic_regression.load()
        assert logistic_regression._model is not None
        logger.info("Testing test_load_logistic_regression: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_load_logistic_regression: Model not loaded")


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_save_logistic_regression(
        logistic_regression_trained: LogisticRegression,
        tmpdir) -> None:
    """_summary_

    Args:
        logistic_regression_trained (LogisticRegression): Fitted estimator
        to be ready for saving
        tmpdir:  Tmp directory to save model
    """
    try:
        logistic_regression_trained.save()
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_save_logistic_regression: SUCCESS")
    except AssertionError:
        logging.error("Testing test_save_logistic_regression: Model not saved")


@pytest.mark.classifier
def test_predict_logistic_regression(
        logistic_regression_trained: LogisticRegression,
        X_train: DataFrame) -> None:
    """Test inference of fitted classifier

    Args:
        logistic_regression_trained (LogisticRegression): Fitted classifier
        X_train (DataFrame): Data set to perform inference on
    """
    try:
        expected_type = np.ndarray
        prediction_result = logistic_regression_trained.predict(X_train)
        assert isinstance(prediction_result, expected_type)
        logger.info("Testing test_predict_logistic_regression: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_predict_logistic_regression: Result is not of expected type")


@pytest.fixture()
def random_forest(tmpdir) -> RandomForest:
    """Fixture to init classifier object

    Args:
        tmpdir : Tmp directory to init classifier

    Returns:
        RandomForest: Initiated but unfitted estimator
    """
    return RandomForest(model_dir=tmpdir)


@pytest.fixture
def random_forest_trained(
    random_forest: RandomForest,
    X_train: DataFrame,
    y_train: DataFrame
) -> RandomForest:
    """Fixture to deliver a fitted model either freshly trained
    or loaded from disk if available

    Args:
        random_forest (RandomForest): Initiated classifier
        X_train (DataFrame): Dependent data set to be used to fit the classifier
        y_train (DataFrame): Independent data set to be used to fit the classifier

    Returns:
        RandomForest: Trained Random Forest estimator
    """
    model_path = "test/artifacts/data/model/random_forest.pkl"
    _model_path = Path(model_path)
    if _model_path.is_file():
        with open(model_path, "rb") as file_handle:
            random_forest._model = pickle.load(file_handle)
    else:
        random_forest.fit(X_train, y_train)
        random_forest._model_path = model_path
        random_forest.save()
    return random_forest


@pytest.mark.classifier
@pytest.mark.slow
def test_fit_and_save_random_forest(
    X_train: DataFrame,
    y_train: DataFrame,
    random_forest: RandomForest
) -> None:
    """Test fit and save a model with random data

    Args:
        X_train (DataFrame): Dependent data set to be used to fit the classifier
        y_train (DataFrame): Independent data set to be used to fit the classifier
        random_forest (RandomForest): Initiated but unfitted classifier
    """
    try:
        random_forest.fit(X_train, y_train)
        random_forest.save()
        logger.info("Testing test_fit_and_save_random_forest: SUCCESS")
    except:
        logging.error(
            "Testing test_fit_and_save_random_forest: Model couldn't be fitted and saved")


@pytest.mark.classifier
def test_predict_random_forest(
        random_forest_trained: RandomForest,
        X_train: DataFrame) -> None:
    """Test inference of fitted classifier

    Args:
        random_forest_trained (RandomForest): Fitted classifier
        X_train (DataFrame): Data set to perform inference on
    """
    try:
        expected_type = np.ndarray
        prediction_result = random_forest_trained.predict(X_train)
        assert isinstance(prediction_result, expected_type)
        logger.info("Testing test_predict_random_forest: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_predict_random_forest: Predictions are not of the expected type")


@pytest.fixture
def barplot_data() -> Series:
    """Create random data to show in bar plot

    Returns:
        Series: Series of data to plot
    """
    barplot_data_ = pd.DataFrame(
        {"a": np.random.randint(low=0, high=10, size=100)})
    return barplot_data_.a.value_counts("normalize")


@pytest.mark.plot
def test_create_bar_plot(barplot_data: Series, tmpdir) -> None:
    """Test creation and save of bar plot

    Args:
        barplot_data (Series): _description_
        tmpdir (Directory): tmp dir object (pytest fixture)
    """
    try:
        barplot = Barplot(plot_dir=tmpdir, figsize=(15, 8))
        barplot.create(data=barplot_data, plot_file_name="barplot.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_bar_plot: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_bar_plot: Barplot couldn't be created")


@pytest.fixture
def dist_data() -> Series:
    """Creates random values to be plot.

    Returns:
        Series: Random values
    """
    return Series(np.random.rand(100))


@pytest.mark.plot
def test_create_dist_plot(dist_data: Series, tmpdir) -> None:
    """Tests the create ans save of dist plot based on random data

    Args:
        dist_data (Series): Random data
        tmpdir (Directory): Tmp directory to test save (pytest fixture)
    """
    try:
        distplot = Distplot(plot_dir=tmpdir, figsize=(15, 8))
        distplot.create(data=dist_data, plot_file_name="distplot.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_dist_plot: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_dist_plot: Distplot couldn't be created")


@pytest.fixture
def feature_names() -> List[str]:
    """Creates list of random characters

    Returns:
        List[str]: Random characters
    """
    return [random.choice(string.ascii_uppercase) for _ in range(N_FEATURES)]


@pytest.fixture
def X_random_forest() -> np.ndarray:
    """Creates array of random values

    Returns:
        np.ndarray: 2 dimension array filled with random values
    """
    return np.random.rand(N_SAMPLES, N_FEATURES)


@pytest.fixture
def y_random_forest() -> np.ndarray:
    """Creates random 1-dim array to be used as independent variable

    Returns:
        np.ndarray: 1-dim array with random values
    """
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def random_forest_estimator(
        X_random_forest: np.ndarray,
        y_random_forest: np.ndarray) -> RandomForestClassifier:
    """Creates and fits estimator with random values

    Args:
        X_random_forest (np.ndarray): 2 dimension array filled with random values
        y_random_forest (np.ndarray): 1-dim array with random values

    Returns:
        RandomForestClassifier: Fitted estimator with random values
    """
    random_forest_estimator = RandomForestClassifier()
    random_forest_estimator.fit(X_random_forest, y_random_forest)
    return random_forest_estimator


@pytest.mark.plot
def test_create_feature_importance_plot(
        feature_names: List[str],
        random_forest_estimator: RandomForestClassifier,
        tmpdir) -> None:
    """Test create and save of feature importance plot
    Args:
        feature_names (List[str]): _description_
        random_forest_estimator (RandomForestClassifier): _description_
        tmpdir (Directory): Tmp directory to test saving of plot
    """
    try:
        importances = random_forest_estimator.feature_importances_

        feature_importance_plot = FeatureImportancePlot(
            plot_dir=tmpdir, figsize=(20, 7))
        feature_importance_plot.create(
            feature_names=feature_names,
            data=importances,
            plot_file_name="feature_importance_plot.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_feature_importance_plot: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_feature_importance_plot: FeatureImportancePlot couldn't be created")


@pytest.fixture
def heatmap_data() -> DataFrame:
    """Creates random data and calculates correlation

    Returns:
        DataFrame: DataFrame with pair-wise correlations
    """
    n_rows = 100
    random_data = pd.DataFrame(
        {
            "a": np.random.random(size=n_rows),
            "b": np.random.random(size=n_rows),
        }
    )
    return random_data.corr()


@pytest.mark.plot
def test_create_heatmap(heatmap_data: DataFrame, tmpdir) -> None:
    """Test to crate heatmap plot

    Args:
        heatmap_data (DataFrame): Correlation coefficients to be plot
        tmpdir (Directory): Tmp directory to test saving of plot
    """
    try:
        heatmap = Heatmap(plot_dir=tmpdir)
        heatmap.create(data=heatmap_data, plot_file_name="heatmap.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_heatmap: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_heatmap: Heatmap couldn't be created")


@pytest.fixture
def hist_data() -> Series:
    """Creates random data for histogram test

    Returns:
        Series: Random data to be ploted in a histogram
    """
    return Series(data=np.random.randint(low=0, high=10, size=100))


@pytest.mark.plot
def test_create_hist(hist_data: Series, tmpdir) -> None:
    """_summary_

    Args:
        hist_data (Series): Data to be used in histogram plot
        tmpdir : Tmp directory to save histogram plot
    """

    try:
        histogram = Histogram(plot_dir=tmpdir)
        histogram.create(hist_data, plot_file_name="hist.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_hist: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_hist: Histogram couldn't be created")


@pytest.fixture
def y_train_model_summary() -> np.ndarray:
    """Creates random y values

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_train_pred_model_summary() -> np.ndarray:
    """Creates random y data

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_test_model_summary() -> np.ndarray:
    """Creates random y data to be used in model summary tests

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.fixture
def y_test_pred_model_summary() -> np.ndarray:
    """Creates random y data for model summary
    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.mark.plot
def test_create_model_summary_plot(
        y_train_model_summary: np.ndarray,
        y_train_pred_model_summary: np.ndarray,
        y_test_model_summary: np.ndarray,
        y_test_pred_model_summary: np.ndarray,
        tmpdir):
    """Test of creating a model summary plot
    Args:
        y_train_model_summary (np.ndarray):
        y_train_pred_model_summary (np.ndarray):
        y_test_model_summary (np.ndarray):
        y_test_pred_model_summary (np.ndarray):
        tmpdir (Directory): Tmp directory to save plot
    """

    try:
        model_summary = ModelSummary(plot_dir=tmpdir, figsize=(6, 6))
        model_summary.create(
            y_train=y_train_model_summary,
            y_train_pred=y_train_pred_model_summary,
            y_test=y_test_model_summary,
            y_test_pred=y_test_pred_model_summary,
            plot_file_name="model_summary.png",
            model_name="my_classifier")
        assert len(tmpdir.listdir()) == 1

        logger.info("Testing test_create_model_summary_plot: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_model_summary_plot: ModelSummary couldn't be created")


@pytest.fixture
def X_array() -> np.ndarray:
    """Fixture to provide dependent values for tests
    Returns:
        np.ndarray: 1-dim array with random numbers
    """
    return np.random.random(size=N_SAMPLES).reshape(-1, 1)


@pytest.fixture
def y_array() -> np.ndarray:
    """Fixture to provide target values for tests
    Returns:
        np.nested_iters: 1-dim array with random numbers
    """
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator_logistic_regression(
        X_array: np.ndarray,
        y_array: np.ndarray,
        tmpdir) -> LogisticRegression:
    """Fixture to be used in tests to provide logistic regression estimator
    Args:
        X_array (np.ndarray): Dependent variables
        y_array (np.ndarray): Independent variable
        tmpdir (str): model directory

    Returns:
        LogisticRegression: Fitted estimator with random values
    """
    logistic_regression_estimator = LogisticRegression(tmpdir)
    logistic_regression_estimator.fit(X=X_array, y=y_array)
    return logistic_regression_estimator


@pytest.mark.plot
def test_create_roc_curve(
        X_random_forest: np.ndarray,
        y_random_forest: np.ndarray,
        random_forest_estimator: RandomForestClassifier,
        tmpdir):
    """Tests create and save of roc curve plot

    Args:
        X_random_forest(np.ndarray): independent variable
        y_random_forest(np.ndarray): dependent variables
        random_forest_estimator (RandomForestClassifier): Estimator to assess quality
        tmpdir (Directory): Tmp directory to save plots
    """
    try:
        roc_curve = RocCurve(plot_dir=tmpdir, figsize=(15, 8))
        roc_curve.create(
            X=X_random_forest,
            y=y_random_forest,
            estimator=random_forest_estimator,
            plot_file_name="roc_plot.png")
        assert len(tmpdir.listdir()) == 1

        logger.info("Testing test_create_roc_curve: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_create_roc_curve: RocCurve couldn't be created")


@pytest.mark.plot
def test_create_shap_plot(
        random_forest_estimator: RandomForestClassifier,
        X_random_forest: np.ndarray,
        tmpdir):
    """test create and save of shap plot

    Args:
        random_forest_estimator (RandomForestClassifier): Fitted tree estimator
        X_random_forest (np.ndarray): Dependent variables
        tmpdir (Directory): Tmp dir to save plot
    """

    try:
        shap_plot = ShapPlot(figsize=(15, 8), plot_dir=tmpdir)
        shap_plot.create(estimator=random_forest_estimator,
                         X=X_random_forest, plot_file_name="shap_plot.png")
        assert len(tmpdir.listdir()) == 1
        logger.info("Testing test_create_shap_plot: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_createst_create_shap_plotte_roc_curve: ShapPlot couldn't be created")


@pytest.mark.factory
@pytest.mark.parametrize("classifier_name",
                         [("random_forest"), ("logistic_regression")])
def test_register_one_classifier(classifier_name: str):
    """Test whether registration of one classifier works

    Args:
        classifier_name (str): Name of the classifier to register
    """

    try:
        churn_prediction_factory = ChurnPredictionFactory()
        churn_prediction_factory.register_classifier(classifier_name)

        assert len(churn_prediction_factory.registered_classifiers) == 1
        assert churn_prediction_factory.registered_classifiers.pop() in [
            RandomForest, LogisticRegression]

        logger.info("Testing test_register_one_classifier: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_register_one_classifier: {} couldn't be registered".format(classifier_name))


@pytest.mark.factory
def test_register_both_classifiers():
    """Test to register both used classifiers
    """

    try:
        churn_prediction_factory = ChurnPredictionFactory()
        churn_prediction_factory.register_classifier("random_forest")
        churn_prediction_factory.register_classifier("logistic_regression")

        assert len(churn_prediction_factory.registered_classifiers) == 2
        logger.info("Testing test_register_both_classifiers: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_register_both_classifiers: Both classifiers couldn't be registered")


@pytest.fixture
def directory_exists(tmpdir) -> Directory:
    """Fixture to create temporary directory if exists
    Args:
        tmpdir (TemporaryDirectory): Temporary directory object create by fixture
    Returns:
        Directory: temporary Directory object
    """
    makedirs(path.join(tmpdir, "test"))
    directory = Directory(path.join(tmpdir, "test"))
    return directory


@pytest.fixture
def directory_not_exists(tmpdir) -> Directory:
    """Fixture to create temporary directory not exists
    Args:
        tmpdir (TemporaryDirectory): Temporary directory object create by fixture
    Returns:
        Directory: temporary Directory object
    """
    test_dir = "test"
    directory = Directory(path.join(tmpdir, test_dir))
    return directory


def test_path_exists(directory_exists):
    """Test to check whether path exist
    Args:
        directory_exists (Directory): directory object create by fixture
    """
    try:
        assert isinstance(directory_exists.directory, pathlib.Path)
        logger.info("Testing test_path_exists: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_path_exists: Type does not match expectation")



def test_path_not_exists(directory_not_exists):
    """Test to check whether path not exist
    Args:
        directory_not_exists (Directory): directory object create by fixture
    """
    try:
        assert isinstance(directory_not_exists.directory, pathlib.Path)
        logger.info("Testing test_path_not_exists: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_path_not_exists: Type does not match expectation")

