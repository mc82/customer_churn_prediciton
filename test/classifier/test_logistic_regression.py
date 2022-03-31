"""Implements tests of logistic_regression module
__author__ = "Maik Goetze"
"""
from pathlib import Path
import numpy as np

from pandas import DataFrame
import pytest

from classifier import LogisticRegression

ARTIFACT_MODEL_DIR = "test/artifacts/data/model/"


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
def test_fit(
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
    logistic_regression.fit(X=X_train, y=y_train)
    assert logistic_regression._model is not None


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_load() -> None:
    """Test loading a pickled model
    """
    logistic_regression = LogisticRegression(model_dir=ARTIFACT_MODEL_DIR)
    logistic_regression.load()
    assert logistic_regression._model is not None


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_save(logistic_regression_trained: LogisticRegression, tmpdir) -> None:
    """_summary_

    Args:
        logistic_regression_trained (LogisticRegression): Fitted estimator
        to be ready for saving
        tmpdir :  Tmp directory to save model
    """
    logistic_regression_trained.save()
    assert len(tmpdir.listdir()) == 1


@pytest.mark.classifier
def test_predict(
        logistic_regression_trained: LogisticRegression,
        X_train: DataFrame) -> None:
    """Test inference of fitted classifier

    Args:
        random_forest_trained (RandomForest): Fitted classifier
        X_train (DataFrame): Data set to perform inference on
    """
    expected_type = np.ndarray

    prediction_result = logistic_regression_trained.predict(X_train)

    assert isinstance(prediction_result, expected_type)
