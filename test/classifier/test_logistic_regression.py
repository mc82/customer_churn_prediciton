"""Implements tests of logistic_regression module
"""
import pickle
from pathlib import Path

from pandas import DataFrame
import pytest

from classifier import LogisticRegression

ARTIFACT_MODEL_PATH = "test/artifacts/data/model/logistic_regression.pkl"


@pytest.fixture
def logistic_regression(tmpdir) -> LogisticRegression:
    """Creates object of classifier

    Args:
        tmpdir: Tmp directory to init classifier

    Returns:
        LogisticRegression: Initialized estimator
    """
    return LogisticRegression(model_path=tmpdir)


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
    model_path = ARTIFACT_MODEL_PATH
    _model_path = Path(model_path)
    if _model_path.is_file():
        with open(model_path, "rb") as f:
            logistic_regression._model = pickle.load(f)
    else:
        logistic_regression.fit(X_train, y_train)
        logistic_regression._model_path = model_path
        logistic_regression.save()
    return logistic_regression


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_fit(
    logistic_regression: LogisticRegression,
    X_train: DataFrame,
    y_train: DataFrame
):
    """Test fitting the model with reandom data

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
    logistic_regression = LogisticRegression(model_path=ARTIFACT_MODEL_PATH)
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
    logistic_regression_trained._model_path = tmpdir + "/loistic_regression_model.pkl"
    logistic_regression_trained.save()
    assert len(tmpdir.listdir()) == 1
