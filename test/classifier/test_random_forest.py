"""Implements test of logistic_regression module
"""
import pickle
from pathlib import Path

import numpy as np
from pandas import DataFrame
import pytest

from classifier import RandomForest


@pytest.fixture()
def random_forest(tmpdir) -> RandomForest:
    """Fixture to init classifier object

    Args:
        tmpdir : Tmp directory to init classifier

    Returns:
        RandomForest: Initiated but unfitted estimator
    """
    return RandomForest(model_path=tmpdir)


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
        RandomForest: _description_
    """
    model_path = "test/artifacts/data/model/model.pkl"
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
def test_fit_and_save(
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
    random_forest.fit(X_train, y_train)
    random_forest.save()


@pytest.mark.classifier
def test_predict(random_forest_trained: RandomForest, X_train: DataFrame) -> None:
    """Test inference of fitted classifier

    Args:
        random_forest_trained (RandomForest): Fitted classifier
        X_train (DataFrame): Data set to perform inference on
    """
    expected_type = np.ndarray

    prediction_result = random_forest_trained.predict(X_train)

    assert isinstance(prediction_result, expected_type)
