"""Implements test of feature_importance module
__author__ = "Maik Goetze"
"""
import random
import string
from typing import List

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from plot import FeatureImportancePlot

N_FEATURES = 10
N_SAMPLES = 100


@pytest.fixture
def feature_names() -> List[str]:
    """Creates list of random characters

    Returns:
        List[str]: Random characters
    """
    return [random.choice(string.ascii_uppercase) for _ in range(N_FEATURES)]


@pytest.fixture
def X() -> np.ndarray:
    """Creates array of random values

    Returns:
        np.ndarray: 2 dimension array filled with random values
    """
    return np.random.rand(N_SAMPLES, N_FEATURES)


@pytest.fixture
def y() -> np.ndarray:
    """Creates random 1-dim array to be used as independent variable

    Returns:
        np.ndarray: 1-dim array with random values
    """
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Creates and fits estimator with random values

    Args:
        X (np.ndarray): 2 dimension array filled with random values
        y (np.ndarray): 1-dim array with random values

    Returns:
        RandomForestClassifier: Fitted estimator with random values
    """
    random_forest_estimator = RandomForestClassifier()
    random_forest_estimator.fit(X, y)
    return random_forest_estimator


@pytest.mark.plot
def test_create(
        feature_names: List[str],
        estimator: RandomForestClassifier,
        tmpdir) -> None:
    """Test create and save of feature importance plot
    Args:
        feature_names (List[str]): _description_
        estimator (RandomForestClassifier): _description_
        tmpdir (_type_): _description_
    """
    importances = estimator.feature_importances_

    feature_importance_plot = FeatureImportancePlot(
        plot_dir=tmpdir, figsize=(20, 7))
    feature_importance_plot.create(
        feature_names=feature_names,
        data=importances,
        plot_file_name="feature_importance_plot.png")
    assert len(tmpdir.listdir()) == 1
