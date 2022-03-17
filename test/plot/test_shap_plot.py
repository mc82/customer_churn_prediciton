
"""Implements test of shap_plot module
"""
import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from plot import ShapPlot

N_SAMPLES = 100


@pytest.fixture
def X() -> np.ndarray:
    """Create random values for dependent variable

    Returns:
        np.ndarray: 1-dim random data
    """
    return np.random.random(size=N_SAMPLES).reshape(-1, 1)


@pytest.fixture
def y() -> np.ndarray:
    """Create random values for independent variable

    Returns:
        np.ndarray: 1-dim random data
    """
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Creates fitted estimator with random data

    Args:
        X (np.ndarray): dependent variable
        y (np.ndarray): independent variable

    Returns:
        RandomForestClassifier: _description_
    """
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X, y)
    return random_forest_classifier


@pytest.mark.plot
def test_create(estimator: RandomForestClassifier, X: np.ndarray, tmpdir):
    """test create and save of shap plot

    Args:
        estimator (RandomForestClassifier): Fitted tree estimator
        X (np.ndarray): Dependent variables
        tmpdir (_type_): Tmp dir to save plot
    """
    shap_plot = ShapPlot(figsize=(15, 8), plot_dir=tmpdir)
    shap_plot.create(estimator=estimator, X=X, plot_file_name="shap_plot.png")
    assert len(tmpdir.listdir()) == 1
