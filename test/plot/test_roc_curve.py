"""Implements test of roc_curve module
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from plot import RocCurve

N_SAMPLES = 100


@pytest.fixture
def X() -> np.ndarray:
    """_summary_

    Returns:
        np.ndarray: 1-dim array with random numbers
    """
    return np.random.random(size=N_SAMPLES).reshape(-1, 1)


@pytest.fixture
def y() -> np.ndarray:
    """_summary_

    Returns:
        np.nested_iters: 1-dim array with randon numbers
    """
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """_summary_

    Args:
        X (np.ndarray): Dependent variables
        y (np.ndarray): Independent variable

    Returns:
        LogisticRegression: Fitted estimator with random values
    """
    logistic_regression_estimator = LogisticRegression()
    logistic_regression_estimator.fit(X=X, y=y)
    return logistic_regression_estimator


@pytest.mark.plot
def test_create(
        X: np.ndarray,
        y: np.ndarray,
        estimator: LogisticRegression,
        tmpdir):
    """Tests create and save of roc curve plot

    Args:
        X (np.ndarray): independent variable
        y (np.ndarray): dependent variables
        estimator (LogisticRegression): Estimator to assess quality
        tmpdir (_type_): Tmp directory to save plots
    """
    roc_curve = RocCurve(plot_dir=tmpdir, figsize=(15, 8))
    roc_curve.create(X=X, y=y, estimator=estimator,
                     plot_file_name="roc_plot.png")
    assert len(tmpdir.listdir()) == 1
