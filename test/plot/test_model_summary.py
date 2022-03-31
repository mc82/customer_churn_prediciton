"""Implements test of model_summary module
__author__ = "Maik Goetze"
"""
import pytest
import numpy as np
from plot import ModelSummary

N_Y_TRAIN = 300
N_Y_TEST = 100


@pytest.fixture
def y_train() -> np.ndarray:
    """Creates random y values

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_train_pred() -> np.ndarray:
    """Creates random y data

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_test() -> np.ndarray:
    """Creates random y data

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.fixture
def y_test_pred() -> np.ndarray:
    """Creates random y data

    Returns:
        np.ndarray: Random 1-dim data
    """
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.mark.plot
def test_create(
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        tmpdir):
    """_summary_

    Args:
        y_train (np.ndarray):
        y_train_pred (np.ndarray):
        y_test (np.ndarray):
        y_test_pred (np.ndarray):
        tmpdir (_type_): Tmp directory to save plot
    """
    model_summary = ModelSummary(plot_dir=tmpdir, figsize=(6, 6))
    model_summary.create(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        plot_file_name="model_summary.png",
        model_name="my_classifier")
    assert len(tmpdir.listdir()) == 1
