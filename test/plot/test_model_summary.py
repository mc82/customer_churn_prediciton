import pytest
import numpy as np
from plot import ModelSummary

N_Y_TRAIN = 300
N_Y_TEST = 100


@pytest.fixture
def y_train():
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_train_pred():
    return np.random.randint(low=0, high=1, size=N_Y_TRAIN)


@pytest.fixture
def y_test():
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.fixture
def y_test_pred():
    return np.random.randint(low=0, high=1, size=N_Y_TEST)


@pytest.mark.plot
def test_create(y_train, y_train_pred, y_test, y_test_pred, tmpdir):
    model_summary = ModelSummary(plot_dir=tmpdir, figsize=(6, 6))
    model_summary.create(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        plot_file_name="model_summary.png",
        model_name="my_classifier")
    assert len(tmpdir.listdir()) == 1
