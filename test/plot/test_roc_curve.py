from matplotlib.pyplot import plot
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from plot import RocCurve


N_SAMPLES = 100


@pytest.fixture
def X():
    return np.random.random(size=N_SAMPLES).reshape(-1, 1)


@pytest.fixture
def y():
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X, y):
    logistic_regression_estimator = LogisticRegression()
    logistic_regression_estimator.fit(X=X, y=y)
    return logistic_regression_estimator


@pytest.mark.plot
def test_create(X, y, estimator, tmpdir):
    roc_curve = RocCurve(plot_dir=tmpdir, figsize=(15, 8))
    roc_curve.create(X=X, y=y, estimator=estimator, plot_name="roc_plot.png")
    assert len(tmpdir.listdir()) == 1
