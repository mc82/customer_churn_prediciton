import pytest
import random
import string
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from plot import FeatureImportancePlot

N_FEATURES = 10
N_SAMPLES = 100


@pytest.fixture
def feature_names():
    return [random.choice(string.ascii_uppercase) for _ in range(N_FEATURES)]


@pytest.fixture
def X():
    return np.random.rand(N_SAMPLES, N_FEATURES)


@pytest.fixture
def y():
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X, y):
    random_forest_estimator = RandomForestClassifier()
    random_forest_estimator.fit(X, y)
    return random_forest_estimator


@pytest.fixture
def importances(estimator):
    return estimator.feature_importances_


@pytest.mark.plot
def test_create(feature_names, importances, tmpdir):
    feature_importance_plot = FeatureImportancePlot(
        plot_dir=tmpdir, figsize=(20, 7))
    feature_importance_plot.create(
        feature_names=feature_names,
        importances=importances,
        plot_name="feature_importance_plot.png")
    assert len(tmpdir.listdir()) == 1
