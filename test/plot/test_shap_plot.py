import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from plot import ShapPlot

N_SAMPLES = 100


@pytest.fixture
def X():
    return np.random.random(size=N_SAMPLES).reshape(-1, 1)


@pytest.fixture
def y():
    return np.random.random_integers(low=0, high=1, size=N_SAMPLES)


@pytest.fixture
def estimator(X, y):
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X, y)
    return random_forest_classifier


@pytest.mark.plot
def test_create(estimator, X,  tmpdir):
    shap_plot = ShapPlot(figsize=(15, 8), plot_dir=tmpdir)
    shap_plot.create(estimator=estimator, X=X, plot_name="shap_plot.png")
    assert len(tmpdir.listdir()) == 1
