import pytest
from pathlib import Path
import pickle
from classifier import LogisticRegression

ARTIFACT_MODEL_PATH = "test/artifacts/data/model/logistic_regression.pkl"


@pytest.fixture
def logistic_regression(tmpdir):
    return LogisticRegression(model_path=tmpdir)


@pytest.fixture
def logistic_regression_trained(logistic_regression, X_train, y_train):
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
def test_fit(logistic_regression, X_train, y_train):
    logistic_regression.fit(X=X_train, y=y_train)
    assert logistic_regression._model is not None


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_load():
    logistic_regression = LogisticRegression(model_path=ARTIFACT_MODEL_PATH)
    logistic_regression.load()
    assert logistic_regression._model is not None


@pytest.mark.classifier
@pytest.mark.logistic_regression
def test_save(logistic_regression_trained, tmpdir):
    logistic_regression_trained._model_path = tmpdir + "/loistic_regression_model.pkl"
    logistic_regression_trained.save()
    assert len(tmpdir.listdir()) == 1
