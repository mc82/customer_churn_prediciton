import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import pickle

from classifier import RandomForest

NUMBER_OF_TEST_SAMPLES = 100


@pytest.fixture
def X_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES),
            "b": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES)
        }
    )


def X_test(X_train) -> pd.DataFrame:
    return X_train


@pytest.fixture
def y_train() -> pd.DataFrame:
    return pd.DataFrame({"y": np.random.randint(low=0, high=1, size=NUMBER_OF_TEST_SAMPLES)})


@pytest.fixture
def y_test(y_train):
    return y_train


@pytest.fixture()
def random_forest(tmpdir):
    return RandomForest(model_path=tmpdir)


@pytest.fixture
def random_forest_trained(random_forest, X_train, y_train):
    model_path = "test/artifacts/data/model/model.pkl"
    _model_path = Path(model_path)
    if _model_path.is_file():
        with open(model_path, "rb") as f:
            random_forest._model = pickle.load(f)
    else:
        random_forest.fit(X_train, y_train)
        random_forest._model_path = model_path
        random_forest.save()
    return random_forest


@pytest.mark.classifier
@pytest.mark.slow
def test_fit_and_save(X_train, y_train, random_forest):
    random_forest.fit(X_train, y_train)
    random_forest.save()


@pytest.mark.classifier
def test_predict(random_forest_trained, X_train):
    expected_type = np.ndarray

    prediction_result = random_forest_trained.predict(X_train)

    assert type(prediction_result) == expected_type
