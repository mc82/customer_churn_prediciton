import numpy as np
import pytest
import pandas as pd

NUMBER_OF_TEST_SAMPLES = 100
BANK_DATA_PATH = "/Users/mgoetze/repos/customer_churn_prediction/test/artifacts/bank_data.csv"


@pytest.fixture
def bank_data():
    return pd.read_csv(BANK_DATA_PATH)


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
    return pd.DataFrame({"y": np.random.randint(
        low=0, high=2, size=NUMBER_OF_TEST_SAMPLES)})


@pytest.fixture
def y_test(y_train):
    return y_train
