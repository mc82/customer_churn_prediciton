"""Implements fixtures to be used in multiple test modules
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

NUMBER_OF_TEST_SAMPLES = 100
BANK_DATA_PATH = "test/artifacts/bank_data.csv"


@pytest.fixture
def bank_data() -> DataFrame:
    """Fixture to deliver bank data

    Returns:
        DataFrame: Bank data from CSV file
    """
    return pd.read_csv(BANK_DATA_PATH)


@pytest.fixture
def X_train() -> pd.DataFrame:
    """Set of dependent data to train a classifier

    Returns:
        pd.DataFrame: Random values
    """
    return pd.DataFrame(
        {
            "a": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES),
            "b": np.random.uniform(size=NUMBER_OF_TEST_SAMPLES)
        }
    )


def X_test(X_train: DataFrame) -> DataFrame:
    """Replicates training data

    Args:
        X_train (DataFrame): Data coming from another fixture

    Returns:
        DataFrame: Random values
    """
    return X_train


@pytest.fixture
def y_train() -> DataFrame:
    """Independet variable

    Returns:
        DataFrame: Random int values to train / verify classifier
    """
    return pd.DataFrame({"y": np.random.randint(
        low=0, high=2, size=NUMBER_OF_TEST_SAMPLES)})


@pytest.fixture
def y_test(y_train: DataFrame) -> DataFrame:
    """Replicates the data of another fixture

    Args:
        y_train (DataFrame): 1-dim data

    Returns:
        DataFrame: Random int values to train / verify classifier
    """
    return y_train
