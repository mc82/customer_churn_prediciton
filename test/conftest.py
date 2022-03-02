import pytest
import pandas as pd

BANK_DATA_PATH = "/Users/mgoetze/repos/customer_churn_prediction/test/artifacts/bank_data.csv"


@pytest.fixture
def bank_data():
    return pd.read_csv(BANK_DATA_PATH)
