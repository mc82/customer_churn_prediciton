"""Implements test of heatmap module
__author__ = "Maik Goetze"
"""
import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame

from plot import Heatmap


@pytest.fixture
def heatmap_data() -> DataFrame:
    """Creates random data and calculates correlation

    Returns:
        DataFrame: DataFrame with pair-wise correlations
    """
    n_rows = 100
    random_data = pd.DataFrame(
        {
            "a": np.random.random(size=n_rows),
            "b": np.random.random(size=n_rows),
        }
    )
    return random_data.corr()


@pytest.mark.plot
def test_create(heatmap_data: DataFrame, tmpdir) -> None:
    """_summary_

    Args:
        heatmap_data (DataFrame): Correlation coefficients to be plot
        tmpdir: Tmp directory to test saving of plot
    """
    heatmap = Heatmap(plot_dir=tmpdir)
    heatmap.create(data=heatmap_data, plot_file_name="heatmap.png")
    assert len(tmpdir.listdir()) == 1
