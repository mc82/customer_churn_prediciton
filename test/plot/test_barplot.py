"""Implements test of barplot module
__author__ = "Maik Goetze"
"""

import pandas as pd
from pandas import Series
import numpy as np
import pytest

from plot import Barplot


@pytest.fixture
def barplot_data() -> Series:
    """Create random data to show in bar plot

    Returns:
        Series: Series of data to plot
    """
    barplot_data_ = pd.DataFrame(
        {"a": np.random.randint(low=0, high=10, size=100)})
    return barplot_data_.a.value_counts("normalize")


@pytest.mark.plot
def test_create(barplot_data: Series, tmpdir) -> None:
    """Test creation and save of bar plot

    Args:
        barplot_data (Series): _description_
        tmpdir : tmp dir object (pytest fixture)
    """
    barplot = Barplot(plot_dir=tmpdir, figsize=(15, 8))
    barplot.create(data=barplot_data, plot_file_name="barplot.png")
    assert len(tmpdir.listdir()) == 1
