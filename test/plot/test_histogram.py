"""Implements test of histogram module
"""
import numpy as np
import pytest
from pandas import Series

from plot.histogram import Histogram


@pytest.fixture
def hist_data() -> Series:
    """Creates random data for histogram test

    Returns:
        Series: Random data to be ploted in a histogram
    """
    return Series(data=np.random.randint(low=0, high=10, size=100))


@pytest.mark.plot
def test_create_hist(hist_data: Series, tmpdir) -> None:
    """_summary_

    Args:
        hist_data (Series): Data to be used in histogram plot
        tmpdir : Tmp directory to save histogram plot
    """
    histogram = Histogram(plot_dir=tmpdir)
    histogram.create(hist_data, plot_file_name="hist.png")
    assert len(tmpdir.listdir()) == 1
