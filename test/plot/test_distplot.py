"""Implements test of distplot module
"""
import numpy as np
import pytest
from pandas import Series

from plot import Distplot


@pytest.fixture
def dist_data() -> Series:
    """Creates random values to be plot.

    Returns:
        Series: Random values
    """
    return Series(np.random.rand(100))


@pytest.mark.plot
def test_create(dist_data: Series, tmpdir) -> None:
    """Tests the create ans save of dist plot based on random data

    Args:
        dist_data (Series): Random data
        tmpdir : Tmp directory to test save (pytest fixture)
    """
    distplot = Distplot(plot_dir=tmpdir, figsize=(15, 8))
    distplot.create(data=dist_data, plot_file_name="distplot.png")
    assert len(tmpdir.listdir()) == 1
