from plot import Distplot
import pytest
from pandas import Series
import numpy as np


@pytest.fixture
def dist_data():
    return Series(np.random.rand(100))


@pytest.mark.plot
def test_create(dist_data, tmpdir):
    distplot = Distplot(plot_dir=tmpdir, figsize=(15, 8))
    distplot.create(dist_data=dist_data, plot_file_name="distplot.png")
    assert len(tmpdir.listdir()) == 1
