from pandas import Series
import numpy as np
import pytest
from plot.histogram import Histogram


@pytest.fixture
def hist_data():
    return Series(data=np.random.randint(low=0, high=10, size=100))


@pytest.mark.plot
def test_create_hist(hist_data, tmpdir):
    histogram = Histogram(plot_dir=tmpdir)
    histogram.create(hist_data, plot_file_name="hist.png")
    assert len(tmpdir.listdir()) == 1
