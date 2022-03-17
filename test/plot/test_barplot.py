import pandas as pd
import numpy as np
import pytest

from plot import Barplot


@pytest.fixture
def barplot_data():
    df = pd.DataFrame({"a": np.random.randint(low=0, high=10, size=100)})
    return df.a.value_counts("normalize")


@pytest.mark.plot
def test_create(barplot_data, tmpdir):
    barplot = Barplot(plot_dir=tmpdir, figsize=(15, 8))
    barplot.create(data=barplot_data, plot_file_name="barplot.png")
    assert len(tmpdir.listdir()) == 1
