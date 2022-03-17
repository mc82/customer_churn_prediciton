import pytest
import numpy as np
import pandas as pd

from plot import Heatmap


@pytest.fixture
def heatmap_data():
    n_rows = 100
    df = pd.DataFrame(
        {
            "a": np.random.random(size=n_rows),
            "b": np.random.random(size=n_rows),
        }
    )
    return df.corr()


@pytest.mark.plot
def test_create(heatmap_data, tmpdir):
    heatmap = Heatmap(plot_dir=tmpdir)
    heatmap.create(data=heatmap_data, plot_file_name="heatmap.png")
    assert len(tmpdir.listdir()) == 1
