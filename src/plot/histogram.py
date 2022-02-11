from pandas import Series

from plot.plot import Plot


class Histogram(Plot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self, data: Series, plot_file_name) -> None:
        self._init_plot()
        ax = data.hist()
        self.save(figure=ax, plot_name=plot_file_name)
