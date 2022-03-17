"""
Implements class to create hist plots.
"""

from pandas import Series

from plot.plot import Plot


class Histogram(Plot):
    """
    Implements method to create and save hist plots leveraging a common interface.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates histogram plot and saves it on disk.
        Args:
            data (Series): Data to plot as histogram
            plot_file_name (str): File name of the created plot
        """
        figure = data.hist()
        self.save(figure=figure, plot_name=plot_file_name)
