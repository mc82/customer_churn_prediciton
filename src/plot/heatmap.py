"""
Implements class to create heat map plot.
__author__ = "Maik Goetze"
"""
import seaborn as sns
from pandas import Series
from .plot import Plot
sns.set()


class Heatmap(Plot):
    """
    Implements method to create heat map plots using a common interace for plots.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates heat map plot based on input data and saves in on disk.
        Args:
            data (Series): Contains the data to plot as heat map
            plot_file_name (str): The file name of the plot
        """
        figure = sns.heatmap(data)
        self.save(figure=figure, plot_name=plot_file_name)
