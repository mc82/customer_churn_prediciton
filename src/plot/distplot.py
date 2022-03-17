"""
Implements class to create dist plots
"""

from pandas import Series
import seaborn as sns
from .plot import Plot
sns.set()


class Distplot(Plot):
    """
    Provides method the create a dist plot using a common interface.
    """

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates a dist plot based on given data and saves it on disk.
        Args:
            data (Series): _description_
            plot_file_name (str): _description_
        """
        figure = sns.distplot(data)
        self.save(figure=figure, plot_name=plot_file_name)
