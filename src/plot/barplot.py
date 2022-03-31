'''
Module provides a class to create bar plots by leveraging a common interface
'''

from pandas import Series
from .plot import Plot


class Barplot(Plot):
    """
    Creates and saves bar plot based on given input.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        """
        Inits the base class
        Args:
            plot_dir (str, optional): Directory where the plots are saved as a file.
                                      Defaults to "plots".
            figsize (_type_, optional): Size of the plot passed as a tuple. Defaults to ....
        """
        super().__init__(plot_dir, figsize)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates barplot based on input data and save it on disk.
        Args:
            barplot_data (Series): Contains the data to plot
            plot_file_name (str): Name of the file
        """
        figure = data.plot(kind='bar')
        self.save(figure=figure, plot_name=plot_file_name)
