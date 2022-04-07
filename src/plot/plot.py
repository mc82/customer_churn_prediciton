"""
Provices class of the common interface of all plots.
__author__ = "Maik Goetze"
"""


from abc import abstractmethod
from os import path
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from pandas import DataFrame, Series


class Plot:
    """
    Implements Interface and some methods to create and save plots.
    """

    def __init__(self, plot_dir: Path, figsize=(20, 30)) -> None:
        self.__plot_dir = plot_dir
        self._figsize = figsize
        self._init_plot()

    @abstractmethod
    def create(self, data: Union[Series, DataFrame], plot_file_name: str):
        """
        Interface to create plots and save it on disk.
        """

    def save(self, figure, plot_name: str) -> None:
        """
        Saves given plot on disk using given plot name.
        Args:
            figure (_type_): The plot to save.
            plot_name (str): File name of the plot
        """
        plot_path = path.join(self.__plot_dir, plot_name)
        figure.figure.savefig(plot_path)

    def _init_plot(self) -> None:
        """
        Set the size of the plot
        """
        self._plt = plt.figure(figsize=self._figsize)

    def __del__(self) -> None:
        """
        Avoids plot of multiple figures into one plot
        """
        plt.close()
