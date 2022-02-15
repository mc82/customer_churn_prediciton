from abc import abstractmethod
import matplotlib.pyplot as plt
from os import path
import os
from pathlib import Path


class Plot(object):

    def __init__(self, plot_dir="plots", figsize=(20, 30)) -> None:
        self.__plot_dir = plot_dir
        self._figsize = figsize
        self._create_plot_dir_if_not_exists()

    @abstractmethod
    def create(self, data, plot_file_name: str):
        pass

    def save(self, figure, plot_name: str):
        plot_path = path.join(self.__plot_dir, plot_name)
        figure.figure.savefig(plot_path)

    def _init_plot(self):
        plt.figure(figsize=self._figsize)

    def _create_plot_dir_if_not_exists(self):
        if not self._check_if_dir_exists():
            os.mkdir(self.__plot_dir)

    def _check_if_dir_exists(self):
        _path = Path(self.__plot_dir)
        return _path.is_dir()
    
    def __del__(self):
        plt.close()
