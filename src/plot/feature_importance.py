"""
Implements class to create plots of feature importance.
"""
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from pandas import Series

from .plot import Plot


class FeatureImportancePlot(Plot):
    """
    Provices method the create plots of feature importance.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(
            self,
            data: Series,
            feature_names: List[str],
            plot_file_name: str):
        """
        Creates plot of feature importance and saves it on disk.

        Args:
            data (Series): _description_
            plot_name (str): _description_
        """
        # Sort feature importances in descending order
        indices = np.argsort(data)[::-1]

        # Rearrange feature feature_names so they match the sorted feature
        # importances
        feature_names = [feature_names[i] for i in indices]

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(len(feature_names)), data[indices])

        # Add feature feature_names as x-axis labels
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=90)
        self.save(figure=self._plt, plot_name=plot_file_name)
