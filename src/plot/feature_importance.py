import numpy as np
from matplotlib import pyplot as plt

from .plot import Plot


class FeatureImportancePlot(Plot):

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, feature_names, importances, plot_name: str):
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature feature_names so they match the sorted feature
        # importances
        feature_names = [feature_names for i in indices]

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(len(feature_names)), importances[indices])

        # Add feature feature_names as x-axis labels
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=90)
        self.save(figure=self._plt, plot_name=plot_name)
