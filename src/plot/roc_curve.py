"""
Implements class to create roc curves.
"""

from sklearn.metrics import plot_roc_curve
from pandas import DataFrame, Series

from .plot import Plot


class RocCurve(Plot):
    """
    Implements method to create and save roc curves.
    """

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, estimator, X: DataFrame, y: Series, plot_file_name: str) -> None:
        """Creates roc curve based on estimator and dependent variables

        Args:
            estimator (_type_): _description_
            X (DataFrame): Dependent variables
            y (Series): Independent variable
            plot_file_name (str): File name of the plot on disk.
        """
        figure = self._plt.gca()
        roc_plot = plot_roc_curve(estimator, X, y, ax=figure, alpha=0.8)
        roc_plot.plot(ax=figure, alpha=0.8)
        self.save(figure=figure, plot_name=plot_file_name)
