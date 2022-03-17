"""
Implements class to provide plot with shap values.
"""

from pandas import DataFrame
import shap

from .plot import Plot


class ShapPlot(Plot):
    """
    Implements method to create shap plot.
    Args:
        Plot (Plot): Base class with common interface
    """

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, estimator, X: DataFrame, plot_file_name: str):
        """_summary_

        Args:
            estimator (_type_): trained tree estimator
            X (DataFrame): Dependend variables to train to model.
            plot_file_name (str): File name of the shap plot.
        """
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        self.save(figure=self._plt, plot_name=plot_file_name)
