from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve

from .plot import Plot


class RocCurve(Plot):

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, estimator, X, y, plot_name: str):
        ax = self._plt.gca()
        roc_plot = plot_roc_curve(estimator, X, y, ax=ax, alpha=0.8)
        roc_plot.plot(ax=ax, alpha=0.8)
        self.save(figure=ax, plot_name=plot_name)
