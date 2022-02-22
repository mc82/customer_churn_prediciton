import shap

from .plot import Plot


class ShapPlot(Plot):
    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, estimator, X, plot_name: str):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        self.save(figure=self._plt, plot_name=plot_name)
