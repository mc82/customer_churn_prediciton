from .plot import Plot
from pandas import Series
import seaborn as sns
sns.set()


class Distplot(Plot):

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, dist_data: Series, plot_file_name: str) -> None:
        ax = sns.distplot(dist_data)
        self.save(figure=ax, plot_name=plot_file_name)
