from .plot import Plot
import seaborn as sns
sns.set()


class Heatmap(Plot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self, heatmap_data, plot_file_name: str) -> None:
        figure = sns.heatmap(heatmap_data)
        self.save(figure=figure, plot_name=plot_file_name)
