from .plot import Plot


class Barplot(Plot):

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, barplot_data, plot_file_name: str) -> None:
        figure = barplot_data.plot(kind='bar')
        self.save(figure=figure, plot_name=plot_file_name)
