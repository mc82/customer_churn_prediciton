from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from .plot import Plot


class ModelSummary(Plot):

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, y_train, y_train_pred, y_test, y_test_pred, model_name, plot_file_name: str):
        plt.rc('figure', figsize=self._figsize)
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_pred)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_pred)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')

        self.save(plt.gcf(), plot_file_name)
