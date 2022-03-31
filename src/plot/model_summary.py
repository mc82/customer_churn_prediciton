"""
Implements class to summarize model performance.
__author__ = "Maik Goetze"
"""

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pandas import Series
from .plot import Plot


class ModelSummary(Plot):
    """
    Implements method to create plots about model
    performance leveraging a common interace.
    """

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(
            self,
            y_train: Series,
            y_train_pred: Series,
            y_test: Series,
            y_test_pred: Series,
            model_name: str,
            plot_file_name: str):
        """
        Creates and saves performance summary as a plot.

        Args:
            y_train (Series): Independent variable used to train the model.
            y_train_pred (Series): Predictions of the model on the training data.
            y_test (Series): Independent variable used to test the model.
            y_test_pred (Series): Predictions of the model on the test data.
            model_name (str): Name of the used model
            plot_file_name (str): File name of the plot.
        """
        plt.rc('figure', figsize=self._figsize)
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_pred)), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_pred)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        self.save(plt.gcf(), plot_file_name)
