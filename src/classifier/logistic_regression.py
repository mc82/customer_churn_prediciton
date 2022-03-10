"""
Contains Logistic Regression interface
"""
from sklearn.linear_model import LogisticRegression as LogisticRegression_
import pandas as pd

from .classifier import Classifier


class LogisticRegression(Classifier):
    '''
    Performs Logistic Regression
    '''

    def __init__(self, model_path: str) -> None:
        '''
        Initializes the base classifier
        Args:
            model_path (str): Path of the model to load and save
        '''
        super().__init__(model_path)
        self._model = LogisticRegression_()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        fits a logistic regression model

        Args:
            X (pandas:DataFrame): independent variables
            y (pandas:DataFrame): dependent variable
        """
        self._model.fit(X=X, y=y)
