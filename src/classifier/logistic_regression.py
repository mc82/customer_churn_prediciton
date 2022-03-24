"""
Contains Logistic Regression interface
"""
from sklearn.linear_model import LogisticRegression as LogisticRegression_

from .classifier import Classifier


class LogisticRegression(Classifier):
    '''
    Performs Logistic Regression
    '''

    name = "logistic_regression"

    def __init__(self, model_path: str) -> None:
        '''
        Initializes the base classifier
        Args:
            model_path (str): Path of the model to load and save
        '''
        super().__init__(model_path)
        self._model = LogisticRegression_()
        
    @property
    def model(self) -> LogisticRegression_:
        return self._model
