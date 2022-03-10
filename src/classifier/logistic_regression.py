from sklearn.linear_model import LogisticRegression as LogisticRegression_

from .classifier import Classifier


class LogisticRegression(Classifier):

    def __init__(self, model_path) -> None:
        super().__init__(model_path)

    def fit(self, X, y):
        self._model = LogisticRegression_()
        self._model.fit(X=X, y=y)
