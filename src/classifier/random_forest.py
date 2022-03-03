from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

from .classifier import Classifier


class RandomForest(Classifier):

    name = "random_forest"

    _param_grid = param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self._classifier = RandomForestClassifier(random_state=42)
        self._model = None

    def fit(self, X_train, y_train):
        self._run_grid_search(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self.best_model.predict(X)

    def _run_grid_search(self, X_train, y_train):
        self._model = GridSearchCV(
            estimator=self._classifier, param_grid=self._param_grid, cv=5)
        self._model.fit(X_train, y_train)

    @property
    def best_model(self):
        return self._model.best_estimator_

    def __str__(self) -> str:
        return super().__str__()
