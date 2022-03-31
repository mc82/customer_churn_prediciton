"""
Implements RandomForest classifier
__author__ = "Maik Goetze"

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from .classifier import Classifier


class RandomForest(Classifier):
    """
        Random Forest classifier using GridSearchCV
    """

    name = "random_forest"

    _param_grid = param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    def __init__(self, model_dir: str) -> None:
        """
        Initialize the Random Forest classifier

        Args:
            model_path (str): path of the model to load and save
        """
        super().__init__(model_dir)
        self._classifier = RandomForestClassifier(random_state=42)
        self._model: GridSearchCV

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Performrs a grid search to fit the model
        Args:
            X (pd.DataFrame): dependend variables to fit the model
            y (pd.DataFrame): independent variables to fit the model
        """
        self._run_grid_search(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts independent values based on dependend variables using best fitted model
        Args:
            X (pd.DataFrame): dependend variales

        Returns:
            np.ndarray: array with the predicted values
        """
        return self.model.predict(X)

    def _run_grid_search(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Runs grid search to find best parameter set to fit the model
        Args:
            X (pd.DataFrame): dependend variables
            y (pd.DataFrame): independent variables
        """
        self._model = GridSearchCV(
            estimator=self._classifier, param_grid=self._param_grid, cv=5)
        self._model.fit(X, y)

    @property
    def model(self) -> RandomForestClassifier:
        """
        Returns:
            RandomForestClassifier: best model found be grid search
        """
        return self._model.best_estimator_
