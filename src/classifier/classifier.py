"""
Implements abstract base class of classifier t
o provice a common interface all derived classifiers.
__author__ = "Maik Goetze"
"""
from abc import ABC
from typing import Any
import pickle
from os import path
import pandas as pd
import numpy as np

from costants import MODEL_EXTENSION


class Classifier(ABC):
    """
    Abstract class so serve a common interface for all implemented classifiers.
    """

    name = ""

    def __init__(self, model_dir: str) -> None:
        self._model: Any
        self._model_dir = model_dir
        self._model_path = self._create_model_path()
        self._classifier = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fits the classifier

        Args:
            X (pandas.DataFrame): depended variables
            y (pandas.DataFrame): independend variables
        """
        self._model.fit(X=X, y=y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def save(self) -> None:
        """
        Save the model as pickle file.
        """
        with open(self._model_path, "wb") as file:
            pickle.dump(self._model, file)

    def load(self) -> None:
        """
        Loads the model from pickle file.
        """
        print(f"Loading model from {self._model_path}")
        with open(self._model_path, "rb") as file:
            self._model = pickle.load(file)
        print("Model successfully loaded")

    def _create_model_path(self) -> str:
        """
        Returns the model path to load and save the model.
        Returns:
            str: path to load and save the model
        """
        return path.join(self._model_dir, self.name + MODEL_EXTENSION)

    def __str__(self) -> str:
        """
        Use name of the classifier for str operations
        Returns:
            str: name of the classifier
        """
        return self.name
