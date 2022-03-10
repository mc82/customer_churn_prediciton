"""
Implements abstract base class of classifier t
o provice a common interface all derived classifiers.
"""
from abc import ABC
from abc import abstractmethod
import pickle
from os import path
import pandas as pd

from costants import MODEL_DIR, MODEL_FILE_NAME, MODEL_EXTENSION


class Classifier(ABC):
    """
    Abstract class so serve a common interface for all implemented classifiers.
    """

    name = ""

    def __init__(self, model_path: str) -> None:
        self._model = None
        self._model_path = model_path
        self._set_model_path()
        self._classifier = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        abstract method to fit the classifier

        Args:
            X (pandas.DataFrame): depended variables
            y (pandas.DataFrame): independend variables
        """

    def save(self) -> None:
        """
        Save the model as pickle file.
        """
        with open(self._model_path, "wb") as file:
            print(file)
            pickle.dump(self._model, file)

    def load(self) -> None:
        """
        Loads the model from pickle file.
        """
        print(f"Loading model from {self._model_path}")
        with open(self._model_path, "rb") as file:
            self._model = pickle.load(file)
        print("Model successfully loaded")

    def _set_model_path(self) -> None:
        """
        Sets to model path to load and save the model.
        """
        self._model_path = path.join(
            MODEL_DIR, MODEL_FILE_NAME + MODEL_EXTENSION)
        print(self._model_path)

    def __str__(self) -> str:
        """
        Use name of the classifier for str operations
        Returns:
            str: name of the classifier
        """
        return self.name
