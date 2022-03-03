from abc import abstractmethod
import pickle
from os import path


from abc import ABC
from costants import MODEL_DIR, MODEL_FILE_NAME, MODEL_EXTENSION


class Classifier(ABC):

    name = ""

    def __init__(self, model_path) -> None:
        self._model = None
        self._model_path = model_path
        self._set_model_path()
        self._classifier = None

    @abstractmethod
    def fit(self):
        pass

    def save(self):
        with open(self._model_path, "wb") as f:
            print(f)
            pickle.dump(self._model, f)

    def load(self):
        print(f"Loading model from {self._model_path}")
        with open(self._model_path, "rb") as f:
            self._model = pickle.load(f)
        print("Model successfully loaded")

    def _set_model_path(self):
        self._model_path = path.join(
            MODEL_DIR, MODEL_FILE_NAME + MODEL_EXTENSION)
        print(self._model_path)

    def __str__(self) -> str:
        return self.name
