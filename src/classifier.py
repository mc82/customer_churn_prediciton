from abc import abstractmethod
import pickle
from os import path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from costants import MODEL_DIR, MODEL_FILE_NAME, MODEL_EXTENSION


class Classifier():

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

    def predict(self, X):
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
