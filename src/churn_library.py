"""Implements all function of the churn library - a ML model to predict churn.
__author__ = "Maik Goetze"
__date__='2023-02-05'
"""

import pickle
import logging
from abc import ABC, abstractmethod
from os import path
from pathlib import Path
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegression_
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import (INPUT_DATA_PATH, MODEL_DIR, MODEL_EXTENSION, PLOT_DIR,
                       TO_BE_ENCODED_COLUMN_NAMES, X_COLUMNS, LOG_DIR)

sns.set()


class Directory:
    """Implements logic to create a directory if not exists and returns the path to it
    """

    def __init__(self, directory_name: str) -> None:
        """Initializes the Directory object
        Args:
            directory_name (str): name of the directory
        """
        self._directory_name = directory_name
        self._path = Path(self._directory_name)

    @property
    def directory(self) -> Path:
        """Provides the directory path

        :return: Path of the directory
        :rtype: Path
        """
        if not self._check_existence():
            self._create()
        return self._path

    def _check_existence(self) -> bool:
        """Checks exists of directory
        :return: returns True if directory exists otherwise False
        :rtype: bool
        """
        exists = self._path.is_dir()
        return exists

    def _create(self) -> None:
        """Creates directory
        """
        Path(self._path).mkdir(parents=True, exist_ok=True)


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
            y (pandas.DataFrame): independed variables
        """
        self._model.fit(X=X, y=y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict based on a fitted classifier

        Args:
            X (pandas.DataFrame): depended variables
        """
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


class LogisticRegression(Classifier):
    '''
    Performs Logistic Regression
    '''

    name = "logistic_regression"

    def __init__(self, model_dir: str) -> None:
        '''
        Initializes the base classifier
        Args:
            model_dir (str): Path of the model to load and save
        '''
        super().__init__(model_dir)
        self._model = LogisticRegression_()

    @property
    def model(self) -> LogisticRegression_:
        return self._model


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
            model_dir (str): path of the model to load and save
        """
        super().__init__(model_dir)
        self._classifier = RandomForestClassifier(random_state=42)
        self._model: GridSearchCV

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Performs a grid search to fit the model
        Args:
            X (pd.DataFrame): dependent variables to fit the model
            y (pd.DataFrame): independent variables to fit the model
        """
        self._run_grid_search(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts independent values based on dependent variables using best fitted model
        Args:
            X (pd.DataFrame): dependent variables

        Returns:
            np.ndarray: array with the predicted values
        """
        return self.model.predict(X)

    def _run_grid_search(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Runs grid search to find best parameter set to fit the model
        Args:
            X (pd.DataFrame): dependent variables
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


class Plot:
    """
    Implements Interface and some methods to create and save plots.
    """

    def __init__(self, plot_dir: Path, figsize=(20, 30)) -> None:
        """
        Inits the base class
        Args:
            plot_dir (str, optional): Directory where the plots are saved as a file.
                                      Defaults to "plots".
            figsize (Tuple, optional): Size of the plot passed as a tuple. Defaults to ....
        """
        self.__plot_dir = plot_dir
        self._figsize = figsize
        self._init_plot()

    @abstractmethod
    def create(self, data: Union[Series, DataFrame], plot_file_name: str):
        """
        Interface to create plots and save it on disk.
        """

    def save(self, figure, plot_name: str) -> None:
        """
        Saves given plot on disk using given plot name.
        Args:
            figure (_type_): The plot to save.
            plot_name (str): File name of the plot
        """
        plot_path = path.join(self.__plot_dir, plot_name)
        figure.figure.savefig(plot_path)

    def _init_plot(self) -> None:
        """
        Set the size of the plot
        """
        self._plt = plt.figure(figsize=self._figsize)

    def __del__(self) -> None:
        """
        Avoids plot of multiple figures into one plot
        """
        plt.close()


class Barplot(Plot):
    """
    Creates and saves bar plot based on given input.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates barplot based on input data and save it on disk.
        Args:
            barplot_data (Series): Contains the data to plot
            plot_file_name (str): Name of the file
        """
        figure = data.plot(kind='bar')
        self.save(figure=figure, plot_name=plot_file_name)


class Distplot(Plot):
    """
    Provides method the create a dist plot using a common interface.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        """Initializes the Distplot object
        Args:
            self (Heatmap): Distplot object
            plot_dir (str): directory to save plot
            figsize (Tuple): size of the plot
        """
        super().__init__(plot_dir, figsize)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates a dist plot based on given data and saves it on disk.
        Args:
            data (Series): _description_
            plot_file_name (str): _description_
        """
        figure = sns.distplot(data)
        self.save(figure=figure, plot_name=plot_file_name)


class FeatureImportancePlot(Plot):
    """
    Provides method the create plots of feature importance.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        super().__init__(plot_dir, figsize)

    def create(
            self,
            data: Series,
            feature_names: List[str],
            plot_file_name: str):
        """
        Creates plot of feature importance and saves it on disk.

        Args:
            data (Series): _description_
            plot_name (str): _description_
        """
        # Sort feature importances in descending order
        indices = np.argsort(data)[::-1]

        # Rearrange feature feature_names so they match the sorted feature
        # importances
        feature_names = [feature_names[i] for i in indices]

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(len(feature_names)), data[indices])

        # Add feature feature_names as x-axis labels
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=90)
        self.save(figure=self._plt, plot_name=plot_file_name)


class Heatmap(Plot):
    """
    Implements method to create heat map plots using a common interace for plots.
    """

    def __init__(self, **kwargs):
        """Initializes the Heatmap object
        Args:
            self (Heatmap): Histogram object
        """
        super().__init__(**kwargs)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates heat map plot based on input data and saves in on disk.
        Args:
            data (Series): Contains the data to plot as heat map
            plot_file_name (str): The file name of the plot
        """
        figure = sns.heatmap(data)
        self.save(figure=figure, plot_name=plot_file_name)


class Histogram(Plot):
    """
    Implements method to create and save hist plots leveraging a common interface.
    """

    def __init__(self, **kwargs):
        """Initializes the Histogram object
        Args:
            self (ShapPlot): Histogram object
            plot_dir (str): directory to save plot
            figsize (Tuple): size of the plot
        """
        super().__init__(**kwargs)

    def create(self, data: Series, plot_file_name: str) -> None:
        """
        Creates histogram plot and saves it on disk.
        Args:
            data (Series): Data to plot as histogram
            plot_file_name (str): File name of the created plot
        """
        figure = data.hist()
        self.save(figure=figure, plot_name=plot_file_name)


class ModelSummary(Plot):
    """
    Implements method to create plots about model
    performance leveraging a common interace.
    """

    def __init__(self, plot_dir="plots", figsize=...) -> None:
        """Initializes the ModelSummary object
        Args:
            self (ShapPlot): ModelSummary object
            plot_dir (str): directory to save plot
            figsize (Tuple): size of the plot
        """
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


class RocCurve(Plot):
    """
    Implements method to create and save roc curves.
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        """Initializes the RocCurve object
        Args:
            self (ShapPlot): RocCurve object
            plot_dir (str): directory to save plot
            figsize (Tuple): size of the plot
        """
        super().__init__(plot_dir, figsize)

    def create(
            self,
            estimator,
            X: DataFrame,
            y: Series,
            plot_file_name: str) -> None:
        """Creates roc curve based on estimator and dependent variables

        Args:
            estimator (_type_): _description_
            X (DataFrame): Dependent variables
            y (Series): Independent variable
            plot_file_name (str): File name of the plot on disk.
        """
        figure = self._plt.gca()
        roc_plot = plot_roc_curve(estimator, X, y, ax=figure, alpha=0.8)
        roc_plot.plot(ax=figure, alpha=0.8)
        self.save(figure=figure, plot_name=plot_file_name)


class ShapPlot(Plot):
    """
    Implements method to create shap plot.
    Args:
        Plot (Plot): Base class with common interface
    """

    def __init__(self, plot_dir, figsize=...) -> None:
        """Initializes the ShapPlot object
        Args:
            self (ShapPlot): ShapPlot object
            plot_dir (str): directory to save plot
            figsize (arry): size of the plot
        """
        super().__init__(plot_dir, figsize)

    def create(self, estimator, X: DataFrame, plot_file_name: str):
        """Creates shap plot
        Args:
            estimator (Estimator)): trained tree estimator
            X (DataFrame): Dependent variables to train to model.
            plot_file_name (str): File name of the shap plot.
        """
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        self.save(figure=self._plt, plot_name=plot_file_name)


class ChurnPrediction():

    def __init__(self, model_dir: Path, plot_dir: Path, classifier) -> None:
        """Initializes the ChurnPrediction object
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        self._model_dir = model_dir
        self._classifier = classifier(model_dir=self._model_dir)
        self._plot_file_name_template = str(
            self._classifier) + "_{plot_name}.png"
        self._plot_dir = plot_dir

    def run(self):
        """Runs the churn prediction workflow
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """

        self._load_data_frame()

        self._print_data_overview()

        self._append_churn_column()

        self._create_eda_plots()

        logger.info("Encoding some columns...")
        for column_name in TO_BE_ENCODED_COLUMN_NAMES:
            logger.info("..encoding {}".format(column_name))
            self._encode_column(column_name)
        logger.info("Encoding some columns is done.")

        self._set_X()
        self._print_X_overview()

        self._set_y()

        self._perform_train_test_split()

        self._fit_predict()

        classification_report_test = classification_report(
            self._y_test, self._y_test_predictions)
        logger.info("Test classification report: {}".format(
            classification_report_test))

        classification_report_train = classification_report(
            self._y_train, self._y_train_predictions)
        logger.info("Train classification report: {}".format(
            classification_report_train))

        logger.info("Creating ROC curve...")
        roc_curve = RocCurve(figsize=(15, 8), plot_dir=self._plot_dir)
        roc_curve.create(
            estimator=self._classifier.model,
            X=self._X_test,
            y=self._y_test,
            plot_file_name=f"{self._classifier}_roc.png")

        # model_path = f'./data/models/{self._classifier}_model.pkl'
        logger.info("Saving model to ...")
        self._classifier.save()
        # joblib.dump(self._classifier.model, model_path)
        logger.info("Saving of model done")

        logger.info("Creating model summary ...")
        model_summary = ModelSummary(figsize=(6, 6), plot_dir=self._plot_dir)
        model_summary.create(
            y_train=self._y_train,
            y_train_pred=self._y_train_predictions,
            y_test=self._y_test,
            y_test_pred=self._y_test_predictions,
            model_name=str(
                self._classifier),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="model_summary"))
        logger.info("Creating model summary done")
        logger.info("Classification done.")

    def _load_data_frame(self):
        """Creates DataFrame form file
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Loading data from {}".format(INPUT_DATA_PATH))
        self._df = pd.read_csv(INPUT_DATA_PATH)
        # self._df = self._df.sample(frac=0.1)  # TODO remove sampling
        logger.info("Loading of data has been finished.")

    def _print_data_overview(self):
        """Logs some key metrics of the data
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Describing loaded data:")
        logger.info("Head of loaded data:")
        logger.info(self._df.head())
        logger.info("Shape of loaded data:")
        logger.info(self._df.shape)
        logger.info("Number of None in loaded data:")
        logger.info(self._df.isnull().sum())
        logger.info("Basic statistics of loaded data:")
        logger.info(self._df.describe())

    def _append_churn_column(self):
        """Encode a passed column of a DataFrame
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Setting values of churn column...")
        self._df['Churn'] = self._df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        logger.info("Setting values of churn column done.")

    def _encode_column(self, column_name: str):
        """Encode a passed column of a DataFrame
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
            column_name (str): Name of the column to encode
        """
        encoded = []
        groups = self._df.groupby(column_name).mean()['Churn']

        for val in self._df[column_name]:
            encoded.append(groups.loc[val])

        self._df[f'{column_name}_Churn'] = encoded

    def _set_X(self):
        """Defines the dependent values
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Creating new DataFrame with all X columns...")
        self._X = pd.DataFrame()
        self._X[X_COLUMNS] = self._df[X_COLUMNS]
        logger.info("Creating new DataFrame with all X columns is done.")

    def _set_y(self):
        """Defines the target values
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Setting churn column as y.")
        self._y = self._df['Churn']

    def _print_X_overview(self):
        """Prints sample of X data
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Head of X data:")
        logger.info(self._X.head())

    def _perform_train_test_split(self):
        """Performs train test split an a given data set
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Splitting into train and test data...")
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=0.3, random_state=42)
        logger.info("Splitting into train and test data finished.")

    def _fit_predict(self):
        """Performs fitting of the estimator and predicts afterwards
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        self._fit()
        self._predict()

    def _fit(self):
        """Performs fitting of the estimator
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Fitting classifier...")
        self._classifier.fit(
            X=self._X_train,
            y=self._y_train
        )
        logger.info("Fitting classifier done.")

    def _predict(self):
        """Performs prediction based on fitted estimator
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Predicting...")
        logger.info("Predicting training data..")
        self._y_train_predictions = self._classifier.predict(
            X=self._X_train
        )
        logger.info("Prediction of training data done.")

        logger.info("Predicting test data..")
        self._y_test_predictions = self._classifier.predict(
            X=self._X_test
        )
        logger.info("Prediction of test data done.")

    def _create_eda_plots(self):
        """Creates all eda plots
        Args:
            self (ChurnPrediction): ChurnPredictionFactory object
        """
        logger.info("Creating EDA plots...")
        logger.info("Creating histogram of churn column...")
        histogram = Histogram(figsize=(20, 10), plot_dir=self._plot_dir)
        histogram.create(
            self._df['Churn'],
            plot_file_name=self._plot_file_name_template.format(
                plot_name="hist_churn")
        )
        histogram.create(
            self._df['Customer_Age'],
            plot_file_name=self._plot_file_name_template.format(
                plot_name="hist_customer_age")
        )
        logger.info("Creating histogram of churn column done")

        logger.info("Creating barplot of marital_status column...")
        barplot = Barplot(figsize=(20, 10), plot_dir=self._plot_dir)
        barplot.create(
            self._df.Marital_Status.value_counts('normalize'),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="bar_marital_status")
        )
        logger.info("Creating barplot of marital_status column done.")

        logger.info("Creating distplot of Total_Trans_Ct column...")
        distplot = Distplot(figsize=(20, 10), plot_dir=self._plot_dir)
        distplot.create(
            self._df['Total_Trans_Ct'],
            plot_file_name=self._plot_file_name_template.format(
                plot_name="dist_total_trans_ct")
        )
        logger.info("Creating distplot of Total_Trans_Ct column done.")

        logger.info("Creating heatmap of of correlated columns...")
        heatmap = Heatmap(
            figsize=(20, 10),
            plot_dir=self._plot_dir
            # annot=False,
            # cmap='Dark2_r',
            # linewidths=2
        )
        heatmap.create(
            self._df.corr(),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="corr")
        )
        logger.info("Creating heatmap of of correlated columns done.")


class ChurnPredictionFactory:
    """Provided interface to run churn prediction with multiple classifiers
    """

    available_classifier = {
        "random_forest": RandomForest,
        "logistic_regression": LogisticRegression
    }

    def __init__(self):
        """Initialise ChurnPredictionFactory object
        Args:
            self (ChurnPredictionFactory): ChurnPredictionFactory object
        """
        self._registered_classifiers: List = []

    def run(self) -> None:
        """Runs the churn prediction with all subscribed classifiers
        """
        logger.info("Running churn prediction...")
        for classifier in self.registered_classifiers:
            logger.info("Running classifier {} ...".format(str(classifier)))
            churn_prediction = ChurnPrediction(
                model_dir=Directory(MODEL_DIR).directory,
                plot_dir=Directory(PLOT_DIR).directory,
                classifier=classifier
            )
            churn_prediction.run()
            logger.info(
                "Churn prediction with classifier {} has been finished.".format(
                    str(classifier)))
        logger.info("Churn prediction has been finished.")

    def register_classifier(self, name: str) -> None:
        """Register multiple classifiers

        Args:
            name (str): nome of the classifier to register
        """
        self._registered_classifiers.append(self.available_classifier[name])

    @property
    def registered_classifiers(self) -> List[str]:
        """Returns list of subscribed classifiers

        Returns:
            List[str]: subscribed classifiers
        """        #
        return self._registered_classifiers


def get_logger(log_level, log_file_name: str) -> logging.Logger:
    """Creates logger object with specif config
    Args:
        log_level (Any): level to log
        log_file_name (str): name of the log file
    Returns:
        logging.Logger: configures Logger object
    """        #

    logger = logging.getLogger("churn_prediction_logger")
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(
        path.join(Directory(LOG_DIR).directory, log_file_name)
    )
    file_handler.setLevel(log_level)

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


logger = get_logger(log_level=logging.DEBUG, log_file_name='churn_library.log')

if __name__ == '__main__':
    churn_prediction_factory = ChurnPredictionFactory()
    churn_prediction_factory.register_classifier("random_forest")
    churn_prediction_factory.register_classifier("logistic_regression")
    churn_prediction_factory.run()
