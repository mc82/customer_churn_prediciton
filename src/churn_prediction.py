from typing import List

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier import LogisticRegression, RandomForest
from costants import MODEL_DIR, TO_BE_ENCODED_COLUMN_NAMES, X_COLUMNS, INPUT_DATA_PATH
from logger import logging
from plot import Barplot, Distplot, Heatmap, Histogram, ModelSummary, RocCurve


class ChurnPrediction():

    def __init__(self, model_dir: str, classifier) -> None:
        self._model_dir = model_dir
        self._classifier = classifier(self._model_dir)
        self._plot_file_name_template = str(
            self._classifier) + "_{plot_name}.png"

    def run(self):

        self._load_data_frame()

        self._print_data_overview()

        self._append_churn_column()

        self._create_eda_plots()

        logging.info("Encoding some columns...")
        for column_name in TO_BE_ENCODED_COLUMN_NAMES:
            logging.info("..encoding {}".format(column_name))
            self._encode_column(column_name)
        logging.info("Encoding some columns is done.")

        self._set_X()
        self._print_X_overview()

        self._set_y()

        self._perform_train_test_split()

        self._fit_predict()

        classification_report_test = classification_report(self._y_test, self._y_test_predictions)
        logging.info("Test classification report: {}".format(classification_report_test))
        
        classification_report_train = classification_report(self._y_train, self._y_train_predictions)
        logging.info("Train classification report: {}".format(classification_report_train))
        
        logging.info("Creating ROC curve...")
        roc_curve = RocCurve(figsize=(15, 8))
        roc_curve.create(
            estimator=self._classifier.model,
            X=self._X_test,
            y=self._y_test,
            plot_file_name=f"{self._classifier}_roc.png")

        model_path = f'./data/models/{self._classifier}_model.pkl'
        logging.info("Saving model to {} ...".format(model_path))
        joblib.dump(self._classifier.model, model_path)
        logging.info("Saving of model done")


        logging.info("Creating model summary ...")
        model_summary = ModelSummary(figsize=(6, 6))
        model_summary.create(
            y_train=self._y_train,
            y_train_pred=self._y_train_predictions,
            y_test=self._y_test,
            y_test_pred=self._y_test_predictions,
            model_name=str(
                self._classifier),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="model_summary"))
        logging.info("Creating model summary done")
        logging.info("Classification done.")

    def _load_data_frame(self):
        logging.info("Loading data from {}".format(INPUT_DATA_PATH))
        self._df = pd.read_csv(INPUT_DATA_PATH)
        self._df = self._df.sample(frac=0.1)
        logging.info("Loading of data has been finished.")

    def _print_data_overview(self):
        logging.info("Describing loaded data:")
        logging.info("Head of loaded data:")
        logging.info(self._df.head())
        logging.info("Shape of loaded data:")
        logging.info(self._df.shape)
        logging.info("Number of None in loaded data:")
        logging.info(self._df.isnull().sum())
        logging.info("Basic statistics of loaded data:")
        logging.info(self._df.describe())

    def _append_churn_column(self):
        logging.info("Setting values of churn column...")
        self._df['Churn'] = self._df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        logging.info("Setting values of churn column done.")

    def _encode_column(self, column_name):
        encoded = []
        groups = self._df.groupby(column_name).mean()['Churn']

        for val in self._df[column_name]:
            encoded.append(groups.loc[val])

        self._df[f'{column_name}_Churn'] = encoded

    def _set_X(self):
        logging.info("Creating new DataFrame with all X columns...")
        self._X = pd.DataFrame()
        self._X[X_COLUMNS] = self._df[X_COLUMNS]
        logging.info("Creating new DataFrame with all X columns is done.")

    def _set_y(self):
        logging.info("Setting churn column as y.")
        self._y = self._df['Churn']

    def _print_X_overview(self):
        logging.info("Head of X data:")
        logging.info(self._X.head())

    def _perform_train_test_split(self):
        logging.info("Splitting into train and test data...")
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=0.3, random_state=42)
        logging.info("Splitting into train and test data finished.")

    def _fit_predict(self):
        self._fit()
        self._predict()

    def _fit(self):
        logging.info("Fitting classifier...")
        self._classifier.fit(
            X=self._X_train,
            y=self._y_train
        )
        logging.info("Fitting classifier done.")

    def _predict(self):
        logging.info("Predicting...")
        logging.info("Predicting training data..")
        self._y_train_predictions = self._classifier.predict(
            X=self._X_train
        )
        logging.info("Prediction of training data done.")

        logging.info("Predicting test data..")
        self._y_test_predictions = self._classifier.predict(
            X=self._X_test
        )
        logging.info("Prediction of test data done.")

    def _create_eda_plots(self):
        logging.info("Creating EDA plots...")
        logging.info("Creating histogram of churn column...")
        histogram = Histogram(figsize=(20, 10))
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
        logging.info("Creating histogram of churn column done")

        logging.info("Creating barplot of marital_status column...")
        barplot = Barplot(figsize=(20, 10))
        barplot.create(
            self._df.Marital_Status.value_counts('normalize'),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="bar_marital_status")
        )
        logging.info("Creating barplot of marital_status column done.")

        logging.info("Creating distplot of Total_Trans_Ct column...")
        distplot = Distplot(figsize=(20, 10))
        distplot.create(
            self._df['Total_Trans_Ct'],
            plot_file_name=self._plot_file_name_template.format(
                plot_name="dist_total_trans_ct")
        )
        logging.info("Creating distplot of Total_Trans_Ct column done.")

        logging.info("Creating heatmap of of correlated columns...")
        heatmap = Heatmap(
            figsize=(20, 10),
            # annot=False,
            # cmap='Dark2_r',
            # linewidths=2
        )
        heatmap.create(
            self._df.corr(),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="corr")
        )
        logging.info("Creating heatmap of of correlated columns done.")


class ChurnPredictionFactory:
    """Provided interface to run churn prediction with multiple classifiers
    """

    available_classifier = {
        "random_forest": RandomForest,
        "logistic_regression": LogisticRegression
    }

    def __init__(self):
        self._registered_classifiers: List = []

    def run(self) -> None:
        """Runs the churn prediction with all subscribed classifiers
        """
        logging.info("Running churn prediction...")
        for classifier in self.registered_classifiers:
            logging.info("Running classifier {} ...".format(str(classifier)))
            churn_prediction = ChurnPrediction(
                model_dir=MODEL_DIR,
                classifier=classifier
            )
            churn_prediction.run()
            logging.info("Churn prediction with classifier {} has been finished.".format(
                str(classifier)))
        logging.info("Churn prediction has been finished.")

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
