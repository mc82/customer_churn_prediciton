from ctypes.wintypes import PFLOAT
from os import path

from matplotlib.style import available
from classifier import RandomForest
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from typing import List

from plot import Histogram, Distplot, Heatmap, Barplot, ModelSummary, RocCurve

from costants import TO_BE_ENCODED_COLUMN_NAMES, X_COLUMNS, MODEL_DIR


class ChurnPrediction():

    def __init__(self, model_dir, classifier) -> None:
        self._model_dir = model_dir
        self._classifier = classifier(self._model_dir)
        self._plot_file_name_template = str(
            self._classifier) + "_{plot_name}.png"

    def run(self):
        self._load_data_frame()

        self._print_data_overview()

        self._append_churn_column()

        self._create_eda_plots()

        for column_name in TO_BE_ENCODED_COLUMN_NAMES:
            self._encode_column(column_name)

        self._set_X()
        self._print_X_overview()

        self._set_y()

        self._perform_train_test_split()

        self._fit_predict()

        # scores
        print(f'{self._classifier} results')
        print('test results')
        print(self._y_train_predictions)
        print(classification_report(self._y_test, self._y_test_predictions))
        print('train results')
        print(classification_report(self._y_train, self._y_train_predictions))

        roc_curve = RocCurve(figsize=(15, 8))
        roc_curve.create(estimator=self._classifier.best_model,
                         X=self._X_test, y=self._y_test, plot_name=f"{self._classifier}_roc.png")

        # save best model
        joblib.dump(self._classifier.best_model,
                    f'./data/models/{self._classifier}_model.pkl')

        model_summary = ModelSummary(figsize=(6, 6))
        model_summary.create(y_train=self._y_train,
                             y_train_pred=self._y_train_predictions,
                             y_test=self._y_test,
                             y_test_pred=self._y_test_predictions,
                             model_name=str(self._classifier),
                             plot_file_name=self._plot_file_name_template.format(
                                 plot_name="model_summary")
                             )

    def _load_data_frame(self):
        self._df = pd.read_csv(r"./data/bank_data.csv")
        self._df = self._df.sample(frac=0.1)

    def _print_data_overview(self):
        print(self._df.head())
        print(self._df.shape)
        print(self._df.isnull().sum())
        print(self._df.describe())

    def _append_churn_column(self):
        self._df['Churn'] = self._df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    def _encode_column(self, column_name):
        encoded = []
        groups = self._df.groupby(column_name).mean()['Churn']

        for val in self._df[column_name]:
            encoded.append(groups.loc[val])

        self._df[f'{column_name}_Churn'] = encoded

    def _set_X(self):
        self._X = pd.DataFrame()
        self._X[X_COLUMNS] = self._df[X_COLUMNS]

    def _set_y(self):
        self._y = self._df['Churn']

    def _print_X_overview(self):
        self._X.head()

    def _perform_train_test_split(self):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X,
            self._y,
            test_size=0.3,
            random_state=42
        )

    def _fit_predict(self):
        self._fit()
        self._predict()

    def _fit(self):
        self._classifier.fit(
            X_train=self._X_train,
            y_train=self._y_train
        )

    def _predict(self):
        self._y_train_predictions = self._classifier.predict(
            X=self._X_train
        )

        self._y_test_predictions = self._classifier.predict(
            X=self._X_test
        )

    def _create_eda_plots(self):
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

        barplot = Barplot(figsize=(20, 10))
        barplot.create(
            self._df.Marital_Status.value_counts('normalize'),
            plot_file_name=self._plot_file_name_template.format(
                plot_name="bar_marital_status")
        )

        distplot = Distplot(figsize=(20, 10))
        distplot.create(
            self._df['Total_Trans_Ct'],
            plot_file_name=self._plot_file_name_template.format(
                plot_name="dist_total_trans_ct")
        )

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


class ChurnPredictionFactory():

    available_classifier = {"random_forest": RandomForest}

    def __init__(self):
        self._registered_classifiers: List = []

    def register_classifier(self, name: str):
        self._registered_classifiers.append(self.available_classifier[name])

    @property
    def registered_classifiers(self):
        return self._registered_classifiers

    def run(self):
        for classifier in self.registered_classifiers:
            churn_prediction = ChurnPrediction(
                model_dir=MODEL_DIR,
                classifier=classifier
            )
            churn_prediction.run()
