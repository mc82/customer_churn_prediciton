from ctypes.wintypes import PFLOAT
from os import path
from classifier import RandomForest
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from plot import Histogram, Distplot, Heatmap, Barplot, ModelSummary, RocCurve

from costants import TO_BE_ENCODED_COLUMN_NAMES, X_COLUMNS


class ChurnPrediction():

    def __init__(self, model_dir) -> None:
        self._model_dir = model_dir

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

        self._fit_predict_random_forest()

        # scores
        print('random forest results')
        print('test results')
        print(self._y_train_predictions_rf)
        print(classification_report(self._y_test, self._y_test_predictions_rf))
        print('train results')
        print(classification_report(self._y_train, self._y_train_predictions_rf))

        roc_curve = RocCurve(figsize=(15, 8))
        roc_curve.create(estimator=self._random_forest_classifier.best_model,
                         X=self._X_test, y=self._y_test, plot_name="random_forest_roc.png")

        # save best model
        joblib.dump(self._random_forest_classifier.best_model,
                    './data/models/rfc_model.pkl')

        model_summary_random_forest = ModelSummary(figsize=(6, 6))
        model_summary_random_forest.create(y_train=self._y_train, y_train_pred=self._y_train_predictions_rf, y_test=self._y_test,
                                           y_test_pred=self._y_test_predictions_rf, model_name="random_forest", plot_file_name="model_summary_random_forest.png")

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

    def _fit_predict_random_forest(self):
        self._fit_random_forest()
        self._predict_random_forest()

    def _fit_random_forest(self):

        self._random_forest_classifier = RandomForest(self._model_dir)
        self._random_forest_classifier.fit(
            X_train=self._X_train,
            y_train=self._y_train
        )

    def _predict_random_forest(self):

        self._y_train_predictions_rf = self._random_forest_classifier.predict(
            X=self._X_train
        )

        self._y_test_predictions_rf = self._random_forest_classifier.predict(
            X=self._X_test
        )

    def _create_eda_plots(self):
        histogram = Histogram(figsize=(20, 10))
        histogram.create(
            self._df['Churn'],
            plot_file_name="hist_churn.png"
        )
        histogram.create(
            self._df['Customer_Age'],
            plot_file_name="hist_customer_age.png"
        )

        barplot = Barplot(figsize=(20, 10))
        barplot.create(
            self._df.Marital_Status.value_counts('normalize'),
            plot_file_name="marital_status.png"
        )

        distplot = Distplot(figsize=(20, 10))
        distplot.create(
            self._df['Total_Trans_Ct'],
            plot_file_name="dist_total_trans_ct.png"
        )

        heatmap = Heatmap(
            figsize=(20, 10),
            # annot=False,
            # cmap='Dark2_r',
            # linewidths=2
        )
        heatmap.create(
            self._df.corr(),
            plot_file_name="corr_plot.png"
        )
