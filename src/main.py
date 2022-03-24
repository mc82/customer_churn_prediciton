"""Implements the script to run the full process
"""

from churn_prediction import ChurnPredictionFactory

if __name__ == '__main__':
    churn_prediction_factory = ChurnPredictionFactory()
    churn_prediction_factory.register_classifier("random_forest")
    churn_prediction_factory.register_classifier("logistic_regression")
    churn_prediction_factory.run()
