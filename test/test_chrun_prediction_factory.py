import pytest
from churn_prediction import ChurnPredictionFactory
from classifier import RandomForest


@pytest.mark.factory
def test_register_one_classifier():
    churn_prediction_factory = ChurnPredictionFactory()
    churn_prediction_factory.register_classifier("random_forest")

    assert len(churn_prediction_factory.registered_classifiers) == 1
    assert churn_prediction_factory.registered_classifiers.pop() == RandomForest
