"""Implements tests to register classifiers
__author__ = "Maik Goetze"
"""
import pytest
from churn_library import ChurnPredictionFactory
from classifier import RandomForest, LogisticRegression


@pytest.mark.factory
@pytest.mark.parametrize("classifier_name",
                         [("random_forest"), ("logistic_regression")])
def test_register_one_classifier(classifier_name: str):
    """Test whether registration of one classifier works

    Args:
        classifier_name (str): Name of the classifier to register
    """
    churn_prediction_factory = ChurnPredictionFactory()
    churn_prediction_factory.register_classifier(classifier_name)

    assert len(churn_prediction_factory.registered_classifiers) == 1
    assert churn_prediction_factory.registered_classifiers.pop() in [
        RandomForest, LogisticRegression]


@pytest.mark.factory
def test_register_both_classifiers():
    """Test to register both used classifiers
    """
    churn_prediction_factory = ChurnPredictionFactory()
    churn_prediction_factory.register_classifier("random_forest")
    churn_prediction_factory.register_classifier("logistic_regression")

    assert len(churn_prediction_factory.registered_classifiers) == 2
