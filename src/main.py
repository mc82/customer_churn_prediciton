from churn_prediction import ChurnPrediction
from costants import MODEL_DIR

if __name__ == '__main__':
    churn_prediction = ChurnPrediction(model_dir=MODEL_DIR)
    churn_prediction.run()
