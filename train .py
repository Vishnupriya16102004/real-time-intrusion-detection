import os
import numpy as np
from preprocess import load_and_preprocess
from model import build_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def reshape_for_lstm(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))


def train():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "data", "combine.csv")

    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

    X_train = reshape_for_lstm(X_train)
    X_test = reshape_for_lstm(X_test)

    model = build_model((X_train.shape[1], 1))

    model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=32,
        validation_split=0.2
    )
    model.save("network_model.keras")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_acc)

    # Predictions
    y_pred = model.predict(X_test)

    threshold = 0.3   # change this value later

    y_pred = (y_pred > threshold).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    


    


if __name__ == "__main__":
    train()