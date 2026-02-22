import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model


def reshape_for_lstm(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))


def compare():

    print("Loading dataset...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "data", "combine.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "network_model.keras")

    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

    # ================= Logistic Regression =================
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # ================= Random Forest =================
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Seaborn confusion matrix ONLY for Random Forest
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ================= CNN-LSTM =================
    print("\nLoading CNN-LSTM model...")
    model = load_model(MODEL_PATH)

    X_test_cnn = reshape_for_lstm(X_test)

    print("Evaluating CNN-LSTM...")
    y_pred_cnn = model.predict(X_test_cnn)

    threshold = 0.3
    y_pred_cnn = (y_pred_cnn > threshold).astype(int).flatten()

    print("\nCNN-LSTM Accuracy:", accuracy_score(y_test, y_pred_cnn))
    print("\nCNN-LSTM Classification Report:")
    print(classification_report(y_test, y_pred_cnn))


if __name__ == "__main__":
    compare()