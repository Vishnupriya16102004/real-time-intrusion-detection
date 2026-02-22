import numpy as np
import time
import tensorflow as tf
from preprocess import load_and_preprocess

def reshape_for_lstm(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

print("Loading trained model...")
model = tf.keras.models.load_model("network_model.keras")

# Load dataset again (only for simulation source)
X_train, X_test, y_train, y_test = load_and_preprocess("data/cicids.csv/combine.csv")

X_test = reshape_for_lstm(X_test)

print("\n🚀 Starting Real-Time Traffic Simulation...\n")

for i in range(20):  # simulate 20 packets
    sample = X_test[i].reshape(1, X_test.shape[1], 1)

    prediction = model.predict(sample, verbose=0)
    probability = prediction[0][0]

    if probability > 0.5:
        print(f"Packet {i+1} → ATTACK ⚠ (Confidence: {probability:.4f})")
    else:
        print(f"Packet {i+1} → BENIGN (Confidence: {probability:.4f})")

    time.sleep(1)