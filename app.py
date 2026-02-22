import numpy as np
import tensorflow as tf


def simulate_realtime():

    model = tf.keras.models.load_model("network_model.h5")

    while True:

        # Fake live network feature vector
        sample = np.random.rand(1, 78, 1)

        prediction = model.predict(sample)

        if prediction > 0.5:
            print("⚠️  Attack Detected")
        else:
            print("✅ Normal Traffic")


if __name__ == "__main__":
    simulate_realtime()
