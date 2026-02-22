import streamlit as st
import numpy as np
import time
import os
import pandas as pd
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess

st.set_page_config(page_title="AI Intrusion Detection", layout="wide")
st.title("🔐 Real-Time AI Intrusion Detection System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "network_model.keras")
DATA_PATH = os.path.join(BASE_DIR, "data", "combine.csv")

model = load_model(MODEL_PATH)
st.success("Model Loaded Successfully")

if st.button("Start Real-Time Simulation"):

    df = pd.read_csv(DATA_PATH, nrows=200)
    df.columns = df.columns.str.strip()

    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])

    X_sample = X.iloc[:20].values

    attack_count = 0
    benign_count = 0
    results = []

    for i in range(20):

        sample = X_sample[i].reshape(1, X_sample.shape[1], 1)
        prediction = model.predict(sample, verbose=0)[0][0]

        if prediction > 0.5:
            attack_count += 1
            results.append(1)
        else:
            benign_count += 1
            results.append(0)

        st.write(f"Packet {i+1} Prediction: {prediction:.4f}")

        st.metric("Total Packets", i+1)
        st.metric("Attacks Detected", attack_count)
        st.metric("Benign Traffic", benign_count)

        st.line_chart(results)
        time.sleep(0.3)

    st.success("Simulation Completed")