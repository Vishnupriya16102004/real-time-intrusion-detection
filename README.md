🔐 Real-Time Intrusion Detection using Hybrid CNN–LSTM
Overview

This project implements a packet-level intrusion detection system using both traditional machine learning and hybrid deep learning architectures. The system classifies network traffic into benign and attack categories and includes a real-time simulation dashboard.

Models Evaluated

Logistic Regression (92% accuracy).
Random Forest (99% accuracy).
CNN–LSTM (98% accuracy with strong attack recall).
Hybrid CNN–LSTM Architecture.
1D Convolutional Layer (feature extraction).
LSTM Layer (temporal modeling).
Dense layers for classification.
Binary cross-entropy loss.
Adam optimizer.
Evaluation Metrics.
Accuracy.
Precision.
Recall.
F1-Score.
Confusion Matrix.

Special focus was placed on minimizing false negatives in attack detection.

Real-Time Dashboard
Implemented using Streamlit:

1-Simulated live packet traffic.
2-Dynamic classification display.
3-Confidence scoring.
4-Attack detection counters.

Tech Stack

*Python
*TensorFlow
*Scikit-Learn
*Streamlit
*Pandas
*NumPy

Future Improvements

1-Transformer-based intrusion detection.
2-Distributed detection systems.
3-Edge deployment.
