import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(path):
    print("Loading dataset...")

    df = pd.read_csv(path, low_memory=False)

    # Remove extra spaces in column names
    df.columns = df.columns.str.strip()

    print("Initial shape:", df.shape)

    # Drop missing values
    df = df.dropna()

    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()

    print("After cleaning:", df.shape)

    # ----- LABEL HANDLING -----
    y = df["Label"]

    # Convert to binary
    y = y.apply(lambda x: 0 if x == "BENIGN" else 1)

    print("Unique labels after conversion:", y.unique())
    print("Label distribution:")
    print(y.value_counts())

    # Remove label column from features
    X = df.drop(columns=["Label"])

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    print("Feature shape:", X.shape)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("Preprocessing completed.")

    return X_train, X_test, y_train, y_test