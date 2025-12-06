import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def create_synthetic_flight_data(sample_size: int = 3000) -> pd.DataFrame:
    """
    Create a synthetic flight delay dataset with realistic-looking values.
    """
    np.random.seed(42)

    airlines = ["AA", "DL", "UA", "WN", "B6", "AS", "F9"]  # example airline codes
    hours = list(range(24))  # 0–23 hours in a day

    data = {
        "AIRLINE": np.random.choice(airlines, sample_size),
        "DEPARTURE_HOUR": np.random.choice(hours, sample_size),
        # Delay in minutes, centered around 20 mins with some spread
        "DELAY": np.abs(
            np.random.normal(loc=20, scale=30, size=sample_size).astype(int)
        ),
    }

    df = pd.DataFrame(data)
    return df


def build_delay_classifier(df: pd.DataFrame):
    """
    Train a simple classifier to predict whether a flight will be delayed
    more than 15 minutes based on departure hour.
    """
    # Target: delayed (1) or not delayed (0)
    df["DELAYED"] = (df["DELAY"] > 15).astype(int)

    X = df[["DEPARTURE_HOUR"]]
    y = df["DELAYED"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    return model, acc, report


def main():
    print("🔹 Creating synthetic flight delay dataset...")
    df = create_synthetic_flight_data(sample_size=3000)

    print("\nSample rows:")
    print(df.head())

    print("\nBasic statistics:")
    print(df.describe())

    print("\n🔹 Training delay classifier (delayed > 15 minutes)...")
    model, acc, report = build_delay_classifier(df)

    print(f"\nModel accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()
