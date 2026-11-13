"""Step One of the original script: data loading and preprocessing."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["query_encoded", "feedback_normalized"]


def load_interactions(csv_path):
    """Load the raw interactions CSV and apply the original cleaning rules
    (drop duplicates, drop rows whose training label is missing)."""
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()
    data = data.dropna(subset=["threat_resolved"])
    return data.reset_index(drop=True)


def preprocess(data):
    """Add the two model features exactly as the original script did:
    integer-factorized query and standardized feedback.

    Returns (data, scaler, query_to_code)."""
    data = data.copy()
    codes, uniques = pd.factorize(data["query"])
    data["query_encoded"] = codes
    query_to_code = {query: int(code) for code, query in enumerate(uniques)}

    scaler = StandardScaler()
    data["feedback_normalized"] = scaler.fit_transform(data[["feedback"]])
    return data, scaler, query_to_code


def split(data, test_size=0.2, random_state=42):
    """Train/test split, same parameters as the original script."""
    return train_test_split(data, test_size=test_size, random_state=random_state)
