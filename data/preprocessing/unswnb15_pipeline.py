"""
data/preprocessing/unswnb15_pipeline.py
UNSW-NB15 dataset preprocessing pipeline.

Handles 49 features, 10 classes, severe class imbalance.
Reference: Guide §3.2 (secondary dataset)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# UNSW-NB15 attack categories (10-class including Normal)
ATTACK_CAT_MAP = {
    "Normal": 0, "Fuzzers": 1, "Analysis": 2, "Backdoors": 3,
    "DoS": 4, "Exploits": 5, "Generic": 6, "Reconnaissance": 7,
    "Shellcode": 8, "Worms": 9,
}
CLASS_NAMES = list(ATTACK_CAT_MAP.keys())

# Columns to drop (non-feature metadata)
DROP_COLS = ["id", "label"]  # 'label' is binary; we use 'attack_cat' for multi-class

CATEGORICAL_COLS = ["proto", "service", "state"]

INPUT_DIM = 49
NUM_CLASSES = 10


def load_unswnb15(train_path: str, test_path: str):
    """Load raw UNSW-NB15 CSV files."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"UNSW-NB15 training file not found: {train_path}\n"
            "Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        )
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def encode_categoricals(df: pd.DataFrame, encoders=None, fit: bool = True):
    """Label-encode categorical columns."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].fillna("unknown").astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )
    return df, encoders


def map_attack_cat(df: pd.DataFrame) -> pd.DataFrame:
    """Map attack_cat string to integer class ID."""
    df = df.copy()
    if "attack_cat" in df.columns:
        df["label"] = (
            df["attack_cat"].fillna("Normal").str.strip()
            .map(ATTACK_CAT_MAP).fillna(0).astype(int)
        )
    else:
        # Fallback: use binary label column (0=Normal,1=Attack→generic)
        df["label"] = df["label"].fillna(0).astype(int)
    return df


def build_pipeline(train_path: str, test_path: str, use_smote: bool = True,
                   seed: int = 42):
    """
    Full UNSW-NB15 preprocessing pipeline.

    Returns:
        X_train, y_train, X_test, y_test, scaler, encoders, class_weights
    """
    train_df, test_df = load_unswnb15(train_path, test_path)

    train_df = map_attack_cat(train_df)
    test_df = map_attack_cat(test_df)

    # Drop non-feature columns
    for col in ["id", "attack_cat"]:
        for df in [train_df, test_df]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Encode categoricals
    train_df, encoders = encode_categoricals(train_df, fit=True)
    test_df, _ = encode_categoricals(test_df, encoders=encoders, fit=False)

    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if use_smote and _SMOTE_AVAILABLE:
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        if min_count >= 2:
            k = min(3, min_count - 1)
            smote = SMOTE(random_state=seed, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.int64)

    weights = compute_class_weight("balanced",
                                    classes=np.unique(y_train), y=y_train)

    print(f"[UNSW-NB15] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler, encoders, weights
