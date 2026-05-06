"""
data/preprocessing/nslkdd_pipeline.py
NSL-KDD dataset preprocessing pipeline.

Downloads, encodes, normalizes, and SMOTE-balances NSL-KDD for FL simulation.
Reference: Guide §3.3
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# ── Column Definitions ──────────────────────────────────────────────────────

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# 5-class mapping: Normal(0), DoS(1), Probe(2), R2L(3), U2R(4)
ATTACK_MAP = {
    "normal": 0,
    # DoS
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "mailbomb": 1, "apache2": 1, "processtable": 1,
    "udpstorm": 1,
    # Probe
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2,
    "mscan": 2, "saint": 2,
    # R2L
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3,
    "phf": 3, "spy": 3, "warezclient": 3, "warezmaster": 3,
    "sendmail": 3, "named": 3, "snmpgetattack": 3, "snmpguess": 3,
    "xlock": 3, "xsnoop": 3, "worm": 3,
    # U2R
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "httptunnel": 4, "ps": 4, "sqlattack": 4, "xterm": 4,
}

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"


# ── Download ─────────────────────────────────────────────────────────────────

def download_nslkdd(train_path: str, test_path: str) -> None:
    """Download NSL-KDD raw files if not present."""
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    for url, path in [(TRAIN_URL, train_path), (TEST_URL, test_path)]:
        if not os.path.exists(path):
            print(f"Downloading {url} → {path}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"[Cache] {path} already exists, skipping download.")


# ── Load ─────────────────────────────────────────────────────────────────────

def load_nslkdd(train_path: str, test_path: str):
    """Load raw NSL-KDD CSV files."""
    train = pd.read_csv(train_path, names=COLUMNS)
    test = pd.read_csv(test_path, names=COLUMNS)
    train.drop(columns=["difficulty"], inplace=True)
    test.drop(columns=["difficulty"], inplace=True)
    return train, test


# ── Encode ───────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, encoders=None, fit: bool = True):
    """Label-encode categorical columns. Fit on train, transform on test."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )
    return df, encoders


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw string labels to integer class IDs."""
    df = df.copy()
    df["label"] = df["label"].str.lower().str.strip()
    df["label"] = df["label"].map(ATTACK_MAP).fillna(-1).astype(int)
    df = df[df["label"] >= 0].reset_index(drop=True)
    return df


# ── Normalize ─────────────────────────────────────────────────────────────────

def normalize_features(X_train: np.ndarray, X_test: np.ndarray):
    """Min-max normalize. Fit ONLY on training data to prevent leakage."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


# ── SMOTE ────────────────────────────────────────────────────────────────────

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE to handle severe class imbalance (U2R in NSL-KDD)."""
    if not _SMOTE_AVAILABLE:
        print("[Warning] imbalanced-learn not installed. Skipping SMOTE.")
        return X, y
    smote = SMOTE(random_state=random_state, k_neighbors=min(3, min(np.bincount(y)) - 1))
    X_res, y_res = smote.fit_resample(X, y)
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_pipeline(train_path: str, test_path: str, use_smote: bool = True,
                   seed: int = 42, val_fraction: float = 0.05):
    """
    Full NSL-KDD preprocessing pipeline.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoders, class_weights
    """
    train_df, test_df = load_nslkdd(train_path, test_path)

    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    train_df, encoders = encode_categoricals(train_df, fit=True)
    test_df, _ = encode_categoricals(test_df, encoders=encoders, fit=False)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_all = train_df[feature_cols].values.astype(np.float32)
    y_all = train_df["label"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    from sklearn.model_selection import train_test_split
    X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
        X_all, y_all, test_size=val_fraction, stratify=y_all, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    if use_smote:
        X_train_scaled, y_train_raw = apply_smote(
            X_train_scaled, y_train_raw, random_state=seed
        )

    y_train = y_train_raw.astype(np.int64)
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    print(
        f"[NSL-KDD] Train: {X_train_scaled.shape} | "
        f"Val: {X_val.shape} | Test: {X_test_scaled.shape}"
    )

    return (
        X_train_scaled, y_train, X_val, y_val.astype(np.int64),
        X_test_scaled, y_test, scaler, encoders, weights
    )


if __name__ == "__main__":
    import sys
    train_path = "data/raw/KDDTrain+.txt"
    test_path = "data/raw/KDDTest+.txt"
    download_nslkdd(train_path, test_path)
    X_tr, y_tr, X_te, y_te, scaler, encoders, weights = build_pipeline(
        train_path, test_path
    )
    print(f"Pipeline complete. Train: {X_tr.shape}, Test: {X_te.shape}")
