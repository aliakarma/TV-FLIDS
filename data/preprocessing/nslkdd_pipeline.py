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
    counts = np.bincount(y)
    min_count = int(counts.min())
    if min_count < 2:
        print(f"[SMOTE] Skipped: minimum class count = {min_count} "
              f"(class {int(counts.argmin())}). Need >= 2 samples.")
        return X, y
    k = min(3, min_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[SMOTE] Applied k_neighbors={k}. "
          f"Samples: {len(X)} -> {len(X_res)}")
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_pipeline(train_path: str, test_path: str, use_smote: bool = True,
                   seed: int = 42, val_size: int = 2000, protocol: str = "main"):
    """
    Full NSL-KDD preprocessing pipeline.

    Two switchable data-preparation protocols (paper §IV-A, §VIII-A):

    - protocol="main" (default, matches the paper's main-table protocol,
      Table V etc.): SMOTE is applied to the *entire* training pool first;
      the `val_size`-sample stratified server validation set D_val is then
      drawn from that SMOTE-balanced pool. This is the leakage-permissive
      protocol the paper uses for its headline numbers.
    - protocol="leakage_free" (paper §VIII-A, Table VI): D_val is drawn
      *before* any SMOTE is applied, from the original (imbalanced) pool.
      SMOTE is deliberately NOT applied inside this function for that
      protocol — it must be applied per-client, after Dirichlet
      partitioning, by the caller (see experiments/run_experiment.py
      ::setup_data), so that no synthetic sample derived from a validation
      example ever leaks into client training data.

    Args:
        val_size: Absolute number of stratified validation samples (paper:
            2,000 — see Table IV). Previously this pipeline only exposed a
            `val_fraction` that produced ~6,299 samples (5% of the pool),
            which never matched the paper; that parameter has been removed.
        protocol: "main" or "leakage_free".

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoders, class_weights
    """
    if protocol not in ("main", "leakage_free"):
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'main' or 'leakage_free'.")

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

    from data.partitioning import split_server_validation_set

    if protocol == "main":
        # SMOTE first (on the full pool), THEN draw D_val from the balanced pool.
        scaler = MinMaxScaler()
        X_all_scaled = scaler.fit_transform(X_all).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        if use_smote:
            X_pool, y_pool = apply_smote(X_all_scaled, y_all, random_state=seed)
        else:
            X_pool, y_pool = X_all_scaled, y_all

        X_train_scaled, y_train, X_val, y_val = split_server_validation_set(
            X_pool, y_pool, val_size=val_size, seed=seed
        )

    else:  # protocol == "leakage_free"
        # Draw D_val BEFORE SMOTE, from the raw (imbalanced) pool. SMOTE is
        # deferred to the caller, to be applied per-client after partitioning.
        X_train_raw, y_train, X_val_raw, y_val = split_server_validation_set(
            X_all, y_all, val_size=val_size, seed=seed
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        if use_smote:
            print("[NSL-KDD] protocol=leakage_free: global SMOTE skipped here; "
                  "apply per-client after partitioning (see setup_data()).")

    y_train = y_train.astype(np.int64)
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    print(
        f"[NSL-KDD] protocol={protocol} | Train: {X_train_scaled.shape} | "
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
    (X_tr, y_tr, X_val, y_val,
     X_te, y_te, scaler, encoders, weights) = build_pipeline(train_path, test_path)
    print(f"[Pipeline] Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")
    print(f"[Classes]  Train labels: {dict(zip(*[v.tolist() for v in __import__('numpy').unique(y_tr, return_counts=True)]))}")

