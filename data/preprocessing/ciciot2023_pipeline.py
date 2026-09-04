"""
data/preprocessing/ciciot2023_pipeline.py
CIC-IoT-2023 dataset preprocessing pipeline (secondary/cross-dataset).

Reference: Neto, E.C.P. et al. (2023), "CICIoT2023: A Real-Time Dataset and
Benchmark for Large-Scale Attacks in IoT Environment", Sensors 23(13):5941
(citation key `neto2023` in the paper's bibliography). Used in the paper's
Supplementary §S-II-B / Table S4 / Figure S1 cross-dataset generalization
study.

Dataset shape (public description): CIC-IoT-2023 is a network-flow-statistics
dataset (NOT raw packet payloads) collected from a 105-device real IoT
testbed. The official release ships as ~169 per-attack-scenario CSV shards,
each with the same 46 numeric flow-feature columns plus a `label` column
holding a FINE-GRAINED attack name (34 distinct values across 7 attack
categories + Benign, e.g. "DDoS-ICMP_Flood", "DoS-TCP_Flood",
"Recon-PortScan", "MITM-ArpSpoofing", "DNS_Spoofing", "Mirai-greeting_flood",
"DictionaryBruteForce", "BenignTraffic", ...).

This repo's convention (matching nslkdd_pipeline.py / unswnb15_pipeline.py)
is two already-split CSV files: `train_file` and `test_file`. A real
deployment of this pipeline must first concatenate the official per-scenario
shards and split them into exactly those two files (the same manual step the
NSL-KDD/UNSW-NB15 download conventions already assume) -- this module does
NOT download or concatenate shards itself (no network calls, by design of
this remediation pass).

Feature columns: the 46-column flow-statistics feature set below is the
column list documented across the dataset's public releases/mirrors (packet
counts/rates, TCP flag counts, protocol one-hot indicators, IAT/size
statistics, and higher-order flow statistics such as Magnitue/Radius/
Covariance/Variance/Weight -- "Magnitue" is the official dataset's own
spelling, kept verbatim here so column names match the real CSVs). Because
different mirrors of this dataset occasionally differ in exact column
name/casing/spacing, `load_ciciot2023()` strips whitespace from column names
on load and raises a clear, actionable error naming any missing columns --
reconcile FEATURE_COLUMNS below against the specific CSV release you
download if that happens.

Taxonomy mapping: the official release exposes both the 34-class fine-grained
`label` and a coarser 7-category-plus-Benign grouping. This pipeline uses the
coarser 8-class scheme (7 attack categories + Benign), for consistency with
how this repo treats NSL-KDD's 5-class (Normal/DoS/Probe/R2L/U2R) and
UNSW-NB15's 10-class (`attack_cat`) taxonomies -- i.e. broad attack-family
classification rather than the full fine-grained subtype set:

    {"Benign": 0, "DDoS": 1, "DoS": 2, "Recon": 3, "WebBased": 4,
     "BruteForce": 5, "Spoofing": 6, "Mirai": 7}

`map_labels()` categorizes each row by matching (in priority order, so that
e.g. "ddos" is checked before the substring-overlapping "dos") normalized
keyword fragments against the fine-grained label string. This is robust to
both the fine-grained official label strings AND to CSVs that already carry
the coarse category name directly in `label` (a keyword equal to the whole
normalized label matches the same way a substring match would).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# ── Column Definitions ──────────────────────────────────────────────────────

# 46 numeric flow-statistics features (see module docstring). All columns are
# already numeric in the official release (including the protocol/service
# indicator columns, which are 0/1 flags or numeric codes, not strings) --
# unlike NSL-KDD/UNSW-NB15, no categorical label-encoding step is required.
FEATURE_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

LABEL_COL = "label"

INPUT_DIM = len(FEATURE_COLUMNS)  # 46
NUM_CLASSES = 8

# 8-class taxonomy: Benign + 7 CIC-IoT-2023 attack categories (see docstring).
CATEGORY_TO_ID = {
    "Benign": 0, "DDoS": 1, "DoS": 2, "Recon": 3,
    "WebBased": 4, "BruteForce": 5, "Spoofing": 6, "Mirai": 7,
}
CLASS_NAMES = ["Benign", "DDoS", "DoS", "Recon", "WebBased",
               "BruteForce", "Spoofing", "Mirai"]

# Keyword -> category, checked in this order against a normalized
# (lowercased, whitespace/hyphen/underscore-stripped) label string. Order
# matters: "ddos" must be checked before "dos" since "dos" is a literal
# substring of "ddos".
_CATEGORY_KEYWORD_PRIORITY = [
    ("benign", "Benign"),
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("recon", "Recon"),
    ("mirai", "Mirai"),
    ("mitm", "Spoofing"),
    ("arpspoof", "Spoofing"),
    ("dnsspoof", "Spoofing"),
    ("spoofing", "Spoofing"),
    ("bruteforce", "BruteForce"),
    ("dictionarybrute", "BruteForce"),
    ("sqlinjection", "WebBased"),
    ("commandinjection", "WebBased"),
    ("backdoormalware", "WebBased"),
    ("uploadingattack", "WebBased"),
    ("xss", "WebBased"),
    ("browserhijacking", "WebBased"),
    ("webbased", "WebBased"),
]


# ── Load ─────────────────────────────────────────────────────────────────────

def load_ciciot2023(train_path: str, test_path: str):
    """Load raw CIC-IoT-2023 train/test CSV files.

    Expects the repo's two-file convention (pre-split by the user from the
    official per-scenario shards -- see module docstring). Column names are
    whitespace-stripped on load to tolerate minor formatting differences
    across dataset mirrors.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"CIC-IoT-2023 training file not found: {train_path}\n"
            "Download the official per-scenario CSV shards from "
            "https://www.unb.ca/cic/datasets/iotdataset-2023.html, "
            "concatenate them, and split into train/test CSVs at the paths "
            "configured in config/dataset_config.yaml (datasets.ciciot2023)."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"CIC-IoT-2023 test file not found: {test_path}")

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    train.columns = [str(c).strip() for c in train.columns]
    test.columns = [str(c).strip() for c in test.columns]

    required = FEATURE_COLUMNS + [LABEL_COL]
    for name, df in (("train", train), ("test", test)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CIC-IoT-2023 {name} file is missing expected columns: {missing}. "
                "The column list in FEATURE_COLUMNS (data/preprocessing/"
                "ciciot2023_pipeline.py) reflects the publicly documented "
                "CIC-IoT-2023 schema, but mirrors of this dataset sometimes "
                "differ slightly in naming/casing -- reconcile FEATURE_COLUMNS "
                "against your actual downloaded CSV header."
            )
    return train, test


# ── Label mapping ────────────────────────────────────────────────────────────

def _normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    for ch in (" ", "-", "_"):
        s = s.replace(ch, "")
    return s


def _categorize(normalized_label: str) -> int:
    for keyword, category in _CATEGORY_KEYWORD_PRIORITY:
        if keyword in normalized_label:
            return CATEGORY_TO_ID[category]
    return -1


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw (fine-grained or already-coarse) label strings to the 8-class
    taxonomy defined by CATEGORY_TO_ID. Rows whose label matches no known
    category keyword are dropped (mirrors nslkdd_pipeline.map_labels'
    unknown-label handling)."""
    df = df.copy()
    normalized = df[LABEL_COL].astype(str).apply(_normalize_label)
    df[LABEL_COL] = normalized.apply(_categorize).astype(int)
    n_before = len(df)
    df = df[df[LABEL_COL] >= 0].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[CIC-IoT-2023] Dropped {n_dropped} row(s) with unrecognized labels.")
    return df


# ── Normalize ────────────────────────────────────────────────────────────────

def normalize_features(X_train: np.ndarray, X_test: np.ndarray):
    """Min-max normalize. Fit ONLY on training data to prevent leakage."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


# ── SMOTE ────────────────────────────────────────────────────────────────────

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE to handle class imbalance (CIC-IoT-2023's rarer categories
    e.g. WebBased/BruteForce are heavily under-represented relative to
    Benign/DDoS -- the same shape of imbalance NSL-KDD/UNSW-NB15 pipelines
    already handle this way)."""
    if not _SMOTE_AVAILABLE:
        print("[Warning] imbalanced-learn not installed. Skipping SMOTE.")
        return X, y
    counts = np.bincount(y)
    min_count = int(counts[counts > 0].min())
    if min_count < 2:
        print(f"[SMOTE] Skipped: minimum non-zero class count = {min_count}. "
              "Need >= 2 samples.")
        return X, y
    k = min(3, min_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[SMOTE] Applied k_neighbors={k}. Samples: {len(X)} -> {len(X_res)}")
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def build_pipeline(train_path: str, test_path: str, use_smote: bool = True,
                    seed: int = 42, val_size: int = 2000, protocol: str = "main"):
    """
    Full CIC-IoT-2023 preprocessing pipeline. Mirrors
    nslkdd_pipeline.build_pipeline's signature/shape and two-protocol
    design (see that module's docstring for full protocol semantics) so
    both pipelines plug into experiments/run_experiment.py::setup_data
    without special-casing.

    - protocol="main" (default; what experiments/run_experiment.py's
      `dataset == "ciciot2023"` branch calls): SMOTE is applied to the
      entire training pool first; the `val_size`-sample stratified server
      validation set D_val is then drawn from that SMOTE-balanced pool.
    - protocol="leakage_free": D_val is drawn *before* SMOTE, from the raw
      (imbalanced) pool; SMOTE is deliberately NOT applied here for that
      protocol -- callers apply it per-client after partitioning.

    Args:
        train_path, test_path: Paths to the pre-split CIC-IoT-2023 CSV files
            (see load_ciciot2023 / module docstring).
        use_smote: Whether to SMOTE-balance the training pool (protocol="main")
            or emit a deferral notice (protocol="leakage_free").
        seed: Random seed for the val split and SMOTE.
        val_size: Absolute number of stratified validation samples.
        protocol: "main" or "leakage_free".

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoders, class_weights
        (encoders is an empty dict: CIC-IoT-2023's 46 feature columns are all
        already numeric, so no categorical LabelEncoders are needed -- this
        slot is kept only so the 9-tuple shape matches the other pipelines.)
    """
    if protocol not in ("main", "leakage_free"):
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'main' or 'leakage_free'.")

    train_df, test_df = load_ciciot2023(train_path, test_path)

    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    encoders = {}  # no categorical columns in CIC-IoT-2023's feature set

    X_all = train_df[FEATURE_COLUMNS].astype(np.float32).values
    y_all = train_df[LABEL_COL].values.astype(np.int64)
    X_test = test_df[FEATURE_COLUMNS].astype(np.float32).values
    y_test = test_df[LABEL_COL].values.astype(np.int64)

    # Guard against NaN/inf in flow-rate columns (Rate/Srate/Drate can be
    # inf when a flow's duration is ~0), matching common practice for this
    # dataset: clip to finite range before scaling.
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=np.finfo(np.float32).max,
                           neginf=np.finfo(np.float32).min)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=np.finfo(np.float32).max,
                            neginf=np.finfo(np.float32).min)

    from data.partitioning import split_server_validation_set

    if protocol == "main":
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
        X_train_raw, y_train, X_val_raw, y_val = split_server_validation_set(
            X_all, y_all, val_size=val_size, seed=seed
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        if use_smote:
            print("[CIC-IoT-2023] protocol=leakage_free: global SMOTE skipped here; "
                  "apply per-client after partitioning (see setup_data()).")

    y_train = y_train.astype(np.int64)
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    print(
        f"[CIC-IoT-2023] protocol={protocol} | Train: {X_train_scaled.shape} | "
        f"Val: {X_val.shape} | Test: {X_test_scaled.shape}"
    )

    return (
        X_train_scaled, y_train, X_val, y_val.astype(np.int64),
        X_test_scaled, y_test, scaler, encoders, weights
    )


if __name__ == "__main__":
    train_path = "data/raw/CICIoT2023_train.csv"
    test_path = "data/raw/CICIoT2023_test.csv"
    (X_tr, y_tr, X_val, y_val,
     X_te, y_te, scaler, encoders, weights) = build_pipeline(train_path, test_path)
    print(f"[Pipeline] Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")
