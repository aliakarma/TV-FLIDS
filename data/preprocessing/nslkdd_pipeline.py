"""
data/preprocessing/nslkdd_pipeline.py
NSL-KDD dataset preprocessing pipeline.

Implements the exact IEEE paper protocol (§VI-A, Table I, §VI-B):
- KDDTrain+ (125,973 records) and KDDTest+ (22,544 records) benchmark split.
- Server validation set D_val (2,000 records) with exact minority class quotas:
    Normal: 1,016, DoS: 693, Probe: 176, R2L: 100, U2R: 15 (Total = 2,000).
- Server tuning set D_tune (12,398 records):
    Stratified 10% draw of remainder after D_val:
    Normal: 6,633, DoS: 4,523, Probe: 1,148, R2L: 90, U2R: 4 (Total = 12,398).
- Client training pool D_client (111,575 records):
    Normal: 59,694, DoS: 40,711, Probe: 10,332, R2L: 805, U2R: 33 (Total = 111,575).
- Test set D_test (22,544 records):
    Normal: 9,711, DoS: 7,458, Probe: 2,421, R2L: 2,754, U2R: 200 (Total = 22,544).
    Contains 3,750 novel attack records (17 attack types absent from KDDTrain+).
- Normalization:
    MinMaxScaler(clip=False) fitted EXCLUSIVELY on server-held data (D_val U D_tune).
    Client training data and test data are NEVER used for scaler fitting (zero leakage).
    Unclipped scaling keeps test values outside fitted range per paper §VI-A.
- SMOTE:
    Client-local SMOTE applied per client after partitioning.
    Target: median non-empty class count M.
    If n_c >= 2: SMOTE with k = min(5, n_c - 1).
    If n_c == 1: duplication to M.
    If n_c == 0: remain 0.
    D_val, D_tune, and D_test are NEVER oversampled.
"""

import os
import urllib.request
from typing import Any, Dict, List, Tuple

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
FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("label", "difficulty")]

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

# Exact quotas from IEEE paper Table I
PAPER_VAL_QUOTAS = {0: 1016, 1: 693, 2: 176, 3: 100, 4: 15}
PAPER_TUNE_QUOTAS = {0: 6633, 1: 4523, 2: 1148, 3: 90, 4: 4}
PAPER_CLIENT_QUOTAS = {0: 59694, 1: 40711, 2: 10332, 3: 805, 4: 33}
PAPER_TEST_QUOTAS = {0: 9711, 1: 7458, 2: 2421, 3: 2754, 4: 200}

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"


# ── Download & Load ──────────────────────────────────────────────────────────

def download_nslkdd(train_path: str, test_path: str) -> None:
    """Download NSL-KDD raw files if not present."""
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    for url, path in [(TRAIN_URL, train_path), (TEST_URL, test_path)]:
        if not os.path.exists(path):
            print(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"[Cache] {path} already exists, skipping download.")


def load_nslkdd(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw NSL-KDD CSV files and drop the non-feature difficulty column."""
    train = pd.read_csv(train_path, names=COLUMNS)
    test = pd.read_csv(test_path, names=COLUMNS)
    if "difficulty" in train.columns:
        train.drop(columns=["difficulty"], inplace=True)
    if "difficulty" in test.columns:
        test.drop(columns=["difficulty"], inplace=True)
    return train, test


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw string labels to integer class IDs (0..4)."""
    df = df.copy()
    df["label"] = df["label"].str.lower().str.strip()
    df["label"] = df["label"].map(ATTACK_MAP).fillna(-1).astype(int)
    df = df[df["label"] >= 0].reset_index(drop=True)
    return df


# ── Exact Paper Quota Computation ───────────────────────────────────────────

def compute_nslkdd_quotas(
    counts: Dict[int, int],
    val_size: int = 2000,
    tune_fraction: float = 0.1,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Compute exact sample quotas for D_val, D_tune, and D_client per paper §VI-A:
    - Classes with < 5,000 records receive max(round(val_size * n / total), 100),
      capped at 30% of available class records.
    - The remaining large classes share the remaining validation quota in proportion
      using largest remainder rounding.
    - D_tune is a stratified 10% of what remains after D_val.
    - The rest forms the client pool D_client.
    """
    total = sum(counts.values())
    val = {}
    for c, n in counts.items():
        if n < 5000:
            prop = round(val_size * n / total)
            val[c] = min(max(prop, 100), int(0.3 * n))

    rest = val_size - sum(val.values())
    big = {c: n for c, n in counts.items() if c not in val}
    big_total = sum(big.values())

    raw = {c: rest * n / big_total for c, n in big.items()}
    floors = {c: int(v) for c, v in raw.items()}
    left = rest - sum(floors.values())
    for c in sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)[:left]:
        floors[c] += 1
    val.update(floors)

    pool = {c: counts[c] - val[c] for c in counts}
    tune = {c: max(1, round(tune_fraction * pool[c])) for c in pool}
    clients = {c: pool[c] - tune[c] for c in pool}
    return val, tune, clients


def extract_nslkdd_splits(
    df_train: pd.DataFrame,
    seed: int = 42,
    val_size: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract exact paper splits D_val (2,000), D_tune (12,398), and D_client (111,575)
    from KDDTrain+ dataframe enforcing Table I quotas and pairwise disjointness.
    """
    if "label" not in df_train.columns:
        raise ValueError("df_train must contain 'label' column.")

    counts = df_train["label"].value_counts().to_dict()
    total_records = sum(counts.values())
    if total_records < val_size:
        raise ValueError(
            f"Cannot extract validation set of size {val_size}: dataset only contains {total_records} records."
        )

    if val_size == 2000:
        val_quotas = PAPER_VAL_QUOTAS.copy()
        tune_quotas = PAPER_TUNE_QUOTAS.copy()
    else:
        val_quotas, tune_quotas, _ = compute_nslkdd_quotas(counts, val_size=val_size)

    rng = np.random.RandomState(seed)
    val_indices: List[int] = []
    tune_indices: List[int] = []
    client_indices: List[int] = []

    for c in range(5):
        c_count = counts.get(c, 0)
        c_indices = df_train.index[df_train["label"] == c].to_numpy().copy()
        v_need = val_quotas.get(c, 0)
        t_need = tune_quotas.get(c, 0)
        total_need = v_need + t_need

        if len(c_indices) < v_need:
            raise ValueError(
                f"Cannot satisfy validation quota for class {c} ({CLASS_NAMES[c]}): "
                f"required {v_need}, but only {len(c_indices)} available."
            )
        if len(c_indices) < total_need:
            raise ValueError(
                f"Cannot satisfy server quota for class {c} ({CLASS_NAMES[c]}): "
                f"required {total_need} (val={v_need}, tune={t_need}), "
                f"but only {len(c_indices)} available."
            )

        rng.shuffle(c_indices)
        val_indices.extend(c_indices[:v_need])
        tune_indices.extend(c_indices[v_need:total_need])
        client_indices.extend(c_indices[total_need:])

    # Pairwise disjointness verification
    set_v = set(val_indices)
    set_t = set(tune_indices)
    set_c = set(client_indices)
    assert len(set_v & set_t) == 0, "D_val and D_tune overlap!"
    assert len(set_v & set_c) == 0, "D_val and D_client overlap!"
    assert len(set_t & set_c) == 0, "D_tune and D_client overlap!"

    val_df = df_train.loc[val_indices].reset_index(drop=True)
    tune_df = df_train.loc[tune_indices].reset_index(drop=True)
    client_df = df_train.loc[client_indices].reset_index(drop=True)

    return val_df, tune_df, client_df


# ── Zero-Leakage Encoding and Scaling ────────────────────────────────────────

def encode_and_scale_splits(
    val_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    client_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    MinMaxScaler,
    Dict[str, LabelEncoder],
]:
    """
    Encode categorical features and min-max scale numeric features.

    CRITICAL SCIENTIFIC REQUIREMENT (Paper §VI-A):
    - Scaler statistics are fitted EXCLUSIVELY on server-held data: D_val U D_tune.
    - Scaler is NEVER fitted on client training data (D_client) or test data (D_test).
    - MinMaxScaler(clip=False) is used so out-of-range test values are preserved.
    - Categorical encoders are fitted on server-held data with unobserved categories mapped to 0.
    """
    val_df = val_df.copy()
    tune_df = tune_df.copy()
    client_df = client_df.copy()
    test_df = test_df.copy()

    # Fit encoders on server data (val + tune)
    server_df = pd.concat([val_df, tune_df], axis=0, ignore_index=True)
    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        if col in server_df.columns:
            le = LabelEncoder()
            le.fit(server_df[col].astype(str))
            encoders[col] = le
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

            server_df[col] = server_df[col].astype(str).map(mapping).fillna(0).astype(int)
            val_df[col] = val_df[col].astype(str).map(mapping).fillna(0).astype(int)
            tune_df[col] = tune_df[col].astype(str).map(mapping).fillna(0).astype(int)
            client_df[col] = client_df[col].astype(str).map(mapping).fillna(0).astype(int)
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(str).map(mapping).fillna(0).astype(int)

    feature_cols = [c for c in server_df.columns if c != "label"]
    X_server = server_df[feature_cols].values.astype(np.float32)

    # Fit scaler strictly on server data (D_val U D_tune)
    scaler = MinMaxScaler(clip=False)
    scaler.fit(X_server)

    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32)).astype(np.float32)
    y_val = val_df["label"].values.astype(np.int64)

    X_tune = scaler.transform(tune_df[feature_cols].values.astype(np.float32)).astype(np.float32)
    y_tune = tune_df["label"].values.astype(np.int64)

    X_client = scaler.transform(client_df[feature_cols].values.astype(np.float32)).astype(np.float32)
    y_client = client_df["label"].values.astype(np.int64)

    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32)).astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    return (
        X_val, y_val,
        X_tune, y_tune,
        X_client, y_client,
        X_test, y_test,
        scaler, encoders,
    )


# ── SMOTE Protocol ───────────────────────────────────────────────────────────

def apply_client_local_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply client-local SMOTE according to paper §VI-A:
    "Each client oversamples locally with SMOTE [chawla2002smote]: every class
    below the client's median non-empty class count is raised to that median,
    with k=min(5, n_c-1) neighbors when the class has n_c >= 2 records,
    duplication when n_c=1, and nothing when n_c=0."
    """
    if len(y) == 0:
        return X, y

    classes = np.unique(y)
    counts = {c: int((y == c).sum()) for c in classes}
    non_empty = [cnt for cnt in counts.values() if cnt > 0]
    if not non_empty:
        return X, y

    M = int(round(float(np.median(non_empty))))

    X_cur = X.copy()
    y_cur = y.copy()

    for c in classes:
        n_c = int((y_cur == c).sum())
        if 0 < n_c < M:
            if n_c == 1:
                idx = np.where(y_cur == c)[0]
                rep = M - 1
                X_rep = np.tile(X_cur[idx], (rep, 1))
                y_rep = np.full(rep, c, dtype=y_cur.dtype)
                X_cur = np.vstack([X_cur, X_rep])
                y_cur = np.concatenate([y_cur, y_rep])
            else:
                if not _SMOTE_AVAILABLE:
                    idx = np.where(y_cur == c)[0]
                    rep = M - n_c
                    choices = np.random.RandomState(random_state).choice(idx, size=rep, replace=True)
                    X_cur = np.vstack([X_cur, X_cur[choices]])
                    y_cur = np.concatenate([y_cur, y_cur[choices]])
                else:
                    k = min(5, n_c - 1)
                    smote = SMOTE(sampling_strategy={c: M}, k_neighbors=k, random_state=random_state)
                    X_cur, y_cur = smote.fit_resample(X_cur, y_cur)

    return X_cur.astype(np.float32), y_cur.astype(np.int64)


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE. Conforms to the paper-defined client-local balancing rule.
    """
    return apply_client_local_smote(X, y, random_state=random_state)


# ── Explicit Splits & Main Pipeline ──────────────────────────────────────────

def get_nslkdd_splits(
    train_path: str,
    test_path: str,
    seed: int = 42,
    val_size: int = 2000,
) -> Dict[str, Any]:
    """
    Load raw NSL-KDD files and produce explicit paper splits:
    D_val (2,000), D_tune (12,398), D_client (111,575), D_test (22,544).
    """
    train_df, test_df = load_nslkdd(train_path, test_path)
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    val_df, tune_df, client_df = extract_nslkdd_splits(train_df, seed=seed, val_size=val_size)
    (
        X_val, y_val,
        X_tune, y_tune,
        X_client, y_client,
        X_test, y_test,
        scaler, encoders,
    ) = encode_and_scale_splits(val_df, tune_df, client_df, test_df)

    weights = compute_class_weight("balanced", classes=np.unique(y_client), y=y_client)

    return {
        "D_val": (X_val, y_val),
        "D_tune": (X_tune, y_tune),
        "D_client": (X_client, y_client),
        "D_test": (X_test, y_test),
        "scaler": scaler,
        "encoders": encoders,
        "class_weights": weights,
    }


def build_pipeline(
    train_path: str,
    test_path: str,
    use_smote: bool = True,
    seed: int = 42,
    val_size: int = 2000,
    protocol: str = "main",
    return_tune: bool = False,
):
    """
    Full NSL-KDD preprocessing pipeline conforming to IEEE paper §VI-A and Table I.

    Protocols:
    - protocol="main": Paper standard. D_val (2,000) and D_tune (12,398) are extracted
      first from KDDTrain+ using exact Table I quotas. Feature scaler is fitted strictly
      on D_val U D_tune. Unclipped scaling preserves test distribution. SMOTE is applied
      locally per-client after Dirichlet partitioning (via apply_client_local_smote).
    - protocol="leakage_free": Synonym for the paper's zero-leakage protocol.

    Args:
        train_path: Path to KDDTrain+.txt
        test_path: Path to KDDTest+.txt
        use_smote: Legacy parameter. SMOTE is never applied globally across the client pool;
                   it is deferred to client-local training per paper §VI-A.
        seed: Random seed for deterministic subset partitioning.
        val_size: Number of validation samples (paper: 2,000).
        protocol: "main" or "leakage_free".
        return_tune: If True, also returns X_tune, y_tune.

    Returns:
        If return_tune is False (default for backward compatibility):
            (X_client, y_client, X_val, y_val, X_test, y_test, scaler, encoders, class_weights)
        If return_tune is True:
            (X_client, y_client, X_val, y_val, X_tune, y_tune, X_test, y_test, scaler, encoders, class_weights)
    """
    if protocol not in ("main", "leakage_free"):
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'main' or 'leakage_free'.")

    splits = get_nslkdd_splits(train_path, test_path, seed=seed, val_size=val_size)

    X_val, y_val = splits["D_val"]
    X_tune, y_tune = splits["D_tune"]
    X_client, y_client = splits["D_client"]
    X_test, y_test = splits["D_test"]
    scaler = splits["scaler"]
    encoders = splits["encoders"]
    weights = splits["class_weights"]

    print(
        f"[NSL-KDD] protocol={protocol} | Client pool: {X_client.shape} | "
        f"Val: {X_val.shape} | Tune: {X_tune.shape} | Test: {X_test.shape}"
    )

    if return_tune:
        return (
            X_client, y_client,
            X_val, y_val,
            X_tune, y_tune,
            X_test, y_test,
            scaler, encoders, weights,
        )

    return (
        X_client, y_client,
        X_val, y_val,
        X_test, y_test,
        scaler, encoders, weights,
    )


if __name__ == "__main__":
    train_file = "data/raw/KDDTrain+.txt"
    test_file = "data/raw/KDDTest+.txt"
    download_nslkdd(train_file, test_file)
    (X_tr, y_tr, X_v, y_v, X_te, y_te, scl, encs, w) = build_pipeline(train_file, test_file)
    print(f"[Pipeline Check] Client pool: {X_tr.shape} | Val: {X_v.shape} | Test: {X_te.shape}")
    val_counts = dict(zip(*np.unique(y_v, return_counts=True)))
    print(f"[Validation Quotas Realized]: {val_counts}")
