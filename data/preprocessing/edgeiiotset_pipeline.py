"""
data/preprocessing/edgeiiotset_pipeline.py
Edge-IIoTset dataset preprocessing pipeline.

Implements the exact IEEE TIFS paper protocol (§VI-A, §VI-B, and Supplementary §S5):
- Feature dimension: d = 61 numeric flow/protocol features.
- Model parameter count closed-form:
    P(61, 6) = 256(61) + 65(6) + 42,304 = 15,616 + 390 + 42,304 = 58,310 parameters.
- Attack taxonomy: 6 classes (Normal: 0, DoS/DDoS: 1, Injection: 2, Scanning: 3, Malware: 4, MITM: 5).
- Fine-grained to 6-class mapping:
    Normal -> 0
    DDoS_UDP, DDoS_ICMP, DDoS_HTTP, DDoS_TCP, DoS_UDP, DoS_ICMP, DoS_HTTP, DoS_TCP -> 1 (DoS/DDoS)
    SQL_injection, XSS, Uploading -> 2 (Injection)
    Port_Scanning, Vulnerability_scanner, Fingerprinting -> 3 (Scanning)
    Backdoor, Password, Ransomware -> 4 (Malware)
    MITM -> 5 (MITM, rarest class per paper §VI-A and Supplementary §S5)
- Time-disjoint split: Each source capture or per-attack/per-sensor file is ordered by timestamp.
  First 80% to training side, last 20% to test side, records within 60 seconds of boundary dropped.
- Per-class caps (Table S5):
    Train: Normal 45k, DoS/DDoS 40k, Injection 25k, Scanning 20k, Malware 15k, MITM 5k (Total 150k).
    Test: Normal 9k, DoS/DDoS 8k, Injection 5k, Scanning 4k, Malware 3k, MITM 1k (Total 30k).
    Caps applied as min(available, cap).
- Server sets:
    D_val (2,000 records) extracted following NSL-KDD rules.
    D_tune (stratified 10% draw of remainder after D_val).
    D_client: remainder client pool (N=20 clients, Dirichlet alpha_D=0.5).
- Zero-leakage scaling:
    MinMaxScaler(clip=False) fitted EXCLUSIVELY on server data (D_val U D_tune).
    Client and test data are NEVER used for scaler fitting.
- Client-local SMOTE applied locally per client after Dirichlet partitioning.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# ── 61 Canonical Feature Columns ───────────────────────────────────────────

FEATURE_COLUMNS = [
    "arp.opcode", "arp.hw.size", "arp.proto.size",
    "icmp.checksum", "icmp.seq_le", "icmp.unused",
    "http.response", "http.tls_port", "http.content_length",
    "dns.qry.name.len", "dns.qry.qu.class", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_response",
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msgtype",
    "mqtt.proto_len", "mqtt.protoname", "mqtt.topic", "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.proto_id", "mbtcp.unit_id",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum",
    "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
    "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.seq",
    "udp.stream", "udp.time_delta",
    "http.request.method", "http.referer", "http.request.version",
    "tcp.flags.cwr", "tcp.flags.ece", "tcp.flags.fin", "tcp.flags.push",
    "tcp.flags.res", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.urg",
    "tcp.hdr_len", "tcp.urgent_pointer", "tcp.window_size",
    "tcp.window_size_scalefactor", "tcp.window_size_value",
    "udp.length",
]

LABEL_COL = "label"
INPUT_DIM = len(FEATURE_COLUMNS)  # 61
NUM_CLASSES = 6

CLASS_NAMES = ["Normal", "DoS/DDoS", "Injection", "Scanning", "Malware", "MITM"]
CATEGORY_TO_ID = {
    "Normal": 0,
    "DoS/DDoS": 1,
    "Injection": 2,
    "Scanning": 3,
    "Malware": 4,
    "MITM": 5,
}

# ── Canonical 14 Raw Attacks + Normal = 15 Multiclass Labels ───────────────
# Reference: Ferrag, M. A. et al. (2022), "Edge-IIoTset: A New Comprehensive
# Realistic Cyber Security Dataset for IoT and IIoT Applications", IEEE Access 10:46790-46806.
# The multiclass Attack_type column contains exactly 14 attack types + Normal = 15 classes.
CANONICAL_RAW_ATTACKS = [
    # 0: Normal
    "Normal",
    # 1: DoS/DDoS (4 attacks)
    "DDoS_UDP", "DDoS_ICMP", "DDoS_HTTP", "DDoS_TCP",
    # 2: Injection (3 attacks)
    "SQL_injection", "XSS", "Uploading",
    # 3: Scanning (3 attacks)
    "Port_Scanning", "Vulnerability_scanner", "Fingerprinting",
    # 4: Malware (3 attacks)
    "Backdoor", "Password", "Ransomware",
    # 5: MITM (1 attack)
    "MITM",
]

# Canonical 15-class mapping to the 6 final classes (§VI-A, line 350)
CANONICAL_RAW_MAP = {
    # 0: Normal
    "Normal": 0,
    # 1: DoS/DDoS (4 attacks)
    "DDoS_UDP": 1,
    "DDoS_ICMP": 1,
    "DDoS_HTTP": 1,
    "DDoS_TCP": 1,
    # 2: Injection (3 attacks)
    "SQL_injection": 2,
    "XSS": 2,
    "Uploading": 2,
    # 3: Scanning (3 attacks)
    "Port_Scanning": 3,
    "Vulnerability_scanner": 3,
    "Fingerprinting": 3,
    # 4: Malware (3 attacks)
    "Backdoor": 4,
    "Password": 4,
    "Ransomware": 4,
    # 5: MITM (1 attack)
    "MITM": 5,
}

# Supported aliases for normalized string parsing.
# - 'os_fingerprinting': alias for 'Fingerprinting' used in some subsets.
# - 'arp_spoofing': underlying protocol mechanism for 'MITM'.
# Note: 'dns_spoofing' is from CIC-IoT-2023 and is intentionally NOT supported here.
RAW_ATTACK_ALIASES = {
    "os_fingerprinting": 3,
    "osfingerprinting": 3,
    "arp_spoofing": 5,
    "arpspoofing": 5,
    # Coarse family names supported as fallbacks
    "normal": 0,
    "ddos": 1,
    "dos": 1,
    "injection": 2,
    "scanning": 3,
    "information_gathering": 3,
    "informationgathering": 3,
    "malware": 4,
    "mitm": 5,
}

def _normalize_string(val: Any) -> str:
    s = str(val).strip().lower()
    for ch in (" ", "-", "_", "/"):
        s = s.replace(ch, "")
    return s

_normalize_label = _normalize_string

# Complete normalized lookup dictionary
RAW_ATTACK_MAP = {
    _normalize_string(k): cid for k, cid in CANONICAL_RAW_MAP.items()
}
for k, cid in RAW_ATTACK_ALIASES.items():
    RAW_ATTACK_MAP[_normalize_string(k)] = cid

# Supplementary Table S5 Per-Class Caps (Train / Test)
PAPER_TRAIN_CAPS = {
    0: 45000,  # Normal
    1: 40000,  # DoS/DDoS
    2: 25000,  # Injection
    3: 20000,  # Scanning
    4: 15000,  # Malware
    5: 5000,   # MITM
}

PAPER_TEST_CAPS = {
    0: 9000,   # Normal
    1: 8000,   # DoS/DDoS
    2: 5000,   # Injection
    3: 4000,   # Scanning
    4: 3000,   # Malware
    5: 1000,   # MITM
}

# Canonical validation quotas for Table S5 capped training set (2,000 samples total)
PAPER_VAL_QUOTAS = {
    0: 600,  # Normal (45k / 150k * 2000)
    1: 533,  # DoS/DDoS (40k / 150k * 2000)
    2: 333,  # Injection (25k / 150k * 2000)
    3: 267,  # Scanning (20k / 150k * 2000)
    4: 200,  # Malware (15k / 150k * 2000)
    5: 67,   # MITM (5k / 150k * 2000)
}

# Compatibility aliases
RAW_ATTACK_MAPPING = RAW_ATTACK_MAP
CANONICAL_CLASSES = CLASS_NAMES
CAP_TRAIN = PAPER_TRAIN_CAPS
CAP_TEST = PAPER_TEST_CAPS
TIMESTAMP_COL = "frame.time"
GUARD_BAND_SECONDS = 60.0


# ── Time-Disjoint Splitting with Guard Band ─────────────────────────────────

def time_disjoint_split_with_guard_band(
    df: pd.DataFrame,
    timestamp_col: str = "frame.time",
    split_ratio: float = 0.8,
    guard_band_seconds: float = 60.0,
    train_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a capture or per-attack/per-sensor file in time order per paper §VI-A:
    "We therefore split each source capture in time order: its first 80% of records
    feed the training pool and its last 20% the test set, and records within 60 seconds
    of the boundary are dropped."

    Args:
        df: Input dataframe containing records from a single capture/file.
        timestamp_col: Name of the timestamp column.
        split_ratio: Fraction for training side (default 0.8).
        guard_band_seconds: Window around split boundary to drop (default 60.0s).
        train_ratio: Alias for split_ratio.

    Returns:
        (train_df, test_df) tuple with boundary records removed.
    """
    if train_ratio is not None:
        split_ratio = train_ratio

    if timestamp_col not in df.columns:
        # If timestamp not present, deterministic index-based 80/20 split
        n = len(df)
        split_idx = int(round(n * split_ratio))
        return df.iloc[:split_idx].copy().reset_index(drop=True), df.iloc[split_idx:].copy().reset_index(drop=True)

    # Convert timestamps to numeric seconds
    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        ts_sec = pd.to_numeric(df[timestamp_col], errors="coerce")
    else:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if ts.isna().all():
            # Try direct numeric
            ts_sec = pd.to_numeric(df[timestamp_col], errors="coerce")
        else:
            ts_sec = ts.astype("int64") / 1e9

    valid_mask = ~ts_sec.isna()
    df_valid = df[valid_mask].sort_values(by=timestamp_col).reset_index(drop=True)
    ts_sorted = ts_sec[valid_mask].sort_values().to_numpy()

    n = len(df_valid)
    if n == 0:
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    split_idx = int(round(n * split_ratio))
    split_idx = min(max(split_idx, 0), n - 1)
    boundary_time = ts_sorted[split_idx]

    train_mask = ts_sorted < (boundary_time - guard_band_seconds)
    test_mask = ts_sorted > (boundary_time + guard_band_seconds)

    train_df = df_valid[train_mask].copy().reset_index(drop=True)
    test_df = df_valid[test_mask].copy().reset_index(drop=True)

    return train_df, test_df


time_disjoint_split_edgeiiotset = time_disjoint_split_with_guard_band


# ── Per-Class Capping ───────────────────────────────────────────────────────

def apply_class_caps(
    df: pd.DataFrame,
    caps: Dict[int, int],
    seed: int = 42,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Apply per-class caps as min(available, cap) per Supplementary Table S5.
    Deterministic selection under the given seed.
    """
    if label_col not in df.columns or len(df) == 0:
        return df.copy()

    rng = np.random.RandomState(seed)
    selected_indices = []

    for c in sorted(caps.keys()):
        cap = caps[c]
        c_indices = df.index[df[label_col] == c].to_numpy().copy()
        if len(c_indices) > cap:
            rng.shuffle(c_indices)
            selected_indices.extend(c_indices[:cap])
        else:
            selected_indices.extend(c_indices)

    return df.loc[selected_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ── Label Mapping ───────────────────────────────────────────────────────────

def map_labels(df: pd.DataFrame, raw_label_col: str = "Attack_type") -> pd.DataFrame:
    """
    Map fine-grained raw attack names to 6-class integer taxonomy (0..5).
    """
    df = df.copy()
    col = raw_label_col if raw_label_col in df.columns else LABEL_COL
    if col not in df.columns:
        raise ValueError(f"Neither '{raw_label_col}' nor '{LABEL_COL}' found in columns.")

    def _resolve(val: Any) -> int:
        if isinstance(val, (int, np.integer)):
            if 0 <= val < NUM_CLASSES:
                return int(val)
        s = _normalize_string(val)
        # 1. Direct exact normalized match
        if s in RAW_ATTACK_MAP:
            return RAW_ATTACK_MAP[s]
        # 2. Substring fallback match
        for k, cid in RAW_ATTACK_MAP.items():
            if k == s or k in s:
                return cid
        return -1

    n_before = len(df)
    df[LABEL_COL] = df[col].apply(_resolve)
    df = df[df[LABEL_COL] >= 0].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[Edge-IIoTset] Dropped {n_dropped} row(s) with unrecognized labels.")
    return df


# ── Server Quotas Computation ───────────────────────────────────────────────

def compute_edgeiiotset_quotas(
    counts: Dict[int, int],
    val_size: int = 2000,
    tune_fraction: float = 0.1,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Compute sample quotas for D_val (2,000), D_tune, and D_client per NSL-KDD rules (§VI-A):
    - Classes with < 5,000 records receive max(round(val_size * n / total), 100),
      scaled proportionally when val_size < 2,000, capped at 30% of available class records.
    - Large classes share remaining validation quota proportionally using largest remainder.
    - D_tune is stratified 10% of remainder after D_val.
    - D_client is remainder for client training pool.
    """
    total = sum(counts.values())
    val_target = min(val_size, total)
    val: Dict[int, int] = {}

    scale = val_target / 2000.0 if val_target < 2000 else 1.0
    rare_floor = max(1, int(round(100 * scale)))

    for c, n in counts.items():
        if n < 5000:
            prop = round(val_target * n / total)
            val[c] = min(max(prop, min(rare_floor, n)), max(1, int(0.3 * n)))

    # If sum(val.values()) exceeds val_target, scale back
    if sum(val.values()) > val_target:
        val_sum = sum(val.values())
        scaled = {c: int(val[c] * val_target / val_sum) for c in val}
        diff = val_target - sum(scaled.values())
        for c in sorted(val.keys(), key=lambda k: val[k] - scaled[k], reverse=True)[:diff]:
            scaled[c] += 1
        val = scaled

    rest = val_target - sum(val.values())
    big = {c: n for c, n in counts.items() if c not in val}
    big_total = sum(big.values())

    if big_total > 0 and rest > 0:
        raw = {c: rest * n / big_total for c, n in big.items()}
        floors = {c: int(v) for c, v in raw.items()}
        left = rest - sum(floors.values())
        for c in sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)[:left]:
            floors[c] += 1
        val.update(floors)
    elif rest > 0:
        for c in sorted(counts.keys(), key=lambda k: counts[k], reverse=True):
            add = min(rest, counts[c] - val.get(c, 0))
            val[c] = val.get(c, 0) + add
            rest -= add
            if rest <= 0:
                break

    pool = {c: counts[c] - val.get(c, 0) for c in counts}
    tune = {c: max(1, round(tune_fraction * pool[c])) if pool[c] > 0 else 0 for c in pool}
    clients = {c: max(0, pool[c] - tune[c]) for c in pool}

    return val, tune, clients


def extract_edgeiiotset_splits(
    df_train: pd.DataFrame,
    seed: int = 42,
    val_size: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract exact paper splits D_val (2,000), D_tune, and D_client from training dataframe.
    Guarantees pairwise disjointness and zero overlap.
    """
    if LABEL_COL not in df_train.columns:
        raise ValueError(f"df_train must contain '{LABEL_COL}' column.")

    counts = df_train[LABEL_COL].value_counts().to_dict()
    val_quotas, tune_quotas, _ = compute_edgeiiotset_quotas(counts, val_size=val_size)

    rng = np.random.RandomState(seed)
    val_indices: List[int] = []
    tune_indices: List[int] = []
    client_indices: List[int] = []

    for c in range(NUM_CLASSES):
        c_indices = df_train.index[df_train[LABEL_COL] == c].to_numpy().copy()
        v_need = val_quotas.get(c, 0)
        t_need = tune_quotas.get(c, 0)
        total_need = v_need + t_need

        if len(c_indices) < v_need:
            raise ValueError(
                f"Cannot satisfy validation quota for class {c} ({CLASS_NAMES[c]}): "
                f"required {v_need}, but only {len(c_indices)} available."
            )

        rng.shuffle(c_indices)
        val_indices.extend(c_indices[:v_need])
        tune_indices.extend(c_indices[v_need:min(total_need, len(c_indices))])
        if len(c_indices) > total_need:
            client_indices.extend(c_indices[total_need:])

    # Verify pairwise disjointness
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


# ── Zero-Leakage Scaling & Encoding ─────────────────────────────────────────

def encode_and_scale_splits(
    val_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    client_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    MinMaxScaler,
    Dict[str, LabelEncoder],
]:
    """
    Zero-leakage encoding and scaling:
    - Scaler fitted EXCLUSIVELY on server data: D_val U D_tune.
    - Scaler NEVER fitted on client training data or test data.
    - MinMaxScaler(clip=False) preserves out-of-range test values.
    """
    cols = feature_columns or FEATURE_COLUMNS
    val_df = val_df.copy()
    tune_df = tune_df.copy()
    client_df = client_df.copy()
    test_df = test_df.copy()

    server_df = pd.concat([val_df, tune_df], axis=0, ignore_index=True)
    encoders: Dict[str, LabelEncoder] = {}

    # Encode any categorical columns present in server data
    for col in cols:
        if col in server_df.columns and server_df[col].dtype == object:
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

    # Clean non-finite values before scaling
    def _extract_clean(df_in: pd.DataFrame) -> np.ndarray:
        arr = df_in[cols].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

    X_server = _extract_clean(server_df)
    scaler = MinMaxScaler(clip=False)
    scaler.fit(X_server)

    X_val = scaler.transform(_extract_clean(val_df)).astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.int64)

    X_tune = scaler.transform(_extract_clean(tune_df)).astype(np.float32)
    y_tune = tune_df[LABEL_COL].values.astype(np.int64)

    X_client = scaler.transform(_extract_clean(client_df)).astype(np.float32)
    y_client = client_df[LABEL_COL].values.astype(np.int64)

    X_test = scaler.transform(_extract_clean(test_df)).astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.int64)

    return (
        X_val, y_val,
        X_tune, y_tune,
        X_client, y_client,
        X_test, y_test,
        scaler, encoders,
    )


# ── Client-Local SMOTE ──────────────────────────────────────────────────────

def apply_client_local_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply client-local SMOTE according to paper §VI-A:
    "Each client oversamples locally with SMOTE: every class below the client's
    median non-empty class count is raised to that median, with k=min(5, n_c-1)
    neighbors when the class has n_c >= 2 records, duplication when n_c=1,
    and nothing when n_c=0."
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
    """Convenience wrapper for client-local SMOTE."""
    return apply_client_local_smote(X, y, random_state=random_state)


# ── Full Pipeline ───────────────────────────────────────────────────────────

def get_edgeiiotset_splits(
    train_path: str,
    test_path: str,
    seed: int = 42,
    val_size: int = 2000,
    apply_caps: bool = True,
    cap_train: Optional[int] = None,
    cap_test: Optional[int] = None,
    train_caps: Optional[Dict[int, int]] = None,
    test_caps: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Load raw or pre-split Edge-IIoTset files and produce explicit paper splits:
    D_val (2,000), D_tune, D_client, D_test.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Edge-IIoTset training file not found: {train_path}\n"
            "Download Edge-IIoTset from IEEE Dataport (10.1109/ACCESS.2022.3165809) "
            "and place the training CSV at this path."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Edge-IIoTset test file not found: {test_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    if apply_caps:
        if train_caps is not None:
            tr_caps = train_caps
        elif cap_train is not None:
            tr_caps = {c: max(1, int(round(cap_train / NUM_CLASSES))) for c in range(NUM_CLASSES)}
        else:
            tr_caps = PAPER_TRAIN_CAPS

        if test_caps is not None:
            te_caps = test_caps
        elif cap_test is not None:
            te_caps = {c: max(1, int(round(cap_test / NUM_CLASSES))) for c in range(NUM_CLASSES)}
        else:
            te_caps = PAPER_TEST_CAPS

        train_df = apply_class_caps(train_df, tr_caps, seed=seed)
        test_df = apply_class_caps(test_df, te_caps, seed=seed)

    val_df, tune_df, client_df = extract_edgeiiotset_splits(train_df, seed=seed, val_size=val_size)
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
    apply_caps: bool = True,
    cap_train: Optional[int] = None,
    cap_test: Optional[int] = None,
    train_caps: Optional[Dict[int, int]] = None,
    test_caps: Optional[Dict[int, int]] = None,
):
    """
    Full Edge-IIoTset preprocessing pipeline conforming to IEEE paper §VI-A and Table S5.

    Returns:
        If return_tune is False:
            (X_client, y_client, X_val, y_val, X_test, y_test, scaler, encoders, class_weights)
        If return_tune is True:
            (X_client, y_client, X_val, y_val, X_tune, y_tune, X_test, y_test, scaler, encoders, class_weights)
    """
    if protocol not in ("main", "leakage_free"):
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'main' or 'leakage_free'.")

    splits = get_edgeiiotset_splits(
        train_path, test_path, seed=seed, val_size=val_size, apply_caps=apply_caps,
        cap_train=cap_train, cap_test=cap_test, train_caps=train_caps, test_caps=test_caps,
    )

    X_val, y_val = splits["D_val"]
    X_tune, y_tune = splits["D_tune"]
    X_client, y_client = splits["D_client"]
    X_test, y_test = splits["D_test"]
    scaler = splits["scaler"]
    encoders = splits["encoders"]
    weights = splits["class_weights"]

    print(
        f"[Edge-IIoTset] protocol={protocol} | Client pool: {X_client.shape} | "
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
