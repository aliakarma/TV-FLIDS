"""
scripts/verify_data.py
Verify NSL-KDD dataset integrity via SHA-256 hashes.

Run: python scripts/verify_data.py
"""
import hashlib
import os
import sys

EXPECTED_HASHES = {
    "data/raw/KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "data/raw/KDDTest+.txt":  "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    all_ok = True
    for path in ["data/raw/KDDTrain+.txt", "data/raw/KDDTest+.txt"]:
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            all_ok = False
            continue
        actual = compute_sha256(path)
        print(f"[{path}] SHA-256: {actual}")
        if actual != EXPECTED_HASHES[path]:
            print(f"[ERROR] Hash mismatch for {path}! Expected {EXPECTED_HASHES[path]}")
            all_ok = False
    if all_ok:
        print("\n[OK] All dataset files present and verified.")
    else:
        print("\n[FAIL] Missing or corrupted files. Run: bash scripts/download_nslkdd.sh")
        sys.exit(1)


if __name__ == "__main__":
    verify()
