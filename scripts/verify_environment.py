"""
scripts/verify_environment.py
Report the ACTUAL runtime environment against the one the paper states.

This script reports; it does not fix and it does not claim equivalence. Run it
before any experimental campaign and paste its output alongside the results, so
that a number is always traceable to the stack that produced it.

    python scripts/verify_environment.py            # human-readable report
    python scripts/verify_environment.py --json     # machine-readable
    python scripts/verify_environment.py --strict   # exit 1 on any mismatch

Paper reference: Section VI-D ("Implementation Details"), which states
Python 3.10.12, PyTorch 2.1.0, Flower 1.6.0, scikit-learn 1.3.2,
imbalanced-learn 0.11.0, NumPy 1.26.2, SciPy 1.11.4, on an NVIDIA RTX 3080
(10 GB), 32 GB RAM, AMD Ryzen 9 5900X.
"""

import argparse
import importlib
import json
import platform
import sys

# (import name, distribution name, version the paper states)
PAPER_STACK = [
    ("torch",     "torch",             "2.1.0"),
    ("flwr",      "flwr",              "1.6.0"),
    ("sklearn",   "scikit-learn",      "1.3.2"),
    ("imblearn",  "imbalanced-learn",  "0.11.0"),
    ("numpy",     "numpy",             "1.26.2"),
    ("scipy",     "scipy",             "1.11.4"),
    ("pandas",    "pandas",            "2.1.3"),
]

PAPER_PYTHON = "3.10.12"

# Paper Section VI-D. Not auto-checkable, reported for the record only.
PAPER_HARDWARE = {
    "gpu": "NVIDIA RTX 3080 (10 GB VRAM)",
    "ram": "32 GB",
    "cpu": "AMD Ryzen 9 5900X",
}


def _installed(import_name, dist_name):
    try:
        mod = importlib.import_module(import_name)
    except Exception as e:                       # noqa: BLE001 - report, don't raise
        return None, f"import failed: {type(e).__name__}: {e}"
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as _v
            version = _v(dist_name)
        except Exception:                        # noqa: BLE001
            version = None
    return version, None


def collect():
    report = {
        "python": {
            "required": PAPER_PYTHON,
            "actual": platform.python_version(),
            "match": platform.python_version() == PAPER_PYTHON,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "full": platform.platform(),
        },
        "packages": {},
        "paper_hardware": PAPER_HARDWARE,
        "accelerator": {},
    }

    for import_name, dist_name, required in PAPER_STACK:
        actual, error = _installed(import_name, dist_name)
        report["packages"][dist_name] = {
            "required": required,
            "actual": actual,
            "error": error,
            "match": (actual == required),
        }

    try:
        import torch
        report["accelerator"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "device_count": (torch.cuda.device_count()
                             if torch.cuda.is_available() else 0),
        }
    except Exception as e:                       # noqa: BLE001
        report["accelerator"] = {"error": f"{type(e).__name__}: {e}"}

    mismatches = [name for name, v in report["packages"].items() if not v["match"]]
    if not report["python"]["match"]:
        mismatches.append("python")
    report["mismatches"] = mismatches
    report["matches_paper_environment"] = not mismatches
    return report


def render(report):
    print("=== TV-FLIDS Environment Verification ===")
    print()
    py = report["python"]
    flag = "OK " if py["match"] else "DIFF"
    print(f"  [{flag}] python            required {py['required']:<10} "
          f"actual {py['actual']}")

    for name, v in report["packages"].items():
        flag = "OK " if v["match"] else "DIFF"
        actual = v["actual"] or f"<{v['error']}>"
        print(f"  [{flag}] {name:<18}required {v['required']:<10} actual {actual}")

    print()
    print(f"  Platform: {report['platform']['full']}")
    acc = report["accelerator"]
    if acc.get("error"):
        print(f"  Accelerator: unavailable ({acc['error']})")
    elif acc.get("cuda_available"):
        print(f"  Accelerator: {acc['device_name']} "
              f"(CUDA {acc['cuda_version']}, {acc['device_count']} device(s))")
    else:
        print("  Accelerator: none detected (CPU only)")
    print(f"  Paper hardware (not auto-checkable): "
          f"{PAPER_HARDWARE['gpu']}, {PAPER_HARDWARE['ram']}, "
          f"{PAPER_HARDWARE['cpu']}")

    print()
    if report["matches_paper_environment"]:
        print("  RESULT: software stack matches the paper's stated versions.")
        print("          Hardware still requires manual confirmation against "
              "Section VI-D.")
    else:
        print(f"  RESULT: {len(report['mismatches'])} mismatch(es): "
              f"{', '.join(report['mismatches'])}")
        print("          Numbers produced here are NOT directly comparable to "
              "the paper's reported values.")
        print("          Reproduce the paper stack with:")
        print("              conda env create -f environment.yml   (Python 3.10.12)")
        print("          or  pip install -r requirements.txt       "
              "(into a 3.10.x interpreter)")

    if report["platform"]["system"] == "Windows":
        print()
        print("  NOTE (Windows): the Flower simulation engine runs clients in "
              "Ray worker")
        print("  processes, and importing PyTorch inside those workers is known "
              "to fail on")
        print("  this platform with:")
        print("      OSError: [WinError 1114] A dynamic link library (DLL) "
              "initialization")
        print("      routine failed. Error loading ...torch/lib/c10.dll")
        print("  Everything that does not go through Ray (unit tests, the "
              "in-process")
        print("  strategy tests, theory/proposition1_verification.py, the "
              "preprocessing")
        print("  pipelines) works normally. The full experimental campaign "
              "needs Linux,")
        print("  or WSL2, matching the paper's stack.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit the report as JSON instead of text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the environment differs from the paper's")
    args = ap.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        render(report)

    if args.strict and not report["matches_paper_environment"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
