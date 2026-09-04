"""
scripts/check_results.py
Pre-submission result completeness AND authenticity check.

Completeness: are the expected result files present, with all strategies x seeds?
Authenticity: does any present file carry a fingerprint of the fabricated
placeholder output quarantined under results/_QUARANTINED_MOCK/? A file that
looks complete but is mock output is worse than a missing one, so the
authenticity scan is a hard failure, not a warning.

Run: python scripts/check_results.py
"""
import glob, json, os, sys

REQUIRED_FILES = [
    "results/tables/full_comparison_results.json",
    "results/tables/ablation_results.json",
    "results/tables/ratio_sweep_results.json",
    "results/figures/fig1_convergence.pdf",
    "results/figures/fig2_trust_evolution.pdf",
    "results/figures/fig3_robustness_curve.pdf",
    "results/figures/fig4_ablation.pdf",
    "results/figures/fig5_adaptive_weights.pdf",
    "results/figures/fig6_confusion.pdf",
]

REQUIRED_STRATEGIES = ["fedavg", "krum", "trimmed_mean", "fltrust",
                        "foolsgold", "flame", "rfa", "tvflids"]
REQUIRED_SEEDS = [42, 123, 456, 789, 1337]

# Fingerprints of the quarantined fabricated artifacts (see
# results/_QUARANTINED_MOCK/README.md for the per-file evidence).
MOCK_FINGERPRINTS = ["mockhash"]
QUARANTINE_DIR = os.path.join("results", "_QUARANTINED_MOCK")


def check_authenticity():
    """Fail if any live result file carries a known fabrication fingerprint.

    Scans results/ but deliberately skips the quarantine directory itself,
    which is archival forensic evidence and is *expected* to match.
    """
    ok = True
    print("=== Result Authenticity Check ===")

    scanned = 0
    for path in glob.glob(os.path.join("results", "**", "*.json"), recursive=True):
        if os.path.normpath(path).startswith(os.path.normpath(QUARANTINE_DIR)):
            continue
        scanned += 1
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                blob = fh.read()
        except OSError as e:
            print(f"  [UNREADABLE] {path}: {e}")
            ok = False
            continue
        hits = [fp for fp in MOCK_FINGERPRINTS if fp in blob]
        if hits:
            print(f"  [FABRICATED] {path} contains {hits} - quarantined mock "
                  f"output sitting where a real result belongs.")
            ok = False

    print(f"  Scanned {scanned} live result file(s) outside {QUARANTINE_DIR}.")

    # Genuine generated results should carry provenance. Proposition 1's output
    # is the one file this repo regenerates deterministically without a full
    # experimental run, so it is checked explicitly.
    prop1 = os.path.join("results", "tables", "proposition1_real.json")
    if os.path.exists(prop1):
        try:
            prov = json.load(open(prop1)).get("provenance", {})
        except json.JSONDecodeError as e:
            print(f"  [CORRUPT] {prop1}: {e}")
            return False
        missing = [k for k in ("script", "git_commit", "timestamp_utc")
                   if not prov.get(k)]
        if missing:
            print(f"  [NO PROVENANCE] {prop1} missing {missing}")
            ok = False
        else:
            print(f"  [OK] {prop1} provenance: {prov['script']} @ "
                  f"{prov['git_commit'][:8]} ({prov['timestamp_utc']})")

    if not ok:
        print("  Regenerate the offending file(s) with the corresponding "
              "runner. Never copy anything out of the quarantine into results/.")
    return ok


def check():
    ok = True
    print("=== Result Completeness Check ===\n")

    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            print(f"  [MISSING] {f}")
            ok = False
            continue
        # A zero-byte file is not a result. Six 0-byte placeholder PDFs sat at
        # the paper's figure paths and passed the old existence-only check.
        size = os.path.getsize(f)
        if size == 0:
            print(f"  [EMPTY] {f} (0 bytes - placeholder, not a generated figure)")
            ok = False
            continue
        if f.endswith(".pdf"):
            with open(f, "rb") as fh:
                if fh.read(5) != b"%PDF-":
                    print(f"  [CORRUPT] {f} is not a PDF")
                    ok = False
                    continue
        print(f"  [OK] {f} ({size:,} bytes)")

    # Check full_comparison_results.json content
    cmp_path = "results/tables/full_comparison_results.json"
    if os.path.exists(cmp_path):
        data = json.load(open(cmp_path))
        raw = data.get("raw", {})
        print("\n=== Strategy × Seed Coverage ===\n")
        for strategy in REQUIRED_STRATEGIES:
            results = raw.get(strategy, [])
            seeds_found = [r.get("seed") for r in results]
            missing = [s for s in REQUIRED_SEEDS if s not in seeds_found]
            if missing:
                print(f"  [INCOMPLETE] {strategy}: missing seeds {missing}")
                ok = False
            else:
                accs = [r.get("final_accuracy", 0) for r in results]
                print(f"  [OK] {strategy}: {len(results)} seeds, "
                      f"acc={sum(accs)/len(accs):.4f}")

    ok = check_authenticity() and ok

    print(f"\n{'[PASS] All checks passed.' if ok else '[FAIL] Fix the items above.'}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if check() else 1)
