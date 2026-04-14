#                  calibrate_rib_p_batch.py
#
#  Find the best spline p for each .mat input by running
#  ribOptimizer_v7 with --match-initial-force and --steps 1.
#
#  Usage:
#    python calibrate_rib_p_batch.py
#
#  Edit RUNS below to add your (filename, reference_force_N) pairs.
#  The script prints a summary table and writes calibrate_rib_p_results.txt.
#
#  Francisco Arriscado -- FEUP, 2026
# ==========================================================================

import subprocess
import sys
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_CANDIDATE = Path(r"C:\Users\Arriscado\envs\crushenv\Scripts\python.exe")
PYTHON = str(PYTHON_CANDIDATE if PYTHON_CANDIDATE.exists() else Path(sys.executable))
SCRIPT = SCRIPT_DIR / "ribOptimizer(v7).py"
FORCES_FILE = SCRIPT_DIR / "Force.txt"
MAT_DIR = SCRIPT_DIR / "data" / "Original from tiago"
OUTPUT_FILE = SCRIPT_DIR / "calibrate_rib_p_results.txt"

# Keep preprocessing aligned with the current v7 default path.
EXTRA_FLAGS = []

# Regex patterns to extract values from optimizer output
RE_P     = re.compile(r"Calibrated spline p:\s*([\d.eE+\-]+)")
RE_MATCH = re.compile(r"Matched force:\s*([\d.eE+\-]+)")
RE_ERR   = re.compile(r"Match error:\s*([+\-]?[\d.eE+\-]+)\s*N\s*\(([+\-]?[\d.eE+\-]+)%\)")
RE_LAYER = re.compile(r"layer(\d+)", re.IGNORECASE)


def extract_layer_number(path: Path) -> int:
    match = RE_LAYER.search(path.name)
    if not match:
        raise ValueError(f"Could not extract layer number from '{path.name}'.")
    return int(match.group(1))


def load_force_values(force_file: Path):
    values = []
    for raw_line in force_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values.append(float(line))
    return values


def build_runs():
    if not MAT_DIR.exists():
        raise FileNotFoundError(f"MAT directory not found:\n  {MAT_DIR}")
    if not FORCES_FILE.exists():
        raise FileNotFoundError(f"Force file not found:\n  {FORCES_FILE}")

    mat_files = sorted(MAT_DIR.glob("*.mat"), key=extract_layer_number)
    forces = load_force_values(FORCES_FILE)

    if len(mat_files) != len(forces):
        raise ValueError(
            f"Found {len(mat_files)} MAT files in '{MAT_DIR}', but "
            f"{len(forces)} force values in '{FORCES_FILE}'."
        )

    runs = []
    for mat_path, target_force in zip(mat_files, forces):
        rel_mat = mat_path.relative_to(SCRIPT_DIR / "data")
        runs.append((str(rel_mat), mat_path.name, target_force))
    return runs


def run_calibration(mat_arg: str, label: str, target_force: float):
    """
    Run ribOptimizer_v7 for a single file with --match-initial-force.
    Returns dict with p, matched_force, error_N, error_pct, or error string.
    """
    cmd = [
        PYTHON, str(SCRIPT),
        mat_arg,
        "maxCF",                              # mode doesn't matter for p calibration
        "--match-initial-force", str(target_force),
        "--steps", "1",                       # exit after 1 epoch; calibration already done
    ] + EXTRA_FLAGS
    print(f"\n[{label}]  target = {target_force:.1f} N")
    print("  Running...", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Timed out after 120 s"}
    except Exception as exc:
        return {"error": str(exc)}

    combined = result.stdout + result.stderr

    m_p     = RE_P.search(combined)
    m_match = RE_MATCH.search(combined)
    m_err   = RE_ERR.search(combined)

    if not m_p:
        # Print raw output so the user can diagnose
        print("  !! Could not parse calibrated p. Raw output below:")
        for line in combined.splitlines():
            print("     " + line)
        return {"error": "Could not parse output"}

    return {
        "p":            float(m_p.group(1)),
        "matched_force": float(m_match.group(1)) if m_match else None,
        "error_N":      float(m_err.group(1))   if m_err   else None,
        "error_pct":    float(m_err.group(2))   if m_err   else None,
    }


def main():
    if not SCRIPT.exists():
        print(f"ERROR: ribOptimizer script not found at:\n  {SCRIPT}")
        sys.exit(1)

    try:
        runs = build_runs()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    results = []
    for mat_arg, label, target_force in runs:
        res = run_calibration(mat_arg, label, target_force)
        results.append((label, target_force, res))

    # Print summary
    sep = "-" * 82
    header = f"{'File':<40} {'Target (N)':>12} {'p':>16} {'Matched (N)':>12} {'Error %':>8}"
    print("\n\n" + "=" * 82)
    print("CALIBRATION SUMMARY")
    print("=" * 82)
    print(header)
    print(sep)

    lines = ["CALIBRATION SUMMARY", header, sep]

    for label, target_force, res in results:
        if "error" in res:
            row = f"{'  ' + label:<40} {target_force:>12.1f} {'FAILED: ' + res['error']:>38}"
        else:
            err_str = f"{res['error_pct']:+.4f}" if res['error_pct'] is not None else "N/A"
            match_str = f"{res['matched_force']:.1f}" if res['matched_force'] is not None else "N/A"
            row = (
                f"{label:<40} {target_force:>12.1f} "
                f"{res['p']:>16.10f} "
                f"{match_str:>12} "
                f"{err_str:>8}"
            )
        print(row)
        lines.append(row)

    print(sep)
    lines.append(sep)

    # Write results file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print the --spline-p flags ready to copy-paste
    print("\nReady-to-use --spline-p flags:")
    for label, _, res in results:
        if "error" not in res:
            print(f"  {label}  ->  --spline-p {res['p']:.10f}")


if __name__ == "__main__":
    main()
