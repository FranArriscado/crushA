#               run_resample_comparison.py
#
#  For each valid partition in the calibration results file, runs
#  ribOptimizer_v7 twice:
#    - resampled   (default, cleanup + uniform resampling)
#    - no-resample (cleanup only, --no-resample flag)
#
#  Both runs use --steps 1 to exit after printing the initial force.
#  The calibrated p from the results file is used for each partition.
#
#  Output .txt/.mat/.pdf files are saved by ribOptimizer to Results2/Rib/
#  and tagged with _resampled or _noresample in the filename.
#
#  A summary CSV is written next to this script.
#
#  Usage:
#    python run_resample_comparison.py
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
DATA_DIR = SCRIPT_DIR / "data"
CALIBRATION_MAT_DIR = DATA_DIR / "Original from tiago"

# Path to the calibration results file produced by calibrate_rib_p_batch.py
CALIB_FILE = SCRIPT_DIR / "calibrate_rib_p_results.txt"

# Where the summary CSV is written
SUMMARY_CSV = SCRIPT_DIR / "resample_comparison.csv"

# Optimisation mode (irrelevant for --steps 1, but required positional arg)
MODE = "maxCF"

# ---- Regex to parse calibration results file ---- #
# Matches lines like:
#   partition_data_layer2_z-132.4.mat   130659.3   0.9994326400   130657.8   -0.0011
RE_VALID = re.compile(
    r"^\s*(partition_data_\S+\.mat)\s+"   # filename
    r"[\d.]+\s+"                           # target force (not needed)
    r"(0\.\d+)\s+"                         # p value
    r"[\d.]+\s+"                           # matched force (not needed)
    r"[+\-][\d.]+\s*$"                     # error % — presence confirms valid
)

# Regex to extract initial force from optimizer output
RE_FORCE = re.compile(r"Force total:\s*([\d.eE+\-]+)\s*N")


def parse_calibration_file(path: Path) -> list[tuple[str, float]]:
    """Return list of (mat_filename, p) for all valid (non-FAILED) entries."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RE_VALID.match(line)
            if m:
                entries.append((m.group(1), float(m.group(2))))
    return entries


def resolve_mat_arg(mat_file: str) -> str:
    """
    Resolve the MAT path used during calibration.

    Calibration currently runs against Rib/data/Original from tiago, while
    the results file stores only the basename. Prefer that folder so the
    comparison replays the exact same inputs. Fall back to Rib/data root
    for older/manual cases.
    """
    preferred = CALIBRATION_MAT_DIR / mat_file
    if preferred.exists():
        return str(Path("Original from tiago") / mat_file)

    fallback = DATA_DIR / mat_file
    if fallback.exists():
        return mat_file

    raise FileNotFoundError(
        f"Could not locate '{mat_file}' in either:\n"
        f"  {preferred}\n"
        f"  {fallback}"
    )


def run_single(mat_file: str, p: float, resample: bool) -> dict:
    """
    Run ribOptimizer_v7 for one file with --steps 1.
    Returns dict with force and success flag.
    """
    tag = "resampled" if resample else "noresample"
    try:
        mat_arg = resolve_mat_arg(mat_file)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    cmd = [
        PYTHON, str(SCRIPT),
        mat_arg, MODE,
        "--spline-p", f"{p:.10f}",
        "--steps", "1",
        "--tag", tag,
    ]
    if not resample:
        cmd.append("--no-resample")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    combined = result.stdout + result.stderr
    m = RE_FORCE.search(combined)
    if not m:
        tail = combined[-800:].strip()
        err = "could not parse force"
        if tail:
            err = f"{err}; tail: {tail}"
        return {"ok": False, "error": err}
    return {"ok": True, "force": float(m.group(1))}


def main():
    if not SCRIPT.exists():
        print(f"ERROR: ribOptimizer script not found:\n  {SCRIPT}")
        sys.exit(1)
    if not CALIB_FILE.exists():
        print(f"ERROR: calibration results file not found:\n  {CALIB_FILE}")
        sys.exit(1)

    entries = parse_calibration_file(CALIB_FILE)
    if not entries:
        print("No valid entries found in calibration file.")
        sys.exit(1)

    print(f"Found {len(entries)} valid partitions. Running 2 passes each...\n")

    rows = []
    for mat_file, p in entries:
        print(f"[{mat_file}]  p = {p:.10f}")

        res_with = run_single(mat_file, p, resample=True)
        f_with = res_with["force"] if res_with["ok"] else None
        status_with = f"{f_with:.1f} N" if f_with is not None else f"FAILED ({res_with['error']})"
        print(f"  resampled  : {status_with}")

        res_without = run_single(mat_file, p, resample=False)
        f_without = res_without["force"] if res_without["ok"] else None
        status_without = f"{f_without:.1f} N" if f_without is not None else f"FAILED ({res_without['error']})"
        print(f"  no-resample: {status_without}")

        if f_with is not None and f_without is not None:
            diff = f_with - f_without
            diff_pct = 100.0 * diff / f_without
            print(f"  diff       : {diff:+.1f} N  ({diff_pct:+.4f}%)")
        else:
            diff = None
            diff_pct = None

        rows.append({
            "file":          mat_file,
            "p":             p,
            "F_resampled":   f_with,
            "F_noresample":  f_without,
            "diff_N":        diff,
            "diff_pct":      diff_pct,
        })
        print()

    # ---- Write CSV ---- #
    with open(SUMMARY_CSV, "w", encoding="utf-8") as f:
        f.write("file,p,F_resampled_N,F_noresample_N,diff_N,diff_pct\n")
        for r in rows:
            diff_n_str   = f"{r['diff_N']:.2f}"   if r['diff_N']   is not None else ""
            diff_pct_str = f"{r['diff_pct']:.4f}" if r['diff_pct'] is not None else ""
            f_res_str    = f"{r['F_resampled']:.1f}"  if r['F_resampled']  is not None else "FAILED"
            f_nor_str    = f"{r['F_noresample']:.1f}" if r['F_noresample'] is not None else "FAILED"
            f.write(
                f"{r['file']},{r['p']:.10f},"
                f"{f_res_str},{f_nor_str},{diff_n_str},{diff_pct_str}\n"
            )

    print(f"Summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
