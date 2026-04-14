#               calibrate_norib_p_batch.py
#
#  Calibrate spline p for each cleaned no-rib partition by running
#  optimizationModule_v21.py as a subprocess with --steps 1.
#
#  The cleaned MAT files are discovered automatically in this folder and
#  paired with the reference forces listed in Force.txt by layer order.
#
#  Francisco Arriscado -- FEUP, 2026
# ==========================================================================

import re
import subprocess
import sys
from pathlib import Path

from scipy.optimize import minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_CANDIDATE = Path(r"C:\Users\Arriscado\envs\crushenv\Scripts\python.exe")
PYTHON = str(PYTHON_CANDIDATE if PYTHON_CANDIDATE.exists() else Path(sys.executable))
SCRIPT = SCRIPT_DIR / "optimizationModule_v21.py"
FORCES_FILE = SCRIPT_DIR / "Force.txt"
OUTPUT_FILE = SCRIPT_DIR / "calibrate_norib_p_results.txt"

RE_LAYER = re.compile(r"layer(\d+)", re.IGNORECASE)
RE_FORCE_ROW = re.compile(
    r"^\s*Force\s+([+\-]?[\d.eE]+)\s+([+\-]?[\d.eE]+)\s+[+\-]?[\d.eE]+%\s*$",
    re.MULTILINE,
)

CANDIDATE_PS = [
    1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
    0.1, 0.2, 0.4, 0.6, 0.8,
    0.9, 0.95, 0.98, 0.99, 0.995,
    0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999999,
]


def extract_layer_number(path: Path) -> int:
    match = RE_LAYER.search(path.name)
    if not match:
        raise ValueError(f"Could not extract layer number from '{path.name}'.")
    return int(match.group(1))


def load_force_values(force_file: Path) -> list[float]:
    values: list[float] = []
    for raw_line in force_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values.append(float(line))
    return values


def discover_cleaned_mats() -> list[Path]:
    mats = []
    for path in SCRIPT_DIR.glob("partition_data_layer*.mat"):
        name = path.name.lower()
        if "uniform" not in name:
            continue
        mats.append(path)
    return sorted(mats, key=extract_layer_number)


def build_runs() -> list[tuple[Path, float]]:
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Optimizer script not found:\n  {SCRIPT}")
    if not FORCES_FILE.exists():
        raise FileNotFoundError(f"Force file not found:\n  {FORCES_FILE}")

    mats = discover_cleaned_mats()
    forces = load_force_values(FORCES_FILE)

    if len(mats) != len(forces):
        raise ValueError(
            f"Found {len(mats)} cleaned MAT files in '{SCRIPT_DIR}', but "
            f"{len(forces)} force values in '{FORCES_FILE}'."
        )

    return list(zip(mats, forces))


def run_v21_force(mat_path: Path, p_val: float) -> float:
    cmd = [
        PYTHON,
        str(SCRIPT),
        str(mat_path),
        "maxCF",
        "--spline-p", f"{p_val:.10f}",
        "--steps", "1",
        "--patience", "1",
        "--tag", "calib_norib_tmp",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )

    combined = result.stdout + result.stderr
    match = RE_FORCE_ROW.search(combined)
    if not match:
        tail = combined[-1200:].strip()
        raise RuntimeError(
            "Could not parse initial force from optimizationModule_v21 output.\n"
            f"Output tail:\n{tail}"
        )

    return float(match.group(1))


def calibrate(mat_path: Path, target_force: float) -> dict:
    cache: dict[float, float] = {}

    def eval_cached(p_val: float) -> float:
        key = round(float(max(min(p_val, 1.0 - 1e-12), 1e-12)), 12)
        if key not in cache:
            cache[key] = run_v21_force(mat_path, key)
        return cache[key]

    samples = [(p, eval_cached(p)) for p in CANDIDATE_PS]
    best_idx = min(
        range(len(samples)),
        key=lambda idx: abs(samples[idx][1] - target_force),
    )
    best_p, best_force = samples[best_idx]
    refined = False

    if 0 < best_idx < len(samples) - 1:
        lo = samples[best_idx - 1][0]
        hi = samples[best_idx + 1][0]
        if hi - lo > 1e-9:
            opt = minimize_scalar(
                lambda p_val: abs(eval_cached(p_val) - target_force),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 30},
            )
            p_refined = float(max(min(opt.x, 1.0 - 1e-12), 1e-12))
            force_refined = eval_cached(p_refined)
            if abs(force_refined - target_force) < abs(best_force - target_force):
                best_p = p_refined
                best_force = force_refined
                refined = True

    return {
        "p": best_p,
        "matched_force": best_force,
        "error_N": best_force - target_force,
        "error_pct": 100.0 * (best_force - target_force) / target_force,
        "refined": refined,
        "edge_hit": best_idx in (0, len(samples) - 1),
    }


def main() -> None:
    try:
        runs = build_runs()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    results: list[tuple[str, float, dict | None]] = []

    for mat_path, target_force in runs:
        label = mat_path.name
        print(f"\n[{label}]  target = {target_force:.1f} N")
        print("  Calibrating...", flush=True)
        try:
            res = calibrate(mat_path, target_force)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append((label, target_force, None))
            continue

        print(f"  Calibrated p : {res['p']:.10f}")
        print(f"  Matched force: {res['matched_force']:.2f} N")
        print(f"  Match error  : {res['error_N']:+.2f} N  ({res['error_pct']:+.4f}%)")
        if res["edge_hit"]:
            print("  Note: best match landed on grid boundary.")
        results.append((label, target_force, res))

    sep = "-" * 86
    header = f"{'File':<44} {'Target (N)':>12} {'p':>16} {'Matched (N)':>12} {'Error %':>8}"
    print("\n\n" + "=" * 86)
    print("CALIBRATION SUMMARY")
    print("=" * 86)
    print(header)
    print(sep)

    lines = ["CALIBRATION SUMMARY", header, sep]
    for label, target_force, res in results:
        if res is None:
            row = f"{label:<44} {target_force:>12.1f} {'FAILED':>38}"
        else:
            row = (
                f"{label:<44} {target_force:>12.1f} "
                f"{res['p']:>16.10f} "
                f"{res['matched_force']:>12.1f} "
                f"{res['error_pct']:>+8.4f}"
            )
        print(row)
        lines.append(row)

    print(sep)
    lines.append(sep)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {OUTPUT_FILE}")

    print("\nReady-to-use --spline-p flags:")
    for label, _, res in results:
        if res is not None:
            print(f"  {label}  ->  --spline-p {res['p']:.10f}")


if __name__ == "__main__":
    main()
