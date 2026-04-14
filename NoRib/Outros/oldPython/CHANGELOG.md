## Optimisation Module — Changelog & Issue Tracker

**Project:** Geometry Optimisation Module for Composite Cross-Section Crush Analysis  
**Continuation by:** Francisco Arriscado, FEUP 2026 
**Date started:** March 2026  
**Base file:** `optimizationModule.py`


---

### 1. Issues identified in the original (`optimizationModule.py`)

The original script successfully demonstrated the optimisation concept — maximising crush force or minimising perimeter of composite cross-sections via PyTorch and the Adam optimiser. However, as documented in the thesis (Chapter 7, Section 7.5), every optimised shape exhibited a visible **discontinuity at the junction between the first and last nodes** of the closed cross-section. The thesis attributed this to "numerical inconsistencies in the RoC calculation near spline endpoints."

After a thorough code review, **three distinct bugs** were identified as contributing to this discontinuity:

**Bug 1 (CRITICAL) — Smoothness penalty does not wrap around the closure**

- **Location:** `smoothnessTerm()` function
- **Problem:** The second-difference computation runs sequentially from node 0 to node N-1. It completely ignores the transition from the last node back to the first. For a closed loop, the smoothness at the closure is equally important as at any interior point. Without this penalty, the optimiser has no incentive to maintain smoothness at the first/last node junction, so it freely creates a sharp kink there.
- **Impact:** This is the primary cause of the visible discontinuity. Interior nodes are held smooth by neighbours on both sides, but the first and last nodes only feel the penalty from one side.

**Bug 2 (MODERATE) — Spline query points misaligned with actual node positions**

- **Location:** `splines()` function, the `sQuery` computation
- **Problem:** `torch.linspace(s[overlapCount], s[-(overlapCount+1)], nOriginal)` creates N evenly-spaced points in the parameter domain. But the actual nodes are NOT evenly spaced along the chord length — their `s` values depend on their 3D positions. This means the RoC is evaluated at slightly wrong locations.
- **Impact:** For interior nodes the error is small. At the boundaries (first and last query points), the misalignment can be significant, particularly for geometries with non-uniform node spacing (e.g., the square cross-section where corner spacing differs from edge spacing).

**Bug 3 (MODERATE) — Phantom RoC reindexing scrambles gradients**

- **Location:** `RoCCalc()` function, the line `RoC = radius[orderedIndices]`
- **Problem:** `radius` is computed at ordered-node positions (radius[i] = RoC of the i-th node in the path). But `orderedIndices` contains original MATLAB node indices (e.g., [2, 1, 0, 27, ...]). This permutation means node 0 in the path gets the RoC of whatever node index 2 corresponds to in the positional array. Since `crushStress.mean()` is used in the forward pass, the scalar output is numerically identical (mean of a permuted array = mean of the original). However, the **gradient flow is scrambled** — each node's radial distance receives the gradient signal intended for a different node.
- **Impact:** The optimiser updates each node based on another node's RoC sensitivity. This creates confused gradient signals especially near the closure region where orderedIndices jump non-sequentially.


---

### 2. Patched version v1 (`optimizationModule_patched.py`)

**Purpose:** Apply the three core bug fixes to resolve the closure discontinuity.

**Changes made:**

| Change | Function | Description |
|--------|----------|-------------|
| FIX 1 | `smoothnessTerm()` | Radial distances are now wrapped with `torch.cat([r[-1:], r, r[:1]])` before computing second differences. The penalty now covers the first-to-last node transition, enforcing smoothness at the closure just like everywhere else. |
| FIX 2 | `splines()` | `sQuery` changed from `torch.linspace(...)` to `s[overlapCount : overlapCount + nOriginal]`. The spline is now evaluated at the exact chord-length parameter values of each original node. |
| FIX 3 | `RoCCalc()` | Removed the `orderedIndices` parameter and the `radius[orderedIndices]` reindexing line. The function now returns `radius` directly since the derivatives are already in ordered-node sequence. The forward pass call was updated accordingly. |
| ADDED | Main script | Runtime timer using `time.perf_counter()` and a comprehensive run summary block printed at the end (model config, initial/final force and perimeter with percentage changes, wall-clock time, time per step). |

**Test result:**

- The fixes clearly engaged — the optimiser pushed significantly harder than before (force up to ~174 kN vs the original ~147 kN plateau, perimeter down to ~548 vs ~580).
- However, the run crashed to NaN at approximately epoch 13799. The optimizer was oscillating between two states in the final epochs before diverging.
- No plot was produced because the final model state was NaN.


---

### 3. Patched version v2 (`optimizationModule_patched_v2.py`)

**Purpose:** Add numerical safeguards to prevent the NaN crash observed in v1, while retaining all three bug fixes.

**Root cause of NaN:** The more effective gradient signals from the fixes caused the optimizer to push nodes into tighter configurations. Eventually, either a chord-length segment collapsed to near-zero (division in `s = s / s[-1]`), or the spline first-derivative norm cubed hit zero (division in `RoCCalc`). One NaN propagated through the computational graph and corrupted Adam's momentum buffers, making recovery impossible. The late-stage oscillation was also a symptom of lr=0.1 being too aggressive once the optimiser was near a solution.

**Changes made (on top of v1):**

| Change | Location | Description |
|--------|----------|-------------|
| SAFE 1a | `chordLength()` | Added `+ 1e-12` to the normalisation denominator: `s / (s[-1] + 1e-12)`. Prevents div-by-zero if total chord length collapses. |
| SAFE 1b | `RoCCalc()` | Added `+ 1e-12` to both `den` (first derivative norm cubed) and the curvature-to-radius inversion. Prevents NaN when the spline has a degenerate segment. |
| SAFE 2 | Optimisation loop | Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()`. Caps gradient magnitude to prevent parameter explosions from a single bad step. |
| SAFE 3 | Optimisation loop | NaN/Inf detection with state rollback. Every epoch, if the loss is finite and better than the best seen so far, the model state is deep-copied. If NaN is detected, the model is rolled back to the best snapshot and the loop breaks cleanly, producing a valid plot and summary. |
| SAFE 4 | Optimiser setup | Added `CosineAnnealingLR` scheduler that decays the learning rate from `learning_rate` (0.1) down to `eta_min` (1e-4) over the full run. Prevents the late-stage oscillations that triggered NaN in v1. |
| ADDED | Main script | Timestamped `.txt` log file via `TeeLogger` class. All `print` output is mirrored to both the console and a file named `optimisation_log_YYYY-MM-DD_HH-MM-SS.txt`. |
| ADDED | Summary | Additional fields: steps completed (vs requested), best epoch, LR schedule info, grad clip norm, early-stop flag. |

**Test result:**

- The run completed all 15000 steps without NaN.
- The closure discontinuity was resolved.
- However, wall-clock time increased by ~19% (565s vs 475s for the original). The overhead was traced to `copy.deepcopy(model.state_dict())` being called on every epoch where the loss improved — potentially 10,000+ full deep copies.


---

### 4. Patched version v3 (`optimizationModule_patched_v3.py`)

**Purpose:** Optimise runtime performance while keeping all fixes and safeguards from v2.

**Changes made (on top of v2):**

| Change | Location | Description |
|--------|----------|-------------|
| PERF 1 | State snapshot | Replaced `copy.deepcopy(model.state_dict())` with `model.r.data.clone()`. Since `r` is the only learnable parameter, cloning the full state dict was wasteful — a single tensor clone is orders of magnitude cheaper. |
| PERF 2 | Snapshot frequency | Snapshots now only occur every `snapshot_interval` (default: 500) epochs when the loss has improved, instead of on every improvement. Reduces ~10,000+ copies to ~30 lightweight clones. |
| PERF 3 | NaN rollback | Changed from `model.load_state_dict(best_state)` to `model.r.data.copy_(best_r)` to match the lightweight snapshot approach. |
| CLEANUP | Imports | Removed `import copy` (no longer needed). |
| CLEANUP | Print statements | Replaced Unicode arrows and dashes (`→`, `—`) with ASCII equivalents (`->`, `-`) in all print statements to prevent `UnicodeEncodeError` on Windows systems using cp1252 encoding. |
| FIX | `TeeLogger` | Log file opened with `encoding='utf-8'` to handle any Unicode that might appear in node coordinate prints. |

**Test result:**

- Pending (runtime comparison in progress).
- Expected to bring wall-clock time back close to the original ~475s while retaining all correctness fixes and safety features.


---

### Notes

**Background on the problem (from thesis Chapter 5):**

The optimisation module extends a MATLAB-based analytical crush prediction tool for composite structures (e.g., Formula 1 Side Impact Structures). Cross-sections are represented by ordered nodes in spherical coordinates, with the radial distance `r` as the learnable parameter. The module uses PyTorch's automatic differentiation and the Adam optimiser to either maximise crush force or minimise perimeter (mass proxy) while maintaining crush performance.

**Remaining known limitation:**

The `NaturalCubicSpline` from `torchcubicspline` imposes zero second derivative at endpoints (natural boundary conditions), which is mathematically incorrect for a closed curve (should be periodic). The overlap technique mitigates this but does not eliminate it entirely. A future improvement would be to use a periodic spline formulation. With the current fixes applied, the overlap technique appears sufficient for practical results.

**Test configurations used:**

- Square cross-section (200mm x 100mm), max crush force, 15000 steps, lr=0.1
- Stadium cross-section (100mm x 200mm), min perimeter (inward), 15000 steps, lr=0.1
- Thickness: 2mm, constant crush stress: 90 MPa, overlap count: 7
