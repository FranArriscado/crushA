#
#  v20 changes from v19:
#    - Separate results folder for sweep tests.
#    - Removed internal coordinate rounding so gradients are preserved.
#    - Keeps the v20 gradient/output improvements while restoring the
#      original inward-only minP behavior for comparison runs.
#    - Objective restore now uses the post-step evaluated state.
#
#  v19 changes from v18:
#    - Removed boxSEA mode.
#    - Added boxMaxCF: maximise crush force within a bounding box.
#      Same objective as maxCF (-force) but the box penalty replaces
#      radial bounds. No radial constraints — the box IS the envelope.
#      Includes smoothness penalty and force floor.
#    - Added boxMinP: minimise perimeter within a bounding box.
#      Same objective as minP (+perimeter, force floor) but the box
#      penalty replaces radial bounds. Nodes can move freely inside
#      the box as long as force is maintained.
#    - Non-box modes (maxCF, minP, maxSEA) completely unchanged.
#
#  v18 changes from v17:
#    - maxCF now allows controlled inward and outward radial movement,
#      instead of outward-only motion. The inward allowance is kept small
#      by default so maxCF remains predominantly expansion-driven while
#      still being able to make limited local inward redistributions.
#    - maxCF now also enforces a force floor relative to the initial
#      force, so inward movement cannot be rewarded if it reduces force.
#    - Early stopping now restores the best objective state before final
#      evaluation, instead of reporting the last state visited.
#    - maxCF now includes a soft inward-motion penalty, so inward moves
#      remain possible but are discouraged unless they produce a clear
#      force benefit.
#
#  v17 changes from v16:
#    - Early stopping: stops when the objective hasn't improved by more
#      than conv_tol (relative) over conv_patience epochs. n_steps is
#      now a hard ceiling. Stopping reason logged in summary.
#    - Edge-length-weighted smoothness: smoothnessTerm now weights
#      second differences by local Cartesian edge length, so dense and
#      sparse regions are penalised equally per unit perimeter.
#      This fixes the asymmetric optimised shapes seen in v16.
#    - .mat output now includes force/perimeter/SEA initial and final.
#    - .txt output is summary-only (no node dump, no verbose prints).
#
#  v16 changes (retained):
#    - Smoothing cubic spline (Reinsch formulation) replacing
#      torchcubicspline.NaturalCubicSpline. Fully differentiable
#      via torch.linalg.solve. Parameter p controls smoothing.
#    - Four optimisation modes: maxCF, minP, maxSEA, boxMaxCF, boxMinP
#    - All physics, constraints, and penalty structure unchanged.
#
#  v15 — Added boxSEA mode: maximise SEA within a bounding box constraint.
#
#  v14 -Cleaned v13 verison,retains what works, removes unnecessary complexity.

#  v13 — Added maxSEA objective, replaced old flag system with mode string.
#
#    NEW (v13): maxSEA mode — maximise Specific Energy Absorption.
#              SEA ∝ Force / Perimeter (thickness & density constant).
#              Replaces the old minP-without-inward mode which always
#              converged to a circle (minimum-perimeter enclosure).
#              maxSEA is self-regulating: the optimizer cannot simply
#              expand (perimeter penalty in denominator) or shrink
#              (force drops in numerator).
#              The SEA objective is scaled by P₀ (initial perimeter) to
#              bring it to force-scale magnitude for proper gradient
#              balance against the smoothness penalty.
#
#    FIX 11 (v13): Smoothness normalised by its initial value.
#              At epoch 0, smooth/smooth₀ = 1.0 regardless of geometry
#              or node count. This makes λ_smooth geometry-independent —
#              the same λ works for all shapes without retuning.
#
#    FIX 12 (v13): Cartesian path smoothness penalty.
#              The radial smoothnessTerm(r) missed corrugations: small
#              bumps on flat sides barely change r but create huge local
#              curvature in the XY path, gaming the RoC calculation.
#              New penalty: Δ²(X,Y,Z) = node[i+1] - 2*node[i] + node[i-1]
#              directly penalises path-level wiggles in all modes.
#
#    REFACTOR (v13): Replaced two boolean flags (minPerimeterFlag,
#              maxCrushForceFlag, inwardSecFlag) with a single string:
#                 run_mode ∈ {'maxCF', 'minP', 'maxSEA'}

#  v12 — Added maxSEA objective, replaced old flag system with mode string.
#
#    NEW (v11): maxSEA mode — maximise Specific Energy Absorption.
#              SEA ∝ Force / Perimeter (thickness & density constant).
#              Replaces the old minP-without-inward mode which always
#              converged to a circle (minimum-perimeter enclosure).
#              maxSEA is self-regulating: the optimizer cannot simply
#              expand (perimeter penalty in denominator) or shrink
#              (force drops in numerator).
#              The SEA objective is scaled by P₀ (initial perimeter) to
#              bring it to force-scale magnitude for proper gradient
#              balance against the smoothness penalty.
#
#    FIX 11 (v11): Smoothness normalised by its initial value.
#              At epoch 0, smooth/smooth₀ = 1.0 regardless of geometry
#              or node count. This makes λ_smooth geometry-independent —
#              the same λ works for all shapes without retuning.
#
#    REFACTOR (v11): Replaced two boolean flags (minPerimeterFlag,
#              maxCrushForceFlag, inwardSecFlag) with a single string:
#                 run_mode ∈ {'maxCF', 'minP', 'maxSEA'}

#  v11 — Added maxSEA objective, replaced old flag system with mode string.
#
#    NEW (v11): maxSEA mode — maximise Specific Energy Absorption.
#              SEA ∝ Force / Perimeter (thickness & density constant).
#              Replaces the old minP-without-inward mode which always
#              converged to a circle (minimum-perimeter enclosure).
#              maxSEA is self-regulating: the optimizer cannot simply
#              expand (perimeter penalty in denominator) or shrink
#              (force drops in numerator).
#
#    REFACTOR (v11): Replaced two boolean flags (minPerimeterFlag,
#              maxCrushForceFlag, inwardSecFlag) with a single string:
#                 run_mode ∈ {'maxCF', 'minP', 'maxSEA'}

#  Patched version v10-Just cleaned code from v9

#  Patched version v9 — v8 + maxCF constraint / scaling fixes:
#
#    FIX 8 (v9): Added inward radial constraint for maxCF mode.
#                Previously maxCF had no constraint preventing nodes from
#                shrinking inward, causing the smoothness penalty to
#                dominate and collapse non-square shapes toward circles.
#                New: radialDist_inward = sum(relu(initial_r - r)^2)
#
#    FIX 9 (v9): Smoothness penalty normalised per-node (divided by N).
#                With lambda=1e10 and N=154 nodes, the raw smoothness
#                term was ~10^6-10^8x larger than force, making the force
#                gradient invisible. Per-node normalisation keeps the
#                penalty scale consistent across shapes with different
#                node counts. Lambda rebalanced accordingly.
#
#    REFACTOR (v9): Loss computation extracted into compute_loss() for
#                   clarity. File naming simplified — removed _FlagTrue/_
#                   FlagFalse from output .txt and .mat filenames.

#  Patched version v8-Just cleaned code from v7

#  Patched version v7 — v6 + MATLAB-alignment fixes:
#
#    FIX 5 (v7): sQuery uses linspace instead of actual node positions,
#                matching MATLAB curvatureSpline: sQuery = linspace(
#                s(overlapCount+1), s(end-overlapCount), nOriginal)
#
#    FIX 6 (v7): overlapCount changed from 7 to 10, matching MATLAB
#                curvatureSpline default.
#
#    FIX 7 (v7): RoC upper-clamped to flat_transition (1221.95 mm for
#                Carbon), matching MATLAB:
#                  RoCMapped = min(max(RoCMapped, 5), flatTransition)
#                Previously Python only had clamp_min(5).
#
#    NOTE:  The spline TYPE differs: Python uses an interpolating cubic
#           spline (NaturalCubicSpline) while MATLAB uses spaps (smoothing
#           spline with tol=1e-2). This causes ~3% residual discrepancy
#           that cannot be eliminated without a differentiable smoothing
#           spline implementation in PyTorch. 
#
#  Inherited from v6:
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 2: (SUPERSEDED by FIX 5) spline query positions
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    FIX 4: per-node edge-length weighted force integration
#    SAFE 1-4, PERF: unchanged from v

#  Patched version v6 — v4 + physics-correct force formula:
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    FIX 4 (v6): force formula replaced with per-node edge-length weighted
#                integration matching MATLAB layersProp:
#                  forceTotal = sum(crushStress[j] * nodeLength[j] * thickness)
#                where nodeLength[j] = (edgeLength[j-1] + edgeLength[j]) / 2
#                Previously: thickness * perimeter * mean(crushStress)
#                The old formula over-weighted high-curvature corner nodes
#                because mean() treats all nodes equally regardless of how
#                much perimeter they represent. On non-uniform geometries
#                this caused the optimizer to game the formula by creating
#                sharp corners, validated at 42% error vs MATLAB on optimized
#                geometry (0.21% error on uniform square).
#    SAFE 1: epsilon guards in RoCCalc and chordLength (prevent div-by-zero)
#    SAFE 2: gradient clipping (prevent exploding gradients)
#    SAFE 3: NaN detection with lightweight rollback (clone r only, periodic)
#    SAFE 4: cosine-annealing LR scheduler (prevents late-stage oscillation)
#    PERF: replaced deepcopy(state_dict) every improvement with
#          r.data.clone() every snapshot_interval steps (~30 clones vs ~15k)
#    ADDED: runtime timer + timestamped log file for convergence studies

#  v5 was skipped in development by mistake

#  Patched version v4
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    SAFE 1: epsilon guards in RoCCalc and chordLength (prevent div-by-zero)
#    SAFE 2: gradient clipping (prevent exploding gradients)
#    SAFE 3: NaN detection with lightweight rollback (clone r only, periodic)
#    SAFE 4: cosine-annealing LR scheduler (prevents late-stage oscillation)
#    PERF: replaced deepcopy(state_dict) every improvement with
#          r.data.clone() every snapshot_interval steps (~30 clones vs ~15k)
#    ADDED: runtime timer + timestamped log file for convergence studies

#  Patched version v3 — v2 fixes + performance-optimised safeguards:
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    SAFE 1: epsilon guards in RoCCalc and chordLength (prevent div-by-zero)
#    SAFE 2: gradient clipping (prevent exploding gradients)
#    SAFE 3: NaN detection with lightweight rollback (clone r only, periodic)
#    SAFE 4: cosine-annealing LR scheduler (prevents late-stage oscillation)
#    PERF: replaced deepcopy(state_dict) every improvement with
#          r.data.clone() every snapshot_interval steps (~30 clones vs ~15k)
#    ADDED: runtime timer + timestamped log file for convergence studies

#  Patched version v2 — all original fixes plus numerical safeguards:
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    SAFE 1: epsilon guards in RoCCalc and chordLength (prevent div-by-zero)
#    SAFE 2: gradient clipping (prevent exploding gradients)
#    SAFE 3: NaN detection with state rollback and early stopping
#    SAFE 4: cosine-annealing LR scheduler (prevents late-stage oscillation)
#    ADDED: runtime timer for convergence studies

#  Patched version v1 — fixes applied to resolve closure discontinuity:
#    FIX 1: smoothnessTerm now wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions (not linspace)
#    FIX 3: RoCCalc no longer reindexes by orderedIndices (derivatives are
#           already in ordered-node sequence)
#    ADDED: runtime timer for convergence studies