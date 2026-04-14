"""
spline_comparison.py
====================
Interactive spline comparison: Python interpolating vs MATLAB spaps.

MATLAB side  — loaded from validation_diagnostics.mat exported by either:
                 ValidateOriginal.m      (for initial partition_data_*.mat shapes)
                 validate_optimizer_v2.m (for optimized shapes)

               Both scripts now export 'splineDense' [1000 x 3] — the real
               spaps curve evaluated at 1000 points — used for the spline fit plot.

Python side  — recomputed here from orderedNodes using CubicSpline (interpolating),
               the same spline type as the optimizer (NaturalCubicSpline).

Output
------
An interactive matplotlib window with three tabs / subplots that can be
zoomed, panned, and inspected freely.

Workflow
--------
1. Run ValidateOriginal.m or validate_optimizer_v2.m in MATLAB
   -> saves validation_diagnostics.mat
2. Edit DIAG_FILE and LABEL below
3. Run: python spline_comparison.py

Francisco Arriscado / FEUP / 2026
"""

import numpy as np
import scipy.io
import scipy.interpolate
import matplotlib
matplotlib.use("TkAgg")          # interactive backend — opens a real window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# =============================================================================
# USER INPUTS
# =============================================================================
DIAG_FILE = "validation_diagnosticsOPT.mat"
LABEL     = "optimized"

# =============================================================================
# PHYSICS CONSTANTS — must match optimizer and MATLAB scripts
# =============================================================================
OVERLAP         = 10
CONSTANT_STRESS = 90.0
THICKNESS       = 2.0
MIN_ROC         = 15.0
FLAT_TRANSITION = 1221.947188

def curvature_eq(roc):
    return 7.951858693686812 * roc**(-0.9409590693634183) + 0.9900990009786546

PY_COLOR   = "#2166AC"
MAT_COLOR  = "#D6604D"
NODE_COLOR = "#222222"


# =============================================================================
# LOAD MATLAB DIAGNOSTICS
# =============================================================================
def load_matlab(diag_file):
    if not os.path.exists(diag_file):
        raise FileNotFoundError(
            f"Cannot find: {diag_file}\n"
            "Run ValidateOriginal.m or validate_optimizer_v2.m in MATLAB first."
        )
    d = scipy.io.loadmat(diag_file)

    if "splineDense" not in d:
        raise KeyError(
            "'splineDense' not found in validation_diagnostics.mat.\n"
            "Update your MATLAB scripts to the latest version and re-run."
        )

    return dict(
        ordered_nodes = d["orderedNodes"].astype(float),
        RoC           = d["RoC"].ravel().astype(float),
        crush_stress  = d["crushStress"].ravel().astype(float),
        node_length   = d["nodeLength"].ravel().astype(float),
        force_node    = d["forceNode"].ravel().astype(float),
        force_total   = float(d["forceTotal"].ravel()[0]),
        spline_dense  = d["splineDense"].astype(float),   # [1000 x 3] real spaps curve
    )


# =============================================================================
# PYTHON PIPELINE
# =============================================================================
def python_pipeline(ordered_nodes):
    n         = len(ordered_nodes)
    nodes_ext = np.vstack([ordered_nodes[-OVERLAP:],
                            ordered_nodes,
                            ordered_nodes[:OVERLAP]])

    diffs = np.diff(nodes_ext, axis=0)
    ds    = np.linalg.norm(diffs, axis=1)
    s_ext = np.concatenate([[0.0], np.cumsum(ds)])
    s_ext /= s_ext[-1]

    # Sparse query (per node) — for RoC and force
    sQ_sparse = np.linspace(s_ext[OVERLAP], s_ext[-OVERLAP], n)

    # Dense query (1000 pts) — for spline fit visualisation
    sQ_dense  = np.linspace(s_ext[OVERLAP], s_ext[-OVERLAP], 1000)

    splines = []
    for dim in range(3):
        sp = scipy.interpolate.CubicSpline(
            s_ext, nodes_ext[:, dim], bc_type="not-a-knot"
        )
        splines.append(sp)

    # Dense curve for plotting
    spline_dense = np.stack([sp(sQ_dense) for sp in splines], axis=1)

    # Derivatives at node positions for RoC
    d1 = np.stack([sp(sQ_sparse, 1) for sp in splines], axis=1)
    d2 = np.stack([sp(sQ_sparse, 2) for sp in splines], axis=1)

    cross_p   = np.cross(d1, d2)
    num       = np.linalg.norm(cross_p, axis=1)
    den       = np.linalg.norm(d1, axis=1) ** 3 + 1e-12
    curvature = num / (den + 1e-12)
    roc       = np.where(curvature < 1e-12, FLAT_TRANSITION, 1.0 / curvature)
    roc       = np.clip(roc, MIN_ROC, FLAT_TRANSITION)

    crush_stress = CONSTANT_STRESS * curvature_eq(roc)

    edge_len    = np.linalg.norm(np.diff(ordered_nodes, axis=0), axis=1)
    edge_len    = np.append(edge_len,
                            np.linalg.norm(ordered_nodes[0] - ordered_nodes[-1]))
    node_length = (edge_len + np.roll(edge_len, 1)) / 2.0

    force_node  = crush_stress * node_length * THICKNESS
    force_total = force_node.sum()

    return dict(
        RoC          = roc,
        crush_stress = crush_stress,
        node_length  = node_length,
        force_node   = force_node,
        force_total  = force_total,
        spline_dense = spline_dense,
    )


# =============================================================================
# INTERACTIVE PLOT — three panels, fully zoomable / pannable
# =============================================================================
def show_interactive(nodes, py, mat, label):
    n   = len(nodes)
    arc = np.arange(n)
    diff_pct = (mat["force_total"] - py["force_total"]) / py["force_total"] * 100

    fig = plt.figure(
        figsize=(18, 7),
        num=f"Spline comparison — {label}"
    )
    fig.suptitle(
        f"{label}    |    "
        f"Python: {py['force_total']:.0f} N    "
        f"MATLAB (spaps): {mat['force_total']:.0f} N    "
        f"Δ = {diff_pct:+.1f}%",
        fontsize=12, fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)

    # ------------------------------------------------------------------
    # Panel 1 — Spline fit over nodes
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    # Real spaps dense curve from MATLAB
    mat_d = np.vstack([mat["spline_dense"], mat["spline_dense"][0]])
    ax1.plot(mat_d[:, 0], mat_d[:, 1],
             color=MAT_COLOR, lw=2.0, ls="--", zorder=3,
             label="spaps — MATLAB (real)")

    # Python dense curve
    py_d = np.vstack([py["spline_dense"], py["spline_dense"][0]])
    ax1.plot(py_d[:, 0], py_d[:, 1],
             color=PY_COLOR, lw=2.0, ls="-", zorder=3,
             label="Interpolating — Python")

    # Nodes with numbers
    cx, cy = nodes[:, 0].mean(), nodes[:, 1].mean()
    ax1.scatter(nodes[:, 0], nodes[:, 1],
                color=NODE_COLOR, s=18, zorder=5, label="Nodes")
    stride = max(1, n // 40)
    for i in range(0, n, stride):
        x, y   = nodes[i, 0], nodes[i, 1]
        dx, dy = x - cx, y - cy
        norm   = max(np.hypot(dx, dy), 1e-6)
        ax1.text(x + 5.0 * dx / norm, y + 5.0 * dy / norm,
                 str(i), fontsize=6, ha="center", va="center",
                 color="#444444", zorder=6)

    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title("Spline fit over nodes\n(zoom in on corners to see difference)")
    ax1.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 2 — Radius of Curvature per node
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.semilogy(arc, py["RoC"],  color=PY_COLOR,  lw=1.8, ls="-",
                 label=f"Python   (min = {MIN_ROC:.0f} mm)")
    ax2.semilogy(arc, mat["RoC"], color=MAT_COLOR, lw=1.8, ls="--",
                 label=f"MATLAB spaps   (min = {MIN_ROC:.0f} mm)")
    ax2.axhline(MIN_ROC,         color=PY_COLOR, ls=":",  lw=1.0, alpha=0.6)
    ax2.axhline(FLAT_TRANSITION, color="gray",   ls="--", lw=0.9, alpha=0.5,
                label=f"Flat transition ({FLAT_TRANSITION:.0f} mm)")

    ax2.set_xticks(np.arange(0, n, max(1, n // 20)))
    ax2.set_xlabel("Node index")
    ax2.set_ylabel("Radius of Curvature [mm]")
    ax2.set_title("Radius of Curvature per node")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_xlim(0, n - 1)

    # ------------------------------------------------------------------
    # Panel 3 — Per-node force contribution
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])

    ax3.plot(arc, py["force_node"],  color=PY_COLOR,  lw=1.8, ls="-",
             label=f"Python   (total = {py['force_total']:.0f} N)")
    ax3.plot(arc, mat["force_node"], color=MAT_COLOR, lw=1.8, ls="--",
             label=f"MATLAB spaps   (total = {mat['force_total']:.0f} N)")

    ax3.set_xticks(np.arange(0, n, max(1, n // 20)))
    ax3.set_xlabel("Node index")
    ax3.set_ylabel("Force per node [N]")
    ax3.set_title(
        f"Per-node force contribution\n"
        f"Δ = {mat['force_total'] - py['force_total']:+.0f} N  ({diff_pct:+.1f}%)"
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)
    ax3.set_xlim(0, n - 1)

    plt.tight_layout()
    print("\nInteractive window open.")
    print("Use the toolbar to zoom and pan.")
    print("Close the window to exit.")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"Loading: {DIAG_FILE}")
    mat   = load_matlab(DIAG_FILE)
    nodes = mat["ordered_nodes"]
    n     = len(nodes)
    print(f"  {n} nodes  |  MATLAB force: {mat['force_total']:.1f} N")

    print("Running Python pipeline...")
    py = python_pipeline(nodes)
    diff_pct = (mat["force_total"] - py["force_total"]) / py["force_total"] * 100
    print(f"  Python force: {py['force_total']:.1f} N  |  Δ = {diff_pct:+.2f}%")

    show_interactive(nodes, py, mat, LABEL)
