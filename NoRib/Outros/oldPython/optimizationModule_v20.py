
#                  optimizationModule_v20.py
#  DEPENDENCIES: torch, scipy, matplotlib, numpy (NO torchcubicspline)
# Francisco Arriscado, FEUP, 2026
#==========================================================================

script_version = 20

import argparse
from typing import Tuple
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
import sys
import time
import numpy as np
import re
from datetime import datetime
from pathlib import Path

# ---- Project paths (config.py at project root) ---- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

RESULTS_OUTROS = Path(__file__).resolve().parent.parent / "results" / "NoRib"
RESULTS_OUTROS.mkdir(parents=True, exist_ok=True)

#==========================================================================
#==========================================================================
#                          User Inputs
#==========================================================================
#==========================================================================

# ---- Geometry / material defaults ---- #
thickness = 2                       # [mm]
dataFileName = 'square_133x133.mat'
constantCrushStress = 90            # [MPa]
flat_transition = 1221.95           # [mm] from DataCurvatureCarbon.xlsx
min_roc = 15.0                      # [mm] minimum RoC clamp (manufacturing constraint)

# ---- Optimisation mode ---- #
# 'maxCF'     -- maximise crush force (outward expansion, radially bounded)
# 'minP'      -- minimise perimeter while preserving force (inward only)
# 'maxSEA'    -- maximise specific energy absorption (Force / Perimeter)
# 'boxMaxCF'  -- maximise crush force inside a bounding box (design envelope)
run_mode = 'maxCF'

# ---- Bounding box (boxMaxCF mode only) ---- #
box_width = 200                      # [mm] box extent in X
box_height = 200                     # [mm] box extent in Y


# ---- Penalty weights ---- #
lambda_crushForce = 1e6
lambda_radialDist = 1e6
lambda_box = 1e10

# ---- Smoothness weight ratio per mode ---- #
smooth_weight_ratio = {
    'maxCF':     0.35,
    'minP':      0.2,
    'maxSEA':    2.0,
    'boxMaxCF':  4,
}

# ---- Geometry movement limits ---- #
max_expansion_ratio = 1.3
# ---- Smoothing spline parameter ---- #
spline_p = None   # None = auto (csaps heuristic)

#==========================================================================
#==========================================================================
#                         Default config for the optimizer
#==========================================================================
#==========================================================================
# ---- Optimizer / training defaults ---- #
n_steps = 100000                    # hard ceiling (early stopping usually triggers first)
learning_rate = 0.1                 # it is not recommended to change this parameter!
grad_clip_norm = 1.0
overlapCount = 10                   # same as matlab code

# ---- Early stopping ---- #
# Stop when the tracked objective hasn't improved by more than conv_tol
conv_patience = 1000                # epochs to wait for improvement
conv_tol = 1e-3                     # relative improvement threshold (0.1%)

#==========================================================================
#==========================================================================
#                    SMOOTHING CUBIC SPLINE (Reinsch form)
#==========================================================================
#==========================================================================

def _build_spline_matrices(t: torch.Tensor):
    """
    Build the Q and R matrices for the Reinsch smoothing spline.

    Given n knot positions t_0 < t_1 < ... < t_{n-1}:
      - h_i = t_{i+1} - t_i                       (n-1 values)
      - Q: (n x m) matrix, m = n-2 interior knots
      - R: (m x m) symmetric tridiagonal matrix

    These satisfy the natural cubic spline relation: Q^T g = R sigma
    where g are spline values at knots and sigma are 2nd derivatives
    at interior knots.
    """
    n = t.shape[0]
    m = n - 2
    h = t[1:] - t[:-1]  # (n-1,)

    # Q matrix: n x m
    Q = torch.zeros(n, m, dtype=t.dtype, device=t.device)
    j = torch.arange(m, device=t.device)
    Q[j, j]     =  1.0 / h[:-1]
    Q[j + 1, j] = -(1.0 / h[:-1] + 1.0 / h[1:])
    Q[j + 2, j] =  1.0 / h[1:]

    # R matrix: m x m symmetric tridiagonal
    R = torch.zeros(m, m, dtype=t.dtype, device=t.device)
    R[j, j] = (h[:-1] + h[1:]) / 3.0
    if m > 1:
        k = torch.arange(m - 1, device=t.device)
        R[k, k + 1] = h[1:-1] / 6.0
        R[k + 1, k] = h[1:-1] / 6.0

    return Q, R, h


def _default_smooth_p(h: torch.Tensor) -> float:
    """MATLAB csaps default: p = 1 / (1 + h_avg^3 / 6)."""
    h_avg = h.mean().item()
    return 1.0 / (1.0 + h_avg**3 / 6.0)


def fit_smoothing_spline(
    t: torch.Tensor,
    y: torch.Tensor,
    p: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit a smoothing cubic spline (Reinsch formulation).

    Minimises:  p * sum(y_i - g_i)^2  +  (1-p) * integral(g''^2 dt)

    Parameters
    ----------
    t : (n,) knot positions, strictly increasing
    y : (n, d) data values (d = 3 for XYZ coordinates)
    p : smoothing parameter in (0, 1].
        p = 1  => interpolation (g = y exactly)
        p -> 0 => maximally smooth (straight line)
        None   => use MATLAB csaps default heuristic

    Returns
    -------
    g : (n, d) smoothed values at knots
    M : (n, d) second derivatives at all knots (natural BC: M[0]=M[-1]=0)
    h : (n-1,) knot spacings
    """
    n = t.shape[0]

    Q, R, h = _build_spline_matrices(t)

    if p is None:
        p = _default_smooth_p(h)

    # Clamp p to avoid division by zero
    p = max(min(p, 1.0 - 1e-15), 1e-15)

    # Reinsch system: (p*R + (1-p)*Q^T Q) sigma = p * Q^T y
    QtQ = Q.t() @ Q                        # (m, m)
    A   = p * R + (1.0 - p) * QtQ          # (m, m)
    rhs = p * (Q.t() @ y)                  # (m, d)

    sigma = torch.linalg.solve(A, rhs)     # (m, d) — differentiable

    # Smoothed values: g = y - ((1-p)/p) * Q @ sigma
    g = y - ((1.0 - p) / p) * (Q @ sigma)  # (n, d)

    # Full second-derivative vector with natural BCs
    M = torch.zeros(n, y.shape[1], dtype=y.dtype, device=y.device)
    M[1:-1] = sigma

    return g, M, h


def eval_spline_derivatives(
    t: torch.Tensor,
    g: torch.Tensor,
    M: torch.Tensor,
    h: torch.Tensor,
    tq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate first and second derivatives of the smoothing cubic spline
    at query points tq.

    The spline on interval [t_i, t_{i+1}] is:
      S(x) = M_i*(t_{i+1}-x)^3/(6*h_i) + M_{i+1}*(x-t_i)^3/(6*h_i)
           + (g_i - M_i*h_i^2/6)*(t_{i+1}-x)/h_i
           + (g_{i+1} - M_{i+1}*h_i^2/6)*(x-t_i)/h_i

    Parameters
    ----------
    t  : (n,)    knot positions
    g  : (n, d)  smoothed values at knots
    M  : (n, d)  second derivatives at knots
    h  : (n-1,)  knot spacings
    tq : (nq,)   query points (must lie within [t[0], t[-1]])

    Returns
    -------
    der1 : (nq, d) first derivatives at query points
    der2 : (nq, d) second derivatives at query points
    """
    n = t.shape[0]

    # Find interval index: t[idx] <= tq < t[idx+1]
    idx = torch.searchsorted(t[1:].contiguous(), tq.contiguous())
    idx = idx.clamp(0, n - 2)

    hi       = h[idx]                          # (nq,)
    dt_right = (t[idx + 1] - tq).unsqueeze(1)  # (nq, 1)
    dt_left  = (tq - t[idx]).unsqueeze(1)       # (nq, 1)
    hi_d     = hi.unsqueeze(1)                  # (nq, 1)

    Mi  = M[idx]        # (nq, d)
    Mi1 = M[idx + 1]    # (nq, d)
    gi  = g[idx]        # (nq, d)
    gi1 = g[idx + 1]    # (nq, d)

    # First derivative of the cubic spline
    der1 = (-Mi  * dt_right**2 / (2.0 * hi_d)
            + Mi1 * dt_left**2  / (2.0 * hi_d)
            + (gi1 - gi) / hi_d
            - (Mi1 - Mi) * hi_d / 6.0)

    # Second derivative of the cubic spline
    der2 = Mi * dt_right / hi_d + Mi1 * dt_left / hi_d

    return der1, der2


#==========================================================================
#==========================================================================
#                         OPTIMIZER MODEL
#==========================================================================
#==========================================================================

class crushOptimizerModel(nn.Module):

    def __init__(self,
        orderedNodes: torch.Tensor,
        orderedIndices: torch.Tensor,
        edges: torch.Tensor,
        constantCrushStress,
        overlapCount,
        thickness,
        flat_transition,
        min_roc,
        spline_p,
        ):

        super().__init__()
        self.centroid = orderedNodes.mean(dim=0, keepdim=True)
        r, theta, phi = Cartesian2Spherical(orderedNodes - self.centroid)

        self.r = nn.Parameter(r)
        self.register_buffer('initial_r', r.clone())
        self.register_buffer('theta', theta)
        self.register_buffer('phi', phi)

        self.orderedIndices = orderedIndices
        self.edges = edges
        self.constantCrushStress = constantCrushStress
        self.overlapCount = overlapCount
        self.thickness = thickness
        self.flat_transition = flat_transition
        self.min_roc = min_roc
        self.spline_p = spline_p

    def forward(self):
        # Reconstruct Cartesian coordinates with fixed angles
        X = self.r * torch.sin(self.theta) * torch.cos(self.phi)
        Y = self.r * torch.sin(self.theta) * torch.sin(self.phi)
        Z = self.r * torch.cos(self.theta)
        orderedNodes = torch.stack([X, Y, Z], dim=1) + self.centroid
        # Extend nodes
        nodesExt = nodesCoords(self.overlapCount, orderedNodes)
        # Chord length
        s = chordLength(nodesExt)
        # Splines — now using smoothing cubic spline
        splineDer1, splineDer2 = splines(
            nodesExt, orderedNodes, s, self.overlapCount, self.spline_p)
        # RoC calculation
        RoC = RoCCalc(splineDer1, splineDer2)
        RoC = RoC.clamp(min=self.min_roc, max=self.flat_transition)
        # Crush stress calculation
        crushStress = self.constantCrushStress * CurvatureEq(RoC)
        # Perimeter and edge lengths calculation
        perimeter, edgeLengths = perimeterCalc(orderedNodes)
        # Per-node edge-length weighted force integration
        nodeLength = (edgeLengths + torch.roll(edgeLengths, 1, dims=0)) / 2
        forceTotal = (crushStress * nodeLength * self.thickness).sum()
        return forceTotal, perimeter

#==========================================================================
# ---- Load Data ---- #
def loadData(name):
    partitionData = scipy.io.loadmat(name)
    nodes = partitionData['nodesCoords']
    edges = partitionData['connectivity']
    edges -= 1
    return partitionData

#==========================================================================
# ---- Find ordered path ---- #
def orderNodes(nodes, edges):
    ordered = [edges[0][0].item()]
    next_node = edges[0][1].item()
    ordered.append(next_node)
    used_edges = {0}

    while len(ordered) < len(edges) + 1:
        for idx, (a, b) in enumerate(edges):
            if idx in used_edges:
                continue
            if a.item() == next_node:
                next_node = b.item()
                ordered.append(next_node)
                used_edges.add(idx)
                break
            elif b.item() == next_node:
                next_node = a.item()
                ordered.append(next_node)
                used_edges.add(idx)
                break

    orderedIndices = ordered[:-1]
    orderedNodes = nodes[orderedIndices]

    return orderedIndices, orderedNodes

#==========================================================================
# ---- Convert from cartesian to spherical coordinates ---- #
def Cartesian2Spherical(xyz):
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    phi = torch.atan2(Y, X)
    theta = torch.acos(Z / r)
    return r, theta, phi

#==========================================================================
# ---- Convert from spherical to cartesian coordinates ---- #
def Spherical2Cartesian(r, theta, phi):
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    return torch.stack([X, Y, Z], dim=1)

#==========================================================================
# ---- Extract nodes coordinates ---- #
def nodesCoords(overlapCount, orderedNodes) -> torch.Tensor:
    X = orderedNodes[:, 0]
    Y = orderedNodes[:, 1]
    Z = orderedNodes[:, 2]
    X_ext = torch.cat([X[-overlapCount:], X, X[:overlapCount]], dim=0)
    Y_ext = torch.cat([Y[-overlapCount:], Y, Y[:overlapCount]], dim=0)
    Z_ext = torch.cat([Z[-overlapCount:], Z, Z[:overlapCount]], dim=0)
    nodesExt = torch.stack([X_ext, Y_ext, Z_ext], dim=1)
    return nodesExt

#==========================================================================
# ---- Parametrize by chord length ---- #
def chordLength(nodesExt) -> torch.Tensor:
    diffs = nodesExt[1:] - nodesExt[:-1]
    ds = torch.norm(diffs, dim=1)
    s = torch.zeros(len(nodesExt), device=nodesExt.device)
    s[1:] = torch.cumsum(ds, dim=0)
    s = s / (s[-1] + 1e-12)
    return s

#==========================================================================
# ---- Fit smoothing splines and evaluate derivatives ---- #
def splines(nodesExt, orderedNodes, s, overlapCount, spline_p
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a smoothing cubic spline to the extended node coordinates and
    evaluate first and second derivatives at the original-node chord
    positions.

    Sampling exactly at the original-node parameters keeps the closed
    loop periodic and avoids giving the start/end seam a special role.
    Reconstructing the query with linspace can bias highly symmetric
    shapes such as squares because the seam then lands slightly off the
    original node locations.
    """
    nOriginal = orderedNodes.shape[0]

    # Fit smoothing spline to extended nodes
    g, M, h = fit_smoothing_spline(s, nodesExt, p=spline_p)

    # Evaluate at the original (non-overlapped) node parameters.
    sQuery = s[overlapCount:-overlapCount]

    # Evaluate derivatives at query points
    splineDer1, splineDer2 = eval_spline_derivatives(s, g, M, h, sQuery)

    return splineDer1, splineDer2

#==========================================================================
# ---- Compute RoC ---- #
def RoCCalc(splineDer1, splineDer2) -> torch.Tensor:
    cross = torch.cross(splineDer1, splineDer2, dim=1)
    num = torch.norm(cross, dim=1)
    den = torch.norm(splineDer1, dim=1) ** 3 + 1e-12
    curvature = num / den
    radius = 1.0 / (curvature + 1e-12)
    return radius

#==========================================================================
# ---- Crush stress dependence - Curvature equation ---- #
def CurvatureEq(x: torch.Tensor) -> torch.Tensor:
    return 7.95 * torch.pow(x, -0.94) + 0.99

#==========================================================================
# ---- Compute Perimeter and Edge Lengths ---- #
def perimeterCalc(orderedNodes):
    nodes_shifted = torch.roll(orderedNodes, -1, dims=0)
    diffs = nodes_shifted - orderedNodes
    edgeLengths = torch.norm(diffs, dim=1)
    return torch.sum(edgeLengths), edgeLengths

#==========================================================================
# ---- Plot ---- #
def plot_initial_and_final(initial_nodes, final_nodes,
                           save_path="optim_result.pdf", box_dims=None):
    def _configure_plot_style(use_tex: bool):
        style = {"font.size": 11, "text.usetex": use_tex}
        if use_tex:
            style.update({
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
        else:
            style.update({
                "font.family": "DejaVu Serif",
                "font.serif": ["DejaVu Serif"],
            })
        plt.rcParams.update(style)

    def _save_plot():
        fig, ax = plt.subplots(figsize=(4.72, 2.25))
        try:
            ax.plot(initial_nodes[:, 0], initial_nodes[:, 1],
                    marker='o', markersize=4, linewidth=0.75,
                    color=(0, 0.447, 0.698), label='Initial')
            ax.plot(final_nodes[:, 0], final_nodes[:, 1],
                    marker='o', markersize=4, linewidth=0.75,
                    color=(0.902, 0.624, 0), label='Optimised')
            if box_dims is not None:
                hw, hh = box_dims[0] / 2, box_dims[1] / 2
                rect = plt.Rectangle((-hw, -hh), box_dims[0], box_dims[1],
                    linewidth=1.0, edgecolor='red', facecolor='none',
                    linestyle='--', label='Box constraint')
                ax.add_patch(rect)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            plt.subplots_adjust(bottom=0.25)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                      ncol=3 if box_dims else 2, frameon=False)
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        _configure_plot_style(use_tex=True)
        _save_plot()
    except Exception as exc:
        print(f"Plot save with LaTeX failed ({exc}); retrying without LaTeX.")
        _configure_plot_style(use_tex=False)
        _save_plot()

#==========================================================================
# ---- Smoothness Penalty (edge-length weighted) ---- #
def smoothnessTerm(r, theta, phi, centroid):
    """
    Second differences of radial distances (closed loop), weighted by
    local Cartesian edge length so that dense and sparse regions are
    penalised equally per unit perimeter.

    Without weighting, regions with closely-spaced nodes accumulate more
    penalty terms per unit arc length, constraining them more heavily
    and causing asymmetric optimised shapes.
    """
    # Reconstruct Cartesian coordinates to get true edge lengths
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    nodes = torch.stack([X, Y, Z], dim=1) + centroid

    # Edge lengths between consecutive nodes (closed loop)
    diffs = torch.roll(nodes, -1, dims=0) - nodes
    edge_len = torch.norm(diffs, dim=1)  # (N,)

    # nodeLength: average of left and right edge for each node
    node_len = (edge_len + torch.roll(edge_len, 1, dims=0)) / 2.0
    node_len = node_len.clamp(min=1e-8)

    # Weight = node_len / mean(node_len): sparse regions (large spacing)
    # get higher weight per node, dense regions get lower weight.
    # Weights sum to N, preserving total penalty magnitude.
    weights = node_len / node_len.mean()

    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    dr = r_wrapped[1:] - r_wrapped[:-1]
    curvature = dr[1:] - dr[:-1]
    return torch.sum(weights * curvature ** 2)

#==========================================================================
# ---- Loss Function ---- #
def _box_penalty(model, box_dims):
    """Quadratic penalty for nodes outside the bounding box."""
    X = model.r * torch.sin(model.theta) * torch.cos(model.phi)
    Y = model.r * torch.sin(model.theta) * torch.sin(model.phi)
    half_w = box_dims[0] / 2.0
    half_h = box_dims[1] / 2.0
    return (torch.sum(torch.relu(X - half_w) ** 2)
          + torch.sum(torch.relu(-half_w - X) ** 2)
          + torch.sum(torch.relu(Y - half_h) ** 2)
          + torch.sum(torch.relu(-half_h - Y) ** 2))


def compute_loss(mode, force_total, perimeter, model,
                 force_initial, smoothness_initial,
                 lambdas, max_expansion_ratio,
                 box_dims=None):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF'     -- maximise crush force (outward, radially bounded).
        'minP'      -- minimise perimeter (inward only, force floor).
        'maxSEA'    -- maximise specific energy absorption (F / P).
        'boxMaxCF'  -- maximise crush force inside a bounding box.
    force_total, perimeter : Tensor
        Current values from forward pass.
    model : crushOptimizerModel
        The model (for accessing r, initial_r).
    force_initial : Tensor
        Force at epoch 0.
    smoothness_initial : Tensor
        Smoothness at epoch 0 (for normalisation).
    lambdas : dict
        Penalty weights: 'smooth', 'force', 'radial', 'box'.
    max_expansion_ratio : float
        Max allowed r / r_initial ratio for outward radial bounds.
    box_dims : tuple or None
        (box_width, box_height) in mm, centered at centroid.
        Required for boxMaxCF.
    """
    # Normalised smoothness: 1.0 at epoch 0, geometry-independent
    smoothness = smoothnessTerm(model.r, model.theta, model.phi, model.centroid) / (smoothness_initial.detach() + 1e-12)

    if mode == 'maxCF':
        # Outward expansion only, radially bounded.
        force_penalty = torch.relu(force_initial - force_total) ** 2
        r_max = max_expansion_ratio * model.initial_r
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        radial_inward  = torch.sum(torch.relu(model.initial_r - model.r) ** 2)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * (radial_inward + radial_outward))

    elif mode == 'minP':
        force_penalty = torch.relu(force_initial - force_total) ** 2
        radial_outward = torch.sum(torch.relu(model.r - model.initial_r) ** 2)
        loss = (perimeter
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * radial_outward)

    elif mode == 'maxSEA':
        sea = force_total / perimeter
        force_penalty = torch.relu(force_initial - force_total) ** 2
        r_min = (1.0 / max_expansion_ratio) * model.initial_r
        r_max = max_expansion_ratio * model.initial_r
        radial_inward  = torch.sum(torch.relu(r_min - model.r) ** 2)
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        loss = (-sea
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * (radial_inward + radial_outward))

    elif mode == 'boxMaxCF':
        # Maximise crush force inside a bounding box.
        # No radial bounds — the box is the only design envelope.
        box_pen = _box_penalty(model, box_dims)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['box'] * box_pen)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Use 'maxCF', 'minP', 'maxSEA', or 'boxMaxCF'.")

    return loss


def objective_value(mode: str,
                    force_total: torch.Tensor,
                    perimeter: torch.Tensor) -> float:
    """Return the scalar objective used for early stopping/restoration."""
    if mode in ('maxCF', 'boxMaxCF'):
        return force_total.item()
    if mode == 'minP':
        return -perimeter.item()
    if mode == 'maxSEA':
        return (force_total / perimeter).item()
    raise ValueError(f"Unknown mode '{mode}'.")


def satisfies_force_floor(force_total: torch.Tensor,
                          force_floor: torch.Tensor,
                          rel_tol: float = 1e-4) -> bool:
    """Treat the force floor as satisfied within a tiny numerical tolerance."""
    return force_total.item() >= force_floor.item() * (1.0 - rel_tol)


def satisfies_box_constraint(model,
                             box_dims,
                             abs_tol: float = 1e-10) -> bool:
    """Treat the box constraint as satisfied when the penalty is numerically zero."""
    if box_dims is None:
        return True
    return _box_penalty(model, box_dims).item() <= abs_tol


def objective_improved(current: float,
                       best: float,
                       rel_tol: float = 1e-12,
                       abs_tol: float = 1e-9) -> bool:
    """Return True when the scalar objective is meaningfully better."""
    if not np.isfinite(best):
        return True
    threshold = abs_tol + rel_tol * max(abs(current), abs(best), 1.0)
    return current > best + threshold

#==========================================================================
#==========================================================================
# MAIN SCRIPT

# ---- Tee logger ---- #
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
        sys.stdout = self.terminal

# ---- Command-line overrides ---- #
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data',       nargs='?', default=None)
parser.add_argument('mode',       nargs='?', default=None)
parser.add_argument('--steps',    type=int,   default=None)
parser.add_argument('--spline-p', type=float, default=None,
                    help='Smoothing spline parameter (0,1]. None=auto.')
parser.add_argument('--patience', type=int,   default=None,
                    help='Early stopping patience (epochs).')
parser.add_argument('--tol',      type=float, default=None,
                    help='Early stopping relative tolerance.')
parser.add_argument('--box-width', type=float, default=None,
                    help='Bounding box width in mm (boxMaxCF).')
parser.add_argument('--box-height', type=float, default=None,
                    help='Bounding box height in mm (boxMaxCF).')
parser.add_argument('--minp-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for minP.')
parser.add_argument('--maxcf-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for maxCF/boxMaxCF.')
parser.add_argument('--maxsea-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for maxSEA.')
parser.add_argument('--tag', type=str, default=None,
                    help='Optional filename suffix for this run.')
args, _ = parser.parse_known_args()
if args.data is not None:
    dataFileName = args.data
if args.mode is not None:
    run_mode = args.mode
if args.steps is not None:
    n_steps = args.steps
if args.spline_p is not None:
    spline_p = args.spline_p
if args.patience is not None:
    conv_patience = args.patience
if args.tol is not None:
    conv_tol = args.tol
if args.box_width is not None:
    box_width = args.box_width
if args.box_height is not None:
    box_height = args.box_height
if args.minp_smooth_ratio is not None:
    smooth_weight_ratio['minP'] = args.minp_smooth_ratio
if args.maxcf_smooth_ratio is not None:
    smooth_weight_ratio['maxCF'] = args.maxcf_smooth_ratio
    smooth_weight_ratio['boxMaxCF'] = args.maxcf_smooth_ratio
if args.maxsea_smooth_ratio is not None:
    smooth_weight_ratio['maxSEA'] = args.maxsea_smooth_ratio

# ---- Validate mode ---- #
valid_modes = {'maxCF', 'minP', 'maxSEA', 'boxMaxCF'}
if run_mode not in valid_modes:
    raise ValueError(f"Invalid run_mode '{run_mode}'. Choose from {valid_modes}.")

run_tag = ''
if args.tag:
    _tag_clean = re.sub(r'[^A-Za-z0-9_-]+', '-', args.tag).strip('-')
    if _tag_clean:
        run_tag = f"_{_tag_clean}"

# ---- Build temporary output filenames (renamed after loop) ---- #
run_timestamp = datetime.now().strftime('%d-%m')
_shape = dataFileName.replace('partition_data_', '').replace('.mat', '')
_tmp_base = f"{_shape}_{run_mode}{run_tag}_running_{run_timestamp}_(v{script_version})"

log_filename = str(RESULTS_OUTROS / f"{_tmp_base}.txt")
mat_filename = str(RESULTS_OUTROS / f"{_tmp_base}.mat")
pdf_filename = str(RESULTS_OUTROS / f"{_tmp_base}.pdf")

tee = TeeLogger(log_filename)
sys.stdout = tee

#==========================================================================
# Load cross section data from MATLAB
_data_path = Path(dataFileName)
if not _data_path.is_absolute():
    _data_path = DATA_DIR / _data_path
partitionData = loadData(str(_data_path))

# Process data
nodes = partitionData['nodesCoords']
edges = partitionData['connectivity']
orderedIndices, orderedNodes = orderNodes(nodes, edges)

# Convert to PyTorch tensors
orderedNodes = torch.tensor(orderedNodes, dtype=torch.float32).detach()
orderedIndices = torch.tensor(orderedIndices, dtype=torch.long)

model = crushOptimizerModel(
    orderedNodes, orderedIndices, edges,
    constantCrushStress, overlapCount, thickness,
    flat_transition, min_roc, spline_p
)

# Compute initial values
with torch.no_grad():
        forceTotal_initial, perimeter_initial = model()
        sea_initial = forceTotal_initial / perimeter_initial
        smoothness_initial = smoothnessTerm(model.r, model.theta, model.phi, model.centroid)
        orderedNodes_initial = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_initial = nodesCoords(overlapCount, orderedNodes_initial).cpu()

# Report effective spline_p (used in summary)
_effective_p = spline_p
if _effective_p is None:
    _tmp_ext = nodesCoords(overlapCount, orderedNodes)
    _tmp_s = chordLength(_tmp_ext)
    _tmp_h = _tmp_s[1:] - _tmp_s[:-1]
    _effective_p = _default_smooth_p(_tmp_h)

# ---- Auto-select lambda_smoothness based on objective scale ---- #
if run_mode in ('maxCF', 'boxMaxCF'):
    objective_scale = forceTotal_initial.item()
elif run_mode == 'minP':
    objective_scale = perimeter_initial.item()
elif run_mode == 'maxSEA':
    objective_scale = sea_initial.item()

lambda_smoothness = smooth_weight_ratio[run_mode] * objective_scale

# Pack penalty weights
lambdas = {
    'smooth': lambda_smoothness,
    'force':  lambda_crushForce,
    'radial': lambda_radialDist,
    'box':    lambda_box,
}

# Optimizer initiation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_steps, eta_min=1e-4)

# Force floor used for early stopping feasibility check:
# boxMaxCF has no force floor (force IS the objective);
# all other modes use initial force.
if run_mode == 'boxMaxCF':
    _force_floor = None
else:
    _force_floor = forceTotal_initial

_active_box_dims = ((box_width, box_height)
                    if run_mode == 'boxMaxCF' else None)

with torch.no_grad():
    initial_loss = compute_loss(
        mode=run_mode,
        force_total=forceTotal_initial,
        perimeter=perimeter_initial,
        model=model,
        force_initial=forceTotal_initial,
        smoothness_initial=smoothness_initial,
        lambdas=lambdas,
        max_expansion_ratio=max_expansion_ratio,
        box_dims=_active_box_dims,
    ).item()
    initial_objective = objective_value(run_mode, forceTotal_initial, perimeter_initial)
    initial_force_ok = (True if _force_floor is None
                        else satisfies_force_floor(forceTotal_initial, _force_floor))
    initial_box_ok = satisfies_box_constraint(model, _active_box_dims)
    initial_feasible = initial_force_ok and initial_box_ok

best_loss = initial_loss
best_r = model.r.data.clone()
best_epoch = -1
nan_detected = False
converged = False

# Early stopping state: track the best feasible objective.
best_objective = initial_objective if initial_feasible else float('-inf')
best_objective_r = model.r.data.clone() if initial_feasible else None
best_objective_epoch = -1 if initial_feasible else None
best_progress_objective = best_objective
epochs_without_improvement = 0

t_start = time.perf_counter()

# Optimization Loop
for epoch in range(n_steps):
        optimizer.zero_grad()
        force_total, perimeter = model()

        loss = compute_loss(
            mode=run_mode,
            force_total=force_total,
            perimeter=perimeter,
            model=model,
            force_initial=forceTotal_initial,
            smoothness_initial=smoothness_initial,
            lambdas=lambdas,
            max_expansion_ratio=max_expansion_ratio,
            box_dims=_active_box_dims,
        )

        # NaN detection
        if torch.isnan(loss) or torch.isinf(loss):
            model.r.data.copy_(best_r)
            nan_detected = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            force_total, perimeter = model()
            loss_eval = compute_loss(
                mode=run_mode,
                force_total=force_total,
                perimeter=perimeter,
                model=model,
                force_initial=forceTotal_initial,
                smoothness_initial=smoothness_initial,
                lambdas=lambdas,
                max_expansion_ratio=max_expansion_ratio,
                box_dims=_active_box_dims,
            )

        if torch.isnan(loss_eval) or torch.isinf(loss_eval):
            model.r.data.copy_(best_r)
            nan_detected = True
            break

        current_loss = loss_eval.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_r = model.r.data.clone()

        # ---- Early stopping check ---- #
        sea_now = (force_total / perimeter).item()
        current_objective = objective_value(run_mode, force_total, perimeter)
        force_floor_ok = (True if _force_floor is None
                          else satisfies_force_floor(force_total, _force_floor))
        box_ok = satisfies_box_constraint(model, _active_box_dims)
        feasible_now = force_floor_ok and box_ok

        # Save the best feasible state even for small genuine improvements.
        if feasible_now and objective_improved(current_objective, best_objective):
            best_objective = current_objective
            best_objective_r = model.r.data.clone()
            best_objective_epoch = epoch

        # Only reset patience for meaningful progress on the feasible objective.
        if feasible_now and (
            not np.isfinite(best_progress_objective) or
            (current_objective - best_progress_objective) /
            (abs(best_progress_objective) + 1e-12) > conv_tol
        ):
            best_progress_objective = current_objective
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= conv_patience and epoch > conv_patience:
            converged = True
            break

        # Terminal-only progress (not written to log file)
        current_lr = optimizer.param_groups[0]['lr']
        tee.terminal.write(
            f"Epoch {epoch}: Perimeter = {perimeter:.4f}, "
            f"Crush Force = {force_total:.4f}, "
            f"SEA = {sea_now:.4f}, "
            f"Loss = {current_loss:.4f}, "
            f"LR = {current_lr:.6f}\n"
        )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

if not nan_detected:
    if best_objective_r is not None:
        model.r.data.copy_(best_objective_r)
    else:
        model.r.data.copy_(best_r)

# ---- Rename output files to final names (without epoch count) ---- #
_final_base = f"{_shape}_{run_mode}{run_tag}_{run_timestamp}_(v{script_version})"

log_filename_final = str(RESULTS_OUTROS / f"{_final_base}.txt")
mat_filename_final = str(RESULTS_OUTROS / f"{_final_base}.mat")
pdf_filename_final = str(RESULTS_OUTROS / f"{_final_base}.pdf")

# Close the tee so the temp log file is flushed and released
tee.close()
# Rename temp files
import shutil
if os.path.exists(log_filename):
    shutil.move(log_filename, log_filename_final)
log_filename = log_filename_final
mat_filename = mat_filename_final
pdf_filename = pdf_filename_final

# Re-open tee to the final log file (append mode for the summary)
tee = TeeLogger.__new__(TeeLogger)
tee.terminal = sys.__stdout__
tee.log = open(log_filename, 'a', encoding='utf-8')
sys.stdout = tee

# ---- Visualize optimized spline ----
with torch.no_grad():
        forceTotal_final, perimeter_final = model()
        sea_final = forceTotal_final / perimeter_final
        final_force_ok = (True if _force_floor is None
                          else satisfies_force_floor(forceTotal_final, _force_floor))
        final_box_ok = satisfies_box_constraint(model, _active_box_dims)
        final_feasible = final_force_ok and final_box_ok
        # Centroid-relative coordinates (for plot — box is drawn at origin)
        orderedNodes_final = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_final = nodesCoords(overlapCount, orderedNodes_final).detach().cpu()
        plot_initial_and_final(nodesExt_initial, nodesExt_final, pdf_filename,
                              box_dims=_active_box_dims)

        # Absolute coordinates (input frame) for .mat export
        orderedNodes_abs = (orderedNodes_final + model.centroid).numpy()

        n = orderedNodes_abs.shape[0]
        connectivity_final = np.column_stack([np.arange(1, n + 1), np.arange(2, n + 2)])
        connectivity_final[-1, 1] = 1

        scipy.io.savemat(
            mat_filename,
            {
                'nodesCoords':    orderedNodes_abs,
                'connectivity':   connectivity_final,
                'forceInitial':   np.float64(forceTotal_initial.item()),
                'forceFinal':     np.float64(forceTotal_final.item()),
                'perimeterInitial': np.float64(perimeter_initial.item()),
                'perimeterFinal':   np.float64(perimeter_final.item()),
                'seaInitial':     np.float64(sea_initial.item()),
                'seaFinal':       np.float64(sea_final.item()),
            }
        )

# ---- Stopping reason ---- #
if nan_detected:
    stop_reason = f'NaN/Inf at epoch {final_epoch} (rolled back to best loss state from epoch {best_epoch})'
elif best_objective_r is None:
    stop_reason = f'No feasible solution found; restored best loss state from epoch {best_epoch}'
elif converged:
    stop_reason = f'Converged at epoch {final_epoch} (restored best feasible objective from epoch {best_objective_epoch})'
else:
    stop_reason = f'Step limit reached ({n_steps}), restored best feasible objective from epoch {best_objective_epoch}'

# ---- Runtime summary (this is the ONLY content written to the .txt) ---- #
mode_labels = {
    'maxCF':     'Max crush force (outward, radially bounded)',
    'minP':      'Min perimeter (inward)',
    'maxSEA':    'Max specific energy absorption',
    'boxMaxCF':  'Max crush force (box constrained)',
}

print("="*60)
print("CONVERGENCE STUDY - RUN SUMMARY")
print("="*60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {mode_labels[run_mode]}")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
if run_mode == 'boxMaxCF':
    print(f"  Box constraint:     {box_width:.1f} x {box_height:.1f} mm")
if run_mode == 'boxMaxCF':
    print(f"  Box satisfied:      {'yes' if final_box_ok else 'no'}")
if _force_floor is not None:
    print(f"  Force floor met:    {'yes' if final_force_ok else 'no'}")
print(f"  Final feasible:     {'yes' if final_feasible else 'no'}")
force_change_pct = ((forceTotal_final.item() - forceTotal_initial.item()) / forceTotal_initial.item()) * 100
perim_change_pct = ((perimeter_final.item() - perimeter_initial.item()) / perimeter_initial.item()) * 100
sea_change_pct   = ((sea_final.item() - sea_initial.item()) / sea_initial.item()) * 100

print("  ---")
print("  {:<8} {:>12} {:>12} {:>10}".format("Metric", "Initial", "Final", "Change"))
print("  {:<8} {:>12} {:>12} {:>10}".format("-" * 8, "-" * 12, "-" * 12, "-" * 10))
print("  {:<8} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Force", forceTotal_initial.item(), forceTotal_final.item(), force_change_pct))
print("  {:<8} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Perim", perimeter_initial.item(), perimeter_final.item(), perim_change_pct))
print("  {:<8} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "SEA", sea_initial.item(), sea_final.item(), sea_change_pct))
print(f"  ---")
print(f"  Epochs:             {final_epoch + 1}")
print(f"  Stop reason:        {stop_reason}")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("="*60)

tee.close()
