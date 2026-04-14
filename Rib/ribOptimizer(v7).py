
#                     ribOptimizer_v7.py
#
#  Rib-aware cross-section geometry optimizer for CrushAnalytica.
#  Phase B: optimises the outer wall shape while accounting for an
#  internal rib's force contribution.
#
#  Architecture:
#    - Wall: spherical parameterisation r(theta,phi) -- same as v19
#    - Rib: straight-line interpolation between junction nodes on wall
#    - Forward: F_total = F_wall + F_rib, P_total = P_wall + P_rib
#    - The rib geometry follows the wall -- no separate rib parameters
#    - If junctions move closer, rib shortens; if farther, rib lengthens
#
#  Spline: Reinsch smoothing cubic spline (from v16+), fully
#          differentiable via torch.linalg.solve. NO torchcubicspline.
#
#  v7 changes:
#    - Per-node thickness loaded from .mat file (wall/junction/rib
#      values extracted automatically from topology roles).
#    - Integrated wall cleanup (short-edge removal) and optional
#      uniform resampling, matching ribRemovalOptimizer workflow.
#      Junction nodes are protected during cleanup and treated as
#      landmarks during resampling.
#    - Removed scalar thickness constant from user inputs.
#
#  v6 changes:
#    - Removed the legacy boxMinP mode to match NoRib optimizer v21.
#    - Reordered the script so user inputs/default config appear at the
#      start of the file, matching the v21 layout more closely.
#
#  v5 changes:
#    - Wall curvature queries now evaluate at the original closed-loop
#      node parameters instead of a linspace approximation.
#    - Box constraints are enforced on the smoothed wall curve, sampled
#      densely between nodes, instead of only at wall nodes.
#    - Best-objective restore now requires all active constraints
#      (force floor, radial envelope, box envelope) to be satisfied.
#    - Added maxFPP naming/CLI aliasing while keeping maxSEA as legacy.
#    - Output naming now uses the resolved input basename, so absolute
#      input paths cannot escape the configured results directory.
#
#  v19-aligned changes:
#    - Smoothing cubic spline (Reinsch form) replacing NaturalCubicSpline.
#    - Edge-length-weighted smoothness term (v17+).
#    - Early stopping with conv_patience/conv_tol and best-objective restore.
#    - maxCF: outward-only with radial cap and force floor.
#
#  Usage:
#    python ribOptimizer.py                                    # defaults
#    python ribOptimizer.py partition_data_sis_rib.mat maxCF   # CLI
#
#  Dependencies: torch, scipy, numpy, matplotlib (NO torchcubicspline)
#
#  Francisco Arriscado -- FEUP, 2026
#==========================================================================

script_version = 7

from typing import Tuple, Dict, Optional
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import minimize_scalar
from collections import defaultdict
import sys
import os
import time
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

# ---- Project paths ---- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Rib" / "data"
RESULTS_OUTROS = PROJECT_ROOT / "results" / "Rib"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTROS.mkdir(parents=True, exist_ok=True)

#==========================================================================
#==========================================================================
#                          User Inputs
#==========================================================================
#==========================================================================

# ---- Geometry / material defaults ---- #
dataFileName = 'partition_data_layer2_z-132.4.mat'
constantCrushStress = 90.0             # [MPa]
flat_transition = 1221.95               # [mm]
min_roc = 15.0                          # [mm]

# ---- Optimisation mode ---- #
# 'maxCF'    -- maximise crush force (outward expansion, capped)
# 'minP'     -- minimise perimeter while preserving force (inward only)
# 'maxFPP'   -- maximise force-per-perimeter ratio (legacy alias: maxSEA)
# 'boxMaxCF' -- maximise crush force within a bounding box
run_mode = 'minP'

# ---- Bounding box (boxMaxCF mode only) ---- #
box_width = 35                          # [mm] box extent in X
box_height = 35                         # [mm] box extent in Y

# ---- Penalty weights ---- #
lambda_crushForce = 1e6
lambda_radialDist = 1e6
lambda_box = 1e10

# ---- Smoothness weight ratio per mode ---- #
smooth_weight_ratio = {
    'maxCF':    0.2,
    'minP':     0.20,
    'maxFPP':   0.75,
    'boxMaxCF': 2.0,
}

# ---- Geometry movement limits ---- #
max_expansion_ratio = 1.3

# ---- Smoothing spline control ---- #
spline_p = None #Value comes from a self study for best value

#==========================================================================
#==========================================================================
#                         Clean up Control
#==========================================================================
#==========================================================================
# ---- Wall cleanup / resampling ---- #
resample_uniform = True
cleanup_short_ratio = 0.35
cleanup_angle_tol_deg = 12.0
cleanup_excess_rel_tol = 0.02
corner_angle_threshold_deg = 150.0

#==========================================================================
#==========================================================================
#                         Default Config
#==========================================================================
#==========================================================================

# ---- Optimizer / training defaults ---- #
n_steps = 100000                        # hard ceiling (early stopping usually triggers first)
learning_rate = 0.1
grad_clip_norm = 1.0
snapshot_interval = 500
overlapCount = 10

# ---- Early stopping ---- #
conv_patience = 1000                    # epochs to wait for improvement
conv_tol = 1e-3                         # relative improvement threshold
force_floor_rel_tol = 1e-4

# ---- Constraint resolution ---- #
# Sample the smoothed wall more densely than the nodes when checking the
# box, so the actual curve stays inside the envelope between nodes too.
box_constraint_samples_per_edge = 16
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

    p = max(min(float(p), 1.0 - 1e-12), 1e-12)

    # Solve for interior second derivatives sigma at knots 1..n-2
    # Reinsch smoothing spline system:
    #   [p R + (1-p) Q^T Q] sigma = p Q^T y
    A = p * R + (1.0 - p) * (Q.t() @ Q)
    rhs = p * (Q.t() @ y)
    sigma = torch.linalg.solve(A, rhs)

    # Smoothed values at knots
    g = y - ((1.0 - p) / p) * (Q @ sigma)

    M = torch.zeros(n, y.shape[1], dtype=y.dtype, device=y.device)
    M[1:-1] = sigma

    return g, M, h


def closed_loop_query_params(
    s: torch.Tensor,
    overlapCount: int,
    samples_per_edge: int = 1,
) -> torch.Tensor:
    """
    Return query positions along the original closed loop.

    samples_per_edge=1 evaluates exactly at the original node parameters.
    Larger values add uniformly-spaced interior samples on each original
    edge segment without duplicating the wrapped endpoint.
    """
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be at least 1.")

    s_nodes = s[overlapCount:-overlapCount]
    s_next = torch.cat([s_nodes[1:], s[-overlapCount].unsqueeze(0)])
    alpha = (
        torch.arange(samples_per_edge, device=s.device, dtype=s.dtype)
        / float(samples_per_edge)
    )
    return (
        s_nodes.unsqueeze(1)
        + (s_next - s_nodes).unsqueeze(1) * alpha
    ).reshape(-1)


def eval_spline_points(
    t: torch.Tensor,
    g: torch.Tensor,
    M: torch.Tensor,
    h: torch.Tensor,
    tq: torch.Tensor,
) -> torch.Tensor:
    """Evaluate spline positions at query points tq."""
    n = t.shape[0]

    idx = torch.searchsorted(t[1:].contiguous(), tq.contiguous())
    idx = idx.clamp(0, n - 2)

    hi = h[idx]
    dt_right = (t[idx + 1] - tq).unsqueeze(1)
    dt_left = (tq - t[idx]).unsqueeze(1)
    hi_d = hi.unsqueeze(1)

    Mi = M[idx]
    Mi1 = M[idx + 1]
    gi = g[idx]
    gi1 = g[idx + 1]

    return (
        Mi * dt_right**3 / (6.0 * hi_d)
        + Mi1 * dt_left**3 / (6.0 * hi_d)
        + (gi - Mi * hi_d**2 / 6.0) * dt_right / hi_d
        + (gi1 - Mi1 * hi_d**2 / 6.0) * dt_left / hi_d
    )


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
#              TOPOLOGY ANALYSIS (NumPy -- no torch needed)
#==========================================================================

def _build_adjacency(edges):
    """Build adjacency list and degree array from 0-based edge list."""
    adj = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    if not adj:
        return adj, np.array([])
    max_node = max(adj.keys())
    degree = np.zeros(max_node + 1, dtype=int)
    for node, nbs in adj.items():
        degree[node] = len(nbs)
    return adj, degree


def _decompose_branches(edges_comp, adj, degree, comp_nodes):
    """Decompose a component into branches. Returns list of node-id lists."""
    comp_set = set(comp_nodes)
    junction_nodes = [n for n in comp_set if degree[n] > 2]
    end_nodes = [n for n in comp_set if degree[n] == 1]

    edge_set = set()
    for a, b in edges_comp:
        a, b = int(a), int(b)
        edge_set.add((min(a, b), max(a, b)))
    visited = set()

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    branches = []
    for start in end_nodes + junction_nodes:
        for nb in adj[start]:
            if nb not in comp_set:
                continue
            ek = edge_key(start, nb)
            if ek not in edge_set or ek in visited:
                continue
            path = [start, nb]
            visited.add(ek)
            cur, prev = nb, start
            while True:
                if degree[cur] != 2:
                    break
                nbs_in = [n for n in adj[cur] if n in comp_set and n != prev]
                if not nbs_in:
                    break
                nxt = nbs_in[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in visited:
                    break
                visited.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt
            branches.append(path)

    # Catch unvisited loops
    for a, b in edges_comp:
        a, b = int(a), int(b)
        ek = edge_key(a, b)
        if ek in visited:
            continue
        loop = [a, b]
        visited.add(ek)
        prev, cur = a, b
        while True:
            nbs_in = [n for n in adj[cur] if n in comp_set and n != prev]
            if not nbs_in:
                break
            nxt = nbs_in[0]
            ek2 = edge_key(cur, nxt)
            if ek2 in visited:
                break
            visited.add(ek2)
            loop.append(nxt)
            prev, cur = cur, nxt
        branches.append(loop)

    return branches, junction_nodes, end_nodes


def extract_wall_and_ribs(nodes, edges):
    """Extract wall closed loop and rib branches from a branched cross-section.

    Assumes: single component, exactly 2 junction nodes, 3 branches.
    The two longest branches form the wall; the shortest is the rib.

    Parameters
    ----------
    nodes : ndarray (N, 3)
    edges : ndarray (M, 2), 0-based

    Returns
    -------
    info : dict with keys:
        'wall_loop'       : list[int] -- ordered wall node indices (no duplicate end)
        'rib_branch'      : list[int] -- full rib node list (including junction endpoints)
        'rib_interior'    : list[int] -- rib nodes excluding junctions
        'junction_nodes'  : list[int] -- [j0, j1]
        'j0_wall_idx'     : int -- index of junction 0 in wall_loop
        'j1_wall_idx'     : int -- index of junction 1 in wall_loop
        'n_rib_interior'  : int
    """
    n_nodes = nodes.shape[0]
    adj, degree = _build_adjacency(edges)

    # Check for junctions
    comp_nodes = list(range(n_nodes))
    branches, junctions, endpoints = _decompose_branches(edges, adj, degree, comp_nodes)

    assert len(junctions) == 2, \
        f"Expected 2 junction nodes, got {len(junctions)}: {junctions}"
    assert len(branches) == 3, \
        f"Expected 3 branches, got {len(branches)}"

    # Measure branch lengths
    branch_lengths = []
    for br in branches:
        xyz = nodes[br]
        length = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        branch_lengths.append(length)

    # Shortest = rib, two longest = wall
    rib_idx = int(np.argmin(branch_lengths))
    wall_indices = [i for i in range(3) if i != rib_idx]

    wall_br0 = branches[wall_indices[0]]
    wall_br1 = branches[wall_indices[1]]
    rib_br = branches[rib_idx]

    # Connect wall branches into a closed loop
    # Both branches connect the same two junctions -- orient and chain them
    if wall_br0[-1] == wall_br1[0]:
        wall_loop = wall_br0 + wall_br1[1:]
    elif wall_br0[-1] == wall_br1[-1]:
        wall_loop = wall_br0 + wall_br1[-2::-1]
    elif wall_br0[0] == wall_br1[0]:
        wall_loop = wall_br0[::-1] + wall_br1[1:]
    elif wall_br0[0] == wall_br1[-1]:
        wall_loop = wall_br1 + wall_br0[1:]
    else:
        raise ValueError("Wall branches don't share junction nodes")

    # Remove duplicate end node (closed loop)
    if wall_loop[-1] == wall_loop[0]:
        wall_loop = wall_loop[:-1]

    # Find junction indices in wall loop
    j0, j1 = junctions
    j0_wall_idx = wall_loop.index(j0)
    j1_wall_idx = wall_loop.index(j1)

    # Rib interior (excluding junction endpoints)
    # Orient rib so it goes from j0 -> ... -> j1
    if rib_br[0] == j0:
        rib_interior = rib_br[1:-1]
    elif rib_br[0] == j1:
        rib_br = rib_br[::-1]
        rib_interior = rib_br[1:-1]
    else:
        raise ValueError("Rib branch doesn't start at a junction node")

    return {
        'wall_loop': wall_loop,
        'rib_branch': rib_br,
        'rib_interior': rib_interior,
        'junction_nodes': [j0, j1],
        'j0_wall_idx': j0_wall_idx,
        'j1_wall_idx': j1_wall_idx,
        'n_rib_interior': len(rib_interior),
    }


#==========================================================================
#              PyTorch PHYSICS FUNCTIONS (aligned with v19)
#==========================================================================

def Cartesian2Spherical(xyz):
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    phi = torch.atan2(Y, X)
    theta = torch.acos(Z / r)
    return r, theta, phi


def Spherical2Cartesian(r, theta, phi):
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    return torch.stack([X, Y, Z], dim=1)


def extend_nodes(overlapCount, orderedNodes):
    X, Y, Z = orderedNodes[:, 0], orderedNodes[:, 1], orderedNodes[:, 2]
    X_ext = torch.cat([X[-overlapCount:], X, X[:overlapCount]])
    Y_ext = torch.cat([Y[-overlapCount:], Y, Y[:overlapCount]])
    Z_ext = torch.cat([Z[-overlapCount:], Z, Z[:overlapCount]])
    return torch.stack([X_ext, Y_ext, Z_ext], dim=1)


def chordLength(nodesExt):
    diffs = nodesExt[1:] - nodesExt[:-1]
    ds = torch.norm(diffs, dim=1)
    s = torch.zeros(len(nodesExt), device=nodesExt.device)
    s[1:] = torch.cumsum(ds, dim=0)
    s = s / (s[-1] + 1e-12)
    return s


def query_spline_derivatives(
    s: torch.Tensor,
    g: torch.Tensor,
    M: torch.Tensor,
    h: torch.Tensor,
    overlapCount: int,
    samples_per_edge: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sQuery = closed_loop_query_params(
        s, overlapCount, samples_per_edge=samples_per_edge)
    return eval_spline_derivatives(s, g, M, h, sQuery)


def splines(nodesExt, orderedNodes, s, overlapCount, spline_p
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a smoothing cubic spline to the extended node coordinates and
    evaluate first and second derivatives at the original node query
    parameters, matching the v21 closed-loop convention.
    """
    # Fit smoothing spline to extended nodes
    g, M, h = fit_smoothing_spline(s, nodesExt, p=spline_p)

    return query_spline_derivatives(
        s, g, M, h, overlapCount, samples_per_edge=1)


def RoCCalc(der1, der2):
    cross = torch.cross(der1, der2, dim=1)
    num = torch.norm(cross, dim=1)
    den = torch.norm(der1, dim=1) ** 3 + 1e-12
    curvature = num / den
    return 1.0 / (curvature + 1e-12)


def CurvatureEq(x):
    return 7.95 * torch.pow(x, -0.94) + 0.99


def perimeterCalc(orderedNodes):
    shifted = torch.roll(orderedNodes, -1, dims=0)
    diffs = shifted - orderedNodes
    edgeLengths = torch.norm(diffs, dim=1)
    return torch.sum(edgeLengths), edgeLengths


def smoothnessTerm(r, theta, phi, centroid):
    """
    Second differences of radial distances (closed loop), weighted by
    local Cartesian edge length so that dense and sparse regions are
    penalised equally per unit perimeter.
    """
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    nodes = torch.stack([X, Y, Z], dim=1) + centroid

    # Edge lengths between consecutive nodes (closed loop)
    diffs = torch.roll(nodes, -1, dims=0) - nodes
    edge_len = torch.norm(diffs, dim=1)

    # nodeLength: average of left and right edge for each node
    node_len = (edge_len + torch.roll(edge_len, 1, dims=0)) / 2.0
    node_len = node_len.clamp(min=1e-8)

    # Weight = node_len / mean(node_len): sparse regions get higher
    # weight per node, dense regions get lower weight.
    weights = node_len / node_len.mean()

    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    dr = r_wrapped[1:] - r_wrapped[:-1]
    curvature = dr[1:] - dr[:-1]
    return torch.sum(weights * curvature ** 2)


def _box_penalty(model, box_dims, samples_per_edge: int):
    """
    Quadratic penalty for the smoothed wall outside the bounding box.

    The force model operates on the smoothed wall curve, so the box
    constraint should be enforced on that same continuous geometry
    instead of on the wall nodes alone.
    """
    curve_points = model.smoothed_curve_points(
        samples_per_edge=samples_per_edge) - model.centroid
    X = curve_points[:, 0]
    Y = curve_points[:, 1]
    half_w = box_dims[0] / 2.0
    half_h = box_dims[1] / 2.0
    return (
        torch.sum(torch.relu(X - half_w) ** 2)
        + torch.sum(torch.relu(-half_w - X) ** 2)
        + torch.sum(torch.relu(Y - half_h) ** 2)
        + torch.sum(torch.relu(-half_h - Y) ** 2)
    ) / float(samples_per_edge)


def _radial_penalty(model, mode, max_expansion_ratio):
    zero = torch.zeros((), dtype=model.r.dtype, device=model.r.device)

    if mode == 'maxCF':
        r_max = max_expansion_ratio * model.initial_r
        return (
            torch.sum(torch.relu(model.initial_r - model.r) ** 2)
            + torch.sum(torch.relu(model.r - r_max) ** 2)
        )

    if mode == 'minP':
        return torch.sum(torch.relu(model.r - model.initial_r) ** 2)

    if mode == 'maxFPP':
        r_min = (1.0 / max_expansion_ratio) * model.initial_r
        r_max = max_expansion_ratio * model.initial_r
        return (
            torch.sum(torch.relu(r_min - model.r) ** 2)
            + torch.sum(torch.relu(model.r - r_max) ** 2)
        )

    return zero


#==========================================================================
#              WALL CLEANUP AND RESAMPLING
#==========================================================================

def _turning_angle_deg(prev_pt, pt, next_pt):
    v1 = prev_pt - pt
    v2 = next_pt - pt
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def cleanup_closed_loop(points, protected_indices=None,
                         short_ratio=0.35, angle_tol_deg=12.0,
                         excess_rel_tol=0.02, excess_abs_tol=1e-3,
                         min_nodes=8):
    """Remove redundant near-colinear short-edge nodes from a closed loop.

    Parameters
    ----------
    points : ndarray (N, 3)
    protected_indices : set[int] or None
        Indices into *points* that must not be removed (e.g. junction nodes).

    Returns
    -------
    cleaned_points : ndarray
    surviving_indices : ndarray of int
        Indices into the original *points* for each surviving node.
    n_removed : int
    """
    pts = points.copy()
    idx_map = np.arange(len(pts), dtype=int)
    protected = set(protected_indices) if protected_indices else set()
    n_removed = 0

    while len(pts) > min_nodes:
        edge_lengths = np.linalg.norm(
            np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        median_edge = float(np.median(edge_lengths))
        changed = False

        for i in range(len(pts)):
            if int(idx_map[i]) in protected:
                continue

            prev_i = (i - 1) % len(pts)
            next_i = (i + 1) % len(pts)
            prev_pt = pts[prev_i]
            pt = pts[i]
            next_pt = pts[next_i]

            len_prev = float(np.linalg.norm(pt - prev_pt))
            len_next = float(np.linalg.norm(next_pt - pt))
            len_chord = float(np.linalg.norm(next_pt - prev_pt))
            longer = max(len_prev, len_next)
            shorter = min(len_prev, len_next)
            if longer <= 0.0:
                continue

            angle = _turning_angle_deg(prev_pt, pt, next_pt)
            excess = (len_prev + len_next) - len_chord

            is_short = shorter <= short_ratio * longer
            is_straight = abs(180.0 - angle) <= angle_tol_deg
            is_low_excess = excess <= max(excess_abs_tol,
                                          excess_rel_tol * median_edge)

            if is_short and is_straight and is_low_excess:
                pts = np.delete(pts, i, axis=0)
                idx_map = np.delete(idx_map, i)
                n_removed += 1
                changed = True
                break

        if not changed:
            break

    return pts, idx_map, n_removed


def allocate_counts_from_lengths(lengths, target_total, min_per_segment=1):
    lengths = np.asarray(lengths, dtype=float)
    if target_total < len(lengths) * min_per_segment:
        raise ValueError("Target total is too small for the number of segments.")

    weights = lengths / np.sum(lengths)
    raw = weights * target_total
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, min_per_segment)

    deficit = target_total - int(np.sum(counts))
    if deficit > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[:deficit]:
            counts[idx] += 1
    elif deficit < 0:
        removable = counts - min_per_segment
        order = np.argsort(raw - np.floor(raw))
        need = -deficit
        for idx in order:
            take = min(removable[idx], need)
            counts[idx] -= take
            need -= take
            if need == 0:
                break
        if need != 0:
            raise ValueError("Could not allocate counts safely across segments.")

    return counts


def resample_closed_loop_uniform(points, target_count=None,
                                  corner_angle_threshold_deg=150.0,
                                  extra_landmark_indices=None):
    """Resample a closed loop uniformly by arc length.

    Sharp corners and extra_landmark_indices are preserved as landmarks.
    """
    if target_count is None:
        target_count = len(points)

    n = len(points)

    # Detect corners
    corners = []
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        angle = _turning_angle_deg(points[prev_i], points[i], points[next_i])
        if angle < corner_angle_threshold_deg:
            corners.append(i)

    # Merge corners with extra landmarks
    landmarks = sorted(set(corners) | set(extra_landmark_indices or []))

    if not landmarks:
        # No landmarks: simple uniform resampling around the full loop
        pts_closed = np.vstack([points, points[:1]])
        seg_vecs = np.diff(pts_closed, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        total = float(np.sum(seg_lengths))
        if total <= 0:
            return points, {}
        cumul = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        sample_s = np.linspace(0.0, total, target_count, endpoint=False)
        result = np.zeros((target_count, points.shape[1]))
        for i, s in enumerate(sample_s):
            idx = int(np.searchsorted(cumul[1:], s, side='right'))
            idx = min(idx, len(seg_lengths) - 1)
            seg_start = cumul[idx]
            seg_len = seg_lengths[idx]
            if seg_len <= 0:
                result[i] = points[idx % n]
            else:
                alpha = (s - seg_start) / seg_len
                result[i] = points[idx % n] + alpha * seg_vecs[idx]
        return result, {}

    # With landmarks: segment-aware resampling
    n_segments = len(landmarks)

    # Compute segment lengths (between consecutive landmarks around loop)
    seg_lengths_list = []
    seg_polylines = []
    for s_idx in range(n_segments):
        start_idx = landmarks[s_idx]
        end_idx = landmarks[(s_idx + 1) % n_segments]
        if end_idx > start_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = list(range(start_idx, n)) + list(range(0, end_idx + 1))
        poly = points[indices]
        seg_polylines.append(poly)
        length = float(np.sum(
            np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        seg_lengths_list.append(length)

    total_length = sum(seg_lengths_list)
    if total_length <= 0:
        return points, {}

    # Allocate node counts per segment proportional to length
    target_intervals = target_count
    counts = allocate_counts_from_lengths(
        np.asarray(seg_lengths_list, dtype=float),
        target_intervals,
        min_per_segment=1,
    )

    # Resample each segment, tracking landmark output indices
    pieces = []
    landmark_output_map = {}
    running_idx = 0
    for i, (poly, n_intervals) in enumerate(zip(seg_polylines, counts)):
        # Record the output index for this landmark
        lm_orig = landmarks[i]
        if lm_orig not in landmark_output_map:
            landmark_output_map[lm_orig] = running_idx

        n_pts = int(n_intervals) + 1
        seg_vecs_local = np.diff(poly, axis=0)
        seg_lens_local = np.linalg.norm(seg_vecs_local, axis=1)
        total_local = float(np.sum(seg_lens_local))
        if total_local <= 0 or n_pts < 2:
            pieces.append(poly[:1])
            running_idx += 1
            continue
        cumul = np.concatenate([[0.0], np.cumsum(seg_lens_local)])
        sample_s = np.linspace(0.0, total_local, n_pts, endpoint=True)
        resampled = np.zeros((n_pts, poly.shape[1]))
        for j, s_val in enumerate(sample_s):
            seg_idx = int(np.searchsorted(cumul[1:], s_val, side='right'))
            seg_idx = min(seg_idx, len(seg_lens_local) - 1)
            seg_start = cumul[seg_idx]
            seg_len = seg_lens_local[seg_idx]
            if seg_len <= 0:
                resampled[j] = poly[seg_idx]
            else:
                alpha = (s_val - seg_start) / seg_len
                resampled[j] = poly[seg_idx] + alpha * seg_vecs_local[seg_idx]
        # Drop last point (it is the start of the next segment)
        pieces.append(resampled[:-1])
        running_idx += len(resampled) - 1

    return np.vstack(pieces), landmark_output_map


#==========================================================================
#              OPTIMIZER MODEL
#==========================================================================

class crushRibOptimizerModel(nn.Module):
    """Rib-aware cross-section optimizer.

    The wall is parameterised in spherical coordinates (r optimisable,
    theta and phi fixed). The rib is a straight line between two junction
    nodes on the wall -- its geometry follows the wall automatically.
    """

    def __init__(self,
                 wall_nodes: torch.Tensor,
                 wall_info: Dict,
                 constantCrushStress: float,
                 overlapCount: int,
                 wall_thickness: float,
                 junction_thicknesses,
                 rib_thickness: float,
                 flat_transition: float,
                 min_roc: float,
                 spline_p: float,
                 ):
        super().__init__()

        # Wall: spherical parameterisation
        self.centroid = wall_nodes.mean(dim=0, keepdim=True)
        r, theta, phi = Cartesian2Spherical(wall_nodes - self.centroid)

        self.r = nn.Parameter(r)
        self.register_buffer('initial_r', r.clone())
        self.register_buffer('theta', theta)
        self.register_buffer('phi', phi)

        # Rib metadata
        self.j0_wall_idx = wall_info['j0_wall_idx']
        self.j1_wall_idx = wall_info['j1_wall_idx']
        self.n_rib_interior = wall_info['n_rib_interior']

        # Per-topology thickness (not optimised)
        n_wall = wall_nodes.shape[0]
        j0_thickness, j1_thickness = junction_thicknesses
        wall_thick_vec = torch.full((n_wall,), wall_thickness,
                                    dtype=wall_nodes.dtype,
                                    device=wall_nodes.device)
        wall_thick_vec[self.j0_wall_idx] = j0_thickness
        wall_thick_vec[self.j1_wall_idx] = j1_thickness
        self.register_buffer('wall_thickness', wall_thick_vec)
        self.rib_thickness = rib_thickness

        # Physics constants
        self.constantCrushStress = constantCrushStress
        self.overlapCount = overlapCount
        self.flat_transition = flat_transition
        self.min_roc = min_roc
        self.spline_p = spline_p
        self._last_spline_cache = None

    def _reconstruct_wall_xyz(self) -> torch.Tensor:
        return Spherical2Cartesian(self.r, self.theta, self.phi) + self.centroid

    def _build_spline_cache(self, wall_xyz: torch.Tensor):
        wall_ext = extend_nodes(self.overlapCount, wall_xyz)
        s = chordLength(wall_ext)
        g, M, h = fit_smoothing_spline(s, wall_ext, p=self.spline_p)
        self._last_spline_cache = {
            'wall_xyz': wall_xyz,
            'wall_ext': wall_ext,
            's': s,
            'g': g,
            'M': M,
            'h': h,
        }
        return self._last_spline_cache

    def smoothed_curve_points(self, samples_per_edge: int = 1) -> torch.Tensor:
        if self._last_spline_cache is None:
            wall_xyz = self._reconstruct_wall_xyz()
            cache = self._build_spline_cache(wall_xyz)
        else:
            cache = self._last_spline_cache

        sQuery = closed_loop_query_params(
            cache['s'], self.overlapCount, samples_per_edge=samples_per_edge)
        return eval_spline_points(
            cache['s'], cache['g'], cache['M'], cache['h'], sQuery)

    def forward(self):
        # ---- 1. Wall Cartesian coordinates ----
        wall_xyz = self._reconstruct_wall_xyz()

        # ---- 2. Junction positions (from wall) ----
        junc0 = wall_xyz[self.j0_wall_idx]
        junc1 = wall_xyz[self.j1_wall_idx]

        # ---- 3. Rib: straight line between junctions ----
        n_rib_total = self.n_rib_interior + 2  # including junction endpoints
        t = torch.linspace(0, 1, n_rib_total, device=wall_xyz.device)
        rib_nodes = junc0.unsqueeze(0) + \
                    t.unsqueeze(1) * (junc1 - junc0).unsqueeze(0)

        # ---- 4. Wall: smoothing spline -> RoC ----
        spline_cache = self._build_spline_cache(wall_xyz)
        der1, der2 = query_spline_derivatives(
            spline_cache['s'],
            spline_cache['g'],
            spline_cache['M'],
            spline_cache['h'],
            self.overlapCount,
            samples_per_edge=1,
        )
        wall_roc = RoCCalc(der1, der2)
        wall_roc = wall_roc.clamp(min=self.min_roc, max=self.flat_transition)

        # ---- 5. Rib RoC: straight -> flat_transition ----
        rib_roc = torch.full((n_rib_total,), self.flat_transition,
                             device=wall_xyz.device)

        # ---- 6. Wall perimeter and node lengths ----
        wall_perimeter, wall_edge_lengths = perimeterCalc(wall_xyz)
        wall_node_length = (wall_edge_lengths +
                            torch.roll(wall_edge_lengths, 1, dims=0)) / 2

        # ---- 7. Rib edge lengths ----
        rib_diffs = rib_nodes[1:] - rib_nodes[:-1]
        rib_edge_lengths = torch.norm(rib_diffs, dim=1)
        rib_length = rib_edge_lengths.sum()

        # Rib node lengths (half-sum of adjacent edges)
        rib_node_length = torch.zeros(n_rib_total, device=wall_xyz.device)
        rib_node_length[0] = rib_edge_lengths[0] / 2
        rib_node_length[-1] = rib_edge_lengths[-1] / 2
        for i in range(1, n_rib_total - 1):
            rib_node_length[i] = (rib_edge_lengths[i - 1] +
                                  rib_edge_lengths[i]) / 2

        # ---- 8. Junction nodes: combine wall + rib contributions ----
        # Average RoC at junctions
        j0_roc = (wall_roc[self.j0_wall_idx] + rib_roc[0]) / 2
        j1_roc = (wall_roc[self.j1_wall_idx] + rib_roc[-1]) / 2

        # Build combined wall RoC (replace junction values with averages)
        roc_wall = wall_roc.clone()
        roc_wall[self.j0_wall_idx] = j0_roc
        roc_wall[self.j1_wall_idx] = j1_roc

        # Add rib edge contribution to junction node lengths
        nl_wall = wall_node_length.clone()
        nl_wall[self.j0_wall_idx] = nl_wall[self.j0_wall_idx] + \
                                     rib_edge_lengths[0] / 2
        nl_wall[self.j1_wall_idx] = nl_wall[self.j1_wall_idx] + \
                                     rib_edge_lengths[-1] / 2

        # ---- 9. Forces ----
        # Wall force (includes junction nodes with rib contributions)
        wall_stress = self.constantCrushStress * CurvatureEq(roc_wall)
        wall_force = (wall_stress * nl_wall * self.wall_thickness).sum()

        # Rib interior force (junctions already counted in wall)
        rib_int_roc = rib_roc[1:-1]
        rib_int_nl = rib_node_length[1:-1]
        rib_int_stress = self.constantCrushStress * CurvatureEq(rib_int_roc)
        rib_force = (rib_int_stress * rib_int_nl * self.rib_thickness).sum()

        # ---- 10. Totals ----
        force_total = wall_force + rib_force
        perimeter_total = wall_perimeter + rib_length

        return force_total, perimeter_total, wall_force, rib_force, \
               wall_perimeter, rib_length


#==========================================================================
#              LOSS FUNCTION
#==========================================================================

def compute_loss(mode, force_total, perimeter, model,
                 force_initial, smoothness_initial,
                 lambdas, max_expansion_ratio,
                 box_constraint_samples_per_edge,
                 box_dims=None):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF'    -- maximise crush force (outward expansion, capped).
        'minP'     -- minimise perimeter (inward only, force floor).
        'maxFPP'   -- maximise force-per-perimeter ratio.
        'boxMaxCF' -- maximise crush force within a bounding box.
    force_total, perimeter : Tensor
        Current values from forward pass.
    model : crushRibOptimizerModel
        The model (for accessing r, initial_r, theta, phi, centroid).
    force_initial : Tensor
        Force at epoch 0.
    smoothness_initial : Tensor
        Smoothness at epoch 0 (for normalisation).
    lambdas : dict
        Penalty weights: 'smooth', 'force', 'radial', 'box'.
    max_expansion_ratio : float
        Max allowed r / r_initial ratio for outward radial bounds.
    box_constraint_samples_per_edge : int
        Dense sampling resolution for box checks on the smoothed wall.
    box_dims : tuple or None
        (box_width, box_height) in mm, centered at centroid. boxMaxCF only.
    """
    smoothness = smoothnessTerm(model.r, model.theta, model.phi, model.centroid) \
                 / (smoothness_initial.detach() + 1e-12)

    if mode == 'maxCF':
        force_penalty = torch.relu(force_initial - force_total) ** 2
        radial_penalty = _radial_penalty(model, mode, max_expansion_ratio)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * radial_penalty)

    elif mode == 'minP':
        # Inward only: prevent outward expansion, maintain force
        force_penalty = torch.relu(force_initial - force_total) ** 2
        radial_penalty = _radial_penalty(model, mode, max_expansion_ratio)
        loss = (perimeter
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * radial_penalty)

    elif mode == 'maxFPP':
        force_per_perimeter = force_total / perimeter
        force_penalty = torch.relu(force_initial - force_total) ** 2
        radial_penalty = _radial_penalty(model, mode, max_expansion_ratio)
        loss = (-force_per_perimeter
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * radial_penalty)

    elif mode == 'boxMaxCF':
        box_penalty = _box_penalty(
            model, box_dims, box_constraint_samples_per_edge)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['box'] * box_penalty)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Use 'maxCF', 'minP', 'maxFPP' (legacy alias: 'maxSEA'), "
            f"or 'boxMaxCF'.")

    return loss


def objective_value(mode: str,
                    force_total: torch.Tensor,
                    perimeter: torch.Tensor) -> float:
    """Return the scalar objective used for early stopping/restoration."""
    if mode in ('maxCF', 'boxMaxCF'):
        return force_total.item()
    if mode == 'minP':
        return -perimeter.item()
    if mode == 'maxFPP':
        return (force_total / perimeter).item()
    raise ValueError(f"Unknown mode '{mode}'.")


def satisfies_force_floor(force_total: torch.Tensor,
                          force_floor: torch.Tensor,
                          rel_tol: float = 1e-4) -> bool:
    """Treat the force floor as satisfied within a tiny numerical tolerance."""
    return force_total.item() >= force_floor.item() * (1.0 - rel_tol)


def satisfies_box_constraint(model,
                             box_dims,
                             box_constraint_samples_per_edge: int,
                             abs_tol: float = 1e-10) -> bool:
    """Treat the box constraint as satisfied when the penalty is numerically zero."""
    if box_dims is None:
        return True
    return _box_penalty(
        model, box_dims, box_constraint_samples_per_edge).item() <= abs_tol


def satisfies_radial_constraint(model,
                                mode,
                                max_expansion_ratio,
                                abs_tol: float = 1e-10) -> bool:
    if mode not in {'maxCF', 'minP', 'maxFPP'}:
        return True
    return _radial_penalty(model, mode, max_expansion_ratio).item() <= abs_tol


def objective_improved(current: float,
                       best: float,
                       rel_tol: float = 1e-12,
                       abs_tol: float = 1e-9) -> bool:
    """Return True when the scalar objective is meaningfully better."""
    if not np.isfinite(best):
        return True
    threshold = abs_tol + rel_tol * max(abs(current), abs(best), 1.0)
    return current > best + threshold


def calibrate_spline_p_to_force(target_force: float,
                                force_eval_fn,
                                auto_p: Optional[float] = None) -> Dict[str, object]:
    """Choose spline p so the initial force best matches a reference target."""
    if target_force <= 0:
        raise ValueError("Target initial force must be positive.")

    cache: Dict[float, float] = {}

    def eval_force(p_val: float) -> float:
        p_val = float(np.clip(p_val, 1e-12, 1.0 - 1e-12))
        key = round(p_val, 12)
        if key not in cache:
            cache[key] = float(force_eval_fn(p_val))
        return cache[key]

    candidate_ps = [
        1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
        0.1, 0.2, 0.4, 0.6, 0.8,
        0.9, 0.95, 0.98, 0.99, 0.995,
        0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999999,
    ]
    if auto_p is not None:
        candidate_ps.append(float(auto_p))
    candidate_ps = sorted(set(
        float(np.clip(p, 1e-12, 1.0 - 1e-12)) for p in candidate_ps))

    samples = []
    for p_val in candidate_ps:
        samples.append((p_val, eval_force(p_val)))

    best_idx = min(
        range(len(samples)),
        key=lambda idx: abs(samples[idx][1] - target_force),
    )
    best_p, best_force = samples[best_idx]
    refined = False

    if len(samples) >= 3 and 0 < best_idx < len(samples) - 1:
        lo = samples[best_idx - 1][0]
        hi = samples[best_idx + 1][0]
        if hi - lo > 1e-9:
            opt = minimize_scalar(
                lambda p_val: abs(eval_force(p_val) - target_force),
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 30},
            )
            p_refined = float(np.clip(opt.x, 1e-12, 1.0 - 1e-12))
            force_refined = eval_force(p_refined)
            if abs(force_refined - target_force) < abs(best_force - target_force):
                best_p = p_refined
                best_force = force_refined
                refined = True

    return {
        'p': best_p,
        'force': best_force,
        'error': best_force - target_force,
        'refined': refined,
        'samples': samples,
        'edge_hit': best_idx in (0, len(samples) - 1),
    }


#==========================================================================
#              PLOTTING
#==========================================================================

def plot_initial_and_final(wall_init, wall_final, rib_init, rib_final,
                           save_path="rib_optim_result.pdf", box_dims=None):
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11
        })
    except Exception:
        plt.rcParams.update({"font.size": 11})

    fig, ax = plt.subplots(figsize=(6, 4))

    # Initial wall
    ax.plot(wall_init[:, 0], wall_init[:, 1],
            'o-', markersize=3, linewidth=0.75,
            color=(0, 0.447, 0.698), label='Initial wall')
    # Final wall
    ax.plot(wall_final[:, 0], wall_final[:, 1],
            'o-', markersize=3, linewidth=0.75,
            color=(0.902, 0.624, 0), label='Optimised wall')

    # Initial rib
    if rib_init is not None and len(rib_init) > 0:
        ax.plot(rib_init[:, 0], rib_init[:, 1],
                's-', markersize=4, linewidth=1.0,
                color=(0, 0.447, 0.698), alpha=0.5, label='Initial rib')
    # Final rib
    if rib_final is not None and len(rib_final) > 0:
        ax.plot(rib_final[:, 0], rib_final[:, 1],
                's-', markersize=4, linewidth=1.0,
                color=(0.902, 0.624, 0), alpha=0.5, label='Optimised rib')

    # Box constraint
    if box_dims is not None:
        hw, hh = box_dims[0] / 2, box_dims[1] / 2
        rect = plt.Rectangle((-hw, -hh), box_dims[0], box_dims[1],
            linewidth=1.0, edgecolor='red', facecolor='none',
            linestyle='--', label='Box constraint')
        ax.add_patch(rect)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect('equal')
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return fig


def _single_role_thickness(values, role_name, decimals=6):
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.size == 0:
        raise ValueError(f"No nodes found for {role_name} thickness extraction.")
    uniq = np.unique(np.round(vals, decimals=decimals))
    if len(uniq) != 1:
        raise ValueError(
            f"{role_name} nodes have {len(uniq)} distinct thickness values: "
            f"{uniq.tolist()}. v7 expects a single {role_name} thickness."
        )
    return float(uniq[0])


#==========================================================================
#              LOGGING
#==========================================================================

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


#==========================================================================
#              MAIN
#==========================================================================

# ---- CLI overrides ----
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data', nargs='?', default=None)
parser.add_argument('mode', nargs='?', default=None)
parser.add_argument('--steps',    type=int,   default=None)
parser.add_argument('--spline-p', type=float, default=None,
                    help='Manual smoothing spline parameter (0,1].')
parser.add_argument('--match-initial-force', type=float, default=None,
                    help='Calibrate spline p so the initial force matches this target in N.')
parser.add_argument('--patience', type=int,   default=None,
                    help='Early stopping patience (epochs).')
parser.add_argument('--tol',      type=float, default=None,
                    help='Early stopping relative tolerance.')
parser.add_argument('--box-width',  type=float, default=None,
                    help='Bounding box width in mm (boxMaxCF).')
parser.add_argument('--box-height', type=float, default=None,
                    help='Bounding box height in mm (boxMaxCF).')
parser.add_argument('--minp-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for minP.')
parser.add_argument('--maxcf-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for maxCF/boxMaxCF.')
parser.add_argument('--maxfpp-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for maxFPP.')
parser.add_argument('--maxsea-smooth-ratio', type=float, default=None,
                    help='Legacy alias for --maxfpp-smooth-ratio.')
parser.add_argument('--tag', type=str, default=None,
                    help='Optional filename suffix for this run.')
parser.add_argument('--no-cleanup', action='store_true',
                    help='Skip wall node cleanup.')
parser.add_argument('--no-resample', action='store_true',
                    help='Skip uniform resampling after cleanup.')
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
    smooth_weight_ratio['maxFPP'] = args.maxsea_smooth_ratio
if args.maxfpp_smooth_ratio is not None:
    smooth_weight_ratio['maxFPP'] = args.maxfpp_smooth_ratio
if args.no_cleanup:
    cleanup_short_ratio = 0.0  # effectively disable
if args.no_resample:
    resample_uniform = False
if args.match_initial_force is not None and args.spline_p is not None:
    raise ValueError(
        "Use either --spline-p or --match-initial-force, not both.")

legacy_mode_aliases = {
    'maxSEA': 'maxFPP',
}
if run_mode in legacy_mode_aliases:
    run_mode = legacy_mode_aliases[run_mode]

valid_modes = {'maxCF', 'minP', 'maxFPP', 'boxMaxCF'}
if run_mode not in valid_modes:
    raise ValueError(f"Invalid run_mode '{run_mode}'. Choose from {valid_modes}.")

run_tag = ''
if args.tag:
    _tag_clean = re.sub(r'[^A-Za-z0-9_-]+', '-', args.tag).strip('-')
    if _tag_clean:
        run_tag = f"_{_tag_clean}"

# ---- Resolve input path and build temporary output filenames ----
_data_path = Path(dataFileName)
if not _data_path.is_absolute():
    _data_path = DATA_DIR / _data_path
_data_path = _data_path.resolve()

run_timestamp = datetime.now().strftime('%d-%m')
_shape = _data_path.stem.replace('partition_data_', '')
_tmp_base = f"{_shape}_rib_{run_mode}{run_tag}_running_{run_timestamp}_(v{script_version})"

log_filename = str(RESULTS_OUTROS / f"{_tmp_base}.txt")
mat_filename = str(RESULTS_OUTROS / f"{_tmp_base}.mat")
pdf_filename = str(RESULTS_OUTROS / f"{_tmp_base}.pdf")

tee = TeeLogger(log_filename)
sys.stdout = tee
print(f"Log file: {log_filename}\n")

#==========================================================================
# ---- Load and decompose geometry ----
print("=" * 60)
print("LOADING AND DECOMPOSING GEOMETRY")
print("=" * 60)

data = loadmat(str(_data_path))
all_nodes = np.array(data['nodesCoords'], dtype=np.float64)
all_edges = np.array(data['connectivity'], dtype=int) - 1  # MATLAB 1-based

# ---- Load per-node thickness ----
if 'thickness' in data:
    all_thickness = np.asarray(data['thickness'], dtype=float).flatten()
    if all_thickness.shape[0] != all_nodes.shape[0]:
        raise ValueError(
            f"Thickness array has {all_thickness.shape[0]} entries for "
            f"{all_nodes.shape[0]} nodes."
        )
    _unique_t = np.sort(np.unique(np.round(all_thickness, decimals=6)))
    if len(_unique_t) >= 5:
        raise ValueError(
            f"Found {len(_unique_t)} distinct thickness values in '{dataFileName}': "
            f"{_unique_t.tolist()}. v7 currently supports wall, rib, and up to "
            "two distinct junction thicknesses."
        )
else:
    raise ValueError("No 'thickness' field found in .mat file. "
                     "Export thickness from MATLAB partition data.")

print(f"File: {dataFileName}")
print(f"Total nodes: {all_nodes.shape[0]}, edges: {all_edges.shape[0]}")

# Extract wall and rib
info = extract_wall_and_ribs(all_nodes, all_edges)

wall_loop = info['wall_loop']
rib_interior = info['rib_interior']
j0, j1 = info['junction_nodes']
j0_wall_idx = info['j0_wall_idx']
j1_wall_idx = info['j1_wall_idx']

wall_non_junction = [idx for idx in wall_loop if idx not in {j0, j1}]
wall_thickness_val = _single_role_thickness(
    all_thickness[wall_non_junction], role_name='wall'
)
if rib_interior:
    rib_thickness_val = _single_role_thickness(
        all_thickness[rib_interior], role_name='rib interior'
    )
else:
    rib_thickness_val = float(_unique_t[0])
j0_thickness_val = float(all_thickness[j0])
j1_thickness_val = float(all_thickness[j1])

print(f"Wall nodes: {len(wall_loop)}")
print(f"Rib interior nodes: {len(rib_interior)}")
print(f"Junction nodes: {j0}, {j1} (wall indices {j0_wall_idx}, {j1_wall_idx})")
print(f"Thickness from .mat: wall={wall_thickness_val:.4f}, "
      f"junctions=[{j0_thickness_val:.4f}, {j1_thickness_val:.4f}], "
      f"rib={rib_thickness_val:.4f} mm")

# Extract wall node coordinates
wall_nodes_np = all_nodes[wall_loop]

# ---- Cleanup: remove redundant short-edge nodes ----
n_before_cleanup = len(wall_nodes_np)
wall_nodes_clean, _surviving_idx, n_removed = cleanup_closed_loop(
    wall_nodes_np,
    protected_indices={j0_wall_idx, j1_wall_idx},
    short_ratio=cleanup_short_ratio,
    angle_tol_deg=cleanup_angle_tol_deg,
    excess_rel_tol=cleanup_excess_rel_tol,
)
if n_removed > 0:
    # Update junction indices after cleanup
    j0_wall_idx_clean = int(np.where(_surviving_idx == j0_wall_idx)[0][0])
    j1_wall_idx_clean = int(np.where(_surviving_idx == j1_wall_idx)[0][0])
    print(f"Cleanup: removed {n_removed} redundant nodes "
          f"({n_before_cleanup} -> {len(wall_nodes_clean)})")
    print(f"  Junction indices updated: {j0_wall_idx}->{j0_wall_idx_clean}, "
          f"{j1_wall_idx}->{j1_wall_idx_clean}")
else:
    j0_wall_idx_clean = j0_wall_idx
    j1_wall_idx_clean = j1_wall_idx

# ---- Resample: uniform arc-length spacing ----
if resample_uniform:
    wall_nodes_final, landmark_map = resample_closed_loop_uniform(
        wall_nodes_clean,
        target_count=len(wall_nodes_clean),
        corner_angle_threshold_deg=corner_angle_threshold_deg,
        extra_landmark_indices={j0_wall_idx_clean, j1_wall_idx_clean},
    )
    # Get junction indices in the resampled array
    j0_wall_idx_final = landmark_map.get(j0_wall_idx_clean, j0_wall_idx_clean)
    j1_wall_idx_final = landmark_map.get(j1_wall_idx_clean, j1_wall_idx_clean)
    print(f"Uniform resampling: {len(wall_nodes_final)} nodes")
    if landmark_map:
        print(f"  Junction indices updated: "
              f"{j0_wall_idx_clean}->{j0_wall_idx_final}, "
              f"{j1_wall_idx_clean}->{j1_wall_idx_final}")
else:
    wall_nodes_final = wall_nodes_clean
    j0_wall_idx_final = j0_wall_idx_clean
    j1_wall_idx_final = j1_wall_idx_clean

# Update info dict with final junction indices
info['j0_wall_idx'] = j0_wall_idx_final
info['j1_wall_idx'] = j1_wall_idx_final
j0_wall_idx = j0_wall_idx_final
j1_wall_idx = j1_wall_idx_final

wall_nodes = torch.tensor(wall_nodes_final, dtype=torch.float32).detach()

#==========================================================================
# ---- Build model ----
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

def _build_model_for_spline_p(p_val):
    return crushRibOptimizerModel(
        wall_nodes=wall_nodes,
        wall_info=info,
        constantCrushStress=constantCrushStress,
        overlapCount=overlapCount,
        wall_thickness=wall_thickness_val,
        junction_thicknesses=(j0_thickness_val, j1_thickness_val),
        rib_thickness=rib_thickness_val,
        flat_transition=flat_transition,
        min_roc=min_roc,
        spline_p=p_val,
    )


calibration_result = None
if args.match_initial_force is not None:
    _tmp_ext = extend_nodes(overlapCount, wall_nodes)
    _tmp_s = chordLength(_tmp_ext)
    _auto_p = _default_smooth_p(_tmp_s[1:] - _tmp_s[:-1])

    def _evaluate_initial_force_for_p(p_val):
        tmp_model = _build_model_for_spline_p(p_val)
        with torch.no_grad():
            force_total, _, _, _, _, _ = tmp_model()
        return force_total.item()

    print("Calibrating spline p to match reference initial force...")
    calibration_result = calibrate_spline_p_to_force(
        target_force=args.match_initial_force,
        force_eval_fn=_evaluate_initial_force_for_p,
        auto_p=_auto_p,
    )
    spline_p = calibration_result['p']
    force_err = calibration_result['error']
    force_err_rel = 100.0 * force_err / args.match_initial_force
    print(f"  Target force:       {args.match_initial_force:.4f} N")
    print(f"  Calibrated spline p:{spline_p:.8f}")
    print(f"  Matched force:      {calibration_result['force']:.4f} N")
    print(f"  Match error:        {force_err:+.4f} N ({force_err_rel:+.4f}%)")
    if calibration_result['edge_hit']:
        print("  Note: best match landed on the calibration grid boundary.")

model = _build_model_for_spline_p(spline_p)

# ---- Compute initial values ----
with torch.no_grad():
    F_init, P_init, Fw_init, Fr_init, Pw_init, Pr_init = model()
    force_per_perimeter_init = F_init / P_init
    smooth_init = smoothnessTerm(model.r, model.theta, model.phi, model.centroid)

    print(f"Force total:     {F_init.item():.2f} N  ({F_init.item()/1000:.2f} kN)")
    print(f"  Wall force:    {Fw_init.item():.2f} N")
    print(f"  Rib force:     {Fr_init.item():.2f} N")
    print(f"  Rib fraction:  {Fr_init.item()/F_init.item()*100:.1f}%")
    print(f"Perimeter total: {P_init.item():.2f} mm")
    print(f"  Wall perim:    {Pw_init.item():.2f} mm")
    print(f"  Rib length:    {Pr_init.item():.2f} mm")
    print(f"F/P:             {force_per_perimeter_init.item():.4f} N/mm")

    # Save initial wall + rib for plotting
    wall_init_cart = Spherical2Cartesian(model.r, model.theta, model.phi)
    wall_init_ext = extend_nodes(overlapCount, wall_init_cart).cpu().numpy()

    # Initial rib nodes
    junc0_init = wall_init_cart[j0_wall_idx]
    junc1_init = wall_init_cart[j1_wall_idx]
    n_rib_total = info['n_rib_interior'] + 2
    t_rib = torch.linspace(
        0, 1, n_rib_total, device=wall_init_cart.device, dtype=wall_init_cart.dtype)
    rib_init_nodes = (junc0_init.unsqueeze(0) +
                      t_rib.unsqueeze(1) * (junc1_init - junc0_init).unsqueeze(0))
    rib_init_np = rib_init_nodes.cpu().numpy()

# ---- Auto-select lambda_smoothness ----
if run_mode in ('maxCF', 'boxMaxCF'):
    objective_scale = F_init.item()
elif run_mode == 'minP':
    objective_scale = P_init.item()
elif run_mode == 'maxFPP':
    objective_scale = force_per_perimeter_init.item()

lambda_smoothness = smooth_weight_ratio[run_mode] * objective_scale
print(f"\nLambda smoothness (auto): {lambda_smoothness:.2f}  "
      f"(ratio={smooth_weight_ratio[run_mode]} x objective={objective_scale:.2f})")

lambdas = {
    'smooth':      lambda_smoothness,
    'force':       lambda_crushForce,
    'radial':      lambda_radialDist,
    'box':         lambda_box,
}

# Force floor used for early stopping feasibility check:
# boxMaxCF has no force floor because force is already the objective.
if run_mode == 'boxMaxCF':
    _force_floor = None
else:
    _force_floor = F_init

_active_box_dims = ((box_width, box_height)
                    if run_mode == 'boxMaxCF' else None)

with torch.no_grad():
    initial_loss = compute_loss(
        mode=run_mode,
        force_total=F_init,
        perimeter=P_init,
        model=model,
        force_initial=F_init,
        smoothness_initial=smooth_init,
        lambdas=lambdas,
        max_expansion_ratio=max_expansion_ratio,
        box_constraint_samples_per_edge=box_constraint_samples_per_edge,
        box_dims=_active_box_dims,
    ).item()
    initial_objective = objective_value(run_mode, F_init, P_init)

#==========================================================================
# ---- Optimisation loop ----
print("\n" + "=" * 60)
print(f"OPTIMISATION: {run_mode} ({n_steps} max steps, "
      f"patience={conv_patience}, tol={conv_tol})")
print("=" * 60)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_steps, eta_min=1e-4)

best_loss = initial_loss
best_r = model.r.data.clone()
best_epoch = -1
nan_detected = False
converged = False

# Early stopping state: track the best feasible objective.
initial_force_ok = (True if _force_floor is None
                    else satisfies_force_floor(
                        F_init, _force_floor, rel_tol=force_floor_rel_tol))
initial_radial_ok = satisfies_radial_constraint(
    model, run_mode, max_expansion_ratio)
initial_box_ok = satisfies_box_constraint(
    model, _active_box_dims, box_constraint_samples_per_edge)
initial_feasible = initial_force_ok and initial_radial_ok and initial_box_ok

best_objective = initial_objective if initial_feasible else float('-inf')
best_objective_r = model.r.data.clone() if initial_feasible else None
best_objective_epoch = -1 if initial_feasible else None
best_progress_objective = best_objective
epochs_without_improvement = 0

t_start = time.perf_counter()

for epoch in range(n_steps):
    optimizer.zero_grad()
    force_total, perimeter, wall_f, rib_f, wall_p, rib_l = model()

    loss = compute_loss(
        mode=run_mode,
        force_total=force_total,
        perimeter=perimeter,
        model=model,
        force_initial=F_init,
        smoothness_initial=smooth_init,
        lambdas=lambdas,
        max_expansion_ratio=max_expansion_ratio,
        box_constraint_samples_per_edge=box_constraint_samples_per_edge,
        box_dims=_active_box_dims,
    )

    if torch.isnan(loss) or torch.isinf(loss):
        model.r.data.copy_(best_r)
        nan_detected = True
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        force_total, perimeter, wall_f, rib_f, wall_p, rib_l = model()
        loss_eval = compute_loss(
            mode=run_mode,
            force_total=force_total,
            perimeter=perimeter,
            model=model,
            force_initial=F_init,
            smoothness_initial=smooth_init,
            lambdas=lambdas,
            max_expansion_ratio=max_expansion_ratio,
            box_constraint_samples_per_edge=box_constraint_samples_per_edge,
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

    # ---- Early stopping check ----
    force_per_perimeter_now = (force_total / perimeter).item()
    current_objective = objective_value(run_mode, force_total, perimeter)
    force_floor_ok = (True if _force_floor is None
                      else satisfies_force_floor(
                          force_total, _force_floor, rel_tol=force_floor_rel_tol))
    radial_ok = satisfies_radial_constraint(model, run_mode, max_expansion_ratio)
    box_ok = satisfies_box_constraint(
        model, _active_box_dims, box_constraint_samples_per_edge)
    feasible_now = force_floor_ok and radial_ok and box_ok

    if feasible_now and objective_improved(current_objective, best_objective):
        best_objective = current_objective
        best_objective_r = model.r.data.clone()
        best_objective_epoch = epoch

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

    # Terminal-only progress
    if epoch % snapshot_interval == 0:
        current_lr = optimizer.param_groups[0]['lr']
        tee.terminal.write(
            f"Epoch {epoch:5d}: F={force_total.item():.0f} "
            f"(wall={wall_f.item():.0f} + rib={rib_f.item():.0f}), "
            f"P={perimeter.item():.1f} "
            f"(wall={wall_p.item():.1f} + rib={rib_l.item():.1f}), "
            f"F/P={force_per_perimeter_now:.2f}, "
            f"Loss={current_loss:.2f}, "
            f"LR={current_lr:.6f}\n"
        )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

# Restore best objective state
if not nan_detected:
    if best_objective_r is not None:
        model.r.data.copy_(best_objective_r)
    else:
        model.r.data.copy_(best_r)

# ---- Rename output files to final names ----
_final_base = f"{_shape}_rib_{run_mode}{run_tag}_{run_timestamp}_(v{script_version})"

log_filename_final = str(RESULTS_OUTROS / f"{_final_base}.txt")
mat_filename_final = str(RESULTS_OUTROS / f"{_final_base}.mat")
pdf_filename_final = str(RESULTS_OUTROS / f"{_final_base}.pdf")

# Close the tee so the temp log file is flushed and released
tee.close()
# Rename temp files
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

#==========================================================================
# ---- Results ----
with torch.no_grad():
    F_final, P_final, Fw_final, Fr_final, Pw_final, Pr_final = model()
    force_per_perimeter_final = F_final / P_final
    final_force_ok = (True if _force_floor is None
                      else satisfies_force_floor(
                          F_final, _force_floor, rel_tol=force_floor_rel_tol))
    final_radial_ok = satisfies_radial_constraint(
        model, run_mode, max_expansion_ratio)
    final_box_ok = satisfies_box_constraint(
        model, _active_box_dims, box_constraint_samples_per_edge)
    final_feasible = final_force_ok and final_radial_ok and final_box_ok

    # Save final geometry
    wall_final_cart = Spherical2Cartesian(model.r, model.theta, model.phi)
    wall_final_ext = extend_nodes(overlapCount, wall_final_cart).cpu().numpy()

    junc0_final = wall_final_cart[j0_wall_idx]
    junc1_final = wall_final_cart[j1_wall_idx]
    rib_final_nodes = (junc0_final.unsqueeze(0) +
                       t_rib.unsqueeze(1) *
                       (junc1_final - junc0_final).unsqueeze(0))
    rib_final_np = rib_final_nodes.cpu().numpy()

    # Plot
    plot_initial_and_final(
        wall_init_ext, wall_final_ext,
        rib_init_np, rib_final_np,
        save_path=pdf_filename,
        box_dims=_active_box_dims,
    )

    # Build connectivity for wall (closed loop)
    wall_final_abs = (wall_final_cart + model.centroid).cpu().numpy()
    rib_final_abs = (rib_final_nodes + model.centroid.squeeze(0)).cpu().numpy()
    n_wall = wall_final_abs.shape[0]
    wall_conn = np.column_stack([np.arange(1, n_wall + 1),
                                 np.roll(np.arange(1, n_wall + 1), -1)])

    savemat(mat_filename, {
        'wall_nodesCoords':       wall_final_abs,
        'wall_connectivity':      wall_conn,
        'rib_nodesCoords':        rib_final_abs,
        'junction_wall_indices':  np.array([j0_wall_idx + 1,
                                            j1_wall_idx + 1]),
        'wallThickness':          np.float64(wall_thickness_val),
        'junctionThicknesses':    np.array([j0_thickness_val,
                                            j1_thickness_val],
                                           dtype=np.float64),
        'ribThickness':           np.float64(rib_thickness_val),
        'splineP':                np.float64(spline_p if spline_p is not None else np.nan),
        'forceInitial':           np.float64(F_init.item()),
        'forceFinal':             np.float64(F_final.item()),
        'perimeterInitial':       np.float64(P_init.item()),
        'perimeterFinal':         np.float64(P_final.item()),
        'forcePerPerimeterInitial': np.float64(force_per_perimeter_init.item()),
        'forcePerPerimeterFinal':   np.float64(force_per_perimeter_final.item()),
        'seaInitial':             np.float64(force_per_perimeter_init.item()),
        'seaFinal':               np.float64(force_per_perimeter_final.item()),
        'wallForceInitial':       np.float64(Fw_init.item()),
        'wallForceFinal':         np.float64(Fw_final.item()),
        'ribForceInitial':        np.float64(Fr_init.item()),
        'ribForceFinal':          np.float64(Fr_final.item()),
    })

# ---- Stopping reason ----
_best_loss_label = 'initial state' if best_epoch < 0 else f'epoch {best_epoch}'
_best_objective_label = (
    'initial state' if best_objective_epoch < 0
    else f'epoch {best_objective_epoch}'
)
if nan_detected:
    stop_reason = (
        f'NaN/Inf at epoch {final_epoch} (rolled back to {_best_loss_label})'
    )
elif best_objective_r is None:
    stop_reason = f'No feasible solution found; restored best loss state from {_best_loss_label}'
elif converged:
    stop_reason = (f'Converged at epoch {final_epoch} '
                   f'(restored best feasible objective from {_best_objective_label})')
else:
    stop_reason = (f'Step limit reached ({n_steps}), '
                   f'restored best feasible objective from {_best_objective_label}')

# ---- Runtime summary ----
mode_labels = {
    'maxCF':    'Max crush force (outward only, force floor)',
    'minP':     'Min perimeter (inward)',
    'maxFPP':   'Max force-per-perimeter ratio',
    'boxMaxCF': 'Max crush force (box constrained)',
}

force_change_pct = ((F_final.item() - F_init.item()) / F_init.item()) * 100
perim_change_pct = ((P_final.item() - P_init.item()) / P_init.item()) * 100
fpp_change_pct = (
    (force_per_perimeter_final.item() - force_per_perimeter_init.item())
    / force_per_perimeter_init.item()
) * 100

print("=" * 60)
print("RIB OPTIMIZER - RUN SUMMARY")
print("=" * 60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {mode_labels[run_mode]}")
print(f"  Spline p:           {spline_p if spline_p is not None else 'auto'}")
if calibration_result is not None:
    print(f"  Target init force:  {args.match_initial_force:.4f} N")
    print(f"  Init force error:   {calibration_result['error']:+.4f} N")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
print(f"  Thickness:          wall={wall_thickness_val:.4f}, "
      f"junc=[{j0_thickness_val:.4f}, {j1_thickness_val:.4f}], "
      f"rib={rib_thickness_val:.4f} mm")
if run_mode == 'boxMaxCF':
    print(f"  Box constraint:     {box_width:.1f} x {box_height:.1f} mm")
if run_mode in {'maxCF', 'minP', 'maxFPP'}:
    print(f"  Radial envelope:    {'yes' if final_radial_ok else 'no'}")
if run_mode == 'boxMaxCF':
    print(f"  Box satisfied:      {'yes' if final_box_ok else 'no'}")
if _force_floor is not None:
    print(f"  Force floor met:    {'yes' if final_force_ok else 'no'}")
print(f"  Final feasible:     {'yes' if final_feasible else 'no'}")
print("  ---")
print("  {:<12} {:>12} {:>12} {:>10}".format(
    "Metric", "Initial", "Final", "Change"))
print("  {:<12} {:>12} {:>12} {:>10}".format(
    "-" * 12, "-" * 12, "-" * 12, "-" * 10))
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Force", F_init.item(), F_final.item(), force_change_pct))
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Perimeter", P_init.item(), P_final.item(), perim_change_pct))
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "F/P", force_per_perimeter_init.item(),
    force_per_perimeter_final.item(), fpp_change_pct))
print("  ---")
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Wall Force", Fw_init.item(), Fw_final.item(),
    ((Fw_final.item() - Fw_init.item()) / Fw_init.item()) * 100))
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Rib Force", Fr_init.item(), Fr_final.item(),
    ((Fr_final.item() - Fr_init.item()) / (Fr_init.item() + 1e-12)) * 100))
print("  {:<12} {:>12.4f} {:>12.4f} {:>+9.2f}%".format(
    "Rib Length", Pr_init.item(), Pr_final.item(),
    ((Pr_final.item() - Pr_init.item()) / Pr_init.item()) * 100))
print(f"  Rib fraction:       "
      f"{Fr_init.item()/F_init.item()*100:.1f}% -> "
      f"{Fr_final.item()/F_final.item()*100:.1f}%")
print(f"  ---")
print(f"  Epochs:             {final_epoch + 1}")
print(f"  Stop reason:        {stop_reason}")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("=" * 60)

tee.close()
