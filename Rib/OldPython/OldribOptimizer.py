
#                     ribOptimizer.py
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
#  Modes:
#    maxCF    -- maximise total crush force (wall + rib)
#    minP     -- minimise total perimeter while maintaining force
#    maxSEA   -- maximise specific energy absorption (F / P)
#    boxMaxCF -- maximise crush force within a bounding box
#    boxMinP  -- minimise perimeter within a bounding box
#
#  v19-aligned changes:
#    - Smoothing cubic spline (Reinsch form) replacing NaturalCubicSpline.
#    - Edge-length-weighted smoothness term (v17+).
#    - Early stopping with conv_patience/conv_tol and best-objective restore.
#    - maxCF: bidirectional bounded with inward soft penalty and force floor (v18).
#    - boxMaxCF / boxMinP modes (v19).
#    - spline_p CLI parameter.
#    - File rename pattern (_running_ -> final).
#    - Removed torchcubicspline dependency.
#
#  Usage:
#    python ribOptimizer.py                                    # defaults
#    python ribOptimizer.py partition_data_sis_rib.mat maxCF   # CLI
#
#  Dependencies: torch, scipy, numpy, matplotlib (NO torchcubicspline)
#
#  Francisco Arriscado -- FEUP, 2026
#==========================================================================

script_version = 3

from typing import Tuple, List, Dict, Optional
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from collections import defaultdict
import sys
import os
import time
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

# ---- Project paths (config.py at project root) ---- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_RIB

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

    sigma = torch.linalg.solve(A, rhs)     # (m, d) -- differentiable

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
    X = torch.round(X * 10000) / 10000
    Y = torch.round(Y * 10000) / 10000
    Z = torch.round(Z * 10000) / 10000
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


def splines(nodesExt, orderedNodes, s, overlapCount, spline_p
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a smoothing cubic spline to the extended node coordinates and
    evaluate first and second derivatives at linspace query points
    matching MATLAB's curvatureSpline convention.
    """
    nOriginal = orderedNodes.shape[0]

    # Fit smoothing spline to extended nodes
    g, M, h = fit_smoothing_spline(s, nodesExt, p=spline_p)

    # Query range: from first original node to last original node
    sStart = s[overlapCount]
    sEnd_matlab = s[-overlapCount]
    sQuery = torch.linspace(
        sStart.item(), sEnd_matlab.item(), nOriginal,
        device=s.device
    )

    # Evaluate derivatives at query points
    splineDer1, splineDer2 = eval_spline_derivatives(s, g, M, h, sQuery)

    return splineDer1, splineDer2


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
                 thickness: float,
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

        # Physics constants
        self.constantCrushStress = constantCrushStress
        self.overlapCount = overlapCount
        self.thickness = thickness
        self.flat_transition = flat_transition
        self.min_roc = min_roc
        self.spline_p = spline_p

    def forward(self):
        # ---- 1. Wall Cartesian coordinates ----
        wall_xyz = Spherical2Cartesian(self.r, self.theta, self.phi) \
                   + self.centroid

        # ---- 2. Junction positions (from wall) ----
        junc0 = wall_xyz[self.j0_wall_idx]
        junc1 = wall_xyz[self.j1_wall_idx]

        # ---- 3. Rib: straight line between junctions ----
        n_rib_total = self.n_rib_interior + 2  # including junction endpoints
        t = torch.linspace(0, 1, n_rib_total, device=wall_xyz.device)
        rib_nodes = junc0.unsqueeze(0) + \
                    t.unsqueeze(1) * (junc1 - junc0).unsqueeze(0)

        # ---- 4. Wall: smoothing spline -> RoC ----
        wall_ext = extend_nodes(self.overlapCount, wall_xyz)
        s = chordLength(wall_ext)
        der1, der2 = splines(wall_ext, wall_xyz, s, self.overlapCount,
                             self.spline_p)
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
        wall_force = (wall_stress * nl_wall * self.thickness).sum()

        # Rib interior force (junctions already counted in wall)
        rib_int_roc = rib_roc[1:-1]
        rib_int_nl = rib_node_length[1:-1]
        rib_int_stress = self.constantCrushStress * CurvatureEq(rib_int_roc)
        rib_force = (rib_int_stress * rib_int_nl * self.thickness).sum()

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
                 lambdas, max_expansion_ratio, max_contraction_ratio,
                 box_dims=None):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF'    -- maximise crush force (bidirectional, radially bounded).
        'minP'     -- minimise perimeter (inward only, force floor).
        'maxSEA'   -- maximise specific energy absorption (F / P).
        'boxMaxCF' -- maximise crush force within a bounding box.
        'boxMinP'  -- minimise perimeter within a bounding box.
    force_total, perimeter : Tensor
        Current values from forward pass.
    model : crushRibOptimizerModel
        The model (for accessing r, initial_r, theta, phi, centroid).
    force_initial : Tensor
        Force at epoch 0.
    smoothness_initial : Tensor
        Smoothness at epoch 0 (for normalisation).
    lambdas : dict
        Penalty weights: 'smooth', 'force', 'radial', 'box', 'inward_soft'.
    max_expansion_ratio : float
        Max allowed r / r_initial ratio for outward radial bounds.
    max_contraction_ratio : float
        Max allowed inward contraction ratio. Lower bound is
        r_initial / max_contraction_ratio.
    box_dims : tuple or None
        (box_width, box_height) in mm, centered at centroid. boxMaxCF/boxMinP only.
    """
    smoothness = smoothnessTerm(model.r, model.theta, model.phi, model.centroid) \
                 / (smoothness_initial.detach() + 1e-12)

    if mode == 'maxCF':
        force_penalty = torch.relu(force_initial - force_total) ** 2
        inward_soft = torch.sum(torch.relu(model.initial_r - model.r) ** 2)
        r_min = (1.0 / max_contraction_ratio) * model.initial_r
        r_max = max_expansion_ratio * model.initial_r
        radial_inward = torch.sum(torch.relu(r_min - model.r) ** 2)
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['inward_soft'] * inward_soft
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
        radial_inward = torch.sum(torch.relu(r_min - model.r) ** 2)
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        loss = (-sea
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * (radial_inward + radial_outward))

    elif mode == 'boxMaxCF':
        force_penalty = torch.relu(force_initial - force_total) ** 2
        r_max = max_expansion_ratio * model.initial_r
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        X = model.r * torch.sin(model.theta) * torch.cos(model.phi)
        Y = model.r * torch.sin(model.theta) * torch.sin(model.phi)
        half_w = box_dims[0] / 2.0
        half_h = box_dims[1] / 2.0
        box_penalty = (torch.sum(torch.relu(X - half_w) ** 2)
                     + torch.sum(torch.relu(-half_w - X) ** 2)
                     + torch.sum(torch.relu(Y - half_h) ** 2)
                     + torch.sum(torch.relu(-half_h - Y) ** 2))
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['radial'] * radial_outward
                + lambdas['box'] * box_penalty)

    elif mode == 'boxMinP':
        force_penalty = torch.relu(force_initial - force_total) ** 2
        X = model.r * torch.sin(model.theta) * torch.cos(model.phi)
        Y = model.r * torch.sin(model.theta) * torch.sin(model.phi)
        half_w = box_dims[0] / 2.0
        half_h = box_dims[1] / 2.0
        box_penalty = (torch.sum(torch.relu(X - half_w) ** 2)
                     + torch.sum(torch.relu(-half_w - X) ** 2)
                     + torch.sum(torch.relu(Y - half_h) ** 2)
                     + torch.sum(torch.relu(-half_h - Y) ** 2))
        loss = (perimeter
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['box'] * box_penalty)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Use 'maxCF', 'minP', 'maxSEA', 'boxMaxCF', or 'boxMinP'.")

    return loss


def objective_value(mode: str,
                    force_total: torch.Tensor,
                    perimeter: torch.Tensor) -> float:
    """Return the scalar objective used for early stopping/restoration."""
    if mode in ('maxCF', 'boxMaxCF'):
        return force_total.item()
    if mode in ('minP', 'boxMinP'):
        return -perimeter.item()
    if mode == 'maxSEA':
        return (force_total / perimeter).item()
    raise ValueError(f"Unknown mode '{mode}'.")


def satisfies_force_floor(force_total: torch.Tensor,
                          force_initial: torch.Tensor,
                          rel_tol: float = 1e-4) -> bool:
    """Treat the force floor as satisfied within a tiny numerical tolerance."""
    return force_total.item() >= force_initial.item() * (1.0 - rel_tol)


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

# ---- User inputs (defaults) ----
thickness = 2.0                         # [mm]
dataFileName = 'partition_data_testing_CU.mat'
constantCrushStress = 90.0              # [MPa]
flat_transition = 1221.95               # [mm]
min_roc = 15.0                          # [mm]

# ---- Smoothing spline parameter ---- #
spline_p = None   # None = auto (csaps heuristic)

# ---- Optimisation mode ---- #
# 'maxCF'    -- maximise crush force (bounded inward/outward movement)
# 'minP'     -- minimise perimeter while preserving force (inward only)
# 'maxSEA'   -- maximise specific energy absorption (Force / Perimeter)
# 'boxMaxCF' -- maximise crush force within a bounding box
# 'boxMinP'  -- minimise perimeter within a bounding box
run_mode = 'minP'

# ---- Bounding box (boxMaxCF / boxMinP modes only) ---- #
box_width  = 35                  # [mm] box extent in X
box_height = 35                  # [mm] box extent in Y

# Hyperparameters
n_steps = 100000                        # hard ceiling (early stopping usually triggers first)
learning_rate = 0.1
grad_clip_norm = 1.0
snapshot_interval = 500
overlapCount = 10

# ---- Early stopping ---- #
conv_patience = 1000                    # epochs to wait for improvement
conv_tol = 1e-3                         # relative improvement threshold

# Penalty weights
lambda_crushForce = 1e6
lambda_radialDist = 1e6
lambda_box = 1e10
lambda_inwardSoft = 1e4

smooth_weight_ratio = {
    'maxCF':    2.0,
    'minP':    0.2,
    'maxSEA':   2.0,
    'boxMaxCF': 2.0,
    'boxMinP': 10.0,
}

max_expansion_ratio = 1.3
max_contraction_ratio = 1.05
force_floor_rel_tol = 1e-4

# ---- CLI overrides ----
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data', nargs='?', default=None)
parser.add_argument('mode', nargs='?', default=None)
parser.add_argument('--steps',    type=int,   default=None)
parser.add_argument('--spline-p', type=float, default=None,
                    help='Smoothing spline parameter (0,1]. None=auto.')
parser.add_argument('--patience', type=int,   default=None,
                    help='Early stopping patience (epochs).')
parser.add_argument('--tol',      type=float, default=None,
                    help='Early stopping relative tolerance.')
parser.add_argument('--max-contraction', type=float, default=None,
                    help='Max inward contraction ratio for maxCF.')
parser.add_argument('--box-width',  type=float, default=None,
                    help='Bounding box width in mm (boxMaxCF/boxMinP).')
parser.add_argument('--box-height', type=float, default=None,
                    help='Bounding box height in mm (boxMaxCF/boxMinP).')
parser.add_argument('--minp-smooth-ratio', type=float, default=None,
                    help='Override smoothness weight ratio for minP/boxMinP.')
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
if args.max_contraction is not None:
    max_contraction_ratio = args.max_contraction
if args.box_width is not None:
    box_width = args.box_width
if args.box_height is not None:
    box_height = args.box_height
if args.minp_smooth_ratio is not None:
    smooth_weight_ratio['minP'] = args.minp_smooth_ratio
    smooth_weight_ratio['boxMinP'] = args.minp_smooth_ratio

valid_modes = {'maxCF', 'minP', 'maxSEA', 'boxMaxCF', 'boxMinP'}
if run_mode not in valid_modes:
    raise ValueError(f"Invalid run_mode '{run_mode}'. Choose from {valid_modes}.")

run_tag = ''
if args.tag:
    _tag_clean = re.sub(r'[^A-Za-z0-9_-]+', '-', args.tag).strip('-')
    if _tag_clean:
        run_tag = f"_{_tag_clean}"

# ---- Build temporary output filenames (renamed after loop) ----
run_timestamp = datetime.now().strftime('%d-%m')
_shape = dataFileName.replace('partition_data_', '').replace('.mat', '')
_tmp_base = f"{_shape}_rib_{run_mode}{run_tag}_running_{run_timestamp}_(v{script_version})"

log_filename = str(RESULTS_RIB / f"{_tmp_base}.txt")
mat_filename = str(RESULTS_RIB / f"{_tmp_base}.mat")
pdf_filename = str(RESULTS_RIB / f"{_tmp_base}.pdf")

tee = TeeLogger(log_filename)
sys.stdout = tee
print(f"Log file: {log_filename}\n")

#==========================================================================
# ---- Load and decompose geometry ----
print("=" * 60)
print("LOADING AND DECOMPOSING GEOMETRY")
print("=" * 60)

_data_path = Path(dataFileName)
if not _data_path.is_absolute():
    _data_path = DATA_DIR / _data_path

data = loadmat(str(_data_path))
all_nodes = np.array(data['nodesCoords'], dtype=np.float64)
all_edges = np.array(data['connectivity'], dtype=int) - 1  # MATLAB 1-based

print(f"File: {dataFileName}")
print(f"Total nodes: {all_nodes.shape[0]}, edges: {all_edges.shape[0]}")

# Extract wall and rib
info = extract_wall_and_ribs(all_nodes, all_edges)

wall_loop = info['wall_loop']
rib_branch = info['rib_branch']
rib_interior = info['rib_interior']
j0, j1 = info['junction_nodes']
j0_wall_idx = info['j0_wall_idx']
j1_wall_idx = info['j1_wall_idx']

print(f"Wall nodes: {len(wall_loop)}")
print(f"Rib interior nodes: {len(rib_interior)}")
print(f"Junction nodes: {j0}, {j1} (wall indices {j0_wall_idx}, {j1_wall_idx})")

# Extract wall node coordinates
wall_nodes_np = all_nodes[wall_loop]
wall_nodes = torch.tensor(wall_nodes_np, dtype=torch.float32).detach()

#==========================================================================
# ---- Build model ----
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

model = crushRibOptimizerModel(
    wall_nodes=wall_nodes,
    wall_info=info,
    constantCrushStress=constantCrushStress,
    overlapCount=overlapCount,
    thickness=thickness,
    flat_transition=flat_transition,
    min_roc=min_roc,
    spline_p=spline_p,
)

# ---- Compute initial values ----
with torch.no_grad():
    F_init, P_init, Fw_init, Fr_init, Pw_init, Pr_init = model()
    sea_init = F_init / P_init
    smooth_init = smoothnessTerm(model.r, model.theta, model.phi, model.centroid)

    print(f"Force total:     {F_init.item():.2f} N  ({F_init.item()/1000:.2f} kN)")
    print(f"  Wall force:    {Fw_init.item():.2f} N")
    print(f"  Rib force:     {Fr_init.item():.2f} N")
    print(f"  Rib fraction:  {Fr_init.item()/F_init.item()*100:.1f}%")
    print(f"Perimeter total: {P_init.item():.2f} mm")
    print(f"  Wall perim:    {Pw_init.item():.2f} mm")
    print(f"  Rib length:    {Pr_init.item():.2f} mm")
    print(f"SEA:             {sea_init.item():.4f} N/mm")

    # Save initial wall + rib for plotting
    wall_init_cart = Spherical2Cartesian(model.r, model.theta, model.phi)
    wall_init_ext = extend_nodes(overlapCount, wall_init_cart).cpu().numpy()

    # Initial rib nodes
    junc0_init = wall_init_cart[j0_wall_idx]
    junc1_init = wall_init_cart[j1_wall_idx]
    n_rib_total = info['n_rib_interior'] + 2
    t_rib = torch.linspace(0, 1, n_rib_total)
    rib_init_nodes = (junc0_init.unsqueeze(0) +
                      t_rib.unsqueeze(1) * (junc1_init - junc0_init).unsqueeze(0))
    rib_init_np = rib_init_nodes.cpu().numpy()

# Report effective spline_p
_effective_p = spline_p
if _effective_p is None:
    _tmp_ext = extend_nodes(overlapCount, wall_nodes)
    _tmp_s = chordLength(_tmp_ext)
    _tmp_h = _tmp_s[1:] - _tmp_s[:-1]
    _effective_p = _default_smooth_p(_tmp_h)

# ---- Auto-select lambda_smoothness ----
if run_mode in ('maxCF', 'boxMaxCF'):
    objective_scale = F_init.item()
elif run_mode in ('minP', 'boxMinP'):
    objective_scale = P_init.item()
elif run_mode == 'maxSEA':
    objective_scale = sea_init.item()

lambda_smoothness = smooth_weight_ratio[run_mode] * objective_scale
print(f"\nLambda smoothness (auto): {lambda_smoothness:.2f}  "
      f"(ratio={smooth_weight_ratio[run_mode]} x objective={objective_scale:.2f})")

lambdas = {
    'smooth':      lambda_smoothness,
    'force':       lambda_crushForce,
    'radial':      lambda_radialDist,
    'box':         lambda_box,
    'inward_soft': lambda_inwardSoft,
}

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
        max_contraction_ratio=max_contraction_ratio,
        box_dims=(box_width, box_height),
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
best_objective = initial_objective
best_objective_r = model.r.data.clone()
best_objective_epoch = -1
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
        max_contraction_ratio=max_contraction_ratio,
        box_dims=(box_width, box_height),
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
            max_contraction_ratio=max_contraction_ratio,
            box_dims=(box_width, box_height),
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
    sea_now = (force_total / perimeter).item()
    current_objective = objective_value(run_mode, force_total, perimeter)
    force_floor_ok = satisfies_force_floor(
        force_total, F_init, rel_tol=force_floor_rel_tol)

    if force_floor_ok and \
       (current_objective - best_objective) / (abs(best_objective) + 1e-12) > conv_tol:
        best_objective = current_objective
        best_objective_r = model.r.data.clone()
        best_objective_epoch = epoch
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
            f"SEA={sea_now:.2f}, "
            f"Loss={current_loss:.2f}, "
            f"LR={current_lr:.6f}\n"
        )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

# Restore best objective state
if not nan_detected:
    model.r.data.copy_(best_objective_r)

# ---- Rename output files to final names ----
_final_base = f"{_shape}_rib_{run_mode}{run_tag}_{run_timestamp}_(v{script_version})"

log_filename_final = str(RESULTS_RIB / f"{_final_base}.txt")
mat_filename_final = str(RESULTS_RIB / f"{_final_base}.mat")
pdf_filename_final = str(RESULTS_RIB / f"{_final_base}.pdf")

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
    sea_final = F_final / P_final

    # Save final geometry
    wall_final_cart = Spherical2Cartesian(model.r, model.theta, model.phi)
    wall_final_np = wall_final_cart.cpu().numpy()
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
        box_dims=(box_width, box_height) if run_mode in ('boxMaxCF', 'boxMinP') else None,
    )

    # Build connectivity for wall (closed loop)
    n_wall = wall_final_np.shape[0]
    wall_conn = np.column_stack([np.arange(1, n_wall + 1),
                                 np.roll(np.arange(1, n_wall + 1), -1)])

    savemat(mat_filename, {
        'wall_nodesCoords':       wall_final_np,
        'wall_connectivity':      wall_conn,
        'rib_nodesCoords':        rib_final_np,
        'junction_wall_indices':  np.array([j0_wall_idx + 1,
                                            j1_wall_idx + 1]),
        'forceInitial':           np.float64(F_init.item()),
        'forceFinal':             np.float64(F_final.item()),
        'perimeterInitial':       np.float64(P_init.item()),
        'perimeterFinal':         np.float64(P_final.item()),
        'seaInitial':             np.float64(sea_init.item()),
        'seaFinal':               np.float64(sea_final.item()),
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
elif converged:
    stop_reason = (f'Converged at epoch {final_epoch} '
                   f'(restored best feasible objective from {_best_objective_label})')
else:
    stop_reason = (f'Step limit reached ({n_steps}), '
                   f'restored best feasible objective from {_best_objective_label}')

# ---- Runtime summary ----
mode_labels = {
    'maxCF':    'Max crush force (bidirectional bounded, force floor)',
    'minP':     'Min perimeter (inward)',
    'maxSEA':   'Max specific energy absorption',
    'boxMaxCF': 'Max crush force (box constrained)',
    'boxMinP':  'Min perimeter (box constrained)',
}

force_change_pct = ((F_final.item() - F_init.item()) / F_init.item()) * 100
perim_change_pct = ((P_final.item() - P_init.item()) / P_init.item()) * 100
sea_change_pct   = ((sea_final.item() - sea_init.item()) / sea_init.item()) * 100

print("=" * 60)
print("RIB OPTIMIZER - RUN SUMMARY")
print("=" * 60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {mode_labels[run_mode]}")
print(f"  Spline p:           {_effective_p:.6f}"
      f"{'  (auto)' if spline_p is None else ''}")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
if run_mode == 'maxCF':
    print(f"  Max contraction:    {max_contraction_ratio:.2f}")
    print(f"  Inward soft pen.:   {lambda_inwardSoft:.1e}")
if run_mode in ('boxMaxCF', 'boxMinP'):
    print(f"  Box constraint:     {box_width:.1f} x {box_height:.1f} mm")
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
    "SEA", sea_init.item(), sea_final.item(), sea_change_pct))
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
