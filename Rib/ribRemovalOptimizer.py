
#                  ribRemovalOptimizer.py
#
#  Rib-removal cross-section optimizer for CrushAnalytica.
#
#  Purpose:
#    Given a ribbed cross-section, answer: can the wall alone achieve the
#    same crush force without the rib?  If so, what wall shape minimises
#    perimeter (= material) while hitting F_target = F_wall + F_rib?
#
#  Pipeline (all self-contained, no subprocess calls):
#    1. Load ribbed .mat (nodesCoords + connectivity)
#    2. Decompose topology: identify wall loop + rib branch
#    3. Compute F_target on the full ribbed geometry (NumPy, same physics
#       as ribAnalysis.py)
#    4. Extract wall-only closed loop, clean short edges, optionally
#       resample uniformly
#    5. Optimise wall shape: minimise perimeter subject to
#       F_wall >= F_target, with smoothness and radial penalties
#    6. Export results: .txt log, .mat, .pdf plot
#
#  Single mode: minP with an externally derived force target.
#  All physics (spline, RoC, curvature eq, force integration) are
#  identical to optimizationModule_v21.
#
#  Usage:
#    python ribRemovalOptimizer.py partition_data_sis_rib.mat
#    python ribRemovalOptimizer.py partition_data_sis_rib.mat --steps 50000
#
#  Dependencies: torch, scipy, numpy, matplotlib (NO torchcubicspline)
#
#  Francisco Arriscado -- FEUP, 2026
#==========================================================================

script_version = 1

from typing import Tuple
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

# ---- Project paths ---- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Rib" / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "RibRemoval"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#==========================================================================
#==========================================================================
#                          User Inputs
#==========================================================================
#==========================================================================

# ---- Geometry / material defaults ---- #
thickness = 2.0                         # [mm]
dataFileName = 'partition_data_L3_uniform.mat'
constantCrushStress = 90.0              # [MPa]
flat_transition = 1221.95               # [mm]
min_roc = 15.0                          # [mm]

# ---- Penalty weights ---- #
lambda_crushForce = 1e6
lambda_radialDist = 1e6

# ---- Smoothness weight ratio ---- #
smooth_weight_ratio = 5

# ---- Geometry movement limits ---- #
max_expansion_ratio = 1.3

# ---- Smoothing spline parameter ---- #
spline_p = None   # None = auto (csaps heuristic)

# ---- Cleanup / resampling ---- #
cleanup_short_ratio = 0.35
cleanup_angle_tol_deg = 12.0
cleanup_excess_rel_tol = 0.02
resample_uniform = True
corner_angle_threshold_deg = 150.0

#==========================================================================
#==========================================================================
#                         Default Config
#==========================================================================
#==========================================================================

n_steps = 100000
learning_rate = 0.1
grad_clip_norm = 1.0
overlapCount = 10

# ---- Early stopping ---- #
conv_patience = 1000
conv_tol = 1e-3
force_floor_rel_tol = 1e-4


#==========================================================================
#==========================================================================
#     PHASE 1: RIBBED GEOMETRY ANALYSIS (topology in NumPy, force in
#              the same PyTorch/Reinsch pipeline as Phase 2)
#==========================================================================
#==========================================================================

# ---- Topology helpers ---- #

def _build_adjacency(edges, node_count):
    adj = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    degree = np.zeros(node_count, dtype=int)
    for node, nbs in adj.items():
        degree[node] = len(nbs)
    return adj, degree


def _decompose_branches(edges, adj, degree):
    comp_nodes = {int(a) for a, _ in edges} | {int(b) for _, b in edges}
    junctions = sorted(n for n in comp_nodes if degree[n] > 2)
    endpoints = sorted(n for n in comp_nodes if degree[n] == 1)
    edge_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}
    visited = set()

    def ek(a, b):
        return (min(a, b), max(a, b))

    branches = []
    for start in endpoints + junctions:
        for nb in adj[start]:
            if nb not in comp_nodes:
                continue
            e = ek(start, nb)
            if e not in edge_set or e in visited:
                continue
            path = [start, nb]
            visited.add(e)
            prev, cur = start, nb
            while True:
                if degree[cur] != 2:
                    break
                nbs = [n for n in adj[cur] if n in comp_nodes and n != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                e2 = ek(cur, nxt)
                if e2 in visited:
                    break
                visited.add(e2)
                path.append(nxt)
                prev, cur = cur, nxt
            branches.append(path)

    # Catch pure loops
    for a, b in edges:
        e = ek(int(a), int(b))
        if e in visited:
            continue
        loop = [int(a), int(b)]
        visited.add(e)
        prev, cur = int(a), int(b)
        while True:
            nbs = [n for n in adj[cur] if n in comp_nodes and n != prev]
            if not nbs:
                break
            nxt = nbs[0]
            e2 = ek(cur, nxt)
            if e2 in visited:
                break
            visited.add(e2)
            loop.append(nxt)
            prev, cur = cur, nxt
        branches.append(loop)

    return branches, junctions, endpoints


def compute_ribbed_force(nodes_np, edges_np, params):
    """Compute total crush force for a branched ribbed section.

    Uses the SAME PyTorch physics pipeline as the Phase 2 optimizer:
    Reinsch smoothing spline, node-parameter queries, CurvatureEq with
    rounded constants (7.95, -0.94, 0.99).  This ensures F_target is
    measured with the identical model that evaluates candidate shapes.

    Topology decomposition is NumPy; force computation is PyTorch under
    torch.no_grad().

    Mirrors the ribOptimizer v6 forward pass:
      - Wall closed loop: Reinsch spline -> RoC at original node params
      - Rib: straight line -> RoC = flat_transition
      - Junction nodes: averaged RoC, combined node lengths
    """
    n_nodes = nodes_np.shape[0]
    adj, degree = _build_adjacency(edges_np, n_nodes)
    branches, junctions, endpoints = _decompose_branches(edges_np, adj, degree)

    assert len(junctions) == 2, f"Expected 2 junctions, got {len(junctions)}"
    assert len(branches) == 3, f"Expected 3 branches, got {len(branches)}"

    # Identify wall vs rib
    lengths = []
    for br in branches:
        xyz = nodes_np[np.array(br)]
        lengths.append(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    rib_branch_idx = int(np.argmin(lengths))
    wall_indices = [i for i in range(3) if i != rib_branch_idx]

    # Build wall closed loop (same as extract_wall_from_ribbed)
    wall_loop = _join_wall_branches(branches[wall_indices[0]],
                                     branches[wall_indices[1]])
    rib_branch = branches[rib_branch_idx]

    # Find junction positions in wall loop
    j0, j1 = junctions
    j0_wall_idx = wall_loop.index(j0)
    j1_wall_idx = wall_loop.index(j1)

    # Rib interior node count (excluding junction endpoints)
    if rib_branch[0] in junctions and rib_branch[-1] in junctions:
        n_rib_interior = len(rib_branch) - 2
    else:
        n_rib_interior = max(0, len(rib_branch) - 2)

    with torch.no_grad():
        wall_xyz = torch.tensor(
            nodes_np[np.array(wall_loop)], dtype=torch.float32)
        oc = params['overlap_count']
        _flat = params['flat_transition']
        _min_roc = params['min_roc']
        _sigma0 = params['sigma0']
        _thick = params['thickness']
        _sp = params.get('spline_p', None)

        # ---- Wall: Reinsch smoothing spline -> RoC at node params ----
        wall_ext = extend_nodes(oc, wall_xyz)
        s = chordLength(wall_ext)
        g, M, h = fit_smoothing_spline(s, wall_ext, p=_sp)
        der1, der2 = query_spline_derivatives(s, g, M, h, oc,
                                               samples_per_edge=1)
        wall_roc = RoCCalc(der1, der2)
        wall_roc = wall_roc.clamp(min=_min_roc, max=_flat)

        # ---- Rib: straight line between junctions -> flat_transition ----
        junc0 = wall_xyz[j0_wall_idx]
        junc1 = wall_xyz[j1_wall_idx]
        n_rib_total = n_rib_interior + 2
        t_rib = torch.linspace(0, 1, n_rib_total)
        rib_nodes = (junc0.unsqueeze(0) +
                     t_rib.unsqueeze(1) * (junc1 - junc0).unsqueeze(0))
        rib_roc = torch.full((n_rib_total,), _flat)

        # ---- Wall perimeter and node lengths ----
        wall_perimeter, wall_edge_lengths = perimeterCalc(wall_xyz)
        wall_node_length = (wall_edge_lengths +
                            torch.roll(wall_edge_lengths, 1, dims=0)) / 2

        # ---- Rib edge lengths ----
        rib_diffs = rib_nodes[1:] - rib_nodes[:-1]
        rib_edge_lengths = torch.norm(rib_diffs, dim=1)
        rib_length = rib_edge_lengths.sum()

        rib_node_length = torch.zeros(n_rib_total)
        rib_node_length[0] = rib_edge_lengths[0] / 2
        rib_node_length[-1] = rib_edge_lengths[-1] / 2
        for i in range(1, n_rib_total - 1):
            rib_node_length[i] = (rib_edge_lengths[i - 1] +
                                  rib_edge_lengths[i]) / 2

        # ---- Junction averaging (same as ribOptimizer v6 forward) ----
        j0_roc = (wall_roc[j0_wall_idx] + rib_roc[0]) / 2
        j1_roc = (wall_roc[j1_wall_idx] + rib_roc[-1]) / 2

        roc_wall = wall_roc.clone()
        roc_wall[j0_wall_idx] = j0_roc
        roc_wall[j1_wall_idx] = j1_roc

        nl_wall = wall_node_length.clone()
        nl_wall[j0_wall_idx] = nl_wall[j0_wall_idx] + rib_edge_lengths[0] / 2
        nl_wall[j1_wall_idx] = nl_wall[j1_wall_idx] + rib_edge_lengths[-1] / 2

        # ---- Forces (same CurvatureEq as Phase 2) ----
        wall_stress = _sigma0 * CurvatureEq(roc_wall)
        wall_force = (wall_stress * nl_wall * _thick).sum()

        rib_int_roc = rib_roc[1:-1]
        rib_int_nl = rib_node_length[1:-1]
        rib_int_stress = _sigma0 * CurvatureEq(rib_int_roc)
        rib_force = (rib_int_stress * rib_int_nl * _thick).sum()

        force_total = wall_force + rib_force
        perimeter_total = wall_perimeter + rib_length

    return {
        'force_total': float(force_total.item()),
        'force_wall': float(wall_force.item()),
        'force_rib': float(rib_force.item()),
        'perimeter': float(perimeter_total.item()),
        'wall_perimeter': float(wall_perimeter.item()),
        'rib_length': float(rib_length.item()),
    }


# ---- Wall extraction ---- #

def _join_wall_branches(br_a, br_b):
    if br_a[-1] == br_b[0]:
        loop = br_a + br_b[1:]
    elif br_a[-1] == br_b[-1]:
        loop = br_a + br_b[-2::-1]
    elif br_a[0] == br_b[0]:
        loop = br_a[::-1] + br_b[1:]
    elif br_a[0] == br_b[-1]:
        loop = br_b + br_a[1:]
    else:
        raise ValueError("Wall branches do not share junction nodes.")
    if loop[-1] == loop[0]:
        loop = loop[:-1]
    return loop


def extract_wall_from_ribbed(nodes, edges):
    """Extract the wall closed loop from a branched ribbed section."""
    n_nodes = nodes.shape[0]
    adj, degree = _build_adjacency(edges, n_nodes)
    branches, junctions, endpoints = _decompose_branches(edges, adj, degree)

    assert len(junctions) == 2, f"Expected 2 junctions, got {len(junctions)}"
    assert len(endpoints) == 0, "Expected closed wall with rib, found open endpoints"
    assert len(branches) == 3, f"Expected 3 branches, got {len(branches)}"

    lengths = []
    for br in branches:
        xyz = nodes[np.array(br)]
        lengths.append(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    rib_idx = int(np.argmin(lengths))
    wall_indices = [i for i in range(3) if i != rib_idx]

    wall_loop = _join_wall_branches(branches[wall_indices[0]],
                                     branches[wall_indices[1]])
    wall_nodes = nodes[np.array(wall_loop)]

    return wall_nodes, {
        'junction_nodes': junctions,
        'rib_branch': branches[rib_idx],
        'rib_length': lengths[rib_idx],
        'wall_node_count': len(wall_loop),
    }


# ---- Node cleanup (from cleanup_partition_geometry_branched.py) ---- #

def _turning_angle_deg(prev_pt, pt, next_pt):
    v1 = prev_pt - pt
    v2 = next_pt - pt
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def cleanup_closed_loop(points, short_ratio=0.35, angle_tol_deg=12.0,
                         excess_rel_tol=0.02, excess_abs_tol=1e-3, min_nodes=8):
    """Remove redundant near-colinear short-edge nodes from a closed loop."""
    pts = points.copy()
    n_removed = 0

    while len(pts) > min_nodes:
        edge_lengths = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        median_edge = float(np.median(edge_lengths))
        changed = False

        for i in range(len(pts)):
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
            is_low_excess = excess <= max(excess_abs_tol, excess_rel_tol * median_edge)

            if is_short and is_straight and is_low_excess:
                pts = np.delete(pts, i, axis=0)
                n_removed += 1
                changed = True
                break

        if not changed:
            break

    return pts, n_removed


def resample_closed_loop_uniform(points, target_count=None,
                                  corner_angle_threshold_deg=150.0):
    """Resample a closed loop uniformly by arc length, preserving corners."""
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

    # Landmarks: corners (preserving order around loop)
    if not corners:
        # No corners: simple uniform resampling around the full loop
        pts_closed = np.vstack([points, points[:1]])
        seg_vecs = np.diff(pts_closed, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        total = float(np.sum(seg_lengths))
        if total <= 0:
            return points
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
        return result

    # With corners: segment-aware resampling
    landmarks = sorted(set(corners))
    n_segments = len(landmarks)

    # Compute segment lengths (between consecutive landmarks around loop)
    seg_lengths = []
    seg_polylines = []
    for s in range(n_segments):
        start_idx = landmarks[s]
        end_idx = landmarks[(s + 1) % n_segments]
        # Build polyline from start to end around the loop
        if end_idx > start_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = list(range(start_idx, n)) + list(range(0, end_idx + 1))
        poly = points[indices]
        seg_polylines.append(poly)
        length = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        seg_lengths.append(length)

    total_length = sum(seg_lengths)
    if total_length <= 0:
        return points

    # Allocate node counts per segment proportional to length
    target_intervals = target_count
    weights = np.array(seg_lengths) / total_length
    raw = weights * target_intervals
    counts = np.maximum(np.floor(raw).astype(int), 1)
    deficit = target_intervals - int(np.sum(counts))
    if deficit > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[:deficit]:
            counts[idx] += 1

    # Resample each segment
    pieces = []
    for i, (poly, n_intervals) in enumerate(zip(seg_polylines, counts)):
        n_pts = int(n_intervals) + 1
        seg_vecs_local = np.diff(poly, axis=0)
        seg_lens_local = np.linalg.norm(seg_vecs_local, axis=1)
        total_local = float(np.sum(seg_lens_local))
        if total_local <= 0 or n_pts < 2:
            pieces.append(poly[:1])
            continue
        cumul = np.concatenate([[0.0], np.cumsum(seg_lens_local)])
        sample_s = np.linspace(0.0, total_local, n_pts, endpoint=True)
        resampled = np.zeros((n_pts, poly.shape[1]))
        for j, s in enumerate(sample_s):
            seg_idx = int(np.searchsorted(cumul[1:], s, side='right'))
            seg_idx = min(seg_idx, len(seg_lens_local) - 1)
            seg_start = cumul[seg_idx]
            seg_len = seg_lens_local[seg_idx]
            if seg_len <= 0:
                resampled[j] = poly[seg_idx]
            else:
                alpha = (s - seg_start) / seg_len
                resampled[j] = poly[seg_idx] + alpha * seg_vecs_local[seg_idx]
        # Drop last point (will be the start of the next segment)
        pieces.append(resampled[:-1])

    return np.vstack(pieces)


#==========================================================================
#==========================================================================
#           PHASE 2: WALL-ONLY OPTIMIZER (PyTorch, from v21)
#==========================================================================
#==========================================================================

# ---- Smoothing cubic spline (Reinsch form) ---- #

def _build_spline_matrices(t):
    n = t.shape[0]
    m = n - 2
    h = t[1:] - t[:-1]
    Q = torch.zeros(n, m, dtype=t.dtype, device=t.device)
    j = torch.arange(m, device=t.device)
    Q[j, j]     =  1.0 / h[:-1]
    Q[j + 1, j] = -(1.0 / h[:-1] + 1.0 / h[1:])
    Q[j + 2, j] =  1.0 / h[1:]
    R = torch.zeros(m, m, dtype=t.dtype, device=t.device)
    R[j, j] = (h[:-1] + h[1:]) / 3.0
    if m > 1:
        k = torch.arange(m - 1, device=t.device)
        R[k, k + 1] = h[1:-1] / 6.0
        R[k + 1, k] = h[1:-1] / 6.0
    return Q, R, h


def _default_smooth_p(h):
    h_avg = h.mean().item()
    return 1.0 / (1.0 + h_avg**3 / 6.0)


def fit_smoothing_spline(t, y, p=None):
    n = t.shape[0]
    Q, R, h = _build_spline_matrices(t)
    if p is None:
        p = _default_smooth_p(h)
    p = max(min(p, 1.0 - 1e-15), 1e-15)
    QtQ = Q.t() @ Q
    A = p * R + (1.0 - p) * QtQ
    rhs = p * (Q.t() @ y)
    sigma = torch.linalg.solve(A, rhs)
    g = y - ((1.0 - p) / p) * (Q @ sigma)
    M = torch.zeros(n, y.shape[1], dtype=y.dtype, device=y.device)
    M[1:-1] = sigma
    return g, M, h


def closed_loop_query_params(s, overlapCount, samples_per_edge=1):
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


def eval_spline_derivatives(t, g, M, h, tq):
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
    der1 = (-Mi * dt_right**2 / (2.0 * hi_d)
            + Mi1 * dt_left**2 / (2.0 * hi_d)
            + (gi1 - gi) / hi_d
            - (Mi1 - Mi) * hi_d / 6.0)
    der2 = Mi * dt_right / hi_d + Mi1 * dt_left / hi_d
    return der1, der2


# ---- PyTorch physics (identical to v21) ---- #

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


def query_spline_derivatives(s, g, M, h, overlapCount, samples_per_edge=1):
    sQuery = closed_loop_query_params(s, overlapCount, samples_per_edge)
    return eval_spline_derivatives(s, g, M, h, sQuery)


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
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    nodes = torch.stack([X, Y, Z], dim=1) + centroid
    diffs = torch.roll(nodes, -1, dims=0) - nodes
    edge_len = torch.norm(diffs, dim=1)
    node_len = (edge_len + torch.roll(edge_len, 1, dims=0)) / 2.0
    node_len = node_len.clamp(min=1e-8)
    weights = node_len / node_len.mean()
    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    dr = r_wrapped[1:] - r_wrapped[:-1]
    curv = dr[1:] - dr[:-1]
    return torch.sum(weights * curv ** 2)


def _radial_penalty_minP(model, max_expansion_ratio):
    """minP: allow shrink + bounded expansion."""
    r_min = (1.0 / max_expansion_ratio) * model.initial_r
    r_max = max_expansion_ratio * model.initial_r
    return (
        torch.sum(torch.relu(r_min - model.r) ** 2)
        + torch.sum(torch.relu(model.r - r_max) ** 2)
    )


# ---- Model ---- #

class WallOptimizerModel(nn.Module):
    """Wall-only optimizer model (spherical parameterisation)."""

    def __init__(self, orderedNodes, constantCrushStress, overlapCount,
                 thickness, flat_transition, min_roc, spline_p):
        super().__init__()
        self.centroid = orderedNodes.mean(dim=0, keepdim=True)
        r, theta, phi = Cartesian2Spherical(orderedNodes - self.centroid)
        self.r = nn.Parameter(r)
        self.register_buffer('initial_r', r.clone())
        self.register_buffer('theta', theta)
        self.register_buffer('phi', phi)
        self.constantCrushStress = constantCrushStress
        self.overlapCount = overlapCount
        self.thickness = thickness
        self.flat_transition = flat_transition
        self.min_roc = min_roc
        self.spline_p = spline_p

    def _reconstruct_nodes(self):
        return Spherical2Cartesian(self.r, self.theta, self.phi) + self.centroid

    def forward(self):
        orderedNodes = self._reconstruct_nodes()
        nodesExt = extend_nodes(self.overlapCount, orderedNodes)
        s = chordLength(nodesExt)
        g, M, h = fit_smoothing_spline(s, nodesExt, p=self.spline_p)
        der1, der2 = query_spline_derivatives(s, g, M, h, self.overlapCount)
        RoC = RoCCalc(der1, der2)
        RoC = RoC.clamp(min=self.min_roc, max=self.flat_transition)
        crushStress = self.constantCrushStress * CurvatureEq(RoC)
        perimeter, edgeLengths = perimeterCalc(orderedNodes)
        nodeLength = (edgeLengths + torch.roll(edgeLengths, 1, dims=0)) / 2
        forceTotal = (crushStress * nodeLength * self.thickness).sum()
        return forceTotal, perimeter


# ---- Loss ---- #

def compute_loss(force_total, perimeter, model, force_target,
                 smoothness_initial, lambdas, max_expansion_ratio):
    smoothness = smoothnessTerm(
        model.r, model.theta, model.phi, model.centroid
    ) / (smoothness_initial.detach() + 1e-12)
    force_penalty = torch.relu(force_target - force_total) ** 2
    radial_penalty = _radial_penalty_minP(model, max_expansion_ratio)
    loss = (perimeter
            + lambdas['smooth'] * smoothness
            + lambdas['force'] * force_penalty
            + lambdas['radial'] * radial_penalty)
    return loss


def satisfies_force_floor(force_total, force_floor, rel_tol=1e-4):
    return force_total.item() >= force_floor.item() * (1.0 - rel_tol)


def satisfies_radial_constraint(model, max_expansion_ratio, abs_tol=1e-10):
    return _radial_penalty_minP(model, max_expansion_ratio).item() <= abs_tol


def objective_improved(current, best, rel_tol=1e-12, abs_tol=1e-9):
    if not np.isfinite(best):
        return True
    threshold = abs_tol + rel_tol * max(abs(current), abs(best), 1.0)
    return current > best + threshold


# ---- Plot ---- #

def plot_results(wall_init, wall_final, save_path="result.pdf"):
    def _configure(use_tex):
        style = {"font.size": 11, "text.usetex": use_tex}
        if use_tex:
            style.update({"font.family": "serif",
                          "font.serif": ["Computer Modern Roman"]})
        else:
            style.update({"font.family": "DejaVu Serif"})
        plt.rcParams.update(style)

    def _save():
        fig, ax = plt.subplots(figsize=(4.72, 2.25))
        try:
            ax.plot(wall_init[:, 0], wall_init[:, 1],
                    marker='o', markersize=4, linewidth=0.75,
                    color=(0, 0.447, 0.698), label='Initial (rib removed)')
            ax.plot(wall_final[:, 0], wall_final[:, 1],
                    marker='o', markersize=4, linewidth=0.75,
                    color=(0.902, 0.624, 0), label='Optimised')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            plt.subplots_adjust(bottom=0.25)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                      ncol=2, frameon=False)
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        _configure(use_tex=True)
        _save()
    except Exception:
        _configure(use_tex=False)
        _save()


# ---- TeeLogger ---- #

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
#==========================================================================
#                          MAIN SCRIPT
#==========================================================================
#==========================================================================

# ---- CLI ---- #
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data',       nargs='?', default=None)
parser.add_argument('--steps',    type=int,   default=None)
parser.add_argument('--spline-p', type=float, default=None)
parser.add_argument('--patience', type=int,   default=None)
parser.add_argument('--tol',      type=float, default=None)
parser.add_argument('--smooth-ratio', type=float, default=None)
parser.add_argument('--tag',      type=str,   default=None)
parser.add_argument('--no-cleanup', action='store_true',
                    help='Skip node cleanup on extracted wall.')
parser.add_argument('--no-resample', action='store_true',
                    help='Skip uniform resampling on extracted wall.')
args, _ = parser.parse_known_args()

if args.data is not None:
    dataFileName = args.data
if args.steps is not None:
    n_steps = args.steps
if args.spline_p is not None:
    spline_p = args.spline_p
if args.patience is not None:
    conv_patience = args.patience
if args.tol is not None:
    conv_tol = args.tol
if args.smooth_ratio is not None:
    smooth_weight_ratio = args.smooth_ratio
if args.no_cleanup:
    cleanup_short_ratio = 0.0  # effectively disable
if args.no_resample:
    resample_uniform = False

run_tag = ''
if args.tag:
    _tag_clean = re.sub(r'[^A-Za-z0-9_-]+', '-', args.tag).strip('-')
    if _tag_clean:
        run_tag = f"_{_tag_clean}"

# ---- Resolve paths ---- #
_data_path = Path(dataFileName)
if not _data_path.is_absolute():
    _data_path = DATA_DIR / _data_path
_data_path = _data_path.resolve()

run_timestamp = datetime.now().strftime('%d-%m')
_shape = _data_path.stem.replace('partition_data_', '')
_tmp_base = f"{_shape}_ribRemoval{run_tag}_running_{run_timestamp}_(v{script_version})"

log_filename = str(RESULTS_DIR / f"{_tmp_base}.txt")
mat_filename = str(RESULTS_DIR / f"{_tmp_base}.mat")
pdf_filename = str(RESULTS_DIR / f"{_tmp_base}.pdf")

tee = TeeLogger(log_filename)
sys.stdout = tee

#==========================================================================
# PHASE 1: Load ribbed geometry, compute F_target, extract wall
#==========================================================================

print("=" * 60)
print("RIB REMOVAL OPTIMIZER")
print("=" * 60)
print(f"\n  Input file: {_data_path.name}")

# Load
raw = loadmat(str(_data_path))
nodes_ribbed = np.asarray(raw['nodesCoords'], dtype=float)
edges_ribbed = np.asarray(raw['connectivity'], dtype=int) - 1  # 0-based

print(f"  Ribbed geometry: {nodes_ribbed.shape[0]} nodes, "
      f"{edges_ribbed.shape[0]} edges")

# Compute F_target on full ribbed geometry
ribbed_params = {
    'thickness': thickness,
    'sigma0': constantCrushStress,
    'flat_transition': flat_transition,
    'min_roc': min_roc,
    'overlap_count': overlapCount,
    'spline_p': spline_p,
}
ribbed_result = compute_ribbed_force(nodes_ribbed, edges_ribbed, ribbed_params)
F_target = ribbed_result['force_total']

print(f"\n  Ribbed section analysis:")
print(f"    F_total  = {F_target:.2f} N  (this is the target)")
print(f"    F_wall   = {ribbed_result['force_wall']:.2f} N")
print(f"    F_rib    = {ribbed_result['force_rib']:.2f} N  "
      f"({ribbed_result['force_rib']/F_target*100:.1f}%)")
print(f"    Perimeter = {ribbed_result['perimeter']:.2f} mm")

# Extract wall
wall_nodes_raw, wall_info = extract_wall_from_ribbed(nodes_ribbed, edges_ribbed)
print(f"\n  Wall extraction: {wall_info['wall_node_count']} nodes")
print(f"    Rib length removed: {wall_info['rib_length']:.2f} mm")

# Cleanup
n_before_cleanup = len(wall_nodes_raw)
wall_nodes_clean, n_removed = cleanup_closed_loop(
    wall_nodes_raw,
    short_ratio=cleanup_short_ratio,
    angle_tol_deg=cleanup_angle_tol_deg,
    excess_rel_tol=cleanup_excess_rel_tol,
)
if n_removed > 0:
    print(f"  Cleanup: removed {n_removed} redundant nodes "
          f"({n_before_cleanup} -> {len(wall_nodes_clean)})")

# Resample
if resample_uniform:
    wall_nodes_final = resample_closed_loop_uniform(
        wall_nodes_clean,
        target_count=len(wall_nodes_clean),
        corner_angle_threshold_deg=corner_angle_threshold_deg,
    )
    print(f"  Uniform resampling: {len(wall_nodes_final)} nodes")
else:
    wall_nodes_final = wall_nodes_clean

#==========================================================================
# PHASE 2: Wall-only optimisation (PyTorch)
#==========================================================================

print(f"\n  Optimizer config:")
print(f"    Mode:            minP (force target from ribbed section)")
print(f"    F_target:        {F_target:.2f} N")
print(f"    Min RoC:         {min_roc:.1f} mm")
print(f"    Max expansion:   {max_expansion_ratio:.2f}")
print(f"    Smooth ratio:    {smooth_weight_ratio}")
print(f"    Steps (max):     {n_steps}")
print(f"    Patience:        {conv_patience}")

# Convert to torch
orderedNodes = torch.tensor(wall_nodes_final, dtype=torch.float32).detach()
force_target_tensor = torch.tensor(F_target, dtype=torch.float32)

model = WallOptimizerModel(
    orderedNodes, constantCrushStress, overlapCount,
    thickness, flat_transition, min_roc, spline_p,
)

# Initial values
with torch.no_grad():
    F_init_wall, P_init_wall = model()
    fpp_init = F_init_wall / P_init_wall
    smooth_init = smoothnessTerm(model.r, model.theta, model.phi, model.centroid)
    nodes_init_rel = Spherical2Cartesian(model.r, model.theta, model.phi)
    nodesExt_init = extend_nodes(overlapCount, nodes_init_rel).cpu()

force_deficit = F_target - F_init_wall.item()
print(f"\n  Wall-only initial force: {F_init_wall.item():.2f} N")
print(f"  Force deficit to cover:  {force_deficit:.2f} N "
      f"({force_deficit/F_target*100:.1f}%)")
print(f"  Wall-only perimeter:     {P_init_wall.item():.2f} mm")

# Effective spline_p
_effective_p = spline_p
if _effective_p is None:
    _tmp_ext = extend_nodes(overlapCount, orderedNodes)
    _tmp_s = chordLength(_tmp_ext)
    _tmp_h = _tmp_s[1:] - _tmp_s[:-1]
    _effective_p = _default_smooth_p(_tmp_h)

# Auto-scale lambda_smoothness
lambda_smoothness = smooth_weight_ratio * P_init_wall.item()
print(f"\n  Lambda smoothness (auto): {lambda_smoothness:.2f} "
      f"(ratio={smooth_weight_ratio} x P_init={P_init_wall.item():.2f})")

lambdas = {
    'smooth': lambda_smoothness,
    'force':  lambda_crushForce,
    'radial': lambda_radialDist,
}

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_steps, eta_min=1e-4)

# Initial loss
with torch.no_grad():
    initial_loss = compute_loss(
        F_init_wall, P_init_wall, model, force_target_tensor,
        smooth_init, lambdas, max_expansion_ratio,
    ).item()

# Objective: minimise perimeter (higher = better, so track -P)
initial_objective = -P_init_wall.item()
initial_force_ok = satisfies_force_floor(F_init_wall, force_target_tensor,
                                          rel_tol=force_floor_rel_tol)
initial_radial_ok = satisfies_radial_constraint(model, max_expansion_ratio)
initial_feasible = initial_force_ok and initial_radial_ok

best_loss = initial_loss
best_r = model.r.data.clone()
best_epoch = -1
nan_detected = False
converged = False

best_objective = initial_objective if initial_feasible else float('-inf')
best_objective_r = model.r.data.clone() if initial_feasible else None
best_objective_epoch = -1 if initial_feasible else None
best_progress_objective = best_objective
epochs_without_improvement = 0

#==========================================================================
# Optimisation loop
#==========================================================================
print("\n" + "=" * 60)
print(f"OPTIMISATION: minP -> F_target={F_target:.0f} N "
      f"({n_steps} max steps, patience={conv_patience})")
print("=" * 60)

t_start = time.perf_counter()

for epoch in range(n_steps):
    optimizer.zero_grad()
    force_total, perimeter = model()

    loss = compute_loss(
        force_total, perimeter, model, force_target_tensor,
        smooth_init, lambdas, max_expansion_ratio,
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
        force_total, perimeter = model()
        loss_eval = compute_loss(
            force_total, perimeter, model, force_target_tensor,
            smooth_init, lambdas, max_expansion_ratio,
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

    # Early stopping
    current_objective = -perimeter.item()  # minimise P => maximise -P
    force_floor_ok = satisfies_force_floor(force_total, force_target_tensor,
                                            rel_tol=force_floor_rel_tol)
    radial_ok = satisfies_radial_constraint(model, max_expansion_ratio)
    feasible_now = force_floor_ok and radial_ok

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
    if epoch % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        fpp_now = (force_total / perimeter).item()
        tee.terminal.write(
            f"Epoch {epoch:5d}: F={force_total.item():.0f} "
            f"(target={F_target:.0f}), "
            f"P={perimeter.item():.1f}, "
            f"F/P={fpp_now:.2f}, "
            f"Loss={current_loss:.2f}, "
            f"LR={current_lr:.6f}\n"
        )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

# Restore best
if not nan_detected:
    if best_objective_r is not None:
        model.r.data.copy_(best_objective_r)
    else:
        model.r.data.copy_(best_r)

# ---- Rename output files ---- #
_final_base = f"{_shape}_ribRemoval{run_tag}_{run_timestamp}_(v{script_version})"
log_filename_final = str(RESULTS_DIR / f"{_final_base}.txt")
mat_filename_final = str(RESULTS_DIR / f"{_final_base}.mat")
pdf_filename_final = str(RESULTS_DIR / f"{_final_base}.pdf")

tee.close()
if os.path.exists(log_filename):
    shutil.move(log_filename, log_filename_final)
log_filename = log_filename_final
mat_filename = mat_filename_final
pdf_filename = pdf_filename_final

tee = TeeLogger.__new__(TeeLogger)
tee.terminal = sys.__stdout__
tee.log = open(log_filename, 'a', encoding='utf-8')
sys.stdout = tee

#==========================================================================
# Results
#==========================================================================
with torch.no_grad():
    F_final, P_final = model()
    fpp_final = F_final / P_final
    final_force_ok = satisfies_force_floor(F_final, force_target_tensor,
                                            rel_tol=force_floor_rel_tol)
    final_radial_ok = satisfies_radial_constraint(model, max_expansion_ratio)
    final_feasible = final_force_ok and final_radial_ok

    # Plot
    nodes_final_rel = Spherical2Cartesian(model.r, model.theta, model.phi)
    nodesExt_final = extend_nodes(overlapCount, nodes_final_rel).detach().cpu()
    plot_results(nodesExt_init.numpy(), nodesExt_final.numpy(), pdf_filename)

    # .mat export
    nodes_final_abs = (nodes_final_rel + model.centroid).numpy()
    n_wall = nodes_final_abs.shape[0]
    conn_final = np.column_stack([np.arange(1, n_wall + 1),
                                   np.arange(2, n_wall + 2)])
    conn_final[-1, 1] = 1

    savemat(mat_filename, {
        'nodesCoords':              nodes_final_abs,
        'connectivity':             conn_final,
        'forceTarget':              np.float64(F_target),
        'forceInitialWall':         np.float64(F_init_wall.item()),
        'forceFinal':               np.float64(F_final.item()),
        'perimeterInitialWall':     np.float64(P_init_wall.item()),
        'perimeterFinal':           np.float64(P_final.item()),
        'forcePerPerimeterInitial': np.float64(fpp_init.item()),
        'forcePerPerimeterFinal':   np.float64(fpp_final.item()),
        'ribbedForceTotal':         np.float64(ribbed_result['force_total']),
        'ribbedForceWall':          np.float64(ribbed_result['force_wall']),
        'ribbedForceRib':           np.float64(ribbed_result['force_rib']),
        'ribbedPerimeter':          np.float64(ribbed_result['perimeter']),
    })

# ---- Stopping reason ---- #
_best_loss_label = 'initial state' if best_epoch < 0 else f'epoch {best_epoch}'
_best_obj_label = (
    'initial state' if best_objective_epoch is None or best_objective_epoch < 0
    else f'epoch {best_objective_epoch}'
)
if nan_detected:
    stop_reason = f'NaN/Inf at epoch {final_epoch} (rolled back to {_best_loss_label})'
elif best_objective_r is None:
    stop_reason = f'No feasible solution found; restored best loss from {_best_loss_label}'
elif converged:
    stop_reason = f'Converged at epoch {final_epoch} (restored best feasible from {_best_obj_label})'
else:
    stop_reason = f'Step limit ({n_steps}), restored best feasible from {_best_obj_label}'

# ---- Summary ---- #
force_change_pct = ((F_final.item() - F_init_wall.item()) /
                    F_init_wall.item()) * 100
perim_change_pct = ((P_final.item() - P_init_wall.item()) /
                    P_init_wall.item()) * 100
fpp_change_pct = ((fpp_final.item() - fpp_init.item()) /
                  fpp_init.item()) * 100
force_error_pct = ((F_final.item() - F_target) / F_target) * 100

# Compare: is the rib-free optimised perimeter smaller than the ribbed total?
ribbed_total_perimeter = ribbed_result['perimeter']
perim_vs_ribbed_pct = ((P_final.item() - ribbed_total_perimeter) /
                       ribbed_total_perimeter) * 100

print("\n" + "=" * 60)
print("RIB REMOVAL OPTIMIZER - RUN SUMMARY")
print("=" * 60)
print(f"  Input file:         {_data_path.name}")
print(f"  Spline p:           {_effective_p:.6f}"
      f"{'  (auto)' if spline_p is None else ''}")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
print(f"  ---")
print(f"  F_target (ribbed):  {F_target:.2f} N")
print(f"  F_final (wall):     {F_final.item():.2f} N  "
      f"(error: {force_error_pct:+.2f}%)")
print(f"  Force target met:   {'yes' if final_force_ok else 'NO'}")
print(f"  Radial satisfied:   {'yes' if final_radial_ok else 'no'}")
print(f"  Final feasible:     {'yes' if final_feasible else 'NO'}")
print(f"  ---")
print(f"  {'Metric':<14} {'Initial':>12} {'Final':>12} {'Change':>10}")
print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
print(f"  {'Wall Force':<14} {F_init_wall.item():>12.2f} {F_final.item():>12.2f} "
      f"{force_change_pct:>+9.2f}%")
print(f"  {'Wall Perim':<14} {P_init_wall.item():>12.2f} {P_final.item():>12.2f} "
      f"{perim_change_pct:>+9.2f}%")
print(f"  {'F/P':<14} {fpp_init.item():>12.4f} {fpp_final.item():>12.4f} "
      f"{fpp_change_pct:>+9.2f}%")
print(f"  ---")
print(f"  Ribbed total P:     {ribbed_total_perimeter:.2f} mm")
print(f"  Optimised wall P:   {P_final.item():.2f} mm  "
      f"({perim_vs_ribbed_pct:+.2f}% vs ribbed)")
if perim_vs_ribbed_pct < 0:
    print(f"  => Wall-only uses LESS material than ribbed section")
else:
    print(f"  => Rib saves {perim_vs_ribbed_pct:.1f}% material vs wall-only")
print(f"  ---")
print(f"  Epochs:             {final_epoch + 1}")
print(f"  Stop reason:        {stop_reason}")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("=" * 60)

tee.close()
