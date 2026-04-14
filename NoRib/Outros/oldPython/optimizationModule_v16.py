
#                  optimizationModule_v16.py
#
#  v16 — Replaced NaturalCubicSpline (interpolating) with a Reinsch-form
#         smoothing cubic spline implemented in pure PyTorch.
#
#  SPLINE CHANGE (v15 -> v16):
#    v15 used torchcubicspline.NaturalCubicSpline (interpolating spline).
#    v16 uses a custom SmoothingCubicSpline class (Reinsch formulation)
#    solved via torch.linalg.solve — fully differentiable through autograd.
#    The smoothing parameter p controls interpolation (p=1) vs smoothing
#    (p->0). Default uses the MATLAB csaps heuristic p = 1/(1 + h^3/6).
#
#  Active features (unchanged from v15):
#    - Four optimisation modes: maxCF, minP, maxSEA, boxSEA
#    - Smoothness normalised by initial value (geometry-independent)
#    - Radial bounds for maxCF (outward only) and maxSEA (symmetric)
#    - Bounding box constraint for boxSEA (centered at centroid)
#    - RoC clamped to [min_roc, flat_transition] (user-configurable)
#    - Per-node edge-length weighted force integration
#    - Overlap=10 with linspace query matching MATLAB curvatureSpline
#
#  Inherited core physics (from v6/v7 through v15):
#    - smoothnessTerm wraps around the closed loop
#    - RoCCalc uses ||r' x r''|| / ||r'||^3
#    - sQuery via linspace matching MATLAB curvatureSpline
#
#  DEPENDENCIES: torch, scipy, matplotlib, numpy (NO torchcubicspline)
#
#  NOTE: The smoothing parameter spline_p can be tuned to match MATLAB
#        spaps(s, y, tol) with tol=1e-2. Default uses csaps heuristic.
#==========================================================================

script_version = 16

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
from datetime import datetime
from pathlib import Path

# ---- Project paths (config.py at project root) ---- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_NORIB

print(torch.cuda.is_available())

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
    X = torch.round(X * 10000) / 10000
    Y = torch.round(Y * 10000) / 10000
    Z = torch.round(Z * 10000) / 10000
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
    evaluate first and second derivatives at linspace query points
    matching MATLAB's curvatureSpline convention.
    """
    nOriginal = orderedNodes.shape[0]

    # Fit smoothing spline to extended nodes
    g, M, h = fit_smoothing_spline(s, nodesExt, p=spline_p)

    # Query range: from first original node to last original node
    # (matching MATLAB: linspace(s(overlapCount+1), s(end-overlapCount), nOrd))
    sStart     = s[overlapCount]
    sEnd_matlab = s[-overlapCount]
    sQuery = torch.linspace(
        sStart.item(), sEnd_matlab.item(), nOriginal,
        device=s.device
    )

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
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11
        })
    except Exception:
        plt.rcParams.update({"font.size": 11})

    fig, ax = plt.subplots(figsize=(4.72, 2.25))
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
    plt.close(fig)

#==========================================================================
# ---- Smoothness Penalty ---- #
def smoothnessTerm(r):
    """Second differences of radial distances (closed loop)."""
    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    dr = r_wrapped[1:] - r_wrapped[:-1]
    curvature = dr[1:] - dr[:-1]
    return torch.sum(curvature ** 2)

#==========================================================================
# ---- Loss Function ---- #
def compute_loss(mode, force_total, perimeter, model,
                 force_initial, smoothness_initial,
                 lambdas, max_expansion_ratio, box_dims=None):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF'  -- maximise crush force (outward expansion, capped).
        'minP'   -- minimise perimeter (inward only, force floor).
        'maxSEA' -- maximise specific energy absorption (F / P).
        'boxSEA' -- maximise SEA within a bounding box constraint.
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
        Max allowed r / r_initial ratio for radial bounds.
    box_dims : tuple or None
        (box_width, box_height) in mm, centered at centroid. boxSEA only.
    """
    # Normalised smoothness: 1.0 at epoch 0, geometry-independent
    smoothness = smoothnessTerm(model.r) / (smoothness_initial.detach() + 1e-12)

    if mode == 'maxCF':
        # Outward only: prevent inward shrinkage, cap outward expansion
        radial_inward = torch.sum(torch.relu(model.initial_r - model.r) ** 2)
        r_max = max_expansion_ratio * model.initial_r
        radial_outward = torch.sum(torch.relu(model.r - r_max) ** 2)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['radial'] * (radial_inward + radial_outward))

    elif mode == 'minP':
        # Inward only: prevent outward expansion, maintain force
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

    elif mode == 'boxSEA':
        sea = force_total / perimeter
        force_penalty = torch.relu(force_initial - force_total) ** 2
        X = model.r * torch.sin(model.theta) * torch.cos(model.phi)
        Y = model.r * torch.sin(model.theta) * torch.sin(model.phi)
        half_w = box_dims[0] / 2.0
        half_h = box_dims[1] / 2.0
        box_penalty = (torch.sum(torch.relu(X - half_w) ** 2)
                     + torch.sum(torch.relu(-half_w - X) ** 2)
                     + torch.sum(torch.relu(Y - half_h) ** 2)
                     + torch.sum(torch.relu(-half_h - Y) ** 2))
        loss = (-sea
                + lambdas['smooth'] * smoothness
                + lambdas['force'] * force_penalty
                + lambdas['box'] * box_penalty)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'maxCF', 'minP', 'maxSEA', or 'boxSEA'.")

    return loss

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

#==========================================================================
#==========================================================================
thickness = 2                       # [mm]
dataFileName = 'partition_data_ellipse.mat'
constantCrushStress = 90            # [MPa]
flat_transition = 1221.95           # [mm] from DataCurvatureCarbon.xlsx
min_roc = 15.0                      # [mm] minimum RoC clamp (manufacturing constraint)

#Don't change this parameter, it's referencing a spline coefficient
spline_p = None   # None = auto (csaps heuristic)

# ---- Optimisation mode ---- #
# 'maxCF'  -- maximise crush force (shape expands outward, capped)
# 'minP'   -- minimise perimeter while preserving force (inward only)
# 'maxSEA' -- maximise specific energy absorption (Force / Perimeter)
# 'boxSEA' -- maximise SEA within a bounding box (design envelope)
run_mode = 'minP'

# ---- Bounding box (boxSEA mode only) ---- #
box_width  = 35                  # [mm] box extent in X
box_height = 35                  # [mm] box extent in Y

# Hyperparameters
n_steps = 1000
learning_rate = 0.1
grad_clip_norm = 1.0
snapshot_interval = 500
overlapCount = 10

# Penalty weights
lambda_crushForce = 1e6
lambda_radialDist = 1e6
lambda_box = 1e6

# Smoothness weight ratio per mode
smooth_weight_ratio = {
    'maxCF':  10.0,
    'minP':   10.0,
    'maxSEA':  2.0,
    'boxSEA':  2.0,
}
# Max radial movement (maxCF and maxSEA modes)
max_expansion_ratio = 1.3
#==========================================================================
#==========================================================================

# ---- Command-line overrides ---- #
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data',       nargs='?', default=None)
parser.add_argument('mode',       nargs='?', default=None)
parser.add_argument('--steps',    type=int,   default=None)
parser.add_argument('--spline-p', type=float, default=None,
                    help='Smoothing spline parameter (0,1]. None=auto.')
args, _ = parser.parse_known_args()
if args.data is not None:
    dataFileName = args.data
if args.mode is not None:
    run_mode = args.mode
if args.steps is not None:
    n_steps = args.steps
if args.spline_p is not None:
    spline_p = args.spline_p

# ---- Validate mode ---- #
valid_modes = {'maxCF', 'minP', 'maxSEA', 'boxSEA'}
if run_mode not in valid_modes:
    raise ValueError(f"Invalid run_mode '{run_mode}'. Choose from {valid_modes}.")

# ---- Build output filenames ---- #
run_timestamp = datetime.now().strftime('%d-%m')
_shape = dataFileName.replace('partition_data_', '').replace('.mat', '')
_steps = f'{n_steps // 1000}k' if n_steps % 1000 == 0 else str(n_steps)
base_filename = f"{_shape}_{run_mode}_{_steps}_{run_timestamp}_(v{script_version})"

log_filename = str(RESULTS_NORIB / f"{base_filename}.txt")
mat_filename = str(RESULTS_NORIB / f"{base_filename}.mat")
pdf_filename = str(RESULTS_NORIB / f"{base_filename}.pdf")

tee = TeeLogger(log_filename)
sys.stdout = tee
print(f"Log file: {log_filename}\n")

#==========================================================================
# Load cross section data from MATLAB
_data_path = Path(dataFileName)
if not _data_path.is_absolute():
    _data_path = DATA_DIR / _data_path
partitionData = loadData(str(_data_path))

# Process data
nodes = partitionData['nodesCoords']
edges = partitionData['connectivity']
print('Nodes Coords:', nodes)
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
        smoothness_initial = smoothnessTerm(model.r)
        print('Force initial:', forceTotal_initial)
        print('Perimeter initial', perimeter_initial)
        print('SEA initial:', sea_initial)
        orderedNodes_initial = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_initial = nodesCoords(overlapCount, orderedNodes_initial).cpu()

# Report effective spline_p
_effective_p = spline_p
if _effective_p is None:
    # Compute what the auto heuristic will give
    _tmp_ext = nodesCoords(overlapCount, orderedNodes)
    _tmp_s = chordLength(_tmp_ext)
    _tmp_h = _tmp_s[1:] - _tmp_s[:-1]
    _effective_p = _default_smooth_p(_tmp_h)
print(f'Spline smoothing p: {_effective_p:.10f}  '
      f'({"auto/csaps" if spline_p is None else "user-specified"})')

# ---- Auto-select lambda_smoothness based on objective scale ---- #
if run_mode == 'maxCF':
    objective_scale = forceTotal_initial.item()
elif run_mode == 'minP':
    objective_scale = perimeter_initial.item()
elif run_mode in ('maxSEA', 'boxSEA'):
    objective_scale = sea_initial.item()

lambda_smoothness = smooth_weight_ratio[run_mode] * objective_scale
print(f'Lambda smoothness (auto): {lambda_smoothness:.2f}  '
      f'(ratio={smooth_weight_ratio[run_mode]} x objective={objective_scale:.2f})')

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

best_loss = float('inf')
best_r = model.r.data.clone()
best_epoch = 0
nan_detected = False

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
            box_dims=(box_width, box_height),
        )

        # NaN detection
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n*** NaN/Inf detected at epoch {epoch}. "
                  f"Rolling back to snapshot from epoch {best_epoch}. ***\n")
            model.r.data.copy_(best_r)
            nan_detected = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            if epoch % snapshot_interval == 0:
                best_r = model.r.data.clone()

        if epoch % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                sea_now = force_total / perimeter
                tee.terminal.write(
                    f"Epoch {epoch}: Perimeter = {perimeter:.4f}, "
                    f"Crush Force = {force_total:.4f}, "
                    f"SEA = {sea_now:.4f}, "
                    f"Loss = {loss.item():.4f}, "
                    f"LR = {current_lr:.6f}\n"
                )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

# ---- Visualize optimized spline ----
with torch.no_grad():
        forceTotal_final, perimeter_final = model()
        sea_final = forceTotal_final / perimeter_final
        print('Force final:', forceTotal_final)
        print('Perimeter final:', perimeter_final)
        print('SEA final:', sea_final)
        orderedNodes_final = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_final = nodesCoords(overlapCount, orderedNodes_final).detach().cpu()
        plot_initial_and_final(nodesExt_initial, nodesExt_final, pdf_filename,
                              box_dims=(box_width, box_height) if run_mode == 'boxSEA' else None)

        orderedNodes_final_np = orderedNodes_final.numpy()
        print("\nFinal Node Coordinates (ordered, for MATLAB validation):")
        print(f"# Shape: {orderedNodes_final_np.shape[0]} nodes x 3 (X, Y, Z) [mm]")
        for i, node in enumerate(orderedNodes_final_np):
            print(f"  Node {i:4d}: [{node[0]:12.6f}, {node[1]:12.6f}, {node[2]:12.6f}]")

        n = orderedNodes_final_np.shape[0]
        connectivity_final = np.column_stack([np.arange(1, n + 1), np.arange(2, n + 2)])
        connectivity_final[-1, 1] = 1

        scipy.io.savemat(
            mat_filename,
            {
                'nodesCoords':  orderedNodes_final_np,
                'connectivity': connectivity_final,
            }
        )
        print(f"\nFinal nodes saved to: {mat_filename}")

# ---- Runtime summary ---- #
mode_labels = {
    'maxCF':  'Max crush force',
    'minP':   'Min perimeter (inward)',
    'maxSEA': 'Max specific energy absorption',
    'boxSEA': 'Max SEA (box constrained)',
}

print("\n" + "="*60)
print("CONVERGENCE STUDY - RUN SUMMARY")
print("="*60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {mode_labels[run_mode]}")
print(f"  Spline type:        Smoothing cubic (Reinsch, p={_effective_p:.10f})")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
if run_mode == 'boxSEA':
    print(f"  Box constraint:     {box_width:.1f} x {box_height:.1f} mm")
print(f"  Smooth weight ratio:{smooth_weight_ratio[run_mode]:.1f}")
print(f"  Lambda smooth:      {lambda_smoothness:.2f} (auto)")
print(f"  Lambda force:       {lambda_crushForce:.0e}")
print(f"  Lambda radial:      {lambda_radialDist:.0e}")
if run_mode == 'boxSEA':
    print(f"  Lambda box:         {lambda_box:.0e}")
print(f"  ---")
print(f"  Force  initial:     {forceTotal_initial.item():.4f}")
print(f"  Force  final:       {forceTotal_final.item():.4f}")
print(f"  Force  change:      {((forceTotal_final.item() - forceTotal_initial.item()) / forceTotal_initial.item()) * 100:+.2f}%")
print(f"  Perim  initial:     {perimeter_initial.item():.4f}")
print(f"  Perim  final:       {perimeter_final.item():.4f}")
print(f"  Perim  change:      {((perimeter_final.item() - perimeter_initial.item()) / perimeter_initial.item()) * 100:+.2f}%")
print(f"  SEA    initial:     {sea_initial.item():.4f}")
print(f"  SEA    final:       {sea_final.item():.4f}")
print(f"  SEA    change:      {((sea_final.item() - sea_initial.item()) / sea_initial.item()) * 100:+.2f}%")
print(f"  ---")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("="*60)

tee.close()
