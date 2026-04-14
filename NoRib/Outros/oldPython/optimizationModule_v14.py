
#                  optimizationModule_v11.py
#
#  Clean version — retains what works, removes unnecessary complexity.
#
#  Active features:
#    - Three optimisation modes: maxCF, minP, maxSEA
#    - Smoothness normalised by initial value (geometry-independent)
#    - Radial bounds for maxCF (outward only) and maxSEA (symmetric)
#    - RoC clamped to [min_roc, flat_transition] (user-configurable)
#    - Per-node edge-length weighted force integration
#    - Spline: NaturalCubicSpline with linspace query + overlap=10
#
#  Inherited core physics (from v6/v7):
#    - smoothnessTerm wraps around the closed loop
#    - RoCCalc uses ‖r'×r''‖ / ‖r'‖³
#    - sQuery via linspace matching MATLAB curvatureSpline
#
#  NOTE: Spline TYPE differs from MATLAB (NaturalCubicSpline vs spaps).
#        This causes ~3% residual discrepancy that is inherent to the
#        library choice, not a bug.
#==========================================================================

from typing import Tuple
import torch
from torch import nn
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
import sys
import time
import numpy as np
from datetime import datetime
print(torch.cuda.is_available())
#==========================================================================
# ---- Define the model / neural network ---- #
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
		# Splines
		splineDer1, splineDer2 = splines(nodesExt, orderedNodes, s, self.overlapCount)
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
	nodesExt = torch.stack([X_ext, Y_ext, Z_ext], dim=1);
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
# ---- Fit smoothing splines ---- #
def splines(nodesExt, orderedNodes, s, overlapCount) -> Tuple[torch.Tensor, torch.Tensor]:
	coeffs = natural_cubic_spline_coeffs(s, nodesExt)
	spline = NaturalCubicSpline(coeffs)
	nOriginal = orderedNodes.shape[0]

	sStart = s[overlapCount]
	sEnd   = s[overlapCount + nOriginal - 1]
	sEnd_matlab = s[-overlapCount]
	sQuery = torch.linspace(
	    sStart.item(), sEnd_matlab.item(), nOriginal,
	    device=s.device
	)

	splineDer1 = spline.derivative(sQuery)
	delta = 1e-5
	sQuery_plus = torch.clamp(sQuery + delta, 0, 1)
	sQuery_minus = torch.clamp(sQuery - delta, 0, 1)
	d1_plus = spline.derivative(sQuery_plus)
	d1_minus = spline.derivative(sQuery_minus)
	splineDer2 = (d1_plus - d1_minus) / (2 * delta)
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
def plot_initial_and_final(initial_nodes, final_nodes, save_path="optim_result.pdf"):
	plt.rcParams.update({
    	"text.usetex": True,
    	"font.family": "serif",
    	"font.serif": ["Computer Modern Roman"],
    	"font.size": 11
	})

	fig, ax = plt.subplots(figsize=(4.72, 2.25))
	ax.plot(initial_nodes[:, 0], initial_nodes[:, 1],
            marker='o', markersize=4, linewidth=0.75, color=(0, 0.447, 0.698), label=r'Initial')
	ax.plot(final_nodes[:, 0], final_nodes[:, 1],
            marker='o', markersize=4, linewidth=0.75, color=(0.902, 0.624, 0), label=r'Optimised')
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")
	ax.set_aspect('equal')
	plt.subplots_adjust(bottom=0.25)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
	plt.show()
	fig.savefig(save_path, format="pdf", bbox_inches="tight")

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
                 lambdas, max_expansion_ratio):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF'  — maximise crush force (outward expansion, capped).
        'minP'   — minimise perimeter (inward only, force floor).
        'maxSEA' — maximise specific energy absorption (F / P).
    force_total, perimeter : Tensor
        Current values from forward pass.
    model : crushOptimizerModel
        The model (for accessing r, initial_r).
    force_initial : Tensor
        Force at epoch 0.
    smoothness_initial : Tensor
        Smoothness at epoch 0 (for normalisation).
    lambdas : dict
        Penalty weights: 'smooth', 'force', 'radial'.
    max_expansion_ratio : float
        Max allowed r / r_initial ratio for radial bounds.
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

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'maxCF', 'minP', or 'maxSEA'.")

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
# User Inputs (defaults — can be overridden via command line)
# Usage: python optimizationModule_v11.py [dataFileName] [run_mode]
#   e.g. python optimizationModule_v11.py partition_data_ellipse.mat maxSEA

thickness = 2                       # [mm]
dataFileName = 'partition_data_square.mat'
constantCrushStress = 90            # [MPa]
flat_transition = 1221.95           # [mm] from DataCurvatureCarbon.xlsx
min_roc = 15.0                      # [mm] minimum RoC clamp (manufacturing constraint)
script_version = 14                 # appended to output filenames for traceability

# ---- Optimisation mode ---- #
# 'maxCF'  — maximise crush force (shape expands outward, capped)
# 'minP'   — minimise perimeter while preserving force (inward only)
# 'maxSEA' — maximise specific energy absorption (Force / Perimeter)
run_mode = 'maxSEA'

# ---- Command-line overrides ---- #
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('data', nargs='?', default=None)
parser.add_argument('mode', nargs='?', default=None)
args, _ = parser.parse_known_args()
if args.data is not None:
    dataFileName = args.data
if args.mode is not None:
    run_mode = args.mode

# Hyperparameters
n_steps = 10000
learning_rate = 0.1
grad_clip_norm = 1.0
snapshot_interval = 500
overlapCount = 10

# Penalty weights
# Smoothness λ is set per-mode to target ~10% of the objective at epoch 0.
# This ensures the smoothness regularisation is meaningful but doesn't
# dominate the objective, regardless of the mode's absolute scale.
# Force and radial λ are constraints, not regularisers — kept high.
lambda_crushForce = 1e6             # minP only: force floor constraint
lambda_radialDist = 1e6             # maxCF and maxSEA: radial bounds

# Smoothness weight ratio per mode: λ_smooth = ratio × |objective_initial|
# At epoch 0, objective is 1/(1+ratio) of total loss.
# maxCF:  force ~100k, needs strong smoothness → ratio=10 (obj = 9%)
# minP:   constraints do the work, ratio is less critical → ratio=10
# maxSEA: SEA ~200, needs gradient influence → ratio=2 (obj = 33%)
smooth_weight_ratio = {
    'maxCF':  10.0,
    'minP':   10.0,
    'maxSEA':  2.0,
}

# Max radial movement (maxCF and maxSEA modes)
# 1.15 = nodes can move ±13-15% from initial position.
max_expansion_ratio = 1.15

# ---- Validate mode ---- #
valid_modes = {'maxCF', 'minP', 'maxSEA'}
if run_mode not in valid_modes:
    raise ValueError(f"Invalid run_mode '{run_mode}'. Choose from {valid_modes}.")

# ---- Build output filenames ---- #
run_timestamp = datetime.now().strftime('%d-%m')
_shape = dataFileName.replace('partition_data_', '').replace('.mat', '')
_steps = f'{n_steps // 1000}k' if n_steps % 1000 == 0 else str(n_steps)
base_filename = f"{_shape}_{run_mode}_{_steps}_{run_timestamp}_(v{script_version})"

log_filename = f"{base_filename}.txt"
mat_filename = f"{base_filename}.mat"
pdf_filename = f"{base_filename}.pdf"

tee = TeeLogger(log_filename)
sys.stdout = tee
print(f"Log file: {log_filename}\n")

#==========================================================================
# Load cross section data from MATLAB
partitionData = loadData(dataFileName)

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
    flat_transition, min_roc
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

# ---- Auto-select lambda_smoothness based on objective scale ---- #
# At epoch 0, normalised smoothness = 1.0, so λ_smooth × 1.0 = λ_smooth.
# We set λ_smooth = smooth_weight_ratio × |objective_initial| so that
# the smoothness term starts at smooth_weight_ratio × the objective.
if run_mode == 'maxCF':
    objective_scale = forceTotal_initial.item()
elif run_mode == 'minP':
    objective_scale = perimeter_initial.item()
elif run_mode == 'maxSEA':
    objective_scale = sea_initial.item()

lambda_smoothness = smooth_weight_ratio[run_mode] * objective_scale
print(f'Lambda smoothness (auto): {lambda_smoothness:.2f}  '
      f'(ratio={smooth_weight_ratio[run_mode]} × objective={objective_scale:.2f})')

# Pack penalty weights
lambdas = {
    'smooth': lambda_smoothness,
    'force':  lambda_crushForce,
    'radial': lambda_radialDist,
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
        plot_initial_and_final(nodesExt_initial, nodesExt_final, pdf_filename)

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
}

print("\n" + "="*60)
print("CONVERGENCE STUDY - RUN SUMMARY")
print("="*60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {mode_labels[run_mode]}")
print(f"  Min RoC clamp:      {min_roc:.1f} mm")
print(f"  Max expansion:      {max_expansion_ratio:.2f}")
print(f"  Smooth weight ratio:{smooth_weight_ratio[run_mode]:.1f}")
print(f"  Lambda smooth:      {lambda_smoothness:.2f} (auto)")
print(f"  Lambda force:       {lambda_crushForce:.0e}")
print(f"  Lambda radial:      {lambda_radialDist:.0e}")
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
