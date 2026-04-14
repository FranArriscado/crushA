
#                  optimizationModule_patched_v9.py
#
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
#
#  Inherited from v8/v7:
#    FIX 5 (v7): sQuery uses linspace matching MATLAB curvatureSpline
#    FIX 6 (v7): overlapCount = 10, matching MATLAB default
#    FIX 7 (v7): RoC upper-clamped to flat_transition
#    NOTE:  Spline TYPE differs (NaturalCubicSpline vs MATLAB spaps)
#           causing ~3% residual discrepancy.
#
#  Inherited from v6:
#    FIX 1: smoothnessTerm wraps around the closed loop
#    FIX 3: RoCCalc no longer reindexes by orderedIndices
#    FIX 4: per-node edge-length weighted force integration
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
		# FIX 7 (v7): clamp to [5, flat_transition] matching MATLAB
		RoC = RoC.clamp(min=5.0, max=self.flat_transition)
		# Crush stress calculation
		crushStress = self.constantCrushStress * CurvatureEq(RoC)
		# Perimeter and edge lengths calculation
		perimeter, edgeLengths = perimeterCalc(orderedNodes)
		# FIX 4 (v6): per-node edge-length weighted force integration
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

	# FIX 5 (v7): linspace query matching MATLAB curvatureSpline
	sStart = s[overlapCount]
	sEnd   = s[overlapCount + nOriginal - 1]  # last original node
	# Note: MATLAB linspace(s(oc+1), s(end-oc), N) where end-oc is the
	# last node of the extended array mapped back to the last original node.
	# In the Python parametrisation, s[-overlapCount] corresponds to s(end-oc).
	sEnd_matlab = s[-overlapCount]
	sQuery = torch.linspace(
	    sStart.item(), sEnd_matlab.item(), nOriginal,
	    device=s.device
	)

	splineDer1 = spline.derivative(sQuery)
	# Second derivative via finite difference (same as v6)
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
def plot_initial_and_final(initial_nodes, final_nodes):
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
	fig.savefig("optim_result.pdf", format="pdf", bbox_inches="tight")

#==========================================================================
# ---- Penalty Constraints ---- #
def smoothnessTerm(r):
    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    dr = r_wrapped[1:] - r_wrapped[:-1]
    curvature = dr[1:] - dr[:-1]
    # FIX 9 (v9): normalise by node count so penalty scales consistently
    smoothness_loss = torch.sum(curvature ** 2) / len(r)
    return smoothness_loss

#==========================================================================
# ---- Loss Function ---- #
# REFACTOR (v9): extracted into function for clarity and maintainability.
def compute_loss(mode, inward_sec, force_total, perimeter, model,
                 force_initial, perimeter_initial, lambdas):
    """
    Compute the optimisation loss for the chosen mode.

    Parameters
    ----------
    mode : str
        'maxCF' for max crush force, 'minP' for min perimeter.
    inward_sec : bool
        If True (minP only), also constrain outward radial expansion.
    force_total : Tensor
        Current total crush force.
    perimeter : Tensor
        Current perimeter.
    model : crushOptimizerModel
        The model (for accessing r, initial_r).
    force_initial : Tensor
        Force at epoch 0, used for penalty constraints.
    perimeter_initial : Tensor
        Perimeter at epoch 0, used for penalty constraints.
    lambdas : dict
        Penalty weights: 'smooth', 'force', 'radial'.

    Returns
    -------
    loss : Tensor
    """
    smoothness = smoothnessTerm(model.r)

    if mode == 'maxCF':
        # FIX 8 (v9): prevent inward shrinkage — nodes can only expand outward
        radial_inward = torch.sum(torch.relu(model.initial_r - model.r) ** 2)
        loss = (-force_total
                + lambdas['smooth'] * smoothness
                + lambdas['radial'] * radial_inward)

    elif mode == 'minP':
        # Prevent force from dropping below initial
        force_penalty = torch.relu(force_initial - force_total) ** 2
        if inward_sec:
            # Also prevent outward expansion
            radial_outward = torch.sum(torch.relu(model.r - model.initial_r) ** 2)
            loss = (perimeter
                    + lambdas['smooth'] * smoothness
                    + lambdas['force'] * force_penalty
                    + lambdas['radial'] * radial_outward)
        else:
            loss = (perimeter
                    + lambdas['smooth'] * smoothness
                    + lambdas['force'] * force_penalty)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'maxCF' or 'minP'.")

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
# User Inputs
thickness = 2                       # [mm]
# Input flags
minPerimeterFlag = True
maxCrushForceFlag = False
inwardSecFlag = True
dataFileName = 'partition_data_ellipse.mat'
constantCrushStress = 90            # [MPa]
flat_transition = 1221.95           # [mm] from DataCurvatureCarbon.xlsx
# Hyperparameters
n_steps = 10000
learning_rate = 0.1
grad_clip_norm = 1.0
snapshot_interval = 500
# Penalty weights — FIX 9 (v9): rebalanced for per-node smoothness
lambda_smoothness = 1e4
lambda_crushForce = 1e4
lambda_radialDist = 1e4
overlapCount = 10

# ---- Derive run mode string ---- #
if maxCrushForceFlag and not minPerimeterFlag:
    run_mode = 'maxCF'
elif minPerimeterFlag and not maxCrushForceFlag:
    run_mode = 'minP'
else:
    raise ValueError("Invalid configuration: set exactly one of minPerimeterFlag / maxCrushForceFlag.")

# ---- Build output filenames ---- #
# REFACTOR (v9): removed _FlagTrue_/_FlagFalse_ from naming
run_timestamp = datetime.now().strftime('%d-%m')
_shape = dataFileName.replace('partition_data_', '').replace('.mat', '')
_inward = '_inward' if inwardSecFlag and run_mode == 'minP' else ''
_steps = f'{n_steps // 1000}k' if n_steps % 1000 == 0 else str(n_steps)
base_filename = f"{_shape}_{run_mode}_{_steps}{_inward}_{run_timestamp}"

log_filename = f"{base_filename}.txt"
mat_filename = f"{base_filename}.mat"

tee = TeeLogger(log_filename)
sys.stdout = tee
print(f"Log file: {log_filename}\n")

# Pack penalty weights into dict for compute_loss()
lambdas = {
    'smooth': lambda_smoothness,
    'force':  lambda_crushForce,
    'radial': lambda_radialDist,
}

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
    flat_transition
)

# compute initial values
with torch.no_grad():
        forceTotal_initial, perimeter_initial = model()
        print('Force initial:', forceTotal_initial)
        print('Perimeter initial', perimeter_initial)
        orderedNodes_initial = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_initial = nodesCoords(overlapCount, orderedNodes_initial).cpu()

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

        # Compute loss via unified function
        loss = compute_loss(
            mode=run_mode,
            inward_sec=inwardSecFlag,
            force_total=force_total,
            perimeter=perimeter,
            model=model,
            force_initial=forceTotal_initial,
            perimeter_initial=perimeter_initial,
            lambdas=lambdas,
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
                tee.terminal.write(
                    f"Epoch {epoch}: Perimeter = {perimeter:.4f}, "
                    f"Crush Force = {force_total:.4f}, "
                    f"Loss = {loss.item():.4f}, "
                    f"LR = {current_lr:.6f}\n"
                )

t_end = time.perf_counter()
elapsed = t_end - t_start
final_epoch = epoch

# ---- Visualize optimized spline ----
with torch.no_grad():
        forceTotal_final, perimeter_final = model()
        print('Force final:', forceTotal_final)
        print('Perimeter final:', perimeter_final)
        orderedNodes_final = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_final = nodesCoords(overlapCount, orderedNodes_final).detach().cpu()
        plot_initial_and_final(nodesExt_initial, nodesExt_final)

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
print("\n" + "="*60)
print("CONVERGENCE STUDY - RUN SUMMARY")
print("="*60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {'Max crush force' if run_mode == 'maxCF' else 'Min perimeter'}"
      f"{'  (inward only)' if inwardSecFlag and run_mode == 'minP' else ''}")
print(f"  Lambda smooth:      {lambda_smoothness:.0e}")
print(f"  Lambda force:       {lambda_crushForce:.0e}")
print(f"  Lambda radial:      {lambda_radialDist:.0e}")
print(f"  ---")
print(f"  Force  initial:     {forceTotal_initial.item():.4f}")
print(f"  Force  final:       {forceTotal_final.item():.4f}")
print(f"  Force  change:      {((forceTotal_final.item() - forceTotal_initial.item()) / forceTotal_initial.item()) * 100:+.2f}%")
print(f"  Perim  initial:     {perimeter_initial.item():.4f}")
print(f"  Perim  final:       {perimeter_final.item():.4f}")
print(f"  Perim  change:      {((perimeter_final.item() - perimeter_initial.item()) / perimeter_initial.item()) * 100:+.2f}%")
print(f"  ---")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("="*60)

tee.close()
