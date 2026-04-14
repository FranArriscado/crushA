#
#                  optimizationModule_patched.py
#                   
#                     Tiago Cardoso
#                      up202006481
#                      FEUP, 2025
#
#  Patched version — fixes applied to resolve closure discontinuity:
#    FIX 1: smoothnessTerm now wraps around the closed loop
#    FIX 2: spline query uses actual node chord-length positions (not linspace)
#    FIX 3: RoCCalc no longer reindexes by orderedIndices (derivatives are
#           already in ordered-node sequence)
#    ADDED: runtime timer for convergence studies
#
# Optimise cross section to minimise perimeter, and therefore mass, while keeping 
# total crush force above target value (initial value)
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
import time
print(torch.cuda.is_available())
#==========================================================================
# ---- Define the model / neural network ---- #
#class defines a feedforward neural network that can be trained with PyTorch optimizers
class crushOptimizerModel(nn.Module):

	# Initialize the neural network layers
	def __init__(self,
		orderedNodes: torch.Tensor,
		orderedIndices: torch.Tensor,
		edges: torch.Tensor,
		constantCrushStress,
		overlapCount,
                thickness,
		):

		super().__init__()
		self.centroid = orderedNodes.mean(dim=0, keepdim=True) # centroid of the cross section
		r, theta, phi = Cartesian2Spherical(orderedNodes - self.centroid) # conversion to spherical coordinates

		# Learnable parameter: r (radial distance)
		self.r = nn.Parameter(r)
		# Store initial radial distances
		self.register_buffer('initial_r', r.clone())
		# Non-learnable parameters: theta, phi
		self.register_buffer('theta', theta)
		self.register_buffer('phi', phi)

		self.orderedIndices = orderedIndices
		self.edges = edges
		self.constantCrushStress = constantCrushStress
		self.overlapCount = overlapCount
		self.thickness = thickness

	# The forward method defines how the input flows through the network
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
		# FIX 3: removed orderedIndices reindexing — derivatives are already
		#         computed at ordered-node positions, so radius[i] corresponds
		#         to the i-th ordered node. The old reindexing scrambled the
		#         gradient attribution across nodes.
		RoC = RoCCalc(splineDer1, splineDer2)
		RoC = RoC.clamp_min(5)
		# Crush stress calculation
		crushStress = self.constantCrushStress * CurvatureEq(RoC)
		# Perimeter calculation
		perimeter = perimeterCalc(orderedNodes)
		# Crush force calculation
		forceTotal = self.thickness * perimeter * crushStress.mean()
		return forceTotal, perimeter

#==========================================================================
# ---- Load Data ---- #
def loadData(name):
	# Load MATLAB variables
	partitionData = scipy.io.loadmat(name)
	# Extract arrays
	nodes = partitionData['nodesCoords']         # NumPy array
	edges = partitionData['connectivity']        # NumPy array
	# Adjust for zero-based indexing
	edges -= 1

	return partitionData

#==========================================================================
# ---- Find ordered path ---- #
def orderNodes(nodes, edges):
    # Orders nodes to follow the path defined by edges.
    # Assumes edges form a single continuous path.

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

    return orderedIndices, orderedNodes  # remove duplicate at end for closed loop

#==========================================================================
# ---- Convert from cartesian to spherical coordinates ---- #
def Cartesian2Spherical(xyz):
	X = xyz[:, 0]
	Y = xyz[:, 1]
	Z = xyz[:, 2]
	r = torch.sqrt(X**2 + Y**2 + Z**2)
	phi = torch.atan2(Y, X) # azimuth angle
	theta = torch.acos(Z / r) # polar angle
	return r, theta, phi

#==========================================================================
# ---- Convert from spherical to cartesian coordinates ---- #
def Spherical2Cartesian(r, theta, phi):
    X = r * torch.sin(theta) * torch.cos(phi)
    Y = r * torch.sin(theta) * torch.sin(phi)
    Z = r * torch.cos(theta)
    # Round to 4 decimal places
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
	diffs = nodesExt[1:] - nodesExt[:-1]  # [N-1, 3]
	ds = torch.norm(diffs, dim=1)  # [N-1]
	s = torch.zeros(len(nodesExt), device=nodesExt.device)
	s[1:] = torch.cumsum(ds, dim=0)
	s = s / s[-1]
	return s

#==========================================================================
# ---- Fit smoothing splines ---- #
def splines(nodesExt, orderedNodes, s, overlapCount) -> Tuple[torch.Tensor, torch.Tensor]:
	coeffs = natural_cubic_spline_coeffs(s, nodesExt)
	spline = NaturalCubicSpline(coeffs)
	nOriginal = orderedNodes.shape[0]   # number of original nodes

	# FIX 2: query the spline at the actual chord-length parameter values
	#         for each original node, instead of a uniform linspace.
	#         The linspace created evaluation points that did not correspond
	#         to the true node positions, introducing RoC errors especially
	#         near the overlap boundaries (first/last nodes).
	sQuery = s[overlapCount : overlapCount + nOriginal]

	splineDer1 = spline.derivative(sQuery)
	# Evaluate first derivative at slightly shifted t
	delta = 1e-5 # evaluate first derivative
	sQuery_plus = sQuery + delta
	sQuery_minus = sQuery - delta
	# Clamp to [0,1]
	sQuery_plus = torch.clamp(sQuery_plus, 0, 1)
	sQuery_minus = torch.clamp(sQuery_minus, 0, 1)
	# First derivative at shifted points
	d1_plus = spline.derivative(sQuery_plus)
	d1_minus = spline.derivative(sQuery_minus)
	# Approximate second derivative using finite difference
	splineDer2 = (d1_plus - d1_minus) / (2 * delta)
	return splineDer1, splineDer2

#==========================================================================
# ---- Compute RoC ---- #
# FIX 3: removed orderedIndices parameter and reindexing
def RoCCalc(splineDer1, splineDer2) -> torch.Tensor:
	cross = torch.cross(splineDer1, splineDer2, dim=1)
	num = torch.norm(cross, dim=1)
	den = torch.norm(splineDer1, dim=1) ** 3
	curvature = num / den
	radius = 1.0 / curvature
	return radius

#==========================================================================
# ---- Crush stress dependence - Curvature equation ---- #
def CurvatureEq(x: torch.Tensor) -> torch.Tensor:
    #return 1.0 / (x ** 0.9409590693634183) * 7.951858693686812 + 0.9900990009786546
    #return (1.0 / (x ** 0.2409590693634183)) * 5.951858693686812 + 0.9900990009786546
    return 7.95 * torch.pow(x, -0.94) + 0.99

#==========================================================================
# ---- Compute Perimeter ---- #
def perimeterCalc(orderedNodes):
    nodes_shifted = torch.roll(orderedNodes, -1, dims=0)
    diffs = nodes_shifted - orderedNodes
    return torch.sum(torch.norm(diffs, dim=1))

#==========================================================================
# ---- Plot ---- #
def plot_initial_and_final(initial_nodes, final_nodes):
	# Enable LaTeX rendering
	plt.rcParams.update({
    	"text.usetex": True,
    	"font.family": "serif",
    	"font.serif": ["Computer Modern Roman"],  # default LaTeX font
    	"font.size": 11  # global font size
	})

	#fig = plt.figure(figsize=(3.94, 3.94))
	#ax = fig.add_subplot(111, projection='3d')

	fig, ax = plt.subplots(figsize=(4.72, 2.25))

    # Plot initial in blue
	ax.plot(initial_nodes[:, 0], initial_nodes[:, 1],
            marker='o', markersize=4, linewidth=0.75, color=(0, 0.447, 0.698), label=r'Initial')

    # Plot final in red
	ax.plot(final_nodes[:, 0], final_nodes[:, 1],
            marker='o', markersize=4, linewidth=0.75, color=(0.902, 0.624, 0), label=r'Optimised')

	#ax.set_title("Initial vs. Optimized Cross Section")
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")
	#ax.set_zlabel(r"$z$")
	# Equal aspect ratio for geometry accuracy
	ax.set_aspect('equal')

	# Adjust bottom space to fit legend
	plt.subplots_adjust(bottom=0.25)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
	#plt.tight_layout()
	plt.show()

	fig.savefig("optim_square100k.pdf", format="pdf", bbox_inches="tight")

#==========================================================================
# ---- Penalty Constraints ---- #

# FIX 1: Smoothness term now wraps around the closed loop.
#         Previously, the second-difference computation was purely sequential
#         from node 0 to node N-1, leaving the first-to-last node transition
#         completely unpenalised. The optimiser could freely create a sharp
#         kink at the closure because no smoothness gradient existed there.
#         Now we prepend r[-1] and append r[0] so the penalty covers the
#         full closed contour.
def smoothnessTerm(r):
    # Wrap radial distances for closed-loop continuity
    r_wrapped = torch.cat([r[-1:], r, r[:1]])
    # First derivative (change in radius)
    dr = r_wrapped[1:] - r_wrapped[:-1]
    # Second derivative approximation (curvature of the r profile)
    curvature = dr[1:] - dr[:-1]
    # Penalise large curvature variations
    smoothness_loss = torch.sum(curvature ** 2)

    return smoothness_loss

#==========================================================================
#==========================================================================
# MAIN SCRIPT

# User Inputs
thickness = 2 # [mm] thickness of partition cross-section
# Input flags
minPerimeterFlag = False
maxCrushForceFlag = True
inwardSecFlag = False
dataFileName = 'partition_data_square.mat'
constantCrushStress = 90 # [MPa]
#==========================================================================
#==========================================================================
# Load cross section data from MATLAB
partitionData = loadData(dataFileName)

# Process data
# Extract arrays
nodes = partitionData['nodesCoords'] # NumPy array
edges = partitionData['connectivity'] # NumPy array
print('Nodes Coords:', nodes)
orderedIndices, orderedNodes = orderNodes(nodes, edges)
overlapCount = 7
# Convert to PyTorch tensors
orderedNodes = torch.tensor(orderedNodes, dtype=torch.float32).detach()
orderedIndices = torch.tensor(orderedIndices, dtype=torch.long)

model = crushOptimizerModel(orderedNodes, orderedIndices, edges, constantCrushStress, overlapCount, thickness)

#==========================================================================
#==========================================================================
# Hyperparameters
n_steps = 15000 # Training loop iterations
learning_rate = 0.1 # Learning rate
# Penalty weights
lambda_smoothness = 1e10
lambda_crushForce = 1e10 # penalty weight for crush force below initial
lambda_radialDist = 1e10 # penalty weight for radial distances above initial
#==========================================================================
#==========================================================================

# compute initial values
with torch.no_grad():
        forceTotal_initial, perimeter_initial = model()
        print('Force initial:', forceTotal_initial)
        print('Perimeter initial', perimeter_initial)
        orderedNodes_initial = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_initial = nodesCoords(overlapCount, orderedNodes_initial).cpu()

# Optimizer initiation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---- Start timer for convergence study ---- #
t_start = time.perf_counter()

# The optimizer will adjust these parameters during training to minimize the loss function
# Optimization Loop
for epoch in range(n_steps):
        optimizer.zero_grad() # reset the gradients of model parameters
        force_total, perimeter = model() # Forward pass
        # Penalty constraints
        smoothness_loss = smoothnessTerm(model.r)
        crushForce_loss = torch.relu(forceTotal_initial - force_total)**2
        radialDist_loss = torch.sum(torch.relu(model.r - model.initial_r)**2)
        # Loss functions to be minimized during training (objective function) including penalties
        if minPerimeterFlag and not maxCrushForceFlag: # Case of minimising perimeter 
        	if inwardSecFlag: # enforce optimised section with all nodes inward
        		loss = perimeter + lambda_smoothness * smoothness_loss + lambda_crushForce * crushForce_loss + lambda_radialDist * radialDist_loss
        	else: # optimised section may have inward or outward nodes
        		loss = perimeter + lambda_smoothness * smoothness_loss + lambda_crushForce * crushForce_loss
        elif maxCrushForceFlag and not minPerimeterFlag: # Case of maximising crush force of the cross-section
        	loss = -force_total + lambda_smoothness * smoothness_loss
        else:
        	print("Invalid configuration: either both flags are set or none.")

        loss.backward() # backpropagate the prediction loss with a call to loss.backward()
        optimizer.step() # adjust/update the parameters by the gradients collected in the backward pass

        if epoch % 1 == 0:
                print(f"Epoch {epoch}: Perimeter = {perimeter:.4f}, "
                      f"Crush Force = {force_total:.4f}, "
                      f"Loss = {loss.item():.4f}")

# ---- Stop timer ---- #
t_end = time.perf_counter()
elapsed = t_end - t_start

# ---- Visualize optimized spline ----
with torch.no_grad():
        forceTotal_final, perimeter_final = model()
        print('Force final:', forceTotal_final)
        print('Perimeter final:', perimeter_final)
        orderedNodes_final = Spherical2Cartesian(model.r, model.theta, model.phi)
        nodesExt_final = nodesCoords(overlapCount, orderedNodes_final).detach().cpu()
        plot_initial_and_final(nodesExt_initial, nodesExt_final)

# ---- Runtime summary for convergence study ---- #
print("\n" + "="*60)
print("CONVERGENCE STUDY — RUN SUMMARY")
print("="*60)
print(f"  Model file:         {dataFileName}")
print(f"  Optimisation mode:  {'Max crush force' if maxCrushForceFlag else 'Min perimeter'}"
      f"{'  (inward only)' if inwardSecFlag and minPerimeterFlag else ''}")
print(f"  Steps (n_steps):    {n_steps}")
print(f"  Learning rate:      {learning_rate}")
print(f"  Overlap count:      {overlapCount}")
print(f"  lambda_smoothness:  {lambda_smoothness:.0e}")
print(f"  lambda_crushForce:  {lambda_crushForce:.0e}")
print(f"  lambda_radialDist:  {lambda_radialDist:.0e}")
print(f"  ---")
print(f"  Force  initial:     {forceTotal_initial.item():.4f}")
print(f"  Force  final:       {forceTotal_final.item():.4f}")
print(f"  Force  change:      {((forceTotal_final.item() - forceTotal_initial.item()) / forceTotal_initial.item()) * 100:+.2f}%")
print(f"  Perim  initial:     {perimeter_initial.item():.4f}")
print(f"  Perim  final:       {perimeter_final.item():.4f}")
print(f"  Perim  change:      {((perimeter_final.item() - perimeter_initial.item()) / perimeter_initial.item()) * 100:+.2f}%")
print(f"  Final loss:         {loss.item():.4f}")
print(f"  ---")
print(f"  Wall-clock time:    {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print(f"  Time per step:      {elapsed/n_steps*1000:.2f} ms")
print("="*60)
