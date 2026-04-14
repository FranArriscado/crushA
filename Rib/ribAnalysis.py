#
#                     ribAnalysis.py
#
#  Branch-aware cross-section force calculator for CrushAnalytica.
#  Phase A of rib support: computes crush force for arbitrary
#  cross-section topologies (closed loops, open chains, branched
#  graphs with internal ribs).
#
#  Faithfully ports the curvatureSpline logic from layersProp.m.
#  Uses scipy/numpy (not PyTorch) — no differentiability needed
#  for force evaluation. This gives more accurate spline fitting
#  than torchcubicspline and avoids the ~3% discrepancy in v15.
#
#  Usage:
#    python ribAnalysis.py                          # run synthetic tests
#    python ribAnalysis.py partition_data_square.mat # analyse a .mat file
#
#  Francisco Arriscado — FEUP, 2026
#==========================================================================

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

#==========================================================================
#                     MATERIAL / PHYSICS CONSTANTS
#==========================================================================
# From DataCurvatureCarbon.xlsx — CFRP curvature crush stress factor
# f(RoC) = a * RoC^b + c  (power law + offset)
CURV_A = 7.9519
CURV_B = -0.9410
CURV_C = 0.9901

# Bounds (for reporting, not used in optimisation here)
CURV_A_LOWER = 4.78983973619878
CURV_B_LOWER = -0.5883999449654764
CURV_C_LOWER = 0.6523314616586323

CURV_A_UPPER = 19.560792535656788
CURV_B_UPPER = -1.3900057011370337
CURV_C_UPPER = 1.151550789276735

# Default flat transition from DataCurvatureCarbon.xlsx
FLAT_TRANSITION_DEFAULT = 1221.947188466746  # [mm]

# Default analysis parameters
DEFAULT_THICKNESS = 2.0          # [mm]
DEFAULT_CRUSH_STRESS = 90.0      # [MPa]
DEFAULT_MIN_ROC = 5.0            # [mm] — matches MATLAB clamp
DEFAULT_OVERLAP_COUNT = 10

#==========================================================================
#                     CURVATURE EQUATION
#==========================================================================

def curvature_eq(roc, a=CURV_A, b=CURV_B, c=CURV_C):
    """Curvature crush stress factor: f(RoC) = a * RoC^b + c"""
    return a * np.power(roc, b) + c


#==========================================================================
#                     GRAPH / TOPOLOGY ANALYSIS
#==========================================================================

def build_adjacency(edges):
    """Build adjacency list and degree array from edge list.

    Parameters
    ----------
    edges : ndarray, shape (M, 2)
        Edge list (0-based node indices).

    Returns
    -------
    adj : dict[int, list[int]]
        Adjacency list.
    degree : ndarray, shape (max_node+1,)
        Degree of each node.
    """
    adj = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)

    if len(adj) == 0:
        return adj, np.array([])

    max_node = max(adj.keys())
    degree = np.zeros(max_node + 1, dtype=int)
    for node, neighbours in adj.items():
        degree[node] = len(neighbours)

    return adj, degree


def find_connected_components(nodes_count, edges):
    """Find connected components using union-find.

    Parameters
    ----------
    nodes_count : int
        Number of nodes.
    edges : ndarray, shape (M, 2)
        Edge list (0-based).

    Returns
    -------
    labels : ndarray, shape (nodes_count,)
        Component label for each node (0-based).
    n_components : int
        Number of components.
    """
    parent = list(range(nodes_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(int(a), int(b))

    # Normalise labels to sequential 0-based
    root_to_label = {}
    labels = np.zeros(nodes_count, dtype=int)
    next_label = 0
    for i in range(nodes_count):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels, next_label


def decompose_branches(edges_component, adj, degree):
    """Decompose a single connected component into branches.

    Mirrors the MATLAB logic in layersProp.m lines 436-531.
    A branch is a maximal path between junction/end nodes,
    or a loop of degree-2 nodes.

    Parameters
    ----------
    edges_component : ndarray, shape (M, 2)
        Edges within this component (0-based).
    adj : dict
        Full adjacency list.
    degree : ndarray
        Full degree array.

    Returns
    -------
    branches : list of list[int]
        Each branch is an ordered list of node indices.
    junction_nodes : list[int]
        Nodes with degree > 2.
    end_nodes : list[int]
        Nodes with degree == 1.
    """
    # Identify special nodes in this component
    component_nodes = set()
    for a, b in edges_component:
        component_nodes.add(int(a))
        component_nodes.add(int(b))

    junction_nodes = [n for n in component_nodes if degree[n] > 2]
    end_nodes = [n for n in component_nodes if degree[n] == 1]

    # Build edge set for fast lookup and visited tracking
    edge_set = set()
    for a, b in edges_component:
        a, b = int(a), int(b)
        edge_set.add((min(a, b), max(a, b)))

    visited_edges = set()

    def mark_edge(a, b):
        return (min(a, b), max(a, b))

    branches = []

    # Start walks from all end nodes and junction nodes
    start_nodes = end_nodes + junction_nodes

    for start in start_nodes:
        for nb in adj[start]:
            if nb not in component_nodes:
                continue
            edge_key = mark_edge(start, nb)
            if edge_key not in edge_set or edge_key in visited_edges:
                continue

            # Walk this branch
            path = [start, nb]
            visited_edges.add(edge_key)

            current = nb
            prev = start
            while True:
                if degree[current] != 2:
                    break  # hit a junction or endpoint

                # Find next node (the neighbour that isn't prev)
                neighbours_in_comp = [n for n in adj[current]
                                      if n in component_nodes]
                next_candidates = [n for n in neighbours_in_comp
                                   if n != prev]
                if not next_candidates:
                    break

                next_node = next_candidates[0]
                edge_key = mark_edge(current, next_node)
                if edge_key in visited_edges:
                    break

                visited_edges.add(edge_key)
                path.append(next_node)
                prev = current
                current = next_node

            branches.append(path)

    # Catch any unvisited loops (pure degree-2 cycles with no junctions)
    for a, b in edges_component:
        a, b = int(a), int(b)
        edge_key = mark_edge(a, b)
        if edge_key in visited_edges:
            continue

        # Start a loop walk
        loop_path = [a, b]
        visited_edges.add(edge_key)
        prev = a
        current = b

        while True:
            neighbours_in_comp = [n for n in adj[current]
                                  if n in component_nodes]
            next_candidates = [n for n in neighbours_in_comp if n != prev]
            if not next_candidates:
                break
            next_node = next_candidates[0]
            edge_key = mark_edge(current, next_node)
            if edge_key in visited_edges:
                break
            visited_edges.add(edge_key)
            loop_path.append(next_node)
            prev = current
            current = next_node

        branches.append(loop_path)

    return branches, junction_nodes, end_nodes


def analyze_topology(nodes, edges):
    """Full topology analysis of a cross-section.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Node coordinates.
    edges : ndarray, shape (M, 2)
        Edge list (0-based).

    Returns
    -------
    result : dict with keys:
        'type' : str — 'closed_loop', 'open_chain', 'branched', or 'multi_component'
        'components' : list of dicts, each with:
            'node_ids' : list[int]
            'edges' : ndarray
            'branches' : list of list[int]
            'junction_nodes' : list[int]
            'end_nodes' : list[int]
            'is_closed' : bool (only for single-branch components)
    """
    n_nodes = nodes.shape[0]
    comp_labels, n_comp = find_connected_components(n_nodes, edges)
    adj, degree = build_adjacency(edges)

    components = []

    for c in range(n_comp):
        comp_node_ids = np.where(comp_labels == c)[0].tolist()
        comp_node_set = set(comp_node_ids)

        # Filter edges to this component
        comp_edges = []
        for a, b in edges:
            a, b = int(a), int(b)
            if a in comp_node_set and b in comp_node_set:
                comp_edges.append([a, b])
        comp_edges = np.array(comp_edges) if comp_edges else np.empty((0, 2), dtype=int)

        # Check for branching
        has_junctions = any(degree[n] > 2 for n in comp_node_ids)
        has_endpoints = any(degree[n] == 1 for n in comp_node_ids)

        if has_junctions:
            branches, junctions, endpoints = decompose_branches(
                comp_edges, adj, degree)
            comp_info = {
                'node_ids': comp_node_ids,
                'edges': comp_edges,
                'branches': branches,
                'junction_nodes': junctions,
                'end_nodes': endpoints,
                'is_branched': True,
                'is_closed': False,
            }
        else:
            # Simple topology: closed loop or open chain
            if has_endpoints:
                # Open chain (degree-1 endpoints exist)
                # Order nodes along the chain
                end_pts = [n for n in comp_node_ids if degree[n] == 1]
                ordered = order_nodes_chain(comp_node_ids, comp_edges,
                                            start_node=end_pts[0])
                comp_info = {
                    'node_ids': comp_node_ids,
                    'edges': comp_edges,
                    'branches': [ordered],
                    'junction_nodes': [],
                    'end_nodes': end_pts,
                    'is_branched': False,
                    'is_closed': False,
                }
            else:
                # Closed loop (all degree 2, no endpoints)
                ordered = order_nodes_chain(comp_node_ids, comp_edges,
                                            start_node=comp_node_ids[0])
                comp_info = {
                    'node_ids': comp_node_ids,
                    'edges': comp_edges,
                    'branches': [ordered],
                    'junction_nodes': [],
                    'end_nodes': [],
                    'is_branched': False,
                    'is_closed': True,
                }

        components.append(comp_info)

    # Classify overall topology
    if n_comp == 1:
        c = components[0]
        if c['is_branched']:
            topo_type = 'branched'
        elif c['is_closed']:
            topo_type = 'closed_loop'
        else:
            topo_type = 'open_chain'
    else:
        topo_type = 'multi_component'

    return {
        'type': topo_type,
        'n_components': n_comp,
        'components': components,
    }


def order_nodes_chain(node_ids, edges, start_node):
    """Order nodes along a simple chain or closed loop (all degree ≤ 2).

    Mirrors MATLAB layersProp.m lines 612-672.
    """
    adj_local = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b)
        adj_local[a].append(b)
        adj_local[b].append(a)

    ordered = [start_node]
    visited_nodes = {start_node}

    current = start_node
    # Pick first neighbour
    if adj_local[current]:
        next_node = adj_local[current][0]
    else:
        return ordered

    ordered.append(next_node)
    visited_nodes.add(next_node)
    prev = current
    current = next_node

    while True:
        neighbours = [n for n in adj_local[current] if n not in visited_nodes]
        if not neighbours:
            break
        next_node = neighbours[0]
        ordered.append(next_node)
        visited_nodes.add(next_node)
        prev = current
        current = next_node

    return ordered


#==========================================================================
#                     SPLINE FITTING & RoC COMPUTATION
#==========================================================================

def compute_roc_closed_loop(xyz, overlap_count=10):
    """Compute per-node RoC for a closed loop using spline with overlap.

    Mirrors MATLAB layersProp.m lines 686-743 (closed loop path).

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Ordered node coordinates for the closed loop.
    overlap_count : int
        Number of nodes to wrap for spline continuity.

    Returns
    -------
    roc : ndarray, shape (N,)
        Radius of curvature at each node.
    """
    n = xyz.shape[0]
    if n < 4:
        # Too few nodes for meaningful spline
        return np.full(n, 1e6)

    # Add overlap nodes for continuity (MATLAB lines 689-692)
    xyz_ext = np.vstack([
        xyz[-overlap_count:],
        xyz,
        xyz[:overlap_count]
    ])

    # Parameterise by chord length (MATLAB lines 701-704)
    diffs = np.diff(xyz_ext, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(xyz_ext))
    s[1:] = np.cumsum(ds)
    s = s / s[-1]  # normalise to [0, 1]

    # Fit cubic splines (MATLAB uses spaps with tol=1e-2;
    # scipy CubicSpline is interpolating, closer to MATLAB's behaviour
    # on well-spaced data than torchcubicspline's NaturalCubicSpline)
    cs_x = CubicSpline(s, xyz_ext[:, 0])
    cs_y = CubicSpline(s, xyz_ext[:, 1])
    cs_z = CubicSpline(s, xyz_ext[:, 2])

    # Query at linspace points within the original (non-overlap) range
    # (MATLAB line 716)
    s_start = s[overlap_count]
    s_end = s[-overlap_count]
    s_query = np.linspace(s_start, s_end, n)

    # First derivatives
    dx = cs_x(s_query, 1)
    dy = cs_y(s_query, 1)
    dz = cs_z(s_query, 1)

    # Second derivatives
    d2x = cs_x(s_query, 2)
    d2y = cs_y(s_query, 2)
    d2z = cs_z(s_query, 2)

    # Curvature: κ = |r' × r''| / |r'|³  (MATLAB lines 729-737)
    T = np.column_stack([dx, dy, dz])
    N = np.column_stack([d2x, d2y, d2z])
    cross_TN = np.cross(T, N)
    num = np.linalg.norm(cross_TN, axis=1)
    den = np.linalg.norm(T, axis=1) ** 3

    curvature = num / (den + 1e-30)
    roc = 1.0 / (curvature + 1e-30)

    return roc


def compute_roc_open_branch(xyz):
    """Compute per-node RoC for an open branch (no overlap).

    Mirrors MATLAB layersProp.m lines 540-571 (branch path).
    Key difference from closed loop: no overlap, sQuery = s (at
    original chord-length positions).

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Ordered node coordinates along the branch.

    Returns
    -------
    roc : ndarray, shape (N,)
        Radius of curvature at each node.
    """
    n = xyz.shape[0]
    if n < 4:
        # Too few nodes — assign large RoC (flat)
        return np.full(n, 1e6)

    # Parameterise by chord length (MATLAB lines 548-550)
    diffs = np.diff(xyz, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s = np.zeros(n)
    s[1:] = np.cumsum(ds)
    s_total = s[-1]
    if s_total < 1e-12:
        return np.full(n, 1e6)
    s = s / s_total

    # Fit cubic spline (MATLAB uses spaps; we use CubicSpline)
    cs_x = CubicSpline(s, xyz[:, 0])
    cs_y = CubicSpline(s, xyz[:, 1])
    cs_z = CubicSpline(s, xyz[:, 2])

    # Evaluate at original chord-length positions (MATLAB line 558: sQuery = s)
    s_query = s

    dx = cs_x(s_query, 1)
    dy = cs_y(s_query, 1)
    dz = cs_z(s_query, 1)

    d2x = cs_x(s_query, 2)
    d2y = cs_y(s_query, 2)
    d2z = cs_z(s_query, 2)

    T = np.column_stack([dx, dy, dz])
    N = np.column_stack([d2x, d2y, d2z])
    cross_TN = np.cross(T, N)
    num = np.linalg.norm(cross_TN, axis=1)
    den = np.linalg.norm(T, axis=1) ** 3

    curvature = num / (den + 1e-30)
    roc = 1.0 / (curvature + 1e-30)

    return roc


#==========================================================================
#              MAIN FORCE CALCULATOR (BRANCH-AWARE)
#==========================================================================

def compute_section_force(nodes, edges, params=None):
    """Compute total crush force for a cross-section of any topology.

    This is the main entry point — handles closed loops, open chains,
    and branched topologies (with ribs) identically.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Node coordinates.
    edges : ndarray, shape (M, 2)
        Edge list (0-based).
    params : dict, optional
        Analysis parameters:
            'thickness' : float [mm]
            'constantCrushStress' : float [MPa]
            'flat_transition' : float [mm]
            'min_roc' : float [mm]
            'overlap_count' : int

    Returns
    -------
    result : dict with keys:
        'force_total' : float — total crush force [N]
        'force_total_lower' : float — lower bound
        'force_total_upper' : float — upper bound
        'perimeter' : float — total perimeter [mm]
        'topology' : str — topology type
        'n_branches' : int
        'per_node' : dict with arrays (roc, crush_stress, force, node_length, area)
        'per_branch' : list of dicts with per-branch breakdown
    """
    # Defaults
    if params is None:
        params = {}
    thickness = params.get('thickness', DEFAULT_THICKNESS)
    sigma0 = params.get('constantCrushStress', DEFAULT_CRUSH_STRESS)
    flat_trans = params.get('flat_transition', FLAT_TRANSITION_DEFAULT)
    min_roc = params.get('min_roc', DEFAULT_MIN_ROC)
    overlap_count = params.get('overlap_count', DEFAULT_OVERLAP_COUNT)

    n_nodes = nodes.shape[0]

    # ---- Topology analysis ----
    topo = analyze_topology(nodes, edges)

    # ---- Per-node arrays ----
    roc_all = np.zeros(n_nodes)
    roc_sum = np.zeros(n_nodes)
    roc_count = np.zeros(n_nodes, dtype=int)

    branch_results = []

    for comp in topo['components']:
        if comp['is_branched']:
            # Branched: per-branch spline, accumulate RoC for averaging
            for branch in comp['branches']:
                branch_nodes = np.array(branch)
                xyz = nodes[branch_nodes]

                # Check if this branch forms a closed loop
                # (first node == last node's neighbour and both are junctions)
                branch_is_closed = (len(branch) > 2 and
                                    branch[0] == branch[-1])
                if branch_is_closed:
                    # Remove the repeated last node for spline fitting
                    xyz = xyz[:-1]
                    branch_roc = compute_roc_closed_loop(xyz, overlap_count)
                    branch_node_ids = branch_nodes[:-1]
                else:
                    branch_roc = compute_roc_open_branch(xyz)
                    branch_node_ids = branch_nodes

                # Accumulate for junction averaging (MATLAB lines 585-601)
                for i, nid in enumerate(branch_node_ids):
                    roc_sum[nid] += branch_roc[i]
                    roc_count[nid] += 1

                branch_results.append({
                    'node_ids': branch_node_ids.tolist(),
                    'roc_raw': branch_roc,
                    'is_closed': branch_is_closed,
                    'n_nodes': len(branch_node_ids),
                })

        elif comp['is_closed']:
            # Single closed loop
            ordered = comp['branches'][0]
            xyz = nodes[ordered]
            loop_roc = compute_roc_closed_loop(xyz, overlap_count)

            for i, nid in enumerate(ordered):
                roc_sum[nid] += loop_roc[i]
                roc_count[nid] += 1

            branch_results.append({
                'node_ids': ordered,
                'roc_raw': loop_roc,
                'is_closed': True,
                'n_nodes': len(ordered),
            })

        else:
            # Open chain
            ordered = comp['branches'][0]
            xyz = nodes[ordered]
            chain_roc = compute_roc_open_branch(xyz)

            for i, nid in enumerate(ordered):
                roc_sum[nid] += chain_roc[i]
                roc_count[nid] += 1

            branch_results.append({
                'node_ids': ordered,
                'roc_raw': chain_roc,
                'is_closed': False,
                'n_nodes': len(ordered),
            })

    # ---- Compute averaged RoC (for junction nodes) ----
    mask = roc_count > 0
    roc_all[mask] = roc_sum[mask] / roc_count[mask]

    # ---- Clamp RoC (MATLAB line 751) ----
    roc_all = np.clip(roc_all, min_roc, flat_trans)

    # ---- Per-node properties ----
    # Node length: half-sum of connected edge lengths (MATLAB lines 207-209)
    node_length = np.zeros(n_nodes)
    for a, b in edges:
        a, b = int(a), int(b)
        elen = np.linalg.norm(nodes[a] - nodes[b])
        node_length[a] += elen / 2
        node_length[b] += elen / 2

    # Area per node
    area = node_length * thickness

    # Crush stress (curvature factor only)
    crush_stress = sigma0 * curvature_eq(roc_all)
    crush_stress_lower = sigma0 * curvature_eq(roc_all,
                                                CURV_A_LOWER, CURV_B_LOWER, CURV_C_LOWER)
    crush_stress_upper = sigma0 * curvature_eq(roc_all,
                                                CURV_A_UPPER, CURV_B_UPPER, CURV_C_UPPER)

    # Per-node force
    force_node = crush_stress * area
    force_node_lower = crush_stress_lower * area
    force_node_upper = crush_stress_upper * area

    # Total force
    force_total = np.sum(force_node)
    force_total_lower = np.sum(force_node_lower)
    force_total_upper = np.sum(force_node_upper)

    # Perimeter: sum of all edge lengths
    perimeter = 0.0
    edge_lengths = []
    for a, b in edges:
        elen = np.linalg.norm(nodes[int(a)] - nodes[int(b)])
        perimeter += elen
        edge_lengths.append(elen)

    # ---- Per-branch force breakdown ----
    # Junction nodes appear in multiple branches. Split their force
    # equally across branches to avoid double-counting.
    node_branch_count = np.zeros(n_nodes, dtype=int)
    for br in branch_results:
        for nid in br['node_ids']:
            node_branch_count[nid] += 1

    for br in branch_results:
        br_node_ids = br['node_ids']
        # Weight each node's force by 1/(number of branches it appears in)
        br_force = 0.0
        for nid in br_node_ids:
            br_force += force_node[nid] / max(node_branch_count[nid], 1)
        br['force'] = br_force
        br['perimeter'] = np.sum(node_length[br_node_ids])

    # ---- Assemble result ----
    total_branches = sum(len(c['branches']) for c in topo['components'])

    result = {
        'force_total': float(force_total),
        'force_total_lower': float(force_total_lower),
        'force_total_upper': float(force_total_upper),
        'perimeter': float(perimeter),
        'sea': float(force_total / perimeter) if perimeter > 0 else 0.0,
        'topology': topo['type'],
        'n_components': topo['n_components'],
        'n_branches': total_branches,
        'per_node': {
            'roc': roc_all,
            'crush_stress': crush_stress,
            'crush_stress_lower': crush_stress_lower,
            'crush_stress_upper': crush_stress_upper,
            'force': force_node,
            'node_length': node_length,
            'area': area,
        },
        'per_branch': branch_results,
        'params': {
            'thickness': thickness,
            'constantCrushStress': sigma0,
            'flat_transition': flat_trans,
            'min_roc': min_roc,
            'overlap_count': overlap_count,
        },
    }

    return result


#==========================================================================
#                     SYNTHETIC TEST GEOMETRIES
#==========================================================================

def make_square(width=150.0, height=100.0, n_per_side=38, z=0.0):
    """Generate a closed-loop square cross-section.

    Matches partition_data_square.mat layout (~154 nodes).
    """
    hw, hh = width / 2, height / 2
    # Generate nodes along each side
    top = np.column_stack([
        np.linspace(-hw, hw, n_per_side + 1)[:-1],
        np.full(n_per_side, hh),
        np.full(n_per_side, z)
    ])
    right = np.column_stack([
        np.full(n_per_side, hw),
        np.linspace(hh, -hh, n_per_side + 1)[:-1],
        np.full(n_per_side, z)
    ])
    bottom = np.column_stack([
        np.linspace(hw, -hw, n_per_side + 1)[:-1],
        np.full(n_per_side, -hh),
        np.full(n_per_side, z)
    ])
    left = np.column_stack([
        np.full(n_per_side, -hw),
        np.linspace(-hh, hh, n_per_side + 1)[:-1],
        np.full(n_per_side, z)
    ])

    nodes = np.vstack([top, right, bottom, left])
    n = nodes.shape[0]
    edges = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])

    return nodes, edges


def make_ellipse(a=75.0, b=50.0, n_nodes=56, z=0.0):
    """Generate a closed-loop ellipse cross-section."""
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    nodes = np.column_stack([
        a * np.cos(theta),
        b * np.sin(theta),
        np.full(n_nodes, z)
    ])
    edges = np.column_stack([np.arange(n_nodes),
                             np.roll(np.arange(n_nodes), -1)])
    return nodes, edges


def make_square_with_rib(width=150.0, height=100.0, n_per_side=20,
                         n_rib=10, z=0.0):
    """Generate a square cross-section with a horizontal internal rib.

    The rib connects the midpoints of the left and right walls.
    This creates a branched topology with 2 junction nodes (degree 3)
    and 3 branches.

    Corner strategy: each wall segment EXCLUDES its end-corner to avoid
    duplication. The next segment starts at its own start-corner.

    Returns
    -------
    nodes : ndarray, shape (N, 3)
    edges : ndarray, shape (M, 2)
    """
    hw, hh = width / 2, height / 2
    n_half = n_per_side // 2

    # Each segment: linspace with n+1 points, exclude LAST to avoid
    # sharing the next segment's first node.

    # Top: (-hw,hh) → (hw,hh), exclude (hw,hh)
    top = np.column_stack([
        np.linspace(-hw, hw, n_per_side + 1)[:-1],
        np.full(n_per_side, hh),
        np.full(n_per_side, z)
    ])
    # Right-top: (hw,hh) → (hw,0), exclude (hw,0) — junction goes next
    right_top = np.column_stack([
        np.full(n_half, hw),
        np.linspace(hh, 0, n_half + 1)[:-1],
        np.full(n_half, z)
    ])
    # Junction right: (hw, 0)
    junction_right = np.array([[hw, 0, z]])
    # Right-bottom: (hw,0-)→(hw,-hh), interior only (exclude both ends)
    right_bottom_interior = np.column_stack([
        np.full(max(n_half - 1, 0), hw),
        np.linspace(0, -hh, n_half + 1)[1:-1],
        np.full(max(n_half - 1, 0), z)
    ]) if n_half > 1 else np.empty((0, 3))
    # Bottom: (hw,-hh) → (-hw,-hh), exclude last
    bottom = np.column_stack([
        np.linspace(hw, -hw, n_per_side + 1)[:-1],
        np.full(n_per_side, -hh),
        np.full(n_per_side, z)
    ])
    # Left-bottom: (-hw,-hh) → (-hw,0), exclude (-hw,0) — junction next
    left_bottom = np.column_stack([
        np.full(n_half, -hw),
        np.linspace(-hh, 0, n_half + 1)[:-1],
        np.full(n_half, z)
    ])
    # Junction left: (-hw, 0)
    junction_left = np.array([[-hw, 0, z]])
    # Left-top: (-hw,0+)→(-hw,hh-), interior only (exclude both ends)
    left_top_interior = np.column_stack([
        np.full(max(n_half - 1, 0), -hw),
        np.linspace(0, hh, n_half + 1)[1:-1],
        np.full(max(n_half - 1, 0), z)
    ]) if n_half > 1 else np.empty((0, 3))

    # Rib interior nodes (between the two junctions, excluding endpoints)
    rib_interior = np.column_stack([
        np.linspace(-hw, hw, n_rib + 2)[1:-1],
        np.zeros(n_rib),
        np.full(n_rib, z)
    ])

    # Assemble all segments
    segments_list = [
        top,                   # seg 0
        right_top,             # seg 1
        junction_right,        # seg 2
        right_bottom_interior, # seg 3
        bottom,                # seg 4
        left_bottom,           # seg 5
        junction_left,         # seg 6
        left_top_interior,     # seg 7
        rib_interior,          # seg 8
    ]

    # Track index ranges
    segments = []
    idx = 0
    for seg in segments_list:
        n_seg = seg.shape[0]
        segments.append((idx, idx + n_seg))
        idx += n_seg

    nodes = np.vstack([s for s in segments_list if s.shape[0] > 0])

    # Build edges
    edges_list = []

    # Outer wall: chain through wall nodes (segs 0..7) in order
    wall_end = segments[7][1]  # end of left_top_interior
    for i in range(wall_end - 1):
        edges_list.append([i, i + 1])
    # Close the wall: last wall node → first wall node
    edges_list.append([wall_end - 1, 0])

    # Rib edges: junction_left → rib interior → junction_right
    idx_jr = segments[2][0]
    idx_jl = segments[6][0]
    rib_start = segments[8][0]
    rib_end = segments[8][1]
    n_rib_actual = rib_end - rib_start

    if n_rib_actual > 0:
        edges_list.append([idx_jl, rib_start])
        for i in range(rib_start, rib_end - 1):
            edges_list.append([i, i + 1])
        edges_list.append([rib_end - 1, idx_jr])
    else:
        edges_list.append([idx_jl, idx_jr])

    edges = np.array(edges_list, dtype=int)

    # Sanity check: no zero-length edges
    for a, b in edges:
        dist = np.linalg.norm(nodes[a] - nodes[b])
        assert dist > 1e-10, \
            f"Zero-length edge: {a}→{b} at {nodes[a]}"

    return nodes, edges


#==========================================================================
#                     REPORTING / VISUALISATION
#==========================================================================

def print_report(result, label=""):
    """Print a formatted analysis report."""
    print("\n" + "=" * 65)
    if label:
        print(f"  CROSS-SECTION ANALYSIS: {label}")
    else:
        print("  CROSS-SECTION ANALYSIS")
    print("=" * 65)

    p = result['params']
    print(f"  Thickness:          {p['thickness']:.1f} mm")
    print(f"  Crush stress (σ₀):  {p['constantCrushStress']:.1f} MPa")
    print(f"  Flat transition:    {p['flat_transition']:.2f} mm")
    print(f"  Min RoC clamp:      {p['min_roc']:.1f} mm")
    print(f"  ---")
    print(f"  Topology:           {result['topology']}")
    print(f"  Components:         {result['n_components']}")
    print(f"  Branches:           {result['n_branches']}")
    print(f"  ---")
    print(f"  Total force:        {result['force_total']:.2f} N"
          f"  ({result['force_total']/1000:.2f} kN)")
    print(f"  Force lower bound:  {result['force_total_lower']:.2f} N")
    print(f"  Force upper bound:  {result['force_total_upper']:.2f} N")
    print(f"  Perimeter:          {result['perimeter']:.2f} mm")
    print(f"  SEA (F/P):          {result['sea']:.4f} N/mm")

    # Per-branch breakdown
    if result['n_branches'] > 1:
        print(f"\n  --- Per-branch breakdown ---")
        for i, br in enumerate(result['per_branch']):
            closed_tag = " (closed)" if br['is_closed'] else " (open)"
            print(f"  Branch {i}: {br['n_nodes']:3d} nodes, "
                  f"force = {br['force']:.2f} N{closed_tag}")

    # Per-node statistics
    pn = result['per_node']
    active_mask = pn['node_length'] > 0
    if np.any(active_mask):
        print(f"\n  --- Per-node RoC statistics ---")
        roc_active = pn['roc'][active_mask]
        print(f"  RoC min:    {roc_active.min():.2f} mm")
        print(f"  RoC max:    {roc_active.max():.2f} mm")
        print(f"  RoC median: {np.median(roc_active):.2f} mm")
        print(f"  RoC mean:   {roc_active.mean():.2f} mm")

        # Nodes clamped at min_roc
        n_clamped_min = np.sum(roc_active <= p['min_roc'] + 0.01)
        n_clamped_max = np.sum(roc_active >= p['flat_transition'] - 0.01)
        print(f"  Nodes at min_roc:       {n_clamped_min} / {np.sum(active_mask)}")
        print(f"  Nodes at flat_trans:    {n_clamped_max} / {np.sum(active_mask)}")

    print("=" * 65)


def plot_section(nodes, edges, result, title="Cross-section analysis"):
    """Plot cross-section with RoC colouring and per-node force bars."""
    pn = result['per_node']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Shape with edges ---
    ax = axes[0]
    for a, b in edges:
        a, b = int(a), int(b)
        ax.plot([nodes[a, 0], nodes[b, 0]],
                [nodes[a, 1], nodes[b, 1]],
                'b-', linewidth=0.8, alpha=0.6)
    # Colour nodes by RoC
    roc_plot = pn['roc'].copy()
    roc_plot[roc_plot < 1] = 1  # avoid log(0)
    sc = ax.scatter(nodes[:, 0], nodes[:, 1],
                    c=np.log10(roc_plot), cmap='viridis',
                    s=20, zorder=5, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='log₁₀(RoC) [mm]')
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Shape & RoC')
    ax.grid(True, alpha=0.3)

    # --- RoC per node ---
    ax = axes[1]
    active = pn['node_length'] > 0
    node_indices = np.arange(len(pn['roc']))
    ax.plot(node_indices[active], pn['roc'][active], '.-', markersize=4)
    ax.set_yscale('log')
    ax.axhline(result['params']['min_roc'], color='r', linestyle='--',
               alpha=0.5, label=f"min_roc = {result['params']['min_roc']}")
    ax.axhline(result['params']['flat_transition'], color='r', linestyle='--',
               alpha=0.3, label='flat_transition')
    ax.set_xlabel('Node index')
    ax.set_ylabel('RoC [mm]')
    ax.set_title('Radius of curvature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Force per node ---
    ax = axes[2]
    ax.bar(node_indices[active], pn['force'][active],
           color=(0.2, 0.6, 0.8), edgecolor='none', width=1.0)
    ax.set_xlabel('Node index')
    ax.set_ylabel('Force [N]')
    ax.set_title(f"Force per node (Total = {result['force_total']:.0f} N)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


#==========================================================================
#                     .MAT FILE LOADER
#==========================================================================

def load_mat_geometry(filepath):
    """Load nodesCoords and connectivity from a .mat file.

    Returns 0-based edges.
    """
    data = loadmat(filepath)
    nodes = np.array(data['nodesCoords'], dtype=float)
    edges = np.array(data['connectivity'], dtype=int) - 1  # MATLAB 1-based → 0-based
    return nodes, edges


#==========================================================================
#                     MAIN
#==========================================================================

def run_synthetic_tests():
    """Run validation on synthetic geometries."""
    print("\n" + "#" * 65)
    print("#  ribAnalysis.py — Phase A validation tests")
    print("#" * 65)

    # ---- Test 1: Square (closed loop) ----
    nodes_sq, edges_sq = make_square(150, 100, n_per_side=38)
    topo_sq = analyze_topology(nodes_sq, edges_sq)
    print(f"\n[Test 1] Square 150×100 ({nodes_sq.shape[0]} nodes)")
    print(f"  Topology: {topo_sq['type']}")
    assert topo_sq['type'] == 'closed_loop', \
        f"Expected closed_loop, got {topo_sq['type']}"
    print("  ✓ Topology correct")

    result_sq = compute_section_force(nodes_sq, edges_sq)
    print_report(result_sq, "Square 150×100")

    # ---- Test 2: Ellipse (closed loop) ----
    nodes_el, edges_el = make_ellipse(75, 50, n_nodes=56)
    topo_el = analyze_topology(nodes_el, edges_el)
    print(f"\n[Test 2] Ellipse a=75, b=50 ({nodes_el.shape[0]} nodes)")
    print(f"  Topology: {topo_el['type']}")
    assert topo_el['type'] == 'closed_loop', \
        f"Expected closed_loop, got {topo_el['type']}"
    print("  ✓ Topology correct")

    result_el = compute_section_force(nodes_el, edges_el)
    print_report(result_el, "Ellipse 75×50")

    # ---- Test 3: Square with rib (branched) ----
    nodes_rib, edges_rib = make_square_with_rib(150, 100,
                                                 n_per_side=20, n_rib=10)
    topo_rib = analyze_topology(nodes_rib, edges_rib)
    print(f"\n[Test 3] Square 150×100 with horizontal rib "
          f"({nodes_rib.shape[0]} nodes, {edges_rib.shape[0]} edges)")
    print(f"  Topology: {topo_rib['type']}")
    assert topo_rib['type'] == 'branched', \
        f"Expected branched, got {topo_rib['type']}"

    comp = topo_rib['components'][0]
    print(f"  Branches: {len(comp['branches'])}")
    print(f"  Junction nodes: {comp['junction_nodes']}")
    print(f"  End nodes: {comp['end_nodes']}")
    print("  ✓ Topology correct")

    result_rib = compute_section_force(nodes_rib, edges_rib)
    print_report(result_rib, "Square 150×100 + horizontal rib")

    # ---- Comparison: with rib vs without rib ----
    # Generate matching square without rib for comparison
    nodes_plain, edges_plain = make_square(150, 100, n_per_side=20)
    result_plain = compute_section_force(nodes_plain, edges_plain)

    print("\n" + "=" * 65)
    print("  RIB EFFECT COMPARISON")
    print("=" * 65)
    f_plain = result_plain['force_total']
    f_rib = result_rib['force_total']
    p_plain = result_plain['perimeter']
    p_rib = result_rib['perimeter']
    print(f"  Without rib:  F = {f_plain:.2f} N, P = {p_plain:.2f} mm, "
          f"SEA = {result_plain['sea']:.4f}")
    print(f"  With rib:     F = {f_rib:.2f} N, P = {p_rib:.2f} mm, "
          f"SEA = {result_rib['sea']:.4f}")
    print(f"  Force gain from rib:     {((f_rib - f_plain) / f_plain) * 100:+.2f}%")
    print(f"  Perimeter increase:      {((p_rib - p_plain) / p_plain) * 100:+.2f}%")
    sea_plain = result_plain['sea']
    sea_rib = result_rib['sea']
    print(f"  SEA change:              {((sea_rib - sea_plain) / sea_plain) * 100:+.2f}%")
    print("=" * 65)

    # ---- Plots ----
    fig1 = plot_section(nodes_sq, edges_sq, result_sq, "Square 150×100")
    fig2 = plot_section(nodes_rib, edges_rib, result_rib,
                        "Square 150×100 + horizontal rib")

    # Save plots
    fig1.savefig("test_square.pdf", bbox_inches='tight')
    fig2.savefig("test_square_rib.pdf", bbox_inches='tight')
    print("\nPlots saved: test_square.pdf, test_square_rib.pdf")

    return result_sq, result_el, result_rib


def run_mat_analysis(filepath):
    """Analyse a .mat file."""
    print(f"\nLoading: {filepath}")
    nodes, edges = load_mat_geometry(filepath)
    print(f"  {nodes.shape[0]} nodes, {edges.shape[0]} edges")

    result = compute_section_force(nodes, edges)
    label = os.path.splitext(os.path.basename(filepath))[0]
    print_report(result, label)

    fig = plot_section(nodes, edges, result, label)
    pdf_path = label + "_analysis.pdf"
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nPlot saved: {pdf_path}")

    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Analyse a .mat file
        mat_path = sys.argv[1]
        if os.path.isfile(mat_path):
            run_mat_analysis(mat_path)
        else:
            print(f"File not found: {mat_path}")
            sys.exit(1)
    else:
        # Run synthetic tests
        run_synthetic_tests()
