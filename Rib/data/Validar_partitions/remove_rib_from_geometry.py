"""
Strip a single rib from a ribbed partition MAT file.

This is intentionally separate from the cleanup scripts:
  - cleanup_partition_geometry.py
  - cleanup_partition_geometry_branched.py

Those tools only improve node spacing. This script removes the rib itself
and writes a plain closed-loop geometry that can be used by the NoRib
optimizer.

Supported inputs
----------------
1. Branched geometry MAT files with:
     - nodesCoords
     - connectivity

   Assumptions:
     - one connected component
     - exactly one rib
     - exactly two junction nodes
     - exactly three branches total
       (two wall branches + one rib branch)

2. Rib optimizer output MAT files with:
     - wall_nodesCoords
     - wall_connectivity   (optional, but used if present)

Output
------
Writes:
  - nodesCoords
  - connectivity

The default output name keeps the original file name and appends
"_ribremoved" before ".mat".
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class Geometry:
    nodes: np.ndarray
    edges: np.ndarray  # 0-based, shape (M, 2)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ribremoved{input_path.suffix}")


def build_adjacency(edges: np.ndarray, node_count: int) -> tuple[dict[int, list[int]], np.ndarray]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    degree = np.zeros(node_count, dtype=int)
    for node, neighbours in adj.items():
        degree[node] = len(neighbours)
    return adj, degree


def ensure_single_component(node_count: int, edges: np.ndarray) -> None:
    if node_count == 0:
        raise ValueError("Geometry has no nodes.")
    if len(edges) == 0:
        raise ValueError("Geometry has no edges.")

    adj, _ = build_adjacency(edges, node_count)
    start = int(edges[0, 0])
    visited: set[int] = set()
    queue: deque[int] = deque([start])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                queue.append(nb)

    used_nodes = {int(a) for a, _ in edges} | {int(b) for _, b in edges}
    if visited != used_nodes:
        raise ValueError(
            "Expected a single connected ribbed geometry, but found multiple components."
        )


def decompose_branches(
    edges: np.ndarray,
    adj: dict[int, list[int]],
    degree: np.ndarray,
) -> tuple[list[list[int]], list[int], list[int]]:
    component_nodes = {int(a) for a, _ in edges} | {int(b) for _, b in edges}
    junction_nodes = sorted(n for n in component_nodes if degree[n] > 2)
    end_nodes = sorted(n for n in component_nodes if degree[n] == 1)

    edge_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}
    visited_edges: set[tuple[int, int]] = set()

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    branches: list[list[int]] = []
    start_nodes = end_nodes + junction_nodes

    for start in start_nodes:
        for nb in adj[start]:
            if nb not in component_nodes:
                continue
            ek = edge_key(start, nb)
            if ek not in edge_set or ek in visited_edges:
                continue

            path = [start, nb]
            visited_edges.add(ek)
            prev = start
            cur = nb

            while True:
                if degree[cur] != 2:
                    break

                next_candidates = [n for n in adj[cur] if n in component_nodes and n != prev]
                if not next_candidates:
                    break

                nxt = next_candidates[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in visited_edges:
                    break

                visited_edges.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt

            branches.append(path)

    unused = edge_set - visited_edges
    if unused:
        raise ValueError(
            "Encountered unused edges while decomposing branches. "
            "This does not look like a single-rib geometry."
        )

    return branches, junction_nodes, end_nodes


def branch_length(nodes: np.ndarray, branch: list[int]) -> float:
    branch_points = nodes[np.asarray(branch, dtype=int)]
    return float(np.sum(np.linalg.norm(np.diff(branch_points, axis=0), axis=1)))


def order_closed_loop(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("Expected nodes to have shape (N, 3).")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected connectivity to have shape (M, 2).")

    adj, degree = build_adjacency(edges, len(nodes))
    if not np.all(degree == 2):
        raise ValueError("Expected a closed loop: every node must have degree 2.")

    start = int(edges[0, 0])
    order = [start]
    prev = -1
    cur = start

    while True:
        next_candidates = [nb for nb in adj[cur] if nb != prev]
        if not next_candidates:
            raise ValueError("Broken loop: reached a dead end while ordering nodes.")

        nxt = int(next_candidates[0])
        if nxt == start:
            break
        if nxt in order:
            raise ValueError("Loop ordering failed: encountered a repeated node.")

        order.append(nxt)
        prev, cur = cur, nxt

        if len(order) > len(nodes):
            raise ValueError("Loop ordering exceeded node count.")

    if len(order) != len(nodes):
        raise ValueError(
            f"Loop ordering reached {len(order)} nodes but geometry has {len(nodes)}."
        )

    return np.asarray(order, dtype=int)


def join_wall_branches(branch_a: list[int], branch_b: list[int]) -> list[int]:
    if branch_a[-1] == branch_b[0]:
        wall_loop = branch_a + branch_b[1:]
    elif branch_a[-1] == branch_b[-1]:
        wall_loop = branch_a + branch_b[-2::-1]
    elif branch_a[0] == branch_b[0]:
        wall_loop = branch_a[::-1] + branch_b[1:]
    elif branch_a[0] == branch_b[-1]:
        wall_loop = branch_b + branch_a[1:]
    else:
        raise ValueError("The two wall branches do not share the same junction nodes.")

    if wall_loop[-1] == wall_loop[0]:
        wall_loop = wall_loop[:-1]
    return wall_loop


def extract_closed_wall_from_branched_geometry(geom: Geometry) -> tuple[np.ndarray, dict[str, object]]:
    nodes = np.asarray(geom.nodes, dtype=float)
    edges = np.asarray(geom.edges, dtype=int)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("Expected nodesCoords to have shape (N, 3).")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected connectivity to have shape (M, 2).")

    ensure_single_component(len(nodes), edges)
    adj, degree = build_adjacency(edges, len(nodes))

    if np.all(degree == 2):
        raise ValueError(
            "Input already looks like a closed loop with no rib. "
            "This script expects a branched geometry."
        )

    branches, junction_nodes, end_nodes = decompose_branches(edges, adj, degree)

    if len(junction_nodes) != 2:
        raise ValueError(
            f"Expected exactly 2 junction nodes for a single-rib geometry, got {len(junction_nodes)}."
        )
    if len(end_nodes) != 0:
        raise ValueError(
            "Expected a closed wall with one internal rib, but found open endpoints."
        )
    if len(branches) != 3:
        raise ValueError(
            f"Expected exactly 3 branches (2 wall + 1 rib), got {len(branches)}."
        )

    lengths = [branch_length(nodes, branch) for branch in branches]
    rib_idx = int(np.argmin(lengths))
    wall_indices = [idx for idx in range(len(branches)) if idx != rib_idx]

    rib_branch = branches[rib_idx]
    wall_branch_a = branches[wall_indices[0]]
    wall_branch_b = branches[wall_indices[1]]

    rib_ends = {rib_branch[0], rib_branch[-1]}
    junction_set = set(junction_nodes)
    if rib_ends != junction_set:
        raise ValueError(
            "The shortest branch does not connect the two junction nodes, so it cannot be treated as the rib."
        )

    wall_loop = join_wall_branches(wall_branch_a, wall_branch_b)
    wall_nodes = nodes[np.asarray(wall_loop, dtype=int)]

    info = {
        "mode": "branched-input",
        "junction_nodes_1based": [j + 1 for j in junction_nodes],
        "branch_lengths": lengths,
        "rib_branch_length": lengths[rib_idx],
        "wall_node_count": len(wall_nodes),
        "rib_node_count": len(rib_branch),
    }
    return wall_nodes, info


def extract_closed_wall_from_rib_optimizer_output(data: dict) -> tuple[np.ndarray, dict[str, object]]:
    wall_nodes = np.asarray(data["wall_nodesCoords"], dtype=float)

    if wall_nodes.ndim != 2 or wall_nodes.shape[1] != 3:
        raise ValueError("Expected wall_nodesCoords to have shape (N, 3).")

    if "wall_connectivity" in data:
        wall_edges = np.asarray(data["wall_connectivity"], dtype=int) - 1
        ordered = order_closed_loop(wall_nodes, wall_edges)
        wall_nodes = wall_nodes[ordered]

    info = {
        "mode": "rib-optimizer-output",
        "wall_node_count": len(wall_nodes),
    }
    return wall_nodes, info


def build_sequential_connectivity(node_count: int) -> np.ndarray:
    if node_count < 3:
        raise ValueError("A closed-loop wall needs at least 3 nodes.")
    connectivity = np.column_stack(
        [np.arange(1, node_count + 1), np.arange(2, node_count + 2)]
    )
    connectivity[-1, 1] = 1
    return connectivity


def load_and_extract_wall(input_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    data = loadmat(input_path)

    if "wall_nodesCoords" in data:
        return extract_closed_wall_from_rib_optimizer_output(data)

    if "nodesCoords" in data and "connectivity" in data:
        geom = Geometry(
            nodes=np.asarray(data["nodesCoords"], dtype=float),
            edges=np.asarray(data["connectivity"], dtype=int) - 1,
        )
        return extract_closed_wall_from_branched_geometry(geom)

    raise ValueError(
        "Unsupported MAT file. Expected either nodesCoords/connectivity or "
        "wall_nodesCoords[/wall_connectivity]."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path, help="Input ribbed MAT file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help='Output MAT file. Defaults to "<input>_ribremoved.mat".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_mat.resolve()
    output_path = (
        args.output.resolve() if args.output is not None else default_output_path(input_path)
    )

    wall_nodes, info = load_and_extract_wall(input_path)
    connectivity = build_sequential_connectivity(len(wall_nodes))

    savemat(
        output_path,
        {
            "nodesCoords": wall_nodes,
            "connectivity": connectivity,
        },
    )

    print(f"Input : {input_path}")
    print(f"Mode  : {info['mode']}")
    print(f"Wall nodes written: {len(wall_nodes)}")
    if "junction_nodes_1based" in info:
        print(f"Junction nodes (1-based): {info['junction_nodes_1based']}")
        print(f"Rib branch length: {info['rib_branch_length']:.6f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
