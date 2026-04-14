"""
Clean uneven node spacing in closed-loop partition MAT files.

This utility targets the partition artefacts introduced when a shell mesh is
intersected after quad-to-triangle conversion. The typical symptom is one very
short edge flanked by two nearly colinear edges, which creates a redundant node
without changing the actual boundary shape.

The cleaner:
  1. loads nodesCoords/connectivity from a MAT file
  2. orders the closed loop
  3. iteratively removes redundant near-colinear short-edge nodes
  4. saves a cleaned MAT file with sequential loop connectivity

It is intentionally conservative and only supports a single closed loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class Geometry:
    nodes: np.ndarray
    edges: np.ndarray  # 0-based


def load_geometry(path: Path) -> Geometry:
    data = loadmat(path)
    nodes = np.asarray(data["nodesCoords"], dtype=float)
    edges = np.asarray(data["connectivity"], dtype=int) - 1
    return Geometry(nodes=nodes, edges=edges)


def build_adjacency(edges: np.ndarray, node_count: int) -> list[list[int]]:
    adj = [[] for _ in range(node_count)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def order_closed_loop(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("Expected nodesCoords to have shape (N, 3).")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected connectivity to have shape (M, 2).")
    if len(nodes) == 0:
        raise ValueError("Geometry has no nodes.")

    adj = build_adjacency(edges, len(nodes))
    degree = np.array([len(nbs) for nbs in adj], dtype=int)
    if not np.all(degree == 2):
        raise ValueError(
            "Cleaner only supports a single closed loop (all nodes must have degree 2)."
        )

    start = int(edges[0, 0])
    order = [start]
    prev = None
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
            raise ValueError("Loop ordering failed: exceeded node count.")

    if len(order) != len(nodes):
        raise ValueError(
            f"Loop ordering reached {len(order)} nodes but geometry has {len(nodes)}."
        )

    return np.asarray(order, dtype=int)


def closed_edge_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def turning_angle_deg(prev_pt: np.ndarray, pt: np.ndarray, next_pt: np.ndarray) -> float:
    v1 = prev_pt - pt
    v2 = next_pt - pt
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def simplify_closed_loop(
    ordered_nodes: np.ndarray,
    ordered_indices: np.ndarray,
    short_ratio: float = 0.35,
    angle_tol_deg: float = 12.0,
    excess_rel_tol: float = 0.02,
    excess_abs_tol: float = 1e-3,
    min_nodes: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    keep_nodes = ordered_nodes.copy()
    keep_indices = ordered_indices.copy()
    removed_original_indices: list[int] = []

    while len(keep_nodes) > min_nodes:
        edge_lengths = closed_edge_lengths(keep_nodes)
        median_edge = float(np.median(edge_lengths))
        changed = False

        for i in range(len(keep_nodes)):
            prev_i = (i - 1) % len(keep_nodes)
            next_i = (i + 1) % len(keep_nodes)

            prev_pt = keep_nodes[prev_i]
            pt = keep_nodes[i]
            next_pt = keep_nodes[next_i]

            len_prev = float(np.linalg.norm(pt - prev_pt))
            len_next = float(np.linalg.norm(next_pt - pt))
            len_chord = float(np.linalg.norm(next_pt - prev_pt))

            longer = max(len_prev, len_next)
            shorter = min(len_prev, len_next)
            if longer <= 0.0:
                continue

            angle = turning_angle_deg(prev_pt, pt, next_pt)
            excess = (len_prev + len_next) - len_chord

            is_short_edge_outlier = shorter <= short_ratio * longer
            is_nearly_straight = abs(180.0 - angle) <= angle_tol_deg
            is_low_excess = excess <= max(excess_abs_tol, excess_rel_tol * median_edge)

            if is_short_edge_outlier and is_nearly_straight and is_low_excess:
                removed_original_indices.append(int(keep_indices[i]))
                keep_nodes = np.delete(keep_nodes, i, axis=0)
                keep_indices = np.delete(keep_indices, i, axis=0)
                changed = True
                break

        if not changed:
            break

    return keep_nodes, keep_indices, removed_original_indices


def build_sequential_connectivity(node_count: int) -> np.ndarray:
    connectivity = np.column_stack(
        [np.arange(1, node_count + 1), np.arange(2, node_count + 2)]
    )
    connectivity[-1, 1] = 1
    return connectivity


def edge_stats(points: np.ndarray) -> dict[str, float]:
    lengths = closed_edge_lengths(points)
    return {
        "count": float(len(lengths)),
        "min": float(np.min(lengths)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "max": float(np.max(lengths)),
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path, help="Input MAT file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MAT file. Defaults to <input>_cleaned.mat",
    )
    parser.add_argument(
        "--short-ratio",
        type=float,
        default=0.35,
        help="Remove a node only if its shorter adjacent edge is <= this ratio of the longer one.",
    )
    parser.add_argument(
        "--angle-tol-deg",
        type=float,
        default=12.0,
        help="Maximum deviation from 180 degrees allowed for redundant-node removal.",
    )
    parser.add_argument(
        "--excess-rel-tol",
        type=float,
        default=0.02,
        help="Relative tolerance on (Lprev + Lnext - chord) against the median edge length.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and report without saving a cleaned MAT file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_mat.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(input_path)

    geom = load_geometry(input_path)
    ordered_indices = order_closed_loop(geom.nodes, geom.edges)
    ordered_nodes = geom.nodes[ordered_indices]

    before = edge_stats(ordered_nodes)
    cleaned_nodes, cleaned_indices, removed = simplify_closed_loop(
        ordered_nodes,
        ordered_indices,
        short_ratio=args.short_ratio,
        angle_tol_deg=args.angle_tol_deg,
        excess_rel_tol=args.excess_rel_tol,
    )
    after = edge_stats(cleaned_nodes)

    print(f"Input : {input_path}")
    print(f"Nodes : {len(ordered_nodes)} -> {len(cleaned_nodes)}")
    print(
        "Edge lengths before: "
        f"min={before['min']:.6f}, median={before['median']:.6f}, "
        f"mean={before['mean']:.6f}, max={before['max']:.6f}"
    )
    print(
        "Edge lengths after : "
        f"min={after['min']:.6f}, median={after['median']:.6f}, "
        f"mean={after['mean']:.6f}, max={after['max']:.6f}"
    )

    if removed:
        removed_1based = [idx + 1 for idx in removed]
        print(f"Removed original node indices ({len(removed_1based)}): {removed_1based}")
    else:
        print("Removed original node indices (0): []")

    if args.dry_run:
        print("Dry run: no file written.")
        return

    connectivity = build_sequential_connectivity(len(cleaned_nodes))
    savemat(
        output_path,
        {
            "nodesCoords": cleaned_nodes,
            "connectivity": connectivity,
        },
    )
    print(f"Saved cleaned geometry to: {output_path}")


if __name__ == "__main__":
    main()
