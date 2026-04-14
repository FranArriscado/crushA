"""
Clean uneven node spacing in branched/open partition MAT files.

This utility targets graphs that contain junction nodes (degree > 2) or
endpoints (degree == 1), such as ribbed cross-sections. It operates
branch-by-branch:

  1. load nodesCoords/connectivity from a MAT file
  2. decompose the graph into maximal branches between special nodes
  3. remove redundant near-colinear short-edge nodes on each branch
  4. optionally redistribute each branch uniformly by arc length while
     preserving shared junction/endpoint nodes and any sharp corners
  5. rebuild and save the cleaned graph as a MAT file

Pure closed loops are intentionally rejected here; use cleanup_partition_geometry.py
for those.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class Geometry:
    nodes: np.ndarray
    edges: np.ndarray  # 0-based


@dataclass
class BranchRecord:
    component_id: int
    branch_id: int
    original_node_ids: list[int]
    cleaned_points: np.ndarray
    cleaned_node_ids: np.ndarray
    removed_original_ids: list[int]


def load_geometry(path: Path) -> Geometry:
    data = loadmat(path)
    nodes = np.asarray(data["nodesCoords"], dtype=float)
    edges = np.asarray(data["connectivity"], dtype=int) - 1
    return Geometry(nodes=nodes, edges=edges)


def build_adjacency(edges: np.ndarray, node_count: int) -> tuple[dict[int, list[int]], np.ndarray]:
    adj = defaultdict(list)
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    degree = np.zeros(node_count, dtype=int)
    for node, neighbours in adj.items():
        degree[node] = len(neighbours)
    return adj, degree


def find_connected_components(node_count: int, edges: np.ndarray) -> tuple[np.ndarray, int]:
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(int(a), int(b))

    root_to_label: dict[int, int] = {}
    labels = np.zeros(node_count, dtype=int)
    next_label = 0
    for i in range(node_count):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels, next_label


def decompose_branches(edges_component: np.ndarray, adj: dict[int, list[int]], degree: np.ndarray) -> tuple[list[list[int]], list[int], list[int]]:
    component_nodes = set()
    for a, b in edges_component:
        component_nodes.add(int(a))
        component_nodes.add(int(b))

    junction_nodes = [n for n in component_nodes if degree[n] > 2]
    end_nodes = [n for n in component_nodes if degree[n] == 1]

    edge_set = set()
    for a, b in edges_component:
        a = int(a)
        b = int(b)
        edge_set.add((min(a, b), max(a, b)))

    visited_edges = set()

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

            current = nb
            prev = start
            while True:
                if degree[current] != 2:
                    break

                next_candidates = [
                    n for n in adj[current] if n in component_nodes and n != prev
                ]
                if not next_candidates:
                    break

                nxt = next_candidates[0]
                ek2 = edge_key(current, nxt)
                if ek2 in visited_edges:
                    break

                visited_edges.add(ek2)
                path.append(nxt)
                prev, current = current, nxt

            branches.append(path)

    # Pure loops are not supported by this script.
    for a, b in edges_component:
        if edge_key(int(a), int(b)) not in visited_edges:
            raise ValueError(
                "Encountered a pure degree-2 loop component. "
                "Use cleanup_partition_geometry.py for closed loops."
            )

    return branches, junction_nodes, end_nodes


def analyze_graph(nodes: np.ndarray, edges: np.ndarray) -> tuple[list[dict], np.ndarray]:
    labels, n_components = find_connected_components(len(nodes), edges)
    adj, degree = build_adjacency(edges, len(nodes))

    components: list[dict] = []
    for comp_id in range(n_components):
        comp_node_ids = np.where(labels == comp_id)[0].tolist()
        comp_set = set(comp_node_ids)
        comp_edges = np.asarray(
            [[int(a), int(b)] for a, b in edges if int(a) in comp_set and int(b) in comp_set],
            dtype=int,
        )

        has_special = any(degree[n] != 2 for n in comp_node_ids)
        if not has_special:
            raise ValueError(
                "This branched cleaner only supports components with junctions or endpoints. "
                "Use cleanup_partition_geometry.py for pure closed loops."
            )

        branches, junctions, endpoints = decompose_branches(comp_edges, adj, degree)
        components.append(
            {
                "component_id": comp_id,
                "node_ids": comp_node_ids,
                "edges": comp_edges,
                "branches": branches,
                "junction_nodes": junctions,
                "end_nodes": endpoints,
            }
        )

    return components, degree


def polyline_edge_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def turning_angle_deg(prev_pt: np.ndarray, pt: np.ndarray, next_pt: np.ndarray) -> float:
    v1 = prev_pt - pt
    v2 = next_pt - pt
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def detect_sharp_corners_open(points: np.ndarray, angle_threshold_deg: float = 150.0) -> np.ndarray:
    corners: list[int] = []
    for i in range(1, len(points) - 1):
        angle = turning_angle_deg(points[i - 1], points[i], points[i + 1])
        if angle < angle_threshold_deg:
            corners.append(i)
    return np.asarray(corners, dtype=int)


def simplify_open_branch(
    points: np.ndarray,
    original_node_ids: np.ndarray,
    short_ratio: float = 0.35,
    angle_tol_deg: float = 12.0,
    excess_rel_tol: float = 0.02,
    excess_abs_tol: float = 1e-3,
    min_nodes: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    keep_points = points.copy()
    keep_ids = original_node_ids.copy()
    removed_ids: list[int] = []

    while len(keep_points) > min_nodes:
        edge_lengths = polyline_edge_lengths(keep_points)
        if len(edge_lengths) == 0:
            break
        median_edge = float(np.median(edge_lengths))
        changed = False

        for i in range(1, len(keep_points) - 1):
            prev_pt = keep_points[i - 1]
            pt = keep_points[i]
            next_pt = keep_points[i + 1]

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
                removed_ids.append(int(keep_ids[i]))
                keep_points = np.delete(keep_points, i, axis=0)
                keep_ids = np.delete(keep_ids, i, axis=0)
                changed = True
                break

        if not changed:
            break

    return keep_points, keep_ids, removed_ids


def resample_open_polyline_uniform(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count < 1:
        raise ValueError("Open-polyline resampling requires at least one sample.")
    if len(points) < 2:
        raise ValueError("Need at least two points to resample an open polyline.")

    seg_vecs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total_length = float(np.sum(seg_lengths))
    if total_length <= 0.0:
        return np.repeat(points[:1], target_count, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    sample_s = np.linspace(0.0, total_length, target_count, endpoint=True)
    resampled = np.zeros((target_count, points.shape[1]), dtype=float)

    for i, s in enumerate(sample_s):
        seg_idx = int(np.searchsorted(cumulative[1:], s, side="right"))
        seg_idx = min(seg_idx, len(seg_lengths) - 1)
        seg_start = cumulative[seg_idx]
        seg_length = seg_lengths[seg_idx]

        if seg_length <= 0.0:
            resampled[i] = points[seg_idx]
            continue

        alpha = (s - seg_start) / seg_length
        resampled[i] = points[seg_idx] + alpha * seg_vecs[seg_idx]

    return resampled


def allocate_counts_from_lengths(lengths: np.ndarray, target_total: int, min_per_segment: int = 1) -> np.ndarray:
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


def branch_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(polyline_edge_lengths(points)))


def resample_open_branch_preserve_landmarks(
    points: np.ndarray,
    target_count: int,
    corner_angle_threshold_deg: float = 150.0,
) -> np.ndarray:
    if target_count < 2:
        raise ValueError("A branch needs at least two nodes after resampling.")
    if len(points) < 2:
        raise ValueError("Need at least two input nodes to resample a branch.")
    if target_count == len(points):
        # Still resample to even out spacing while preserving landmarks.
        pass

    corners = detect_sharp_corners_open(points, angle_threshold_deg=corner_angle_threshold_deg)
    landmark_idx = np.unique(np.concatenate(([0], corners, [len(points) - 1]))).astype(int)

    segment_polylines = []
    segment_lengths = []
    for i in range(len(landmark_idx) - 1):
        start = landmark_idx[i]
        end = landmark_idx[i + 1]
        poly = points[start : end + 1]
        segment_polylines.append(poly)
        segment_lengths.append(branch_path_length(poly))

    target_intervals = target_count - 1
    intervals_per_segment = allocate_counts_from_lengths(
        np.asarray(segment_lengths, dtype=float),
        target_intervals,
        min_per_segment=1,
    )

    pieces = []
    for i, (poly, n_intervals) in enumerate(zip(segment_polylines, intervals_per_segment)):
        segment_points = resample_open_polyline_uniform(poly, int(n_intervals) + 1)
        if i < len(segment_polylines) - 1:
            pieces.append(segment_points[:-1])
        else:
            pieces.append(segment_points)

    return np.vstack(pieces)


def branch_stats(points: np.ndarray) -> dict[str, float]:
    lengths = polyline_edge_lengths(points)
    if len(lengths) == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(lengths)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "max": float(np.max(lengths)),
    }


def default_output_path(input_path: Path, resample_uniform: bool) -> Path:
    suffix = "_uniform" if resample_uniform else "_cleaned"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path, help="Input MAT file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MAT file. Defaults to <input>_cleaned.mat or <input>_uniform.mat.",
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
        help="Relative tolerance on (Lprev + Lnext - chord) against the median branch edge length.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and report without saving a cleaned MAT file.",
    )
    parser.add_argument(
        "--resample-uniform",
        action="store_true",
        help="After cleanup, redistribute each branch uniformly by arc length.",
    )
    parser.add_argument(
        "--corner-angle-threshold-deg",
        type=float,
        default=150.0,
        help="Angles below this on a branch are treated as sharp corners and preserved during resampling.",
    )
    return parser.parse_args()


def rebuild_geometry(
    processed_branches: list[BranchRecord],
    shared_original_ids: set[int],
) -> Geometry:
    nodes_out: list[np.ndarray] = []
    edges_out: list[tuple[int, int]] = []
    shared_map: dict[int, int] = {}

    def get_shared_node(original_id: int, coord: np.ndarray) -> int:
        if original_id not in shared_map:
            shared_map[original_id] = len(nodes_out)
            nodes_out.append(np.asarray(coord, dtype=float))
        return shared_map[original_id]

    for record in processed_branches:
        branch_global_ids: list[int] = []
        branch_points = record.cleaned_points
        branch_orig = record.cleaned_node_ids

        for idx, coord in enumerate(branch_points):
            is_endpoint = idx == 0 or idx == len(branch_points) - 1
            if is_endpoint and int(branch_orig[idx]) in shared_original_ids:
                gid = get_shared_node(int(branch_orig[idx]), coord)
            else:
                gid = len(nodes_out)
                nodes_out.append(np.asarray(coord, dtype=float))
            branch_global_ids.append(gid)

        for a, b in zip(branch_global_ids[:-1], branch_global_ids[1:]):
            if a != b:
                edges_out.append((a, b))

    # Stable dedupe
    seen = set()
    unique_edges = []
    for a, b in edges_out:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append((a, b))

    return Geometry(nodes=np.asarray(nodes_out, dtype=float), edges=np.asarray(unique_edges, dtype=int))


def degree_histogram(node_count: int, edges: np.ndarray) -> dict[int, int]:
    _, degree = build_adjacency(edges, node_count)
    return {int(k): int(np.sum(degree == k)) for k in sorted(set(degree.tolist()))}


def main() -> None:
    args = parse_args()
    input_path = args.input_mat.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(input_path, args.resample_uniform)
    )

    geom = load_geometry(input_path)
    components, degree = analyze_graph(geom.nodes, geom.edges)
    shared_original_ids = set(np.where(degree != 2)[0].tolist())

    print(f"Input : {input_path}")
    print(f"Nodes : {len(geom.nodes)}")
    print(f"Edges : {len(geom.edges)}")
    print(f"Degree histogram before: {degree_histogram(len(geom.nodes), geom.edges)}")
    print(f"Special nodes (1-based): {[idx + 1 for idx in sorted(shared_original_ids)]}")

    processed: list[BranchRecord] = []
    total_removed: list[int] = []

    for comp in components:
        print()
        print(
            f"Component {comp['component_id'] + 1}: "
            f"{len(comp['branches'])} branches, "
            f"junctions={[n + 1 for n in comp['junction_nodes']]}, "
            f"endpoints={[n + 1 for n in comp['end_nodes']]}"
        )

        for branch_idx, branch in enumerate(comp["branches"], start=1):
            orig_ids = np.asarray(branch, dtype=int)
            points = geom.nodes[orig_ids]
            before = branch_stats(points)

            cleaned_points, cleaned_ids, removed_ids = simplify_open_branch(
                points,
                orig_ids,
                short_ratio=args.short_ratio,
                angle_tol_deg=args.angle_tol_deg,
                excess_rel_tol=args.excess_rel_tol,
            )
            total_removed.extend(removed_ids)
            after_cleanup = branch_stats(cleaned_points)

            final_points = cleaned_points
            if args.resample_uniform:
                final_points = resample_open_branch_preserve_landmarks(
                    cleaned_points,
                    target_count=len(cleaned_points),
                    corner_angle_threshold_deg=args.corner_angle_threshold_deg,
                )
            after_final = branch_stats(final_points)

            print(
                f"  Branch {branch_idx}: "
                f"nodes {len(points)} -> {len(cleaned_points)} -> {len(final_points)} | "
                f"before min/med/max={before['min']:.6f}/{before['median']:.6f}/{before['max']:.6f} | "
                f"final min/med/max={after_final['min']:.6f}/{after_final['median']:.6f}/{after_final['max']:.6f}"
            )
            if removed_ids:
                print(f"    Removed original node indices (1-based): {[idx + 1 for idx in removed_ids]}")

            processed.append(
                BranchRecord(
                    component_id=comp["component_id"],
                    branch_id=branch_idx,
                    original_node_ids=branch,
                    cleaned_points=final_points,
                    cleaned_node_ids=cleaned_ids,
                    removed_original_ids=removed_ids,
                )
            )

    rebuilt = rebuild_geometry(processed, shared_original_ids)
    print()
    print(f"Rebuilt nodes: {len(rebuilt.nodes)}")
    print(f"Rebuilt edges: {len(rebuilt.edges)}")
    print(f"Degree histogram after : {degree_histogram(len(rebuilt.nodes), rebuilt.edges)}")

    if total_removed:
        removed_1based = sorted(idx + 1 for idx in total_removed)
        print(f"All removed original node indices (1-based): {removed_1based}")
    else:
        print("All removed original node indices (1-based): []")

    if args.dry_run:
        print("Dry run: no file written.")
        return

    connectivity = rebuilt.edges + 1  # MATLAB 1-based
    savemat(
        output_path,
        {
            "nodesCoords": rebuilt.nodes,
            "connectivity": connectivity,
        },
    )
    print(f"Saved cleaned geometry to: {output_path}")


if __name__ == "__main__":
    main()
