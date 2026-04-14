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


def detect_sharp_corners(points: np.ndarray, angle_threshold_deg: float = 150.0) -> np.ndarray:
    corners: list[int] = []
    for i in range(len(points)):
        prev_i = (i - 1) % len(points)
        next_i = (i + 1) % len(points)
        angle = turning_angle_deg(points[prev_i], points[i], points[next_i])
        if angle < angle_threshold_deg:
            corners.append(i)
    return np.asarray(corners, dtype=int)


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
    sample_s = np.linspace(0.0, total_length, target_count, endpoint=False)
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


def allocate_counts_from_lengths(lengths: np.ndarray, target_count: int) -> np.ndarray:
    if target_count < len(lengths):
        raise ValueError("Target node count must be at least the number of segments.")

    weights = lengths / np.sum(lengths)
    raw = weights * target_count
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    deficit = target_count - int(np.sum(counts))

    if deficit > 0:
        remainders = raw - np.floor(raw)
        order = np.argsort(-remainders)
        for idx in order[:deficit]:
            counts[idx] += 1
    elif deficit < 0:
        removable = counts - 1
        remainders = raw - np.floor(raw)
        order = np.argsort(remainders)
        need = -deficit
        for idx in order:
            take = min(removable[idx], need)
            counts[idx] -= take
            need -= take
            if need == 0:
                break
        if need != 0:
            raise ValueError("Could not allocate per-segment counts safely.")

    return counts


def resample_closed_loop_preserve_corners(
    points: np.ndarray,
    corner_indices: np.ndarray,
    target_count: int,
) -> np.ndarray:
    if len(corner_indices) == 0:
        return resample_closed_loop_uniform(points, target_count)

    ordered_corners = np.sort(corner_indices)
    rotated_points = np.roll(points, -int(ordered_corners[0]), axis=0)
    rotated_corners = (ordered_corners - ordered_corners[0]) % len(points)
    rotated_corners = np.sort(rotated_corners)

    segment_lengths = []
    segment_polylines = []
    m = len(rotated_corners)

    for i in range(m):
        start = int(rotated_corners[i])
        end = int(rotated_corners[(i + 1) % m])
        if end <= start:
            poly = np.vstack([rotated_points[start:], rotated_points[: end + 1]])
        else:
            poly = rotated_points[start : end + 1]

        segment_polylines.append(poly)
        seg_len = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        segment_lengths.append(seg_len)

    counts = allocate_counts_from_lengths(np.asarray(segment_lengths, dtype=float), target_count)
    pieces = [
        resample_open_polyline_uniform(poly, int(count))
        for poly, count in zip(segment_polylines, counts)
    ]
    return np.vstack(pieces)


def resample_closed_loop_uniform(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count < 3:
        raise ValueError("Uniform resampling requires at least 3 nodes.")
    if len(points) < 3:
        raise ValueError("Need at least 3 input nodes to resample a closed loop.")

    rolled = np.roll(points, -1, axis=0)
    seg_vecs = rolled - points
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total_length = float(np.sum(seg_lengths))
    if total_length <= 0.0:
        raise ValueError("Cannot resample a degenerate loop with zero perimeter.")

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    sample_s = np.linspace(0.0, total_length, target_count, endpoint=False)
    resampled = np.zeros((target_count, points.shape[1]), dtype=float)

    for i, s in enumerate(sample_s):
        seg_idx = int(np.searchsorted(cumulative[1:], s, side="right"))
        seg_idx = min(seg_idx, len(points) - 1)
        seg_start = cumulative[seg_idx]
        seg_length = seg_lengths[seg_idx]

        if seg_length <= 0.0:
            resampled[i] = points[seg_idx]
            continue

        alpha = (s - seg_start) / seg_length
        resampled[i] = points[seg_idx] + alpha * seg_vecs[seg_idx]

    return resampled


def edge_stats(points: np.ndarray) -> dict[str, float]:
    lengths = closed_edge_lengths(points)
    return {
        "count": float(len(lengths)),
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
    parser.add_argument(
        "--resample-uniform",
        action="store_true",
        help="After cleanup, redistribute nodes uniformly by arc length around the closed loop.",
    )
    parser.add_argument(
        "--resample-count",
        type=int,
        default=None,
        help="Target node count for uniform resampling. Defaults to the cleaned node count.",
    )
    parser.add_argument(
        "--corner-angle-threshold-deg",
        type=float,
        default=150.0,
        help="If uniform resampling is enabled, angles below this are treated as sharp corners and preserved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_mat.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else default_output_path(input_path, args.resample_uniform)
    )

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
    after_cleanup = edge_stats(cleaned_nodes)

    final_nodes = cleaned_nodes
    detected_corners = np.array([], dtype=int)
    if args.resample_uniform:
        target_count = (
            args.resample_count if args.resample_count is not None else len(cleaned_nodes)
        )
        detected_corners = detect_sharp_corners(
            cleaned_nodes, angle_threshold_deg=args.corner_angle_threshold_deg
        )
        if len(detected_corners) > 0:
            final_nodes = resample_closed_loop_preserve_corners(
                cleaned_nodes,
                detected_corners,
                target_count,
            )
        else:
            final_nodes = resample_closed_loop_uniform(cleaned_nodes, target_count)
    after_final = edge_stats(final_nodes)

    print(f"Input : {input_path}")
    if args.resample_uniform:
        print(
            f"Nodes : {len(ordered_nodes)} -> {len(cleaned_nodes)} -> {len(final_nodes)}"
        )
    else:
        print(f"Nodes : {len(ordered_nodes)} -> {len(cleaned_nodes)}")
    print(
        "Edge lengths before: "
        f"min={before['min']:.6f}, median={before['median']:.6f}, "
        f"mean={before['mean']:.6f}, max={before['max']:.6f}"
    )
    print(
        "Edge lengths clean : "
        f"min={after_cleanup['min']:.6f}, median={after_cleanup['median']:.6f}, "
        f"mean={after_cleanup['mean']:.6f}, max={after_cleanup['max']:.6f}"
    )
    if args.resample_uniform:
        print(
            "Edge lengths final : "
            f"min={after_final['min']:.6f}, median={after_final['median']:.6f}, "
            f"mean={after_final['mean']:.6f}, max={after_final['max']:.6f}"
        )
    else:
        print(
            "Edge lengths final : "
            f"min={after_final['min']:.6f}, median={after_final['median']:.6f}, "
            f"mean={after_final['mean']:.6f}, max={after_final['max']:.6f}"
        )

    if removed:
        removed_1based = [idx + 1 for idx in removed]
        print(f"Removed original node indices ({len(removed_1based)}): {removed_1based}")
    else:
        print("Removed original node indices (0): []")

    if args.resample_uniform:
        detected_1based = [idx + 1 for idx in detected_corners]
        print(f"Detected sharp corners ({len(detected_1based)}): {detected_1based}")

    if args.dry_run:
        print("Dry run: no file written.")
        return

    connectivity = build_sequential_connectivity(len(final_nodes))
    savemat(
        output_path,
        {
            "nodesCoords": final_nodes,
            "connectivity": connectivity,
        },
    )
    print(f"Saved cleaned geometry to: {output_path}")


if __name__ == "__main__":
    main()
