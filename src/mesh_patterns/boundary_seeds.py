"""
Synthetic boundary seeds placed on horizontal rings at border heights.

These virtual seeds anchor Voronoi cells at the top and bottom margins without
clipping the drawable surface mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass(slots=True)
class BorderSeedSet:
    """Pattern seeds plus optional synthetic border rings for Voronoi math."""

    all_seeds: np.ndarray
    all_normals: np.ndarray
    pattern_mask: np.ndarray
    boundary_mask: np.ndarray

    @property
    def pattern_seeds(self) -> np.ndarray:
        return self.all_seeds[self.pattern_mask]

    @property
    def pattern_normals(self) -> np.ndarray:
        return self.all_normals[self.pattern_mask]

    @property
    def boundary_seeds(self) -> np.ndarray:
        return self.all_seeds[self.boundary_mask]

    @property
    def pattern_count(self) -> int:
        return int(np.sum(self.pattern_mask))

    @property
    def boundary_count(self) -> int:
        return int(np.sum(self.boundary_mask))


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _edge_plane_crossing(
    point_a: np.ndarray,
    point_b: np.ndarray,
    z_level: float,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray | None:
    z_a = float(point_a[2])
    z_b = float(point_b[2])

    if abs(z_a - z_level) <= tolerance and abs(z_b - z_level) <= tolerance:
        return None

    if (z_a - z_level) * (z_b - z_level) > tolerance:
        return None

    if abs(z_b - z_a) <= tolerance:
        return None

    parameter = (z_level - z_a) / (z_b - z_a)
    if parameter < -tolerance or parameter > 1.0 + tolerance:
        return None

    crossing = point_a + parameter * (point_b - point_a)
    crossing[2] = z_level
    return crossing


def _triangle_plane_segments(
    vertices: np.ndarray,
    z_level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    crossings: list[np.ndarray] = []
    for start, end in ((0, 1), (1, 2), (2, 0)):
        point = _edge_plane_crossing(vertices[start], vertices[end], z_level)
        if point is not None:
            crossings.append(point)

    if len(crossings) != 2:
        return []

    return [(crossings[0], crossings[1])]


def _canonical_points(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    tolerance: float = 0.2,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    points: list[np.ndarray] = []
    tree: cKDTree | None = None

    def point_id(point: np.ndarray) -> int:
        nonlocal tree
        point = np.asarray(point, dtype=np.float64)
        if tree is not None:
            distance, index = tree.query(point)
            if float(distance) <= tolerance:
                return int(index)

        points.append(point)
        tree = cKDTree(np.asarray(points, dtype=np.float64))
        return len(points) - 1

    edges: list[tuple[int, int]] = []
    for start, end in segments:
        left = point_id(start)
        right = point_id(end)
        if left == right:
            continue
        edges.append((left, right))

    return points, edges


def _trace_loops(
    points: list[np.ndarray],
    edges: list[tuple[int, int]],
) -> list[np.ndarray]:
    if not edges:
        return []

    adjacency: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()
    for left, right in edges:
        edge = (min(left, right), max(left, right))
        edge_set.add(edge)
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    point_array = np.asarray(points, dtype=np.float64)
    loops: list[np.ndarray] = []
    used: set[tuple[int, int]] = set()

    for start, end in edges:
        edge = (min(start, end), max(start, end))
        if edge in used:
            continue

        chain = [start, end]
        used.add(edge)
        previous, current = start, end

        while True:
            candidates = [
                candidate
                for candidate in adjacency.get(current, [])
                if candidate != previous
                and (min(current, candidate), max(current, candidate)) in edge_set
                and (min(current, candidate), max(current, candidate)) not in used
            ]
            if len(candidates) != 1:
                break

            previous, current = current, candidates[0]
            used.add((min(previous, current), max(previous, current)))
            chain.append(current)
            if current == start:
                break

        if len(chain) >= 4 and chain[0] == chain[-1]:
            loops.append(point_array[np.asarray(chain, dtype=np.int64)])

    return loops


def _mesh_plane_segments(
    mesh: trimesh.Trimesh,
    z_level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for face in mesh.faces:
        vertices = mesh.vertices[face]
        segments.extend(_triangle_plane_segments(vertices, z_level))
    return segments


def _largest_radius_loop(loops: list[np.ndarray]) -> np.ndarray | None:
    if not loops:
        return None

    return max(
        loops,
        key=lambda loop: float(np.mean(np.linalg.norm(loop[:, :2], axis=1))),
    )


def _resample_closed_loop(
    loop: np.ndarray,
    spacing: float,
) -> np.ndarray:
    loop = np.asarray(loop, dtype=np.float64)
    if len(loop) < 2:
        return loop

    if not np.allclose(loop[0], loop[-1]):
        closed = np.vstack([loop, loop[0]])
    else:
        closed = loop

    edge_vectors = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    total_length = float(np.sum(lengths))
    if total_length <= 1e-9:
        return closed[:-1]

    target_count = max(8, int(np.ceil(total_length / spacing)))
    sample_distances = np.linspace(0.0, total_length, target_count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    points: list[np.ndarray] = []
    segment_index = 0
    for distance in sample_distances:
        while (
            segment_index < len(lengths) - 1
            and cumulative[segment_index + 1] < distance
        ):
            segment_index += 1

        segment_length = lengths[segment_index]
        if segment_length <= 1e-9:
            points.append(closed[segment_index].copy())
            continue

        local = (distance - cumulative[segment_index]) / segment_length
        points.append(
            closed[segment_index]
            + local * (closed[segment_index + 1] - closed[segment_index])
        )

    return np.asarray(points, dtype=np.float64)


def _mesh_has_plane_crossing(mesh: trimesh.Trimesh, z_level: float) -> bool:
    return bool(_mesh_plane_segments(mesh, z_level))


def effective_ring_z(
    mesh: trimesh.Trimesh,
    z_level: float,
    *,
    prefer: str,
) -> float | None:
    """
    Find a usable ring height when the requested world-z misses the surface.
    """

    if _mesh_has_plane_crossing(mesh, z_level):
        return z_level

    z_min = float(mesh.bounds[0][2])
    z_max = float(mesh.bounds[1][2])
    samples = np.linspace(z_min, z_max, 250)

    if prefer == "lowest":
        for z_value in samples:
            if z_value >= z_level - 1e-6 and _mesh_has_plane_crossing(mesh, float(z_value)):
                return float(z_value)
    else:
        for z_value in reversed(samples):
            if z_value <= z_level + 1e-6 and _mesh_has_plane_crossing(mesh, float(z_value)):
                return float(z_value)

    return None


def horizontal_ring_on_mesh(
    mesh: trimesh.Trimesh,
    z_level: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a ring of synthetic seeds where the mesh crosses a horizontal plane.
    """

    segments = _mesh_plane_segments(mesh, z_level)
    points, edges = _canonical_points(segments)
    loops = _trace_loops(points, edges)
    outer_loop = _largest_radius_loop(loops)
    if outer_loop is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    ring_points = _resample_closed_loop(outer_loop, spacing)
    if len(ring_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    snapped, _distances, face_ids = mesh.nearest.on_surface(ring_points)
    normals = mesh.face_normals[face_ids]
    return np.asarray(snapped, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def border_z_levels(
    bounds: np.ndarray,
    *,
    bottom_border: float,
    top_border: float,
) -> tuple[float | None, float | None]:
    bottom_level = float(bounds[0][2]) + bottom_border if bottom_border > 0 else None
    top_level = float(bounds[1][2]) - top_border if top_border > 0 else None
    return bottom_level, top_level


def build_border_seed_rings(
    mesh: trimesh.Trimesh,
    bounds: np.ndarray,
    *,
    bottom_border: float,
    top_border: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic seed rings at the configured border heights.
    """

    bottom_level, top_level = border_z_levels(
        bounds,
        bottom_border=bottom_border,
        top_border=top_border,
    )

    ring_points: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []

    for z_level, prefer in (
        (bottom_level, "lowest"),
        (top_level, "highest"),
    ):
        if z_level is None:
            continue
        effective_z = effective_ring_z(mesh, z_level, prefer=prefer)
        if effective_z is None:
            continue
        points, normals = horizontal_ring_on_mesh(mesh, effective_z, spacing)
        if len(points):
            ring_points.append(points)
            ring_normals.append(normals)

    if not ring_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    return np.vstack(ring_points), np.vstack(ring_normals)


def combine_pattern_and_border_seeds(
    pattern_seeds: np.ndarray,
    pattern_normals: np.ndarray,
    boundary_seeds: np.ndarray,
    boundary_normals: np.ndarray,
) -> BorderSeedSet:
    """
    Merge real and synthetic seeds for Voronoi calculations.
    """

    if len(boundary_seeds) == 0:
        pattern_mask = np.ones(len(pattern_seeds), dtype=bool)
        return BorderSeedSet(
            all_seeds=np.asarray(pattern_seeds, dtype=np.float64),
            all_normals=np.asarray(pattern_normals, dtype=np.float64),
            pattern_mask=pattern_mask,
            boundary_mask=np.zeros(len(pattern_seeds), dtype=bool),
        )

    all_seeds = np.vstack([pattern_seeds, boundary_seeds])
    all_normals = np.vstack([pattern_normals, boundary_normals])
    pattern_mask = np.zeros(len(all_seeds), dtype=bool)
    pattern_mask[: len(pattern_seeds)] = True
    boundary_mask = ~pattern_mask
    return BorderSeedSet(
        all_seeds=all_seeds,
        all_normals=all_normals,
        pattern_mask=pattern_mask,
        boundary_mask=boundary_mask,
    )
