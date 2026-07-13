"""
Poisson-disk sampling on mesh surfaces.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        return vector
    return vector / length


def tangent_plane_distance(
    origin: np.ndarray,
    other: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Distance between two points measured in a tangent plane at origin.
    """

    relative = other - origin
    unit_normal = _normalize(normal)
    projected = relative - np.dot(relative, unit_normal) * unit_normal
    return float(np.linalg.norm(projected))



def _spacing_satisfied(
    point: np.ndarray,
    normal: np.ndarray,
    accepted_points: np.ndarray,
    accepted_normals: np.ndarray,
    *,
    min_spacing: float,
    spacing_metric: str,
    search_tree: cKDTree | None,
    search_radius: float,
) -> bool:
    if len(accepted_points) == 0:
        return True

    if spacing_metric == "surface":
        for accepted, accepted_normal in zip(accepted_points, accepted_normals):
            if tangent_plane_distance(point, accepted, normal) < min_spacing:
                return False
            if tangent_plane_distance(accepted, point, accepted_normal) < min_spacing:
                return False
        return True

    if search_tree is None:
        neighbor_indices = np.arange(len(accepted_points))
    else:
        neighbor_indices = search_tree.query_ball_point(point, search_radius)

    for neighbor_index in neighbor_indices:
        accepted = accepted_points[neighbor_index]
        distance = float(np.linalg.norm(point - accepted))
        if distance < min_spacing:
            return False

    return True


def poisson_disk_on_surface(
    mesh: trimesh.Trimesh,
    min_spacing: float,
    *,
    oversample_factor: int = 30,
    seed: int | None = None,
    spacing_metric: str = "surface",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scatter points on a mesh with approximately uniform spacing.

    By default spacing is measured in each seed's tangent plane, which matches
    the local Voronoi construction and avoids over-packing on narrowing curved
    regions where 3D Euclidean distance stays large but surface distance
    shrinks.

    Returns
        points
        face_indices for the triangle each point was sampled on
    """

    if min_spacing <= 0:
        raise ValueError("min_spacing must be positive")

    if spacing_metric not in {"surface", "euclidean"}:
        raise ValueError("spacing_metric must be 'surface' or 'euclidean'")

    area = mesh.area
    target_count = max(4, int(area / (min_spacing**2)))
    candidate_count = target_count * oversample_factor

    rng = np.random.default_rng(seed)
    candidates, face_indices = trimesh.sample.sample_surface(mesh, candidate_count)
    candidate_normals = mesh.face_normals[face_indices]
    order = rng.permutation(len(candidates))

    accepted_points = np.empty((0, 3), dtype=np.float64)
    accepted_faces: list[int] = []
    accepted_normals = np.empty((0, 3), dtype=np.float64)
    tree: cKDTree | None = None
    search_radius = min_spacing if spacing_metric == "euclidean" else min_spacing * 3.0

    for index in order:
        point = candidates[index]
        face_id = int(face_indices[index])
        normal = candidate_normals[index]

        if _spacing_satisfied(
            point,
            normal,
            accepted_points,
            accepted_normals,
            min_spacing=min_spacing,
            spacing_metric=spacing_metric,
            search_tree=tree if spacing_metric == "euclidean" else None,
            search_radius=search_radius,
        ):
            accepted_points = np.vstack([accepted_points, point])
            accepted_faces.append(face_id)
            accepted_normals = np.vstack([accepted_normals, normal])
            tree = cKDTree(accepted_points)

    return (
        np.asarray(accepted_points, dtype=np.float64),
        np.asarray(accepted_faces, dtype=np.int64),
    )
