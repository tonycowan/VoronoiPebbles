"""
Vertical border constraints for pattern placement.
"""

from __future__ import annotations

import numpy as np
import trimesh


def face_min_heights(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.vertices[mesh.faces][:, :, 2].min(axis=1)


def face_max_heights(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.vertices[mesh.faces][:, :, 2].max(axis=1)


def submesh_boundary_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Return vertices lying on the open boundary edges of a mesh.
    """

    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(left, right)), int(max(left, right)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_vertices: set[int] = set()
    for (left, right), count in edge_counts.items():
        if count == 1:
            boundary_vertices.add(left)
            boundary_vertices.add(right)

    if not boundary_vertices:
        return np.empty((0, 3), dtype=np.float64)

    return mesh.vertices[np.asarray(sorted(boundary_vertices), dtype=np.int64)]


def tangent_plane_boundary_radius(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float | None:
    """
    Distance from a seed to the nearest open mesh edge in its tangent plane.
    """

    from mesh_patterns.pebble_shapes import _project_to_local, _tangent_basis

    boundary_points = submesh_boundary_points(mesh)
    if len(boundary_points) == 0:
        return None

    tangent_u, tangent_v = _tangent_basis(normal)
    boundary_uv = _project_to_local(boundary_points, origin, tangent_u, tangent_v)
    distances = np.linalg.norm(boundary_uv, axis=1)
    valid = distances[distances > 1e-6]
    return float(np.min(valid)) if len(valid) else None


def seed_reach_margin(min_spacing: float) -> float:
    """
    Conservative estimate of how far a pebble can extend from its seed.
    """

    return min_spacing * 0.55


def filter_seeds_by_borders(
    seeds: np.ndarray,
    face_ids: np.ndarray,
    bounds: np.ndarray,
    *,
    bottom_border: float,
    top_border: float,
    reach_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop seeds that cannot fit fully inside the bordered region.
    """

    if len(seeds) == 0:
        return seeds, face_ids

    keep = np.ones(len(seeds), dtype=bool)
    bottom_z = bounds[0][2]
    top_z = bounds[1][2]

    if bottom_border > 0:
        keep &= seeds[:, 2] >= bottom_z + bottom_border + reach_margin

    if top_border > 0:
        keep &= seeds[:, 2] <= top_z - top_border - reach_margin

    return seeds[keep], face_ids[keep]


def _clip_box(z_min: float | None, z_max: float | None, size: float = 4000.0) -> trimesh.Trimesh:
    if z_min is not None and z_max is not None:
        height = z_max - z_min
        center_z = z_min + height * 0.5
    elif z_min is not None:
        height = size
        center_z = z_min + height * 0.5
    elif z_max is not None:
        height = size
        center_z = z_max - height * 0.5
    else:
        raise ValueError("At least one clip bound is required")

    box = trimesh.creation.box(extents=[size, size, height])
    box.apply_translation([0.0, 0.0, center_z])
    return box


def clip_to_vertical_borders(
    mesh: trimesh.Trimesh,
    bounds: np.ndarray,
    *,
    bottom_border: float = 0.0,
    top_border: float = 0.0,
    engine: str = "manifold",
) -> trimesh.Trimesh | None:
    """
    Keep only geometry inside the allowed vertical band.
    """

    result = mesh
    bottom_z = bounds[0][2]
    top_z = bounds[1][2]

    if bottom_border > 0:
        allowed_min = bottom_z + bottom_border
        if result.bounds[0][2] < allowed_min:
            clip = _clip_box(z_min=allowed_min, z_max=None)
            result = trimesh.boolean.intersection([result, clip], engine=engine)
            if result is None or len(result.faces) == 0:
                return None

    if top_border > 0:
        allowed_max = top_z - top_border
        if result.bounds[1][2] > allowed_max:
            clip = _clip_box(z_min=None, z_max=allowed_max)
            result = trimesh.boolean.intersection([result, clip], engine=engine)
            if result is None or len(result.faces) == 0:
                return None

    if isinstance(result, list):
        result = trimesh.util.concatenate(result)

    return result
