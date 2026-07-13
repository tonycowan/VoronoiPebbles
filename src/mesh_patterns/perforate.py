"""
Boolean perforation of mesh surfaces.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def hole_drill_depth(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    inward: np.ndarray,
    *,
    fallback: float = 25.0,
    margin: float = 1.0,
) -> float:
    """
    Measure how far to drill inward from a surface seed point.
    """

    origin = seed + inward * 0.05
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[inward],
    )

    if len(locations) == 0:
        return fallback

    return float(np.linalg.norm(locations - origin, axis=1).max() + margin)


def build_hole_cutter(
    seed: np.ndarray,
    inward: np.ndarray,
    radius: float,
    depth: float,
    *,
    sections: int = 12,
    outer_margin: float = 0.5,
) -> trimesh.Trimesh:
    """
    Build a cylindrical cutter that passes through the wall.
    """

    start = seed - inward * outer_margin
    end = seed + inward * depth
    return trimesh.creation.cylinder(
        radius=radius,
        segment=(start, end),
        sections=sections,
    )


def build_hole_cutters(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    *,
    sections: int = 12,
) -> list[trimesh.Trimesh]:
    """
    Build cylindrical cutters for each perforation.
    """

    inward = _normalize(-normals)
    cutters: list[trimesh.Trimesh] = []

    for seed, normal, radius in zip(seeds, inward, radii):
        depth = hole_drill_depth(mesh, seed, normal)
        cutters.append(
            build_hole_cutter(
                seed,
                normal,
                float(radius),
                depth,
                sections=sections,
            )
        )

    return cutters


def union_cutters(
    cutters: list[trimesh.Trimesh],
    *,
    batch_size: int = 40,
    engine: str = "manifold",
) -> trimesh.Trimesh:
    """
    Union many cutters into one mesh for a single boolean subtraction.
    """

    if not cutters:
        raise ValueError("At least one cutter is required")

    if len(cutters) == 1:
        return cutters[0]

    unioned = cutters[0]

    for index in range(1, len(cutters), batch_size):
        chunk = trimesh.util.concatenate(cutters[index : index + batch_size])
        unioned = trimesh.boolean.union([unioned, chunk], engine=engine)

    return unioned


def perforate_mesh(
    mesh: trimesh.Trimesh,
    cutters: list[trimesh.Trimesh],
    *,
    batch_size: int = 40,
    engine: str = "manifold",
) -> trimesh.Trimesh:
    """
    Subtract hole cutters from a mesh.
    """

    cutter_mesh = union_cutters(cutters, batch_size=batch_size, engine=engine)
    result = trimesh.boolean.difference([mesh, cutter_mesh], engine=engine)

    if isinstance(result, list):
        return trimesh.util.concatenate(result)

    return result
