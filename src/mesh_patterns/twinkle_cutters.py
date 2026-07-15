"""
Tiny cylindrical perforations aimed at a shared axial light source.

Each seed on the outer shell gets a through-wall cylinder whose axis is the line
from the light source through that seed. Random seed placement produces random
angles of incidence against the surface; as an observer walks around the lamp,
different bores line up with the eye and the light for a twinkling effect.
"""

from __future__ import annotations

import numpy as np
import trimesh

from mesh_patterns.radial_cutters import vertical_axis_xy


def light_source_position(
    mesh: trimesh.Trimesh,
    *,
    axis_xy: np.ndarray | None = None,
    light_source_offset: float = 30.0,
) -> np.ndarray:
    """
    Point light on the lamp's vertical axis, ``light_source_offset`` mm below the top.
    """

    if axis_xy is None:
        axis_xy = vertical_axis_xy(mesh)

    top_z = float(np.asarray(mesh.bounds, dtype=np.float64)[1, 2])
    return np.array(
        [float(axis_xy[0]), float(axis_xy[1]), top_z - light_source_offset],
        dtype=np.float64,
    )


def bore_direction_from_light(
    seed: np.ndarray,
    light: np.ndarray,
) -> np.ndarray:
    """
    Unit direction from the light through ``seed`` (outward through the shell).
    """

    delta = np.asarray(seed, dtype=np.float64) - np.asarray(light, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return delta / length


def measure_bore_through_wall(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    outward: np.ndarray,
    *,
    ray_start_distance: float = 8.0,
    fallback_thickness: float = 4.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Find outer and inner shell hits along the bore axis through ``seed``.

    Returns ``(outer_hit, inner_hit)`` with both points on the bore line, or
    ``None`` when the wall cannot be resolved.
    """

    origin = seed + outward * ray_start_distance
    locations, _index_ray, _index_tri = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[-outward],
    )
    if len(locations) < 2:
        if len(locations) == 1:
            outer = locations[0]
            inner = outer - outward * fallback_thickness
            return outer, inner
        return None

    distances = np.linalg.norm(locations - origin, axis=1)
    order = np.argsort(distances)
    hits = locations[order]
    return hits[0], hits[1]


def build_twinkle_cutter(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    light: np.ndarray,
    *,
    radius: float,
    outer_margin: float = 0.5,
    exit_margin: float = 0.5,
    sections: int = 16,
) -> trimesh.Trimesh | None:
    """
    Build one through-wall cylinder aimed at ``light``.
    """

    outward = bore_direction_from_light(seed, light)
    hits = measure_bore_through_wall(mesh, seed, outward)
    if hits is None:
        return None

    outer_hit, inner_hit = hits
    start = outer_hit + outward * outer_margin
    end = inner_hit - outward * exit_margin

    segment_length = float(np.linalg.norm(end - start))
    if segment_length < 1e-3:
        return None

    return trimesh.creation.cylinder(
        radius=radius,
        segment=(start, end),
        sections=sections,
    )


def build_twinkle_cutters(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    light: np.ndarray | None = None,
    axis_xy: np.ndarray | None = None,
    light_source_offset: float = 30.0,
    hole_radius: float = 0.8,
    outer_margin: float = 0.5,
    exit_margin: float = 0.5,
    sections: int = 16,
) -> tuple[list[trimesh.Trimesh], dict[str, float | int]]:
    """
    Build light-aimed cylindrical cutters for every surface seed.
    """

    if light is None:
        light = light_source_position(
            mesh,
            axis_xy=axis_xy,
            light_source_offset=light_source_offset,
        )

    cutters: list[trimesh.Trimesh] = []
    failures = 0
    incidence_angles: list[float] = []

    for index, seed in enumerate(seeds):
        cutter = build_twinkle_cutter(
            mesh,
            seed,
            light,
            radius=hole_radius,
            outer_margin=outer_margin,
            exit_margin=exit_margin,
            sections=sections,
        )
        if cutter is None:
            failures += 1
            continue
        cutters.append(cutter)
        if normals is not None:
            incidence_angles.append(
                incidence_angle_degrees(seed, light, normals[index])
            )

    stats: dict[str, float | int] = {
        "cutter_count": len(cutters),
        "failures": failures,
        "pattern_seeds": len(seeds),
        "hole_radius": hole_radius,
        "light_source_offset": light_source_offset,
        "light_x": float(light[0]),
        "light_y": float(light[1]),
        "light_z": float(light[2]),
        "mean_incidence_deg": (
            float(np.mean(incidence_angles)) if incidence_angles else 0.0
        ),
    }

    return cutters, stats


def incidence_angle_degrees(
    seed: np.ndarray,
    light: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Angle between the bore axis and the outward surface normal, in degrees.
    """

    outward = bore_direction_from_light(seed, light)
    unit_normal = np.asarray(normal, dtype=np.float64)
    unit_normal = unit_normal / max(float(np.linalg.norm(unit_normal)), 1e-9)
    cosine = float(np.clip(np.dot(outward, unit_normal), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
