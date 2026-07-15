"""
Square reflector perforations for axial point-light redirection.

Each square through-cut is oriented so light from a point source on the lamp's
vertical axis (offset below the top) specularly reflects off one face of the
hole and exits horizontally. At the light height, incident and exit directions
coincide and the holes become horizontal square tunnels.
"""

from __future__ import annotations

import numpy as np
import trimesh

from mesh_patterns.radial_cutters import vertical_axis_xy
from mesh_patterns.twinkle_cutters import light_source_position, measure_bore_through_wall


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector / length


def horizontal_exit_direction(
    point: np.ndarray,
    axis_xy: np.ndarray,
) -> np.ndarray:
    """
    Horizontal unit vector from the vertical axis through ``point``.
    """

    delta = np.array(
        [
            float(point[0]) - float(axis_xy[0]),
            float(point[1]) - float(axis_xy[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return delta / length


def incident_direction(
    seed: np.ndarray,
    light: np.ndarray,
) -> np.ndarray:
    """
    Unit travel direction of light from ``light`` to ``seed``.
    """

    return _normalize(np.asarray(seed, dtype=np.float64) - np.asarray(light, dtype=np.float64))


def reflector_frame(
    seed: np.ndarray,
    light: np.ndarray,
    axis_xy: np.ndarray,
    *,
    colinear_epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Build the local frame for a square reflector cutter.

    Returns ``(through_axis, mirror_normal, side_axis, is_horizontal)`` where
    ``through_axis`` is the tunnel direction, ``mirror_normal`` is the reflecting
    face normal (specular bisector of incident and horizontal exit), and
    ``side_axis`` completes a right-handed orthonormal frame.
    """

    incident = incident_direction(seed, light)
    exit_dir = horizontal_exit_direction(seed, axis_xy)
    alignment = float(np.clip(np.dot(incident, exit_dir), -1.0, 1.0))

    if alignment > 1.0 - colinear_epsilon:
        through_axis = exit_dir
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side_axis = _normalize(np.cross(through_axis, world_up))
        if float(np.linalg.norm(side_axis)) < 1e-9:
            side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        mirror_normal = _normalize(np.cross(side_axis, through_axis))
        return through_axis, mirror_normal, side_axis, True

    # Unit I and R imply A = I+R and N = R-I are orthogonal.
    through_axis = _normalize(incident + exit_dir)
    mirror_normal = _normalize(exit_dir - incident)
    side_axis = _normalize(np.cross(through_axis, mirror_normal))
    if float(np.linalg.norm(side_axis)) < 1e-9:
        side_axis = _normalize(
            np.cross(through_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        )
        mirror_normal = _normalize(np.cross(side_axis, through_axis))

    # Re-orthonormalize for numerical safety.
    mirror_normal = _normalize(np.cross(side_axis, through_axis))
    return through_axis, mirror_normal, side_axis, False


def build_square_prism(
    center: np.ndarray,
    through_axis: np.ndarray,
    mirror_normal: np.ndarray,
    side_axis: np.ndarray,
    *,
    hole_size: float,
    length: float,
) -> trimesh.Trimesh:
    """
    Axis-aligned box in a local frame mapped to the reflector orientation.
    """

    half = 0.5 * hole_size
    half_length = 0.5 * length
    corners = np.array(
        [
            [-half, -half, -half_length],
            [half, -half, -half_length],
            [half, half, -half_length],
            [-half, half, -half_length],
            [-half, -half, half_length],
            [half, -half, half_length],
            [half, half, half_length],
            [-half, half, half_length],
        ],
        dtype=np.float64,
    )

    # Local axes: x = side, y = mirror normal, z = through.
    local_to_world = np.column_stack([side_axis, mirror_normal, through_axis])
    vertices = corners @ local_to_world.T + np.asarray(center, dtype=np.float64)

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def build_reflector_cutter(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    light: np.ndarray,
    axis_xy: np.ndarray,
    *,
    hole_size: float,
    outer_margin: float = 0.5,
    exit_margin: float = 0.5,
) -> trimesh.Trimesh | None:
    """
    Build one square through-wall cutter with a specular reflecting face.
    """

    through_axis, mirror_normal, side_axis, _is_horizontal = reflector_frame(
        seed,
        light,
        axis_xy,
    )
    hits = measure_bore_through_wall(mesh, seed, through_axis)
    if hits is None:
        return None

    outer_hit, inner_hit = hits
    start = outer_hit + through_axis * outer_margin
    end = inner_hit - through_axis * exit_margin
    length = float(np.linalg.norm(end - start))
    if length < 1e-3:
        return None

    center = 0.5 * (start + end)
    return build_square_prism(
        center,
        through_axis,
        mirror_normal,
        side_axis,
        hole_size=hole_size,
        length=length,
    )


def build_reflector_cutters(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    *,
    light: np.ndarray | None = None,
    axis_xy: np.ndarray | None = None,
    light_source_offset: float = 30.0,
    hole_size: float = 1.5,
    outer_margin: float = 0.5,
    exit_margin: float = 0.5,
) -> tuple[list[trimesh.Trimesh], dict[str, float | int]]:
    """
    Build square reflector cutters for every surface seed.
    """

    if axis_xy is None:
        axis_xy = vertical_axis_xy(mesh)
    if light is None:
        light = light_source_position(
            mesh,
            axis_xy=axis_xy,
            light_source_offset=light_source_offset,
        )

    cutters: list[trimesh.Trimesh] = []
    failures = 0
    horizontal_count = 0

    for seed in seeds:
        _through_axis, _mirror_normal, _side_axis, is_horizontal = reflector_frame(
            seed,
            light,
            axis_xy,
        )
        if is_horizontal:
            horizontal_count += 1

        cutter = build_reflector_cutter(
            mesh,
            seed,
            light,
            axis_xy,
            hole_size=hole_size,
            outer_margin=outer_margin,
            exit_margin=exit_margin,
        )
        if cutter is None or not cutter.is_volume:
            failures += 1
            continue
        cutters.append(cutter)

    stats: dict[str, float | int] = {
        "cutter_count": len(cutters),
        "failures": failures,
        "pattern_seeds": len(seeds),
        "hole_size": hole_size,
        "light_source_offset": light_source_offset,
        "light_x": float(light[0]),
        "light_y": float(light[1]),
        "light_z": float(light[2]),
        "horizontal_count": horizontal_count,
    }
    return cutters, stats
