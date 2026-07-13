"""
Local Voronoi cells from perpendicular bisectors on a tangent plane.

For each seed, gather nearby partners inside a search sphere, draw
seed-to-partner chords and perpendicular bisectors on the local surface
tangent plane, then clip bisectors at their intersections to form a closed
cell boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import LineString, Polygon

from mesh_patterns.boundary_seeds import BorderSeedSet
from mesh_patterns.pebble_shapes import (
    _project_to_local,
    _tangent_basis,
    local_voronoi_cell,
    polygon_to_surface_loop,
    shrink_voronoi_cell,
)


@dataclass(slots=True)
class BisectorSegment:
    """One perpendicular bisector between a seed and a partner."""

    partner_index: int
    midpoint_uv: np.ndarray
    midpoint_3d: np.ndarray
    chord_3d: np.ndarray
    full_perpendicular_3d: np.ndarray
    clipped_perpendicular_3d: np.ndarray | None = None


@dataclass(slots=True)
class LocalVoronoiCell:
    """Voronoi cell for one seed built from local partner bisectors."""

    seed_index: int
    partner_indices: np.ndarray
    bisectors: list[BisectorSegment] = field(default_factory=list)
    boundary_loop_3d: np.ndarray | None = None
    search_radius: float = 0.0


def characteristic_seed_spacing(
    seeds: np.ndarray,
    *,
    min_spacing: float,
) -> float:
    """
    Typical nearest-neighbor spacing, ignoring outlier gaps.
    """

    if len(seeds) < 2:
        return min_spacing

    _, neighbor_distances = cKDTree(seeds).query(seeds, k=2)
    nearest = neighbor_distances[:, 1]
    valid = nearest[nearest <= min_spacing * 2.5]
    if len(valid) == 0:
        return min_spacing

    return float(np.percentile(valid, 90))


def max_seed_neighbor_distance(seeds: np.ndarray) -> float:
    """
    Largest nearest-neighbor distance in the seed set.
    """

    if len(seeds) < 2:
        return 0.0

    _, neighbor_distances = cKDTree(seeds).query(seeds, k=2)
    return float(neighbor_distances[:, 1].max())


def voronoi_partner_indices(
    seed_index: int,
    seeds: np.ndarray,
    *,
    search_radius: float,
) -> np.ndarray:
    """
    Return seed indices inside the search sphere around one seed.
    """

    distances, indices = cKDTree(seeds).query(
        seeds[seed_index],
        k=len(seeds),
        distance_upper_bound=search_radius,
    )
    mask = np.isfinite(distances) & (indices != seed_index)
    return np.asarray(indices[mask], dtype=np.int64)


def _snap_to_surface(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    locations, _distances, _face_ids = mesh.nearest.on_surface(
        np.asarray(points, dtype=np.float64)
    )
    return np.asarray(locations, dtype=np.float64)


def _uv_to_surface(
    uv_points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    tangent_u, tangent_v = _tangent_basis(normal)
    lifted = (
        origin
        + uv_points[:, 0:1] * tangent_u
        + uv_points[:, 1:2] * tangent_v
    )
    return _snap_to_surface(mesh, lifted)


def _surface_polyline(
    mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 12,
) -> np.ndarray:
    points = start + np.linspace(0.0, 1.0, samples)[:, None] * (end - start)
    return _snap_to_surface(mesh, points)


def _bisector_line_uv(
    partner_uv: np.ndarray,
    *,
    half_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return midpoint, unit perpendicular direction, and full segment in UV.
    """

    midpoint = 0.5 * partner_uv
    chord = partner_uv
    length = np.linalg.norm(chord)
    if length <= 1e-9:
        raise ValueError("Seed and partner are coincident in the tangent plane.")

    perpendicular = np.array([-chord[1], chord[0]], dtype=np.float64) / length
    segment = np.vstack(
        [
            midpoint - half_length * perpendicular,
            midpoint + half_length * perpendicular,
        ]
    )
    return midpoint, perpendicular, segment


def _clip_bisector_to_polygon(
    bisector_segment_uv: np.ndarray,
    cell_polygon: Polygon,
) -> np.ndarray | None:
    """
    Keep the portion of a bisector segment that lies on the cell boundary.
    """

    line = LineString(bisector_segment_uv)
    boundary = cell_polygon.boundary
    intersection = line.intersection(boundary)
    if intersection.is_empty:
        return None

    if intersection.geom_type == "LineString":
        coords = np.asarray(intersection.coords, dtype=np.float64)
        if len(coords) < 2:
            return None
        return coords

    if intersection.geom_type == "MultiLineString":
        longest = max(intersection.geoms, key=lambda geom: geom.length)
        return np.asarray(longest.coords, dtype=np.float64)

    if intersection.geom_type == "Point":
        return None

    return None


def _finite_voronoi_polygon(
    seed_index: int,
    partner_indices: np.ndarray,
    seeds: np.ndarray,
    normal: np.ndarray,
    *,
    clip_radius: float,
) -> Polygon | None:
    """
    Build the Voronoi cell polygon in the seed tangent plane.
    """

    tangent_u, tangent_v = _tangent_basis(normal)
    origin = seeds[seed_index]
    local_indices = np.concatenate(
        [[seed_index], np.asarray(partner_indices, dtype=np.int64)]
    )
    seeds_2d = _project_to_local(seeds[local_indices], origin, tangent_u, tangent_v)
    return local_voronoi_cell(
        0,
        seeds_2d,
        clip_radius=clip_radius,
    )


def partner_search_radius(
    seeds: np.ndarray,
    *,
    min_spacing: float,
    margin: float,
) -> float:
    """
    Search radius for local Voronoi partner selection.
    """

    return characteristic_seed_spacing(seeds, min_spacing=min_spacing) + margin


def _active_partner_count(
    seed_index: int,
    partner_indices: np.ndarray,
    seeds: np.ndarray,
    normal: np.ndarray,
    *,
    clip_radius: float,
) -> int:
    """
    Partners that share a finite Voronoi ridge with the seed (cell neighbors).
    """

    tangent_u, tangent_v = _tangent_basis(normal)
    origin = seeds[seed_index]
    local_indices = np.concatenate(
        [[seed_index], np.asarray(partner_indices, dtype=np.int64)]
    )
    seeds_2d = _project_to_local(seeds[local_indices], origin, tangent_u, tangent_v)
    if len(seeds_2d) < 2:
        return 0

    voronoi = Voronoi(seeds_2d)
    active_local: set[int] = set()
    for ridge_i, (p1, p2) in enumerate(voronoi.ridge_points):
        if -1 in voronoi.ridge_vertices[ridge_i]:
            continue
        if p1 == 0:
            active_local.add(int(p2))
        elif p2 == 0:
            active_local.add(int(p1))
    return len(active_local)


def build_local_voronoi_cell_polygon(
    seed_index: int,
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    partner_indices: np.ndarray | None = None,
) -> Polygon | None:
    """
    Build the tangent-plane Voronoi cell polygon for one seed.
    """

    if partner_indices is None:
        partner_indices = voronoi_partner_indices(
            seed_index,
            seeds,
            search_radius=search_radius,
        )
    if len(partner_indices) == 0:
        return None

    normal = normals[seed_index]
    clip_radius = max(perpendicular_half_length, search_radius)
    return _finite_voronoi_polygon(
        seed_index,
        partner_indices,
        seeds,
        normal,
        clip_radius=clip_radius,
    )


def build_local_voronoi_boundary_loop(
    seed_index: int,
    seeds: np.ndarray,
    normals: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    partner_indices: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """
    Build a closed local Voronoi boundary loop for one seed.

    Returns the loop and the number of partners whose bisectors form cell edges.
    """

    if partner_indices is None:
        partner_indices = voronoi_partner_indices(
            seed_index,
            seeds,
            search_radius=search_radius,
        )
    if len(partner_indices) == 0:
        return None, 0

    origin = seeds[seed_index]
    normal = normals[seed_index]
    clip_radius = max(perpendicular_half_length, search_radius)
    cell_polygon = build_local_voronoi_cell_polygon(
        seed_index,
        seeds,
        normals,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        partner_indices=partner_indices,
    )
    if cell_polygon is None:
        return None, 0

    active_partners = _active_partner_count(
        seed_index,
        partner_indices,
        seeds,
        normal,
        clip_radius=clip_radius,
    )

    loop = polygon_to_surface_loop(cell_polygon, origin, normal)
    if loop is None:
        return None, active_partners

    return _snap_to_surface(mesh, loop), active_partners


def build_local_shrunk_boundary_loop(
    seed_index: int,
    seeds: np.ndarray,
    normals: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    partner_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Shrink a local Voronoi cell inward by half the minimum pebble gap.
    """

    cell_polygon = build_local_voronoi_cell_polygon(
        seed_index,
        seeds,
        normals,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        partner_indices=partner_indices,
    )
    if cell_polygon is None:
        return None

    shrunk_polygon = shrink_voronoi_cell(cell_polygon, gap)
    if shrunk_polygon is None:
        return None

    origin = seeds[seed_index]
    normal = normals[seed_index]
    loop = polygon_to_surface_loop(shrunk_polygon, origin, normal)
    if loop is None:
        return None

    return _snap_to_surface(mesh, loop)


def local_shrunk_boundary_loops_for_pattern(
    seed_set: BorderSeedSet,
    mesh: trimesh.Trimesh,
    *,
    min_spacing: float,
    margin: float,
    perpendicular_half_length: float,
    gap: float,
    search_radius: float | None = None,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """
    Build inset local Voronoi boundaries for pattern seeds.
    """

    if search_radius is None:
        search_radius = partner_search_radius(
            seed_set.all_seeds,
            min_spacing=min_spacing,
            margin=margin,
        )

    loops: list[np.ndarray] = []
    pattern_indices = np.where(seed_set.pattern_mask)[0]

    for seed_index in pattern_indices:
        partner_indices = voronoi_partner_indices(
            int(seed_index),
            seed_set.all_seeds,
            search_radius=search_radius,
        )
        loop = build_local_shrunk_boundary_loop(
            int(seed_index),
            seed_set.all_seeds,
            seed_set.all_normals,
            mesh,
            search_radius=search_radius,
            perpendicular_half_length=perpendicular_half_length,
            gap=gap,
            partner_indices=partner_indices,
        )
        if loop is not None:
            loops.append(loop)

    stats: dict[str, float | int] = {
        "search_radius": search_radius,
        "inset_per_side": gap * 0.5,
        "boundary_count": len(loops),
        "shrink_failures": len(pattern_indices) - len(loops),
    }
    return loops, stats


def local_voronoi_boundary_loops(
    seeds: np.ndarray,
    normals: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    min_spacing: float,
    margin: float,
    perpendicular_half_length: float,
    search_radius: float | None = None,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """
    Build closed local Voronoi boundaries for every seed.
    """

    if search_radius is None:
        search_radius = partner_search_radius(
            seeds,
            min_spacing=min_spacing,
            margin=margin,
        )

    loops: list[np.ndarray] = []
    partner_counts: list[int] = []
    active_partner_counts: list[int] = []

    for seed_index in range(len(seeds)):
        partner_indices = voronoi_partner_indices(
            seed_index,
            seeds,
            search_radius=search_radius,
        )
        partner_counts.append(len(partner_indices))
        loop, active_partners = build_local_voronoi_boundary_loop(
            seed_index,
            seeds,
            normals,
            mesh,
            search_radius=search_radius,
            perpendicular_half_length=perpendicular_half_length,
            partner_indices=partner_indices,
        )
        active_partner_counts.append(active_partners)
        if loop is not None:
            loops.append(loop)

    stats: dict[str, float | int] = {
        "search_radius": search_radius,
        "boundary_count": len(loops),
        "missing_boundaries": len(seeds) - len(loops),
        "avg_partners": int(np.mean(partner_counts)) if partner_counts else 0,
        "max_partners": int(np.max(partner_counts)) if partner_counts else 0,
        "avg_active_partners": (
            int(np.mean(active_partner_counts)) if active_partner_counts else 0
        ),
        "max_active_partners": (
            int(np.max(active_partner_counts)) if active_partner_counts else 0
        ),
    }
    return loops, stats


def local_voronoi_boundary_loops_for_pattern(
    seed_set: BorderSeedSet,
    mesh: trimesh.Trimesh,
    *,
    min_spacing: float,
    margin: float,
    perpendicular_half_length: float,
    search_radius: float | None = None,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """
    Build Voronoi loops for pattern seeds using synthetic border partners.
    """

    if search_radius is None:
        search_radius = partner_search_radius(
            seed_set.all_seeds,
            min_spacing=min_spacing,
            margin=margin,
        )

    loops: list[np.ndarray] = []
    partner_counts: list[int] = []
    active_partner_counts: list[int] = []
    pattern_indices = np.where(seed_set.pattern_mask)[0]

    for seed_index in pattern_indices:
        partner_indices = voronoi_partner_indices(
            int(seed_index),
            seed_set.all_seeds,
            search_radius=search_radius,
        )
        partner_counts.append(len(partner_indices))
        loop, active_partners = build_local_voronoi_boundary_loop(
            int(seed_index),
            seed_set.all_seeds,
            seed_set.all_normals,
            mesh,
            search_radius=search_radius,
            perpendicular_half_length=perpendicular_half_length,
            partner_indices=partner_indices,
        )
        active_partner_counts.append(active_partners)
        if loop is not None:
            loops.append(loop)

    stats: dict[str, float | int] = {
        "search_radius": search_radius,
        "boundary_count": len(loops),
        "missing_boundaries": len(pattern_indices) - len(loops),
        "avg_partners": int(np.mean(partner_counts)) if partner_counts else 0,
        "max_partners": int(np.max(partner_counts)) if partner_counts else 0,
        "avg_active_partners": (
            int(np.mean(active_partner_counts)) if active_partner_counts else 0
        ),
        "max_active_partners": (
            int(np.max(active_partner_counts)) if active_partner_counts else 0
        ),
    }
    return loops, stats


def build_local_voronoi_cell(
    seed_index: int,
    seeds: np.ndarray,
    normals: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    chord_samples: int = 12,
) -> LocalVoronoiCell:
    """
    Build one local Voronoi cell from sphere-selected partners and bisectors.
    """

    partner_indices = voronoi_partner_indices(
        seed_index,
        seeds,
        search_radius=search_radius,
    )
    cell = LocalVoronoiCell(
        seed_index=seed_index,
        partner_indices=partner_indices,
        search_radius=search_radius,
    )

    if len(partner_indices) == 0:
        return cell

    origin = seeds[seed_index]
    normal = normals[seed_index]
    tangent_u, tangent_v = _tangent_basis(normal)
    partners_uv = _project_to_local(
        seeds[partner_indices],
        origin,
        tangent_u,
        tangent_v,
    )

    clip_radius = max(perpendicular_half_length, search_radius)
    cell_polygon = _finite_voronoi_polygon(
        seed_index,
        partner_indices,
        seeds,
        normal,
        clip_radius=clip_radius,
    )
    if cell_polygon is None:
        return cell

    for partner_index, partner_uv in zip(partner_indices, partners_uv):
        midpoint_uv, _perpendicular_uv, full_segment_uv = _bisector_line_uv(
            partner_uv,
            half_length=perpendicular_half_length,
        )
        midpoint_3d = _uv_to_surface(midpoint_uv[None], origin, normal, mesh)[0]
        partner_point = seeds[int(partner_index)]
        chord_3d = _surface_polyline(
            mesh,
            origin,
            partner_point,
            samples=chord_samples,
        )
        full_perpendicular_3d = _uv_to_surface(
            full_segment_uv,
            origin,
            normal,
            mesh,
        )

        clipped_uv = _clip_bisector_to_polygon(full_segment_uv, cell_polygon)
        clipped_3d = None
        if clipped_uv is not None and len(clipped_uv) >= 2:
            clipped_3d = _uv_to_surface(clipped_uv, origin, normal, mesh)

        cell.bisectors.append(
            BisectorSegment(
                partner_index=int(partner_index),
                midpoint_uv=midpoint_uv,
                midpoint_3d=midpoint_3d,
                chord_3d=chord_3d,
                full_perpendicular_3d=full_perpendicular_3d,
                clipped_perpendicular_3d=clipped_3d,
            )
        )

    cell.boundary_loop_3d = polygon_to_surface_loop(
        cell_polygon,
        origin,
        normal,
    )
    if cell.boundary_loop_3d is not None:
        cell.boundary_loop_3d = _snap_to_surface(mesh, cell.boundary_loop_3d)

    return cell


def build_local_voronoi_cells(
    seed_indices: np.ndarray,
    seeds: np.ndarray,
    normals: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    margin: float,
    perpendicular_half_length: float,
    min_spacing: float,
    search_radius: float | None = None,
) -> list[LocalVoronoiCell]:
    """
    Build local Voronoi cells for a subset of seeds.
    """

    if search_radius is None:
        search_radius = characteristic_seed_spacing(seeds, min_spacing=min_spacing) + margin

    return [
        build_local_voronoi_cell(
            int(seed_index),
            seeds,
            normals,
            mesh,
            search_radius=search_radius,
            perpendicular_half_length=perpendicular_half_length,
        )
        for seed_index in seed_indices
    ]


def select_cluster_seed_indices(
    seeds: np.ndarray,
    *,
    count: int,
    center_index: int | None = None,
) -> np.ndarray:
    """
    Pick a compact cluster of seeds for small debug exports.
    """

    if len(seeds) <= count:
        return np.arange(len(seeds), dtype=np.int64)

    if center_index is None:
        center_index = int(len(seeds) // 2)

    _, indices = cKDTree(seeds).query(seeds[center_index], k=count)
    return np.asarray(indices, dtype=np.int64)


def partner_sphere_mesh(
    center: np.ndarray,
    radius: float,
    *,
    segments: int = 24,
) -> trimesh.Trimesh:
    """
    Wireframe sphere showing the partner search radius.
    """

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(center)
    return sphere
