"""
Convert local Voronoi cells into rounded pebble cutter geometry.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import Voronoi, cKDTree
from shapely import affinity
from shapely.geometry import Polygon

from mesh_patterns.boundary_seeds import BorderSeedSet
from mesh_patterns.rounded_pebble import round_polygon_vertices
from procedural_pebbles.polygons import round_polygons, shrink_voronoi_maps
from procedural_pebbles.voronoi import finite_polygons


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    return tangent_u, tangent_v


def _project_to_local(
    points: np.ndarray,
    origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    relative = points - origin
    return np.column_stack(
        [
            relative @ tangent_u,
            relative @ tangent_v,
        ]
    )


def _as_single_polygon(geometry) -> Polygon | None:
    if geometry.is_empty:
        return None

    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda geom: geom.area)

    if geometry.geom_type != "Polygon":
        return None

    return geometry


def local_voronoi_cell(
    seed_index: int,
    seeds_2d: np.ndarray,
    *,
    clip_radius: float,
) -> Polygon | None:
    if len(seeds_2d) < 2:
        return None

    voronoi = Voronoi(seeds_2d)
    regions, vertices = finite_polygons(voronoi, radius=clip_radius * 3)

    region = regions[seed_index]
    if len(region) < 3:
        return None

    polygon = Polygon(vertices[region])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return _as_single_polygon(polygon)


def shrink_voronoi_cell(cell: Polygon, minimum_pebble_distance: float) -> Polygon | None:
    """
    Shrink a Voronoi map by half the minimum distance between pebbles.
    """

    shrunk = shrink_voronoi_maps([cell], minimum_pebble_distance)
    if not shrunk:
        return None

    return _as_single_polygon(shrunk[0])


def shape_pebble_polygon(
    polygon: Polygon,
    *,
    round_radius: float,
) -> Polygon | None:
    if polygon.is_empty or polygon.area <= 0:
        return None

    if round_radius <= 0:
        return _as_single_polygon(polygon)

    rounded = round_polygons([polygon], radius=round_radius)
    if not rounded:
        return None

    return _as_single_polygon(rounded[0])


def build_voronoi_cell_polygon(
    seed_index: int,
    seeds: np.ndarray,
    normal: np.ndarray,
) -> Polygon | None:
    tangent_u, tangent_v = _tangent_basis(normal)
    origin = seeds[seed_index]
    seeds_2d = _project_to_local(seeds, origin, tangent_u, tangent_v)

    tree = cKDTree(seeds)
    distances, _ = tree.query(origin, k=2)
    spacing = float(distances[1])

    return local_voronoi_cell(
        seed_index,
        seeds_2d,
        clip_radius=spacing * 1.5,
    )


def build_shrunk_cell_polygon(
    seed_index: int,
    seeds: np.ndarray,
    normal: np.ndarray,
    *,
    gap: float,
) -> Polygon | None:
    """
    Shrink a Voronoi cell by half the minimum distance between pebbles.
    """

    cell = build_voronoi_cell_polygon(seed_index, seeds, normal)
    if cell is None:
        return None

    return shrink_voronoi_cell(cell, gap)


def polygon_to_surface_loop(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray | None:
    """
    Map a tangent-plane polygon back to a closed 3D loop on the surface.
    """

    tangent_u, tangent_v = _tangent_basis(normal)
    coordinates = np.asarray(polygon.exterior.coords, dtype=np.float64)
    if len(coordinates) < 4:
        return None

    return (
        origin
        + coordinates[:, 0:1] * tangent_u
        + coordinates[:, 1:2] * tangent_v
    )


def shrunk_cell_boundary_loop(
    seed_index: int,
    seeds: np.ndarray,
    normal: np.ndarray,
    *,
    gap: float,
) -> np.ndarray | None:
    """
    Return the inset Voronoi boundary for one seed as a closed 3D loop.
    """

    polygon = build_shrunk_cell_polygon(
        seed_index,
        seeds,
        normal,
        gap=gap,
    )
    if polygon is None:
        return None

    return polygon_to_surface_loop(polygon, seeds[seed_index], normal)


def shrunk_cell_boundary_loops(
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    gap: float,
) -> list[np.ndarray]:
    """
    Build inset Voronoi boundary loops for every seed that survives shrinking.
    """

    loops: list[np.ndarray] = []
    for seed_index, normal in enumerate(normals):
        loop = shrunk_cell_boundary_loop(
            seed_index,
            seeds,
            normal,
            gap=gap,
        )
        if loop is not None:
            loops.append(loop)

    return loops


def build_pebble_polygon(
    seed_index: int,
    seeds: np.ndarray,
    normal: np.ndarray,
    *,
    gap: float,
    round_radius: float,
) -> Polygon | None:
    cell = build_voronoi_cell_polygon(seed_index, seeds, normal)
    if cell is None:
        return None

    shrunk = shrink_voronoi_cell(cell, gap)
    if shrunk is None:
        return None

    return shape_pebble_polygon(
        shrunk,
        round_radius=round_radius,
    )


def build_local_shrunk_cell_polygon(
    seed_index: int,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    partner_indices: np.ndarray | None = None,
) -> Polygon | None:
    """
    Build a shrunk local Voronoi cell polygon in the seed tangent plane.
    """

    from mesh_patterns.local_voronoi import (
        build_local_voronoi_cell_polygon,
        voronoi_partner_indices,
    )

    if partner_indices is None:
        partner_indices = voronoi_partner_indices(
            seed_index,
            seed_set.all_seeds,
            search_radius=search_radius,
        )

    cell = build_local_voronoi_cell_polygon(
        seed_index,
        seed_set.all_seeds,
        seed_set.all_normals,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        partner_indices=partner_indices,
    )
    if cell is None:
        return None

    return shrink_voronoi_cell(cell, gap)


def build_local_pebble_polygon(
    seed_index: int,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    round_radius: float = 0.0,
    partner_indices: np.ndarray | None = None,
) -> Polygon | None:
    """
    Build a cutter polygon from a local Voronoi cell.

    When ``round_radius`` is zero, returns the exact shrunk Voronoi cell.
    """

    polygon = build_local_shrunk_cell_polygon(
        seed_index,
        seed_set,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        gap=gap,
        partner_indices=partner_indices,
    )
    if polygon is None or round_radius <= 0:
        return polygon

    return shape_pebble_polygon(
        polygon,
        round_radius=round_radius,
    )


def build_local_rounded_pebble_polygon(
    seed_index: int,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    rounding_distance: float,
    spline_samples: int = 8,
    rounding_fullness: float = 1.0,
    partner_indices: np.ndarray | None = None,
) -> Polygon | None:
    """
    Build a rounded cutter polygon from a shrunk local Voronoi cell.
    """

    polygon = build_local_shrunk_cell_polygon(
        seed_index,
        seed_set,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        gap=gap,
        partner_indices=partner_indices,
    )
    if polygon is None or rounding_distance <= 0.0:
        return polygon

    return round_polygon_vertices(
        polygon,
        rounding_distance,
        spline_samples=spline_samples,
        rounding_fullness=rounding_fullness,
    )


def _frame_transform(
    origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    drill: np.ndarray,
    *,
    outer_margin: float,
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 0] = tangent_u
    transform[:3, 1] = tangent_v
    transform[:3, 2] = drill
    transform[:3, 3] = origin - drill * outer_margin
    return transform


def extrude_pebble_cutter(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    drill: np.ndarray,
    depth: float,
    *,
    outer_margin: float = 0.05,
) -> trimesh.Trimesh:
    tangent_u, tangent_v = _tangent_basis(normal)
    extruded = trimesh.creation.extrude_polygon(
        polygon,
        height=depth + outer_margin,
    )
    transform = _frame_transform(
        origin,
        tangent_u,
        tangent_v,
        drill,
        outer_margin=outer_margin,
    )
    extruded.apply_transform(transform)
    return extruded


def build_surface_skirt_cutter(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    inward: np.ndarray,
    *,
    skirt_depth: float,
) -> trimesh.Trimesh:
    return extrude_pebble_cutter(
        polygon,
        origin,
        normal,
        inward,
        skirt_depth,
    )


def build_through_pebble_cutter(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    inward: np.ndarray,
    *,
    start_depth: float,
    depth: float,
    scale: float = 0.6,
) -> trimesh.Trimesh | None:
    remaining = depth - start_depth
    if remaining <= 0.2:
        return None

    scaled = _as_single_polygon(
        affinity.scale(polygon, xfact=scale, yfact=scale, origin="centroid")
    )
    if scaled is None or scaled.is_empty:
        return None

    start = origin + inward * max(start_depth - 0.05, 0.0)
    return extrude_pebble_cutter(
        scaled,
        start,
        normal,
        inward,
        remaining,
    )


def build_pebble_cutters(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    gap: float,
    round_radius: float = 1.0,
    skirt_depth: float = 4.0,
    through_scale: float = 0.6,
    bottom_border: float = 0.0,
    top_border: float = 0.0,
) -> list[trimesh.Trimesh]:
    """
    Build rounded pebble cutters for each seed.

    A full-size rounded pebble shapes the visible opening. On thicker walls a
    smaller rounded pebble continues the through hole below that surface layer.
    """

    from .borders import clip_to_vertical_borders
    from .perforate import hole_drill_depth

    cutters: list[trimesh.Trimesh] = []

    for seed_index, (origin, normal) in enumerate(zip(seeds, normals)):
        pebble = build_pebble_polygon(
            seed_index,
            seeds,
            normal,
            gap=gap,
            round_radius=round_radius,
        )
        if pebble is None:
            continue

        inward = -normal / np.linalg.norm(normal)
        wall_depth = hole_drill_depth(mesh, origin, inward)
        surface_depth = min(skirt_depth, wall_depth)

        skirt = build_surface_skirt_cutter(
            pebble,
            origin,
            normal,
            inward,
            skirt_depth=surface_depth,
        )
        skirt = clip_to_vertical_borders(
            skirt,
            mesh.bounds,
            bottom_border=bottom_border,
            top_border=top_border,
        )
        if skirt is not None and len(skirt.faces) > 0 and skirt.is_volume:
            cutters.append(skirt)

        through = build_through_pebble_cutter(
            pebble,
            origin,
            normal,
            inward,
            start_depth=surface_depth,
            depth=wall_depth,
            scale=through_scale,
        )
        if through is None:
            continue

        through = clip_to_vertical_borders(
            through,
            mesh.bounds,
            bottom_border=bottom_border,
            top_border=top_border,
        )
        if through is not None and len(through.faces) > 0 and through.is_volume:
            cutters.append(through)

    return cutters


def build_local_pebble_cutters(
    mesh: trimesh.Trimesh,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    skirt_depth: float = 4.0,
    through_scale: float = 0.6,
    bottom_border: float = 0.0,
    top_border: float = 0.0,
) -> list[trimesh.Trimesh]:
    """
    Build cutters from exact shrunk local Voronoi cells for pattern seeds.
    """

    from .borders import clip_to_vertical_borders
    from .perforate import hole_drill_depth
    from mesh_patterns.local_voronoi import voronoi_partner_indices

    cutters: list[trimesh.Trimesh] = []
    pattern_indices = np.where(seed_set.pattern_mask)[0]

    for seed_index in pattern_indices:
        seed_index = int(seed_index)
        origin = seed_set.all_seeds[seed_index]
        normal = seed_set.all_normals[seed_index]
        partner_indices = voronoi_partner_indices(
            seed_index,
            seed_set.all_seeds,
            search_radius=search_radius,
        )

        opening = build_local_shrunk_cell_polygon(
            seed_index,
            seed_set,
            search_radius=search_radius,
            perpendicular_half_length=perpendicular_half_length,
            gap=gap,
            partner_indices=partner_indices,
        )
        if opening is None:
            continue

        inward = -normal / np.linalg.norm(normal)
        wall_depth = hole_drill_depth(mesh, origin, inward)
        surface_depth = min(skirt_depth, wall_depth)

        skirt = build_surface_skirt_cutter(
            opening,
            origin,
            normal,
            inward,
            skirt_depth=surface_depth,
        )
        skirt = clip_to_vertical_borders(
            skirt,
            mesh.bounds,
            bottom_border=bottom_border,
            top_border=top_border,
        )
        if skirt is not None and len(skirt.faces) > 0 and skirt.is_volume:
            cutters.append(skirt)

        through = build_through_pebble_cutter(
            opening,
            origin,
            normal,
            inward,
            start_depth=surface_depth,
            depth=wall_depth,
            scale=through_scale,
        )
        if through is None:
            continue

        through = clip_to_vertical_borders(
            through,
            mesh.bounds,
            bottom_border=bottom_border,
            top_border=top_border,
        )
        if through is not None and len(through.faces) > 0 and through.is_volume:
            cutters.append(through)

    return cutters
