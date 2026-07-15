"""
Build perforation cutters by sweeping shrunk Voronoi loops toward the vertical axis.

Each boundary point on the outer surface is connected to the inner wall by a ray cast
toward the lamp's central vertical axis, forming a ruled through-wall volume.
"""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from mesh_patterns.boundary_seeds import BorderSeedSet
from mesh_patterns.local_voronoi import (
    build_local_rounded_shrunk_boundary_loop,
    build_local_shrunk_boundary_loop,
    voronoi_partner_indices,
)
from mesh_patterns.pebble_shapes import (
    _tangent_basis,
    build_local_rounded_pebble_polygon,
    build_local_shrunk_cell_polygon,
)


def vertical_axis_xy(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    XY center of the vertical axis used for radial ray casting.
    """

    centroid = np.asarray(mesh.centroid, dtype=np.float64)
    if np.linalg.norm(centroid[:2]) < 1.0:
        return centroid[:2]

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return 0.5 * (bounds[0, :2] + bounds[1, :2])


def inward_direction_toward_axis(
    point: np.ndarray,
    axis_xy: np.ndarray,
) -> np.ndarray:
    """
    Horizontal unit vector from ``point`` toward the vertical axis.
    """

    delta = np.array(
        [axis_xy[0] - point[0], axis_xy[1] - point[1], 0.0],
        dtype=np.float64,
    )
    length = np.linalg.norm(delta)
    if length < 1e-6:
        xy = point[:2]
        xy_length = np.linalg.norm(xy)
        if xy_length < 1e-6:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([-xy[0] / xy_length, -xy[1] / xy_length, 0.0], dtype=np.float64)

    return delta / length


def outward_direction_from_axis(
    point: np.ndarray,
    axis_xy: np.ndarray,
) -> np.ndarray:
    """
    Horizontal unit vector from the axis toward ``point``.
    """

    return -inward_direction_toward_axis(point, axis_xy)


def project_to_exterior_surface(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    axis_xy: np.ndarray,
    *,
    outer_margin: float = 0.0,
    ray_start_distance: float = 10.0,
) -> np.ndarray:
    """
    Re-seat a point on the exterior shell by casting from outside toward the axis.
    """

    outward = outward_direction_from_axis(point, axis_xy)
    origin = point + outward * ray_start_distance
    locations, _index_ray, _index_tri = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[-outward],
    )
    if len(locations) == 0:
        snapped, _distances, _face_ids = mesh.nearest.on_surface(
            [np.asarray(point, dtype=np.float64)]
        )
        return np.asarray(snapped[0], dtype=np.float64) + outward * outer_margin

    distances = np.linalg.norm(locations - origin, axis=1)
    hit = locations[int(np.argmin(distances))]
    return hit + outward * outer_margin


def anchor_outer_loop_on_surface(
    mesh: trimesh.Trimesh,
    outer_loop: np.ndarray,
    axis_xy: np.ndarray,
    *,
    outer_margin: float = 0.08,
    ray_start_distance: float = 10.0,
) -> np.ndarray:
    """
    Re-seat outer loop vertices on the exterior surface, then nudge slightly outward.

    Tangent-plane outlines can sit slightly inside the wall after snapping. Casting
    from outside locates the true outer hit; a tiny outward nudge ensures the cutter
    breaches the exterior without starting deep in free space.
    """

    outer = _open_loop(outer_loop)
    return np.asarray(
        [
            project_to_exterior_surface(
                mesh,
                point,
                axis_xy,
                outer_margin=outer_margin,
                ray_start_distance=ray_start_distance,
            )
            for point in outer
        ],
        dtype=np.float64,
    )


def backoff_outer_loop_from_axis(
    outer_loop: np.ndarray,
    axis_xy: np.ndarray,
    *,
    backoff_distance: float = 10.0,
) -> np.ndarray:
    """
    Move each outer boundary vertex farther from the axis along its radial ray.

    The expanded points are placed in space at ``radius + backoff_distance`` without
    re-projecting to the exterior shell, which would collapse back to the same surface
    hit for a given azimuth.
    """

    outer = _open_loop(outer_loop)
    return np.asarray(
        [
            point + outward_direction_from_axis(point, axis_xy) * backoff_distance
            for point in outer
        ],
        dtype=np.float64,
    )


def horizontal_radius(point: np.ndarray, axis_xy: np.ndarray) -> float:
    delta = np.asarray(point, dtype=np.float64)[:2] - axis_xy
    return float(np.linalg.norm(delta))


def uv_points_to_3d(
    uv_points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    tangent_u, tangent_v = _tangent_basis(normal)
    return (
        origin
        + uv_points[:, 0:1] * tangent_u
        + uv_points[:, 1:2] * tangent_v
    )


def sample_polygon_uv_points(
    polygon: Polygon,
    *,
    sample_spacing: float = 1.0,
) -> np.ndarray:
    """
    Sample the polygon boundary and interior on a local UV parameterization.
    """

    coords = list(polygon.exterior.coords[:-1])
    minx, miny, maxx, maxy = polygon.bounds
    xs = (
        np.array([(minx + maxx) * 0.5], dtype=np.float64)
        if maxx - minx < sample_spacing
        else np.arange(minx, maxx + sample_spacing * 0.5, sample_spacing)
    )
    ys = (
        np.array([(miny + maxy) * 0.5], dtype=np.float64)
        if maxy - miny < sample_spacing
        else np.arange(miny, maxy + sample_spacing * 0.5, sample_spacing)
    )

    for x in xs:
        for y in ys:
            point = Point(float(x), float(y))
            if polygon.contains(point) or polygon.boundary.distance(point) < 1e-6:
                coords.append((float(x), float(y)))

    return np.asarray(coords, dtype=np.float64)


def sample_cut_region_on_exterior_surface(
    mesh: trimesh.Trimesh,
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    axis_xy: np.ndarray,
    *,
    sample_spacing: float = 1.0,
    outer_margin: float = 0.0,
) -> np.ndarray:
    """
    Map UV samples from the cut polygon onto the exterior mesh surface.

    Tangent-plane lifting can sit above or below the curved shell, so each sample is
    re-seated with an exterior ray cast instead of a blind nearest-point snap.
    """

    uv_points = sample_polygon_uv_points(polygon, sample_spacing=sample_spacing)
    lifted = uv_points_to_3d(uv_points, origin, normal)
    surface_points = [
        project_to_exterior_surface(
            mesh,
            point,
            axis_xy,
            outer_margin=outer_margin,
        )
        for point in lifted
    ]
    surface_points.append(
        project_to_exterior_surface(mesh, origin, axis_xy, outer_margin=outer_margin)
    )
    return np.asarray(surface_points, dtype=np.float64)


def radial_extents_of_cut_cell(
    mesh: trimesh.Trimesh,
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    axis_xy: np.ndarray,
    *,
    outer_loop: np.ndarray | None = None,
    sample_spacing: float = 1.0,
    outer_margin: float = 0.0,
) -> tuple[float, float]:
    """
    Minimum and maximum horizontal distance from the axis across the cut region.

    Samples lie on the exterior mesh surface within the planned cut boundary, not on
    the seed tangent plane.
    """

    surface_points = sample_cut_region_on_exterior_surface(
        mesh,
        polygon,
        origin,
        normal,
        axis_xy,
        sample_spacing=sample_spacing,
        outer_margin=outer_margin,
    )
    if outer_loop is not None:
        anchored = anchor_outer_loop_on_surface(
            mesh,
            outer_loop,
            axis_xy,
            outer_margin=outer_margin,
        )
        surface_points = np.vstack([surface_points, anchored])

    radii = np.linalg.norm(surface_points[:, :2] - axis_xy, axis=1)
    return float(np.min(radii)), float(np.max(radii))


def measure_wall_thickness(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    axis_xy: np.ndarray,
    *,
    fallback: float = 4.0,
) -> float:
    """
    Measure shell thickness by ray casting through the wall toward the axis.
    """

    outward = outward_direction_from_axis(point, axis_xy)
    origin = point + outward * 0.15
    locations, _index_ray, _index_tri = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[-outward],
    )
    if len(locations) < 2:
        return fallback

    distances = np.sort(np.linalg.norm(locations - origin, axis=1))
    return float(distances[1] - distances[0])


def compute_radial_cut_depth(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    mesh: trimesh.Trimesh,
    axis_xy: np.ndarray,
    *,
    outer_loop: np.ndarray | None = None,
    sample_spacing: float = 1.0,
    inward_margin: float = 0.2,
) -> float:
    """
    Cut depth toward the axis: radial cell span on the exterior surface plus wall thickness.
    """

    min_radius, max_radius = radial_extents_of_cut_cell(
        mesh,
        polygon,
        origin,
        normal,
        axis_xy,
        outer_loop=outer_loop,
        sample_spacing=sample_spacing,
    )
    wall_thickness = measure_wall_thickness(mesh, origin, axis_xy)
    return (max_radius - min_radius) + wall_thickness + inward_margin


def inner_point_at_cut_depth(
    outer_point: np.ndarray,
    axis_xy: np.ndarray,
    cut_depth: float,
) -> np.ndarray:
    inward = inward_direction_toward_axis(outer_point, axis_xy)
    return outer_point + inward * cut_depth


def trace_inner_wall_point(
    mesh: trimesh.Trimesh,
    outer_point: np.ndarray,
    axis_xy: np.ndarray,
    *,
    exit_margin: float = 0.35,
) -> np.ndarray | None:
    """
    Ray cast through the wall toward the axis and return the interior-wall hit.

    When the outer point sits on or just outside the exterior, the first mesh hit is
    the outer skin and the second is the inner wall. Using only the first hit would
    create a superficial surface groove.
    """

    direction = inward_direction_toward_axis(outer_point, axis_xy)
    outward = -direction
    origin = outer_point + outward * 0.15
    locations, _index_ray, _index_tri = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[direction],
    )
    if len(locations) == 0:
        return None

    distances = np.linalg.norm(locations - origin, axis=1)
    order = np.argsort(distances)
    hits = locations[order]

    if len(hits) >= 2:
        inner_point = hits[1]
    else:
        inner_point = hits[0]

    return inner_point + direction * exit_margin


def _open_loop(loop: np.ndarray) -> np.ndarray:
    points = np.asarray(loop, dtype=np.float64)
    if len(points) >= 2 and np.allclose(points[0], points[-1]):
        return points[:-1]
    return points


def build_ruled_through_cutter(
    outer_loop: np.ndarray,
    inner_loop: np.ndarray,
) -> trimesh.Trimesh | None:
    """
    Build a watertight ruled mesh between matching outer and inner loops.
    """

    outer = _open_loop(outer_loop)
    inner = _open_loop(inner_loop)
    if len(outer) < 3 or len(inner) != len(outer):
        return None

    count = len(outer)
    outer_center = outer.mean(axis=0)
    inner_center = inner.mean(axis=0)

    vertices = np.vstack([outer, inner, outer_center[None], inner_center[None]])
    outer_center_index = 2 * count
    inner_center_index = 2 * count + 1

    faces: list[list[int]] = []
    for index in range(count):
        next_index = (index + 1) % count
        outer_a = index
        outer_b = next_index
        inner_a = count + index
        inner_b = count + next_index
        faces.append([outer_a, outer_b, inner_b])
        faces.append([outer_a, inner_b, inner_a])
        faces.append([outer_center_index, outer_b, outer_a])
        faces.append([inner_center_index, inner_a, inner_b])

    cutter = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    cutter.merge_vertices()
    return cutter if cutter.is_volume else None


def build_radial_cutter_from_loop(
    mesh: trimesh.Trimesh,
    outer_loop: np.ndarray,
    axis_xy: np.ndarray,
    *,
    cut_depth: float,
    outer_margin: float = 0.08,
    outer_backoff: float = 1.0,
) -> trimesh.Trimesh | None:
    """
    Sweep one outer boundary loop toward the axis to form a through-wall cutter.
    """

    outer = anchor_outer_loop_on_surface(
        mesh,
        outer_loop,
        axis_xy,
        outer_margin=outer_margin,
    )
    if outer_backoff > 0.0:
        outer = backoff_outer_loop_from_axis(
            outer,
            axis_xy,
            backoff_distance=outer_backoff,
        )
    if len(outer) < 3:
        return None

    inner = np.asarray(
        [
            inner_point_at_cut_depth(point, axis_xy, cut_depth)
            for point in outer
        ],
        dtype=np.float64,
    )
    return build_ruled_through_cutter(outer, inner)


def build_local_radial_pebble_cutters(
    mesh: trimesh.Trimesh,
    pattern_mesh: trimesh.Trimesh,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    axis_xy: np.ndarray | None = None,
    outer_margin: float = 0.08,
    outer_backoff: float = 1.0,
    cut_depth_margin: float = 1.0,
    rounding_distance: float = 0.0,
    spline_samples: int = 8,
    rounding_fullness: float = 1.0,
    max_overhang_degrees: float = 55.0,
) -> tuple[list[trimesh.Trimesh], dict[str, int]]:
    """
    Build radial through-wall cutters for every pattern seed.
    """

    if axis_xy is None:
        axis_xy = vertical_axis_xy(mesh)

    cutters: list[trimesh.Trimesh] = []
    pattern_indices = np.where(seed_set.pattern_mask)[0]
    failures = 0

    for seed_index in pattern_indices:
        seed_index = int(seed_index)
        origin = seed_set.all_seeds[seed_index]
        normal = seed_set.all_normals[seed_index]
        partner_indices = voronoi_partner_indices(
            seed_index,
            seed_set.all_seeds,
            search_radius=search_radius,
        )
        cut_polygon = (
            build_local_rounded_pebble_polygon(
                seed_index,
                seed_set,
                search_radius=search_radius,
                perpendicular_half_length=perpendicular_half_length,
                gap=gap,
                rounding_distance=rounding_distance,
                spline_samples=spline_samples,
                rounding_fullness=rounding_fullness,
                max_overhang_degrees=max_overhang_degrees,
                partner_indices=partner_indices,
            )
            if rounding_distance > 0.0
            else build_local_shrunk_cell_polygon(
                seed_index,
                seed_set,
                search_radius=search_radius,
                perpendicular_half_length=perpendicular_half_length,
                gap=gap,
                partner_indices=partner_indices,
            )
        )
        if cut_polygon is None:
            failures += 1
            continue

        outer_loop = (
            build_local_rounded_shrunk_boundary_loop(
                seed_index,
                seed_set,
                pattern_mesh,
                search_radius=search_radius,
                perpendicular_half_length=perpendicular_half_length,
                gap=gap,
                rounding_distance=rounding_distance,
                spline_samples=spline_samples,
                rounding_fullness=rounding_fullness,
                max_overhang_degrees=max_overhang_degrees,
                partner_indices=partner_indices,
            )
            if rounding_distance > 0.0
            else build_local_shrunk_boundary_loop(
                seed_index,
                seed_set.all_seeds,
                seed_set.all_normals,
                pattern_mesh,
                search_radius=search_radius,
                perpendicular_half_length=perpendicular_half_length,
                gap=gap,
                partner_indices=partner_indices,
            )
        )
        if outer_loop is None:
            failures += 1
            continue

        cut_depth = compute_radial_cut_depth(
            cut_polygon,
            origin,
            normal,
            mesh,
            axis_xy,
            outer_loop=outer_loop,
            inward_margin=cut_depth_margin,
        )

        cutter = build_radial_cutter_from_loop(
            mesh,
            outer_loop,
            axis_xy,
            cut_depth=cut_depth,
            outer_margin=outer_margin,
            outer_backoff=outer_backoff,
        )
        if cutter is None:
            failures += 1
            continue

        cutters.append(cutter)

    stats = {
        "cutter_count": len(cutters),
        "failures": failures,
        "pattern_seeds": len(pattern_indices),
    }
    return cutters, stats


def build_local_rounded_radial_pebble_cutters(
    mesh: trimesh.Trimesh,
    pattern_mesh: trimesh.Trimesh,
    seed_set: BorderSeedSet,
    *,
    search_radius: float,
    perpendicular_half_length: float,
    gap: float,
    rounding_distance: float,
    axis_xy: np.ndarray | None = None,
    outer_margin: float = 0.08,
    outer_backoff: float = 1.0,
    cut_depth_margin: float = 1.0,
    spline_samples: int = 8,
    rounding_fullness: float = 1.0,
    max_overhang_degrees: float = 55.0,
) -> tuple[list[trimesh.Trimesh], dict[str, int]]:
    """
    Build radial cutters from rounded shrunk Voronoi boundaries.
    """

    return build_local_radial_pebble_cutters(
        mesh,
        pattern_mesh,
        seed_set,
        search_radius=search_radius,
        perpendicular_half_length=perpendicular_half_length,
        gap=gap,
        axis_xy=axis_xy,
        outer_margin=outer_margin,
        outer_backoff=outer_backoff,
        cut_depth_margin=cut_depth_margin,
        rounding_distance=rounding_distance,
        spline_samples=spline_samples,
        rounding_fullness=rounding_fullness,
        max_overhang_degrees=max_overhang_degrees,
    )


def build_local_radial_pebble_cutters_from_loops(
    mesh: trimesh.Trimesh,
    loops: list[np.ndarray],
    *,
    axis_xy: np.ndarray | None = None,
    outer_backoff: float = 1.0,
) -> tuple[list[trimesh.Trimesh], dict[str, int]]:
    """
    Build radial cutters from precomputed outer boundary loops.
    """

    if axis_xy is None:
        axis_xy = vertical_axis_xy(mesh)

    cutters: list[trimesh.Trimesh] = []
    failures = 0

    for outer_loop in loops:
        outer = _open_loop(outer_loop)
        if len(outer) < 3:
            failures += 1
            continue
        radii = np.array([horizontal_radius(point, axis_xy) for point in outer])
        wall_thickness = measure_wall_thickness(mesh, outer.mean(axis=0), axis_xy)
        cut_depth = float(np.max(radii) - np.min(radii)) + wall_thickness + 0.2
        cutter = build_radial_cutter_from_loop(
            mesh,
            outer_loop,
            axis_xy,
            cut_depth=cut_depth,
            outer_backoff=outer_backoff,
        )
        if cutter is None:
            failures += 1
            continue
        cutters.append(cutter)

    stats = {
        "cutter_count": len(cutters),
        "failures": failures,
        "pattern_seeds": len(loops),
    }
    return cutters, stats
