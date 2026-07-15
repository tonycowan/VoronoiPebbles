"""
Notch nearly-horizontal upper cut boundaries for support-free printing.

After shrunk Voronoi outlines are formed (and before corner rounding), replace
nearly-horizontal spans in the upper half of each cell with a steep V so the
remaining structure stays within a maximum angle from vertical.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def local_up_uv(normal: np.ndarray) -> np.ndarray | None:
    """
    Project world +Z into the seed tangent plane as a unit UV vector.

    Returns None when the surface is nearly horizontal (no usable local up).
    """

    normal = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return None
    normal = normal / length

    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_uv = np.array(
        [
            float(np.dot(world_up, tangent_u)),
            float(np.dot(world_up, tangent_v)),
        ],
        dtype=np.float64,
    )
    up_length = float(np.linalg.norm(up_uv))
    if up_length < 1e-9:
        return None

    return up_uv / up_length


def angle_from_vertical(direction: np.ndarray, up: np.ndarray) -> float:
    """
    Absolute angle in radians between ``direction`` and the local vertical axis.
    """

    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return 0.5 * np.pi

    direction = direction / length
    cosine = abs(float(np.dot(direction, up)))
    return float(np.arccos(np.clip(cosine, 0.0, 1.0)))


def _as_single_polygon(geometry) -> Polygon | None:
    if geometry is None or geometry.is_empty:
        return None

    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda geom: geom.area)

    if geometry.geom_type != "Polygon":
        return None

    return geometry


def _open_coords(coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if len(points) >= 2 and np.allclose(points[0], points[-1]):
        return points[:-1]
    return points


def _clean_ring(
    coords: np.ndarray,
    *,
    min_edge_length: float = 1e-6,
) -> np.ndarray:
    if len(coords) == 0:
        return coords

    cleaned = [coords[0]]
    for point in coords[1:]:
        if float(np.linalg.norm(point - cleaned[-1])) > min_edge_length:
            cleaned.append(point)

    if len(cleaned) > 1 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) <= min_edge_length:
        cleaned.pop()

    return np.asarray(cleaned, dtype=np.float64)


def _edge_length(coords: np.ndarray, edge_index: int) -> float:
    count = len(coords)
    return float(
        np.linalg.norm(coords[(edge_index + 1) % count] - coords[edge_index])
    )


def _point_on_edge(coords: np.ndarray, edge_index: int, parameter: float) -> np.ndarray:
    count = len(coords)
    start = coords[edge_index]
    end = coords[(edge_index + 1) % count]
    return start + (end - start) * parameter


def _normalize_print_orientation(print_orientation: str) -> str:
    orientation = print_orientation.strip().lower()
    if orientation not in {"upright", "inverted"}:
        raise ValueError(
            "print_orientation must be 'upright' or 'inverted', "
            f"got {print_orientation!r}"
        )
    return orientation


def _qualifying_edges(
    coords: np.ndarray,
    up: np.ndarray,
    max_radians: float,
    *,
    print_orientation: str,
) -> list[bool]:
    """
    Nearly-horizontal edges in the overhang half of the clock face.

    ``upright`` uses 9–12–3 (upper half); ``inverted`` uses 3–6–9 (lower half).
    """

    orientation = _normalize_print_orientation(print_orientation)
    count = len(coords)
    flags: list[bool] = []
    for index in range(count):
        start = coords[index]
        end = coords[(index + 1) % count]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1e-12:
            flags.append(False)
            continue

        midpoint = 0.5 * (start + end)
        height = float(np.dot(midpoint, up))
        if orientation == "upright":
            if height <= 0.0:
                flags.append(False)
                continue
        elif height >= 0.0:
            flags.append(False)
            continue

        if angle_from_vertical(edge, up) > max_radians + 1e-9:
            flags.append(True)
        else:
            flags.append(False)

    return flags


def _runs_from_flags(flags: list[bool]) -> list[tuple[int, int]]:
    """
    Return inclusive edge-index runs ``(start, end)`` of consecutive True flags.

    Handles wrap-around so a run spanning the ring seam is one interval with
    ``start > end`` numerically (interpreted as wrapping). Non-wrapping runs
    have ``start <= end``.
    """

    count = len(flags)
    if count == 0 or not any(flags):
        return []

    if all(flags):
        return [(0, count - 1)]

    runs: list[tuple[int, int]] = []
    index = 0
    while index < count:
        if not flags[index]:
            index += 1
            continue
        start = index
        while index < count and flags[index]:
            index += 1
        runs.append((start, index - 1))

    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == count - 1:
        first_end = runs[0][1]
        last_start = runs[-1][0]
        runs = runs[1:-1] + [(last_start, first_end)]

    return runs


def _run_edge_indices(start: int, end: int, count: int) -> list[int]:
    if start <= end:
        return list(range(start, end + 1))

    return list(range(start, count)) + list(range(0, end + 1))


def _run_center(
    coords: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, int, float]:
    """
    Arc-length midpoint of a run of edges.

    Returns ``(point, edge_index, parameter)`` with parameter in ``[0, 1)``.
    """

    count = len(coords)
    edges = _run_edge_indices(start, end, count)
    lengths = [_edge_length(coords, edge) for edge in edges]
    total = float(sum(lengths))
    if total < 1e-12:
        edge = edges[0]
        return coords[edge].copy(), edge, 0.0

    target = 0.5 * total
    traveled = 0.0
    for edge, length in zip(edges, lengths):
        if traveled + length >= target - 1e-12:
            remaining = target - traveled
            parameter = 0.0 if length < 1e-12 else remaining / length
            parameter = float(np.clip(parameter, 0.0, 1.0 - 1e-12))
            return _point_on_edge(coords, edge, parameter), edge, parameter
        traveled += length

    edge = edges[-1]
    return _point_on_edge(coords, edge, 1.0 - 1e-12), edge, 1.0 - 1e-12


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    *,
    min_ray_t: float = 1e-6,
) -> tuple[float, float, np.ndarray] | None:
    """
    Intersect ray ``origin + t * direction`` (t > min_ray_t) with segment.

    Returns ``(t, s, point)`` with segment parameter ``s`` in ``[0, 1]``.
    """

    edge = edge_end - edge_start
    denominator = _cross2(direction, edge)
    if abs(denominator) < 1e-14:
        return None

    offset = edge_start - origin
    ray_t = _cross2(offset, edge) / denominator
    if ray_t <= min_ray_t:
        return None

    segment_s = _cross2(offset, direction) / denominator
    if segment_s < -1e-9 or segment_s > 1.0 + 1e-9:
        return None

    segment_s = float(np.clip(segment_s, 0.0, 1.0))
    point = origin + direction * ray_t
    return float(ray_t), segment_s, point


def _first_ray_hit(
    coords: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    skip_edges: set[int] | None = None,
) -> tuple[np.ndarray, int, float] | None:
    """
    Nearest forward ray hit on the ring, optionally skipping some edges.
    """

    count = len(coords)
    skip_edges = skip_edges or set()
    best: tuple[float, np.ndarray, int, float] | None = None

    for edge_index in range(count):
        if edge_index in skip_edges:
            continue

        hit = _ray_segment_intersection(
            origin,
            direction,
            coords[edge_index],
            coords[(edge_index + 1) % count],
        )
        if hit is None:
            continue

        ray_t, segment_s, point = hit
        if best is None or ray_t < best[0]:
            best = (ray_t, point, edge_index, segment_s)

    if best is None:
        return None

    return best[1], best[2], best[3]


def _overhang_ray_directions(
    up: np.ndarray,
    max_radians: float,
    *,
    print_orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inward rays at ``±max_radians`` from vertical.

    ``upright`` casts toward ``-up`` (into the upper overhang ceiling);
    ``inverted`` casts toward ``+up`` (into the lower overhang floor when the
    mesh will be printed upside down).
    """

    orientation = _normalize_print_orientation(print_orientation)
    inward = -up if orientation == "upright" else up
    side = np.array([-up[1], up[0]], dtype=np.float64)
    cosine = float(np.cos(max_radians))
    sine = float(np.sin(max_radians))
    dir_plus_side = cosine * inward + sine * side
    dir_minus_side = cosine * inward - sine * side
    return dir_plus_side, dir_minus_side


def _plus_arc_contains_edge(
    start_edge: int,
    start_parameter: float,
    end_edge: int,
    end_parameter: float,
    target_edge: int,
    target_parameter: float,
    count: int,
) -> bool:
    """
    True if ``target`` lies on the +direction ring arc from start to end.
    """

    if start_edge == end_edge:
        if end_parameter + 1e-9 >= start_parameter:
            return (
                target_edge == start_edge
                and start_parameter - 1e-9 <= target_parameter <= end_parameter + 1e-9
            )
        # Wrap within a single-index coincidence is treated as the long way;
        # require an explicit multi-edge walk below by using the wrap path.
        if target_edge == start_edge and (
            target_parameter >= start_parameter - 1e-9
            or target_parameter <= end_parameter + 1e-9
        ):
            return True

    if target_edge == start_edge and target_parameter >= start_parameter - 1e-9:
        return True
    if target_edge == end_edge and target_parameter <= end_parameter + 1e-9:
        return True

    index = (start_edge + 1) % count
    for _ in range(count):
        if index == end_edge:
            return False
        if index == target_edge:
            return True
        index = (index + 1) % count

    return False


def _order_stops_for_rebuild(
    coords: np.ndarray,
    center: np.ndarray,
    hit_a: np.ndarray,
    hit_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Order hits so +ring traversal is ``stop_neg → center → stop_pos``.
    """

    center_edge, center_parameter = _locate_point_on_ring(coords, center)
    edge_a, param_a = _locate_point_on_ring(coords, hit_a)
    edge_b, param_b = _locate_point_on_ring(coords, hit_b)
    count = len(coords)

    if _plus_arc_contains_edge(
        edge_a,
        param_a,
        edge_b,
        param_b,
        center_edge,
        center_parameter,
        count,
    ):
        return hit_a, hit_b

    return hit_b, hit_a


def _locate_point_on_ring(
    coords: np.ndarray,
    point: np.ndarray,
) -> tuple[int, float]:
    """
    Nearest edge index and parameter for ``point`` on the closed ring.
    """

    count = len(coords)
    best_edge = 0
    best_parameter = 0.0
    best_distance = float("inf")

    for edge_index in range(count):
        start = coords[edge_index]
        end = coords[(edge_index + 1) % count]
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq < 1e-18:
            parameter = 0.0
            candidate = start
        else:
            parameter = float(np.clip(np.dot(point - start, edge) / length_sq, 0.0, 1.0))
            candidate = start + edge * parameter

        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_distance = distance
            best_edge = edge_index
            best_parameter = parameter

    return best_edge, best_parameter


def _append_unique(points: list[np.ndarray], point: np.ndarray) -> None:
    if not points:
        points.append(point)
        return
    if float(np.linalg.norm(points[-1] - point)) > 1e-9:
        points.append(point)


def _rebuild_with_notch(
    coords: np.ndarray,
    stop_neg: np.ndarray,
    center: np.ndarray,
    stop_pos: np.ndarray,
) -> np.ndarray | None:
    """
    Replace the boundary arc from ``stop_neg`` through ``center`` to ``stop_pos``
    with the two segments ``stop_neg → center → stop_pos``.
    """

    count = len(coords)
    neg_edge, neg_parameter = _locate_point_on_ring(coords, stop_neg)
    pos_edge, pos_parameter = _locate_point_on_ring(coords, stop_pos)

    new_points: list[np.ndarray] = []
    _append_unique(new_points, stop_neg)
    _append_unique(new_points, center)
    _append_unique(new_points, stop_pos)

    # Unwalked arc: ring vertices from just after stop_pos through the start of
    # the edge that contains stop_neg. stop_neg itself closes the ring.
    if not (pos_edge == neg_edge and neg_parameter > pos_parameter + 1e-12):
        index = (pos_edge + 1) % count
        for _ in range(count):
            _append_unique(new_points, coords[index])
            if index == neg_edge:
                break
            index = (index + 1) % count

    if len(new_points) < 3:
        return None

    return np.asarray(new_points, dtype=np.float64)


def _notch_once(
    coords: np.ndarray,
    up: np.ndarray,
    max_radians: float,
    *,
    print_orientation: str,
) -> np.ndarray | None:
    flags = _qualifying_edges(
        coords,
        up,
        max_radians,
        print_orientation=print_orientation,
    )
    runs = _runs_from_flags(flags)
    if not runs:
        return None

    def run_length(run: tuple[int, int]) -> float:
        edges = _run_edge_indices(run[0], run[1], len(coords))
        return float(sum(_edge_length(coords, edge) for edge in edges))

    start, end = max(runs, key=run_length)
    center, center_edge, _center_parameter = _run_center(coords, start, end)
    run_edges = set(_run_edge_indices(start, end, len(coords)))

    dir_plus, dir_minus = _overhang_ray_directions(
        up,
        max_radians,
        print_orientation=print_orientation,
    )
    hit_plus = _first_ray_hit(coords, center, dir_plus, skip_edges=run_edges)
    hit_minus = _first_ray_hit(coords, center, dir_minus, skip_edges=run_edges)
    if hit_plus is None or hit_minus is None:
        skip = {center_edge}
        if hit_plus is None:
            hit_plus = _first_ray_hit(coords, center, dir_plus, skip_edges=skip)
        if hit_minus is None:
            hit_minus = _first_ray_hit(coords, center, dir_minus, skip_edges=skip)
    if hit_plus is None or hit_minus is None:
        return None

    stop_a = hit_plus[0]
    stop_b = hit_minus[0]
    if float(np.linalg.norm(stop_a - stop_b)) < 1e-6:
        return None

    stop_neg, stop_pos = _order_stops_for_rebuild(coords, center, stop_a, stop_b)
    return _rebuild_with_notch(coords, stop_neg, center, stop_pos)


def notch_nearly_horizontal_overhangs(
    polygon: Polygon,
    normal: np.ndarray,
    *,
    max_overhang_degrees: float = 55.0,
    print_orientation: str = "upright",
    max_iterations: int = 32,
) -> Polygon | None:
    """
    Notch nearly-horizontal overhang spans into printable V roofs.

    Operates in the seed tangent plane where the seed is at the origin.

    ``print_orientation='upright'`` notches the upper half (9–12–3).
    ``print_orientation='inverted'`` notches the lower half (3–6–9) for meshes
    that will be printed upside down.

    Spans whose direction exceeds ``max_overhang_degrees`` from local vertical
    are replaced by chords from the span center to the first inward ray hits at
    ``±max_overhang_degrees`` from vertical.
    """

    original = _as_single_polygon(polygon)
    if original is None:
        return None

    orientation = _normalize_print_orientation(print_orientation)

    up = local_up_uv(normal)
    if up is None:
        return original

    if max_overhang_degrees <= 0.0:
        return original

    max_radians = float(np.deg2rad(max_overhang_degrees))
    coords = _clean_ring(_open_coords(np.asarray(original.exterior.coords, dtype=np.float64)))
    if len(coords) < 3:
        return original

    working = coords
    for _ in range(max_iterations):
        updated = _notch_once(
            working,
            up,
            max_radians,
            print_orientation=orientation,
        )
        if updated is None:
            break
        working = _clean_ring(updated)
        if len(working) < 3:
            return original

    result = Polygon(working)
    if not result.is_valid or not result.is_simple:
        result = result.buffer(0)

    result = _as_single_polygon(result)
    if result is None or result.is_empty or result.area <= 0.0:
        return original

    return result
