"""
Smooth sharp vertices on shrunk pebble cut polygons.

Each corner is softened by walking a fixed distance along the two incident edges,
then replacing the vertex with a cubic Bezier spline that is tangent to both
boundary segments at those trim points.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def _as_single_polygon(geometry) -> Polygon | None:
    if geometry.is_empty:
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
    """
    Drop consecutive duplicate/near-duplicate ring vertices.
    """

    if len(coords) == 0:
        return coords

    cleaned = [coords[0]]
    for point in coords[1:]:
        if float(np.linalg.norm(point - cleaned[-1])) > min_edge_length:
            cleaned.append(point)

    if len(cleaned) > 1 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) <= min_edge_length:
        cleaned.pop()

    return np.asarray(cleaned, dtype=np.float64)


def _walk_from_vertex(
    vertex: np.ndarray,
    neighbor: np.ndarray,
    distance: float,
) -> np.ndarray:
    edge = neighbor - vertex
    length = float(np.linalg.norm(edge))
    if length < 1e-12:
        return vertex.copy()

    if distance >= length:
        return neighbor.copy()

    return vertex + edge * (distance / length)


def _walk_along_boundary(
    coords: np.ndarray,
    start_index: int,
    direction: int,
    distance: float,
) -> np.ndarray:
    """
    Walk along the closed polygon boundary from a vertex by ``distance``.
    """

    count = len(coords)
    remaining = distance
    current_index = start_index
    current_point = coords[current_index]

    while remaining > 1e-9:
        next_index = (current_index + direction) % count
        next_point = coords[next_index]
        edge = next_point - current_point
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 1e-12:
            current_index = next_index
            current_point = next_point
            continue

        if remaining >= edge_length - 1e-9:
            remaining -= edge_length
            current_index = next_index
            current_point = next_point
            continue

        return current_point + edge * (remaining / edge_length)

    return current_point.copy()


def _prev_active(index: int, skipped: set[int], count: int) -> int:
    cursor = (index - 1) % count
    while cursor in skipped:
        cursor = (cursor - 1) % count
    return cursor


def _next_active(index: int, skipped: set[int], count: int) -> int:
    cursor = (index + 1) % count
    while cursor in skipped:
        cursor = (cursor + 1) % count
    return cursor


def _interior_angle(
    previous: np.ndarray,
    vertex: np.ndarray,
    nxt: np.ndarray,
) -> float:
    incoming = vertex - previous
    outgoing = nxt - vertex
    in_len = float(np.linalg.norm(incoming))
    out_len = float(np.linalg.norm(outgoing))
    if in_len < 1e-12 or out_len < 1e-12:
        return np.pi

    incoming /= in_len
    outgoing /= out_len
    cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
    return float(np.arccos(cosine))


def bezier_handle_length(
    trim_distance: float,
    interior_angle: float,
    *,
    fullness: float = 1.0,
) -> float:
    """
    Tangent-handle length for a corner-smoothing cubic Bezier.

    Tangency alone does not fix curvature. We scale the standard cubic Bezier
    approximation to a circular arc of the same corner deflection:

        handle = fullness * (4/3) * tan(deflection / 4) * trim_distance

    where ``deflection = pi - interior_angle``.

    ``fullness = 1`` is a balanced default. Lower values pull the spline closer
    to the chord (tighter corner); higher values let it bulge outward.
    """

    deflection = np.pi - interior_angle
    if deflection < 1e-6:
        return 0.0

    return fullness * (4.0 / 3.0) * np.tan(deflection * 0.25) * trim_distance


def _sample_cubic_bezier(
    start: np.ndarray,
    control_a: np.ndarray,
    control_b: np.ndarray,
    end: np.ndarray,
    *,
    samples: int,
) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, max(samples, 2))
    one_minus = 1.0 - parameter
    points = (
        one_minus[:, None] ** 3 * start
        + 3.0 * one_minus[:, None] ** 2 * parameter[:, None] * control_a
        + 3.0 * one_minus[:, None] * parameter[:, None] ** 2 * control_b
        + parameter[:, None] ** 3 * end
    )
    return np.asarray(points, dtype=np.float64)


def _sample_tangent_corner_spline(
    previous: np.ndarray,
    vertex: np.ndarray,
    nxt: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    trim_distance: float,
    interior_angle: float,
    fullness: float,
    samples: int,
) -> np.ndarray:
    """
    Cubic Bezier from ``start`` to ``end``, tangent to both boundary legs.
    """

    arrival = vertex - previous
    departure = nxt - vertex
    in_len = float(np.linalg.norm(arrival))
    out_len = float(np.linalg.norm(departure))
    if in_len < 1e-12 or out_len < 1e-12:
        return np.asarray([start, end], dtype=np.float64)

    tangent_in = arrival / in_len
    tangent_out = departure / out_len

    handle = bezier_handle_length(
        trim_distance,
        interior_angle,
        fullness=fullness,
    )
    handle_in = min(handle, float(np.linalg.norm(start - vertex)) * 0.99)
    handle_out = min(handle, float(np.linalg.norm(end - vertex)) * 0.99)

    chord = float(np.linalg.norm(end - start))
    if chord > 1e-12:
        handle_in = min(handle_in, chord * 0.49)
        handle_out = min(handle_out, chord * 0.49)

    if handle_in < 1e-9 and handle_out < 1e-9:
        return np.asarray([start, end], dtype=np.float64)

    control_a = start + tangent_in * handle_in
    control_b = end - tangent_out * handle_out
    return _sample_cubic_bezier(
        start,
        control_a,
        control_b,
        end,
        samples=samples,
    )


def _skipped_vertices(
    coords: np.ndarray,
    rounding_distance: float,
) -> set[int]:
    count = len(coords)
    skipped: set[int] = set()

    for index in range(count):
        if index in skipped:
            continue

        vertex = coords[index]
        for neighbor_index in ((index - 1) % count, (index + 1) % count):
            if neighbor_index in skipped:
                continue

            neighbor = coords[neighbor_index]
            if float(np.linalg.norm(neighbor - vertex)) <= rounding_distance + 1e-9:
                skipped.add(neighbor_index)

    return skipped


def _append_point(points: list[np.ndarray], point: np.ndarray) -> None:
    if not points:
        points.append(point)
        return

    if float(np.linalg.norm(points[-1] - point)) > 1e-9:
        points.append(point)


def _append_boundary_segment(
    points: list[np.ndarray],
    coords: np.ndarray,
    start_index: int,
    end_point: np.ndarray,
    end_index: int,
    *,
    skipped: set[int],
) -> None:
    """
    Continue along straight boundary edges to ``end_point`` without re-inserting
    skipped corner vertices.
    """

    if points and float(np.linalg.norm(points[-1] - end_point)) <= 1e-9:
        return

    count = len(coords)
    cursor = (start_index + 1) % count
    while cursor != end_index:
        if cursor not in skipped:
            _append_point(points, coords[cursor])
        cursor = (cursor + 1) % count

    _append_point(points, end_point)


def round_polygon_vertices(
    polygon: Polygon,
    rounding_distance: float,
    *,
    spline_samples: int = 8,
    rounding_fullness: float = 1.0,
) -> Polygon | None:
    """
    Replace sharp polygon corners with tangent cubic Bezier splines.

    ``rounding_distance`` is measured along each incident edge from the corner.
    Neighboring vertices consumed by that walk are omitted from smoothing.

    ``rounding_fullness`` scales tangent-handle length and therefore curvature.
    See ``bezier_handle_length`` for the default mapping.
    """

    if rounding_distance <= 0.0:
        return _as_single_polygon(polygon)

    coords = _clean_ring(
        _open_coords(np.asarray(polygon.exterior.coords, dtype=np.float64))
    )
    if len(coords) < 3:
        return None

    skipped = _skipped_vertices(coords, rounding_distance)
    rounded: list[np.ndarray] = []
    count = len(coords)

    for index in range(count):
        if index in skipped:
            continue

        previous_index = _prev_active(index, skipped, count)
        next_index = _next_active(index, skipped, count)
        vertex = coords[index]

        trim_in = _walk_along_boundary(
            coords,
            index,
            direction=-1,
            distance=rounding_distance,
        )
        trim_out = _walk_along_boundary(
            coords,
            index,
            direction=1,
            distance=rounding_distance,
        )

        angle = _interior_angle(
            coords[previous_index],
            vertex,
            coords[next_index],
        )
        if angle < 1e-6 or angle > np.pi - 1e-6:
            _append_point(rounded, trim_in)
            _append_point(rounded, trim_out)
        else:
            spline = _sample_tangent_corner_spline(
                coords[previous_index],
                vertex,
                coords[next_index],
                trim_in,
                trim_out,
                trim_distance=rounding_distance,
                interior_angle=angle,
                fullness=rounding_fullness,
                samples=spline_samples,
            )
            if len(spline) > 0:
                if len(rounded) == 0:
                    rounded.extend(spline.tolist())
                else:
                    rounded.extend(spline[1:].tolist())

        following = _next_active(index, skipped, count)
        if following == index:
            continue

        trim_next_in = _walk_along_boundary(
            coords,
            following,
            direction=-1,
            distance=rounding_distance,
        )
        _append_boundary_segment(
            rounded,
            coords,
            index,
            trim_next_in,
            following,
            skipped=skipped,
        )

    if len(rounded) < 3:
        return None

    result = Polygon(np.asarray(rounded, dtype=np.float64))
    if not result.is_valid or not result.is_simple:
        result = result.buffer(0)

    return _as_single_polygon(result)
