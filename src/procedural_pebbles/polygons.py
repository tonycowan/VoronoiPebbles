"""
Polygon utilities.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box


def build_polygons(regions, vertices, width: float, height: float):
    """
    Convert Voronoi regions into clipped Shapely polygons.
    """

    tile = box(0.0, 0.0, width, height)

    result = []

    for region in regions:

        if len(region) < 3:
            continue

        poly = Polygon(vertices[i] for i in region)

        if not poly.is_valid:
            poly = poly.buffer(0)

        poly = poly.intersection(tile)

        if poly.is_empty:
            continue

        if poly.geom_type == "Polygon":
            result.append(poly)

        elif poly.geom_type == "MultiPolygon":
            result.extend(poly.geoms)

    return result


def shrink_voronoi_maps(polygons, minimum_pebble_distance: float):
    """
    Shrink each Voronoi cell by half the minimum pebble spacing.

    This leaves ``minimum_pebble_distance`` between the nearest points on
    any two adjacent pebbles, while junctions where three or more meet stay
    wider.
    """

    return inset_polygons(polygons, minimum_pebble_distance * 0.5)


def inset_polygons(polygons, gap):
    result = []

    for poly in polygons:
        p = poly.buffer(-gap)

        if p.is_empty:
            continue

        if p.geom_type == "Polygon":
            result.append(p)

        elif p.geom_type == "MultiPolygon":
            result.extend(p.geoms)

    return result


def _chaikin_closed(coords: np.ndarray, iterations: int, ratio: float = 0.25) -> np.ndarray:
    points = np.asarray(coords, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    for _ in range(iterations):
        corners = []
        count = len(points)
        for index in range(count):
            start = points[index]
            end = points[(index + 1) % count]
            corners.append((1.0 - ratio) * start + ratio * end)
            corners.append(ratio * start + (1.0 - ratio) * end)
        points = np.asarray(corners)

    return np.vstack([points, points[:1]])


def chaikin_polygon(polygon: Polygon, iterations: int = 3) -> Polygon:
    """
    Smooth polygon corners with Chaikin subdivision.
    """

    coords = _chaikin_closed(np.asarray(polygon.exterior.coords), iterations)
    smoothed = Polygon(coords)
    if not smoothed.is_valid:
        smoothed = smoothed.buffer(0)

    return smoothed


def _cap_polygon_vertices(polygon: Polygon, max_vertices: int) -> Polygon:
    coords = list(polygon.exterior.coords)
    if len(coords) <= max_vertices + 1:
        return polygon

    low = 0.001
    high = max(0.01, polygon.length / 8.0)
    best = polygon

    for _ in range(14):
        tolerance = (low + high) * 0.5
        candidate = polygon.simplify(tolerance, preserve_topology=True)
        if candidate.is_empty:
            high = tolerance
            continue

        if candidate.geom_type == "MultiPolygon":
            candidate = max(candidate.geoms, key=lambda geom: geom.area)

        count = len(candidate.exterior.coords)
        if count > max_vertices + 1:
            low = tolerance
        else:
            best = candidate
            high = tolerance

    return best


def round_polygons(
    polygons,
    radius=0.8,
    *,
    iterations: int = 3,
    max_vertices: int = 64,
):
    """
    Round pebble corners using Chaikin subdivision plus a small fillet.

    Chaikin depth stays fixed for performance. Larger ``radius`` values add
    a buffer fillet instead of more subdivision passes, which keeps cutter
    complexity bounded for boolean perforation.
    """

    result = []

    for poly in polygons:
        smoothed = chaikin_polygon(poly, iterations=iterations)
        if smoothed.is_empty:
            continue

        if radius > 0:
            fillet = min(radius, smoothed.length / 16.0)
            if fillet > 0:
                smoothed = (
                    smoothed.buffer(fillet, join_style="round", resolution=8)
                    .buffer(-fillet, join_style="round", resolution=8)
                )
                if smoothed.is_empty:
                    continue
                if smoothed.geom_type == "MultiPolygon":
                    smoothed = max(smoothed.geoms, key=lambda geom: geom.area)

        smoothed = _cap_polygon_vertices(smoothed, max_vertices)

        if smoothed.geom_type == "Polygon":
            result.append(smoothed)
        else:
            result.extend(smoothed.geoms)

    return result
