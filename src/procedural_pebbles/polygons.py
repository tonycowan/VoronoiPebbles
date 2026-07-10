"""
Polygon utilities.
"""

from __future__ import annotations

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

def round_polygons(polygons, radius=0.8):
    result = []

    for poly in polygons:
        p = (
            poly
            .buffer(radius, join_style="round")
            .buffer(-radius, join_style="round")
            .simplify(0.05, preserve_topology=True)
        )

        if p.is_empty:
            continue

        if p.geom_type == "Polygon":
            result.append(p)
        else:
            result.extend(p.geoms)

    return result
