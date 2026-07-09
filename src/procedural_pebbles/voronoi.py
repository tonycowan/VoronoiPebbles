"""
Finite Voronoi helper.

Adapted from the SciPy cookbook implementation and
refactored into a reusable function.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi


def finite_polygons(vor: Voronoi, radius=None):
    """
    Reconstruct infinite Voronoi regions into finite regions.

    Returns

        regions
        vertices
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input.")

    if radius is None:
        mins = vor.points.min(axis=0)
        maxs = vor.points.max(axis=0)

        radius = np.linalg.norm(maxs - mins) * 2

    new_regions = []

    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(
        vor.ridge_points,
        vor.ridge_vertices,
    ):

        all_ridges.setdefault(
            p1,
            [],
        ).append((p2, v1, v2))

        all_ridges.setdefault(
            p2,
            [],
        ).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):

        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]

        new_region = [
            v for v in vertices if v >= 0
        ]

        for p2, v1, v2 in ridges:

            if v2 < 0:
                v1, v2 = v2, v1

            if v1 >= 0:
                continue

            tangent = (
                vor.points[p2]
                - vor.points[p1]
            )

            tangent /= np.linalg.norm(
                tangent
            )

            normal = np.array(
                [
                    -tangent[1],
                    tangent[0],
                ]
            )

            midpoint = vor.points[
                [p1, p2]
            ].mean(axis=0)

            direction = (
                np.sign(
                    np.dot(
                        midpoint - center,
                        normal,
                    )
                )
                * normal
            )

            far = (
                vor.vertices[v2]
                + direction * radius
            )

            new_region.append(
                len(new_vertices)
            )

            new_vertices.append(
                far.tolist()
            )

        polygon = np.asarray(
            [
                new_vertices[v]
                for v in new_region
            ]
        )

        centroid = polygon.mean(axis=0)

        angles = np.arctan2(
            polygon[:, 1] - centroid[1],
            polygon[:, 0] - centroid[0],
        )

        new_region = np.asarray(
            new_region
        )[np.argsort(angles)]

        new_regions.append(
            new_region.tolist()
        )

    return (
        new_regions,
        np.asarray(new_vertices),
    )
