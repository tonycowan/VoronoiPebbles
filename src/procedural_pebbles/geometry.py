"""
Geometry primitives for Procedural Pebbles.

This module currently implements

- deterministic random point generation
- Lloyd relaxation
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import Voronoi


@dataclass
class PointSet:
    width: float
    height: float
    points: np.ndarray
    rng: np.random.Generator

    @classmethod
    def random(
        cls,
        width: float,
        height: float,
        count: int,
        seed: int | None = None,
    ) -> "PointSet":

        rng = np.random.default_rng(seed)

        pts = np.empty((count, 2), dtype=float)

        pts[:, 0] = rng.uniform(0.0, width, count)
        pts[:, 1] = rng.uniform(0.0, height, count)

        return cls(
            width=width,
            height=height,
            points=pts,
            rng=rng,
        )

    def voronoi(self) -> Voronoi:
        return Voronoi(self.points)

    def relax(self, iterations: int = 1) -> None:
        """
        Lloyd relaxation.

        Infinite regions are ignored for now.
        """

        for _ in range(iterations):

            vor = Voronoi(self.points)

            new_points = []

            for point_index, region_index in enumerate(vor.point_region):

                region = vor.regions[region_index]

                if -1 in region:
                    new_points.append(
                        self.points[point_index]
                    )
                    continue

                vertices = vor.vertices[region]

                centroid = vertices.mean(axis=0)

                centroid[0] = np.clip(
                    centroid[0],
                    0,
                    self.width,
                )

                centroid[1] = np.clip(
                    centroid[1],
                    0,
                    self.height,
                )

                new_points.append(centroid)

            self.points = np.asarray(new_points)
