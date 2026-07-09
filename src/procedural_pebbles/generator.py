"""
Public API.
"""

from __future__ import annotations

from .geometry import PointSet
from .svg import save_points
from scipy.spatial import Voronoi
from .voronoi import finite_polygons


class PebbleGenerator:

    def __init__(
        self,
        width: float,
        height: float,
        seed: int | None = None,
    ):

        self.width = width
        self.height = height
        self.seed = seed

        self.points = None

    def generate(self, count: int = 100):

        self.points = PointSet.random(
            self.width,
            self.height,
            count,
            self.seed,
        )

    def relax(self, iterations: int = 3):

        if self.points is None:
            raise RuntimeError("generate() first")

        self.points.relax(iterations)

    def save_points_svg(
        self,
        filename: str,
    ):

        if self.points is None:
            raise RuntimeError("generate() first")

        save_points(
            filename,
            self.width,
            self.height,
            self.points.points,
        )
    def voronoi(self):

        if self.points is None:
            raise RuntimeError(
                "generate() first"
            )

        vor = Voronoi(
            self.points.points
        )

        return finite_polygons(vor)
