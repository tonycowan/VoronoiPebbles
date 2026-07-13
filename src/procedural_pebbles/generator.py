"""
generator.py

High-level API.
"""

from __future__ import annotations

from .geometry import PointSet


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

        self.points: PointSet | None = None

    def generate(
        self,
        count: int = 150,
    ) -> None:

        self.points = PointSet.random(
            width=self.width,
            height=self.height,
            count=count,
            seed=self.seed,
        )

    @property
    def point_count(self) -> int:

        if self.points is None:
            return 0

        return self.points.count
