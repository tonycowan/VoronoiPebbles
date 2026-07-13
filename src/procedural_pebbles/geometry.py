"""
geometry.py

Simple point storage and generation.

This module intentionally contains no Voronoi or SVG code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class PointSet:
    """
    Collection of 2D points.
    """

    width: float
    height: float
    points: np.ndarray

    @classmethod
    def random(
        cls,
        width: float,
        height: float,
        count: int,
        seed: int | None = None,
    ) -> "PointSet":
        """
        Generate uniformly distributed random points.
        """

        rng = np.random.default_rng(seed)

        pts = np.empty((count, 2), dtype=np.float64)

        pts[:, 0] = rng.uniform(0.0, width, count)
        pts[:, 1] = rng.uniform(0.0, height, count)

        return cls(
            width=width,
            height=height,
            points=pts,
        )

    @property
    def count(self) -> int:
        return len(self.points)

    def copy(self) -> "PointSet":
        return PointSet(
            width=self.width,
            height=self.height,
            points=self.points.copy(),
        )

    def bounds(self):
        return (
            0.0,
            0.0,
            self.width,
            self.height,
        )
