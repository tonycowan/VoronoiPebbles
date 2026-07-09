"""
SVG debug output.
"""

from __future__ import annotations

import svgwrite


def save_points(
    filename: str,
    width: float,
    height: float,
    points,
    radius: float = 1.0,
) -> None:

    svg = svgwrite.Drawing(
        filename,
        size=(f"{width}mm", f"{height}mm"),
        viewBox=f"0 0 {width} {height}",
    )

    for x, y in points:

        svg.add(
            svg.circle(
                center=(x, y),
                r=radius,
                fill="black",
            )
        )

    svg.save()
