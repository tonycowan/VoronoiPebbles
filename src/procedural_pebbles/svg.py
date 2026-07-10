"""
SVG rendering.
"""

from __future__ import annotations

import svgwrite

def save_points(
    filename,
    width,
    height,
    points,
    radius=1.0,
):

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

def save_polygons(
    filename,
    width,
    height,
    polygons,
):

    svg = svgwrite.Drawing(
        filename,
        size=(f"{width}mm", f"{height}mm"),
        viewBox=f"0 0 {width} {height}",
    )

    #
    # One compound path.
    #

    commands = []

    for poly in polygons:

        coords = list(poly.exterior.coords)

        if len(coords) < 3:
            continue

        commands.append(
            f"M {coords[0][0]:.3f} {coords[0][1]:.3f}"
        )

        for x, y in coords[1:]:

            commands.append(
                f"L {x:.3f} {y:.3f}"
            )

        commands.append("Z")

    svg.add(
        svg.path(
            d=" ".join(commands),
            fill="none",
            stroke="black",
            stroke_width=0.15,
        )
    )

    svg.save()
