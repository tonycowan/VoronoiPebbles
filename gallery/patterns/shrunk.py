"""
Visualize local Voronoi cells shrunk inward by half the minimum pebble gap.

Usage:
    python gallery/patterns/shrunk.py gallery/shapes/CurvyLamp.stl
    python gallery/patterns/shrunk.py gallery/shapes/CurvyLamp.stl --gap 2 --no-outer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from mesh_patterns.debug_artifacts import (
    TessellationDebugConfig,
    build_tessellation_seed_set,
    export_shrunk_artifact,
)

PATTERN = "shrunk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export shrunk local Voronoi cell boundaries on a mesh surface.",
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the target mesh STL",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=10.0,
        help="Minimum center-to-center spacing between seeds in mm",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=2.0,
        help="Minimum distance between pebbles in mm (each cell shrinks by half this amount)",
    )
    parser.add_argument(
        "--spacing-metric",
        choices=("surface", "euclidean"),
        default="euclidean",
        help="Distance metric used during Poisson-disk acceptance",
    )
    parser.add_argument(
        "--bottom-border",
        type=float,
        default=5.0,
        help="Unpatterned margin in mm to leave at the base of the shape",
    )
    parser.add_argument(
        "--top-border",
        type=float,
        default=20.0,
        help="Unpatterned margin in mm to leave at the top of the shape",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Added to typical seed spacing for partner search sphere in mm",
    )
    parser.add_argument(
        "--perpendicular-half-length",
        type=float,
        default=50.0,
        help="Bisector half-length used when clipping local Voronoi cells in mm",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=1.5,
        help="Radius of seed marker spheres in mm",
    )
    parser.add_argument(
        "--outer-line-radius",
        type=float,
        default=0.18,
        help="Radius of the original Voronoi outline tubes in mm",
    )
    parser.add_argument(
        "--shrunk-line-radius",
        type=float,
        default=0.28,
        help="Radius of the shrunk boundary tubes in mm",
    )
    parser.add_argument(
        "--no-outer",
        action="store_true",
        help="Hide the original Voronoi tessellation outlines",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point scattering",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview window after export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    shape = mesh_path.stem

    mesh = trimesh.load(mesh_path, force="mesh")
    config = TessellationDebugConfig(
        min_spacing=args.min_spacing,
        gap=args.gap,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        seed=args.seed,
        spacing_metric=args.spacing_metric,
        margin=args.margin,
        perpendicular_half_length=args.perpendicular_half_length,
        marker_radius=args.marker_radius,
        outer_line_radius=args.outer_line_radius,
        shrunk_line_radius=args.shrunk_line_radius,
        include_outer_voronoi=not args.no_outer,
    )
    tessellation = build_tessellation_seed_set(mesh, config)
    _, glb_path = export_shrunk_artifact(mesh, shape, tessellation, config)

    if args.show:
        trimesh.load(glb_path).show()


if __name__ == "__main__":
    main()
