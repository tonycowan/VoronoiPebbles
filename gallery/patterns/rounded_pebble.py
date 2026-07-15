"""
Generate rounded pebble perforations on a mesh surface.

Usage:
    python gallery/patterns/rounded_pebble.py gallery/shapes/CurvyLamp.stl
    python gallery/patterns/rounded_pebble.py gallery/shapes/CurvyLamp.stl --export-artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from mesh_patterns import RoundedPebblePipeline
from mesh_patterns.debug_artifacts import (
    TessellationDebugConfig,
    export_tessellation_artifacts,
    tessellation_from_result,
)
from mesh_patterns.gallery_paths import next_run_number

PATTERN = "rounded_pebble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a rounded pebble perforation pattern to a mesh.",
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
        help="Minimum center-to-center spacing between pebbles in mm",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=2.0,
        help="Minimum distance between pebbles in mm (each Voronoi cell shrinks by half this amount)",
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
        "--export-artifacts",
        action="store_true",
        help=(
            "Also export Voronoi, shrunk tessellation, and rounded pebble cutter "
            "preview debug views (GLB + STL) using the exact same seed data as the cut"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point scattering",
    )
    parser.add_argument(
        "--outer-boundary-backoff",
        type=float,
        default=1.0,
        help=(
            "Expand each cut's outer boundary outward from the central axis by this "
            "many mm before building the cutter"
        ),
    )
    parser.add_argument(
        "--cut-depth-margin",
        type=float,
        default=1.0,
        help=(
            "Extra mm added to each cutter's inward depth beyond the auto-computed "
            "radial cell span and wall thickness"
        ),
    )
    parser.add_argument(
        "--rounding-distance",
        type=float,
        default=1.0,
        help=(
            "Distance in mm to walk along each cut boundary edge before replacing a "
            "sharp vertex with a rounded fillet"
        ),
    )
    parser.add_argument(
        "--arc-samples",
        "--spline-samples",
        dest="spline_samples",
        type=int,
        default=8,
        help="Number of samples along each tangent corner spline",
    )
    parser.add_argument(
        "--rounding-fullness",
        type=float,
        default=1.0,
        help=(
            "Curvature scale for corner splines (1.0 = balanced default; lower is "
            "tighter, higher is fuller)"
        ),
    )
    parser.add_argument(
        "--max-overhang-degrees",
        type=float,
        default=55.0,
        help=(
            "Maximum angle from vertical allowed on upper cut boundaries before "
            "notching for support-free printing"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    shape = mesh_path.stem

    mesh = trimesh.load(mesh_path, force="mesh")

    pipeline = RoundedPebblePipeline(
        min_spacing=args.min_spacing,
        gap=args.gap,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        seed=args.seed,
        spacing_metric=args.spacing_metric,
        outer_boundary_backoff=args.outer_boundary_backoff,
        cut_depth_margin=args.cut_depth_margin,
        rounding_distance=args.rounding_distance,
        rounding_fullness=args.rounding_fullness,
        spline_samples=args.spline_samples,
        max_overhang_degrees=args.max_overhang_degrees,
    )

    result = pipeline.run(mesh)
    run_number = next_run_number(shape, PATTERN)

    if args.export_artifacts:
        debug_config = TessellationDebugConfig(
            min_spacing=args.min_spacing,
            gap=args.gap,
            bottom_border=args.bottom_border,
            top_border=args.top_border,
            seed=args.seed,
            spacing_metric=args.spacing_metric,
            rounding_distance=args.rounding_distance,
            rounding_fullness=args.rounding_fullness,
            spline_samples=args.spline_samples,
            max_overhang_degrees=args.max_overhang_degrees,
        )
        export_tessellation_artifacts(
            mesh,
            shape,
            debug_config,
            pebble_preview=True,
            rounded_pebble_preview=True,
            tessellation=tessellation_from_result(result),
        )

    print("Building rounded pebble cutters...")
    perforated = pipeline.build_perforated_mesh(result)
    output_path = pipeline.export_creation(
        perforated,
        shape=shape,
        pattern=PATTERN,
        run_number=run_number,
    )

    print(f"Loaded mesh: {mesh_path}")
    print(f"Bottom border: {args.bottom_border:.1f} mm")
    print(f"Top border: {args.top_border:.1f} mm")
    print(f"Selected surface faces: {result.selection.face_count:,}")
    print(f"Selected surface area: {result.selection.area:,.1f} mm^2")
    print(f"Seed points: {len(result.seeds):,}")
    print(f"Rounding distance: {args.rounding_distance:.1f} mm")
    print(f"Rounding fullness: {args.rounding_fullness:.2f}")
    print(f"Max overhang: {args.max_overhang_degrees:.1f} deg")
    print(f"Outer boundary backoff: {args.outer_boundary_backoff:.1f} mm")
    print(f"Cut depth margin: {args.cut_depth_margin:.1f} mm")
    print(f"Perforated mesh faces: {len(perforated.faces):,}")
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
