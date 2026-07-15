"""
Generate twinkle perforations on a mesh surface.

Tiny cylinders are placed with Poisson-disk spacing and aimed at a shared point
light on the lamp's vertical axis (offset below the top). As an observer walks
around the shade, different bores line up with the light for a twinkling effect.

Usage:
    python gallery/patterns/twinkle.py gallery/shapes/CurvyLamp.stl
    python gallery/patterns/twinkle.py gallery/shapes/CurvyLamp.stl \\
        --min-spacing 6 --hole-radius 0.8 --light-source-offset 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from mesh_patterns import TwinklePipeline
from mesh_patterns.gallery_paths import next_run_number

PATTERN = "twinkle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a twinkle perforation pattern to a mesh.",
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the target mesh STL",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=6.0,
        help="Minimum center-to-center spacing between holes in mm",
    )
    parser.add_argument(
        "--hole-radius",
        type=float,
        default=0.8,
        help="Radius of each cylindrical bore in mm",
    )
    parser.add_argument(
        "--light-source-offset",
        type=float,
        default=30.0,
        help=(
            "Distance in mm below the top of the shade where the nominal light "
            "source sits on the central axis"
        ),
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for point scattering",
    )
    parser.add_argument(
        "--cylinder-sections",
        type=int,
        default=16,
        help="Number of radial segments used for each cylinder cutter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    shape = mesh_path.stem

    mesh = trimesh.load(mesh_path, force="mesh")

    pipeline = TwinklePipeline(
        min_spacing=args.min_spacing,
        hole_radius=args.hole_radius,
        light_source_offset=args.light_source_offset,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        seed=args.seed,
        spacing_metric=args.spacing_metric,
        cylinder_sections=args.cylinder_sections,
        outer_cut_margin=0.5,
    )

    result = pipeline.run(mesh)
    run_number = next_run_number(shape, PATTERN)

    print("Building twinkle cutters...")
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
    print(f"Hole radius: {args.hole_radius:.2f} mm")
    print(f"Light source offset below top: {args.light_source_offset:.1f} mm")
    print(f"Perforated mesh faces: {len(perforated.faces):,}")
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
