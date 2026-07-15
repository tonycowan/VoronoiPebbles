# Procedural Pebbles

Generate stylized pebble mosaics and mesh perforation patterns suitable for:

- SVG artwork
- CNC / laser cutting
- 3D-printed lamp shades and shells
- architectural patterns
- decorative textures

The generator is based on Voronoi tessellation, Lloyd relaxation, polygon
offsets, and corner smoothing rather than attempting to simulate actual stones.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run gallery scripts with the project interpreter so dependencies resolve:

```bash
.venv/bin/python gallery/patterns/rounded_pebble.py gallery/shapes/CurvyLamp.stl
```

Outputs are written under `gallery/creations/` as
`<shape>.<pattern>.<run>.stl` (plus optional debug GLB/STL artifacts).

## Gallery pattern CLIs

All perforation / debug scripts take a positional `mesh` path (typically an
STL under `gallery/shapes/`).

Shared seeding / border options used by most scripts:

| Argument | Default | Description |
| --- | --- | --- |
| `--min-spacing` | `10.0` | Minimum center-to-center seed spacing (mm) |
| `--spacing-metric` | `euclidean` (`surface` for `points.py`) | Distance metric for Poisson-disk acceptance |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--seed` | `42` | Random seed for point scattering |

---

### `gallery/patterns/rounded_pebble.py`

Rounded shrunk-Voronoi pebble perforations with printable overhang notching
(upper or lower half depending on print orientation), then corner spline
rounding. Primary production cutter.

```bash
.venv/bin/python gallery/patterns/rounded_pebble.py gallery/shapes/CurvyLamp.stl \
  --export-artifacts \
  --rounding-distance 1 \
  --rounding-fullness 0.75
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between pebbles (mm) |
| `--gap` | `2.0` | Minimum distance between pebbles (mm); each Voronoi cell shrinks by half this amount |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` for Poisson-disk acceptance |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--export-artifacts` | off | Also export Voronoi, shrunk, and rounded cutter preview debug views (GLB + STL) |
| `--seed` | `42` | Random seed for point scattering |
| `--outer-boundary-backoff` | `1.0` | Expand each cut’s outer boundary outward from the central axis (mm) before building the cutter |
| `--cut-depth-margin` | `1.0` | Extra inward cutter depth beyond auto-computed radial span and wall thickness (mm) |
| `--rounding-distance` | `1.0` | Distance to walk along each cut edge before replacing a sharp vertex with a fillet (mm) |
| `--arc-samples` / `--spline-samples` | `8` | Number of samples along each tangent corner spline |
| `--rounding-fullness` | `1.0` | Corner spline curvature scale (`1.0` balanced; lower = tighter, higher = fuller) |
| `--max-overhang-degrees` | `55.0` | Max angle from vertical on overhang cut boundaries before notching for support-free printing |
| `--print-orientation` | `upright` | `upright` notches the upper half (9–12–3); `inverted` notches the lower half (3–6–9) for meshes printed upside down |

---

### `gallery/patterns/pebble.py`

Same radial pebble perforations as above, but without corner rounding or overhang notching (exact shrunk Voronoi cut outlines).

```bash
.venv/bin/python gallery/patterns/pebble.py gallery/shapes/CurvyLamp.stl --export-artifacts
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between pebbles (mm) |
| `--gap` | `2.0` | Minimum distance between pebbles (mm); each cell shrinks by half this amount |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--export-artifacts` | off | Also export Voronoi, shrunk, and pebble cutter preview debug views |
| `--seed` | `42` | Random seed for point scattering |
| `--outer-boundary-backoff` | `1.0` | Expand each cut’s outer boundary outward from the central axis (mm) |
| `--cut-depth-margin` | `1.0` | Extra inward cutter depth (mm) |

---

### `gallery/patterns/points.py`

Debug export of the Poisson-disk seed cloud on the selected outer surface.

```bash
.venv/bin/python gallery/patterns/points.py gallery/shapes/CurvyLamp.stl --show
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between seeds (mm) |
| `--spacing-metric` | `surface` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--marker-radius` | `1.5` | Radius of seed marker spheres (mm) |
| `--seed` | `42` | Random seed for point scattering |
| `--show` | off | Open an interactive preview window after export |

---

### `gallery/patterns/voronoi.py`

Debug export of local Voronoi cell boundaries on the mesh surface.

```bash
.venv/bin/python gallery/patterns/voronoi.py gallery/shapes/CurvyLamp.stl --show
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between seeds (mm) |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--margin` | `10.0` | Added to typical seed spacing for the partner search sphere (mm) |
| `--perpendicular-half-length` | `50.0` | Bisector half-length used when clipping local Voronoi cells (mm) |
| `--marker-radius` | `1.5` | Radius of seed marker spheres (mm) |
| `--line-radius` | `0.25` | Radius of Voronoi outline tubes (mm) |
| `--line-lift` | `0.35` | Lift Voronoi outlines above the surface (mm) |
| `--seed` | `42` | Random seed for point scattering |
| `--show` | off | Open an interactive preview window after export |

---

### `gallery/patterns/voronoi_local.py`

Small local Voronoi bisector test (optionally limited to a subset of seeds).

```bash
.venv/bin/python gallery/patterns/voronoi_local.py gallery/shapes/CurvyLamp.stl --seed-count 12 --show
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between seeds (mm) |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--seed-count` | `0` | Number of seeds to test (`0` = all seeds) |
| `--margin` | `10.0` | Added to max nearest-neighbor distance for the partner search sphere (mm) |
| `--perpendicular-half-length` | `50.0` | Length of each perpendicular bisector side (mm) |
| `--marker-radius` | `1.5` | Radius of seed marker spheres (mm) |
| `--line-radius` | `0.22` | Radius of bisector / outline tubes (mm) |
| `--seed` | `42` | Random seed for point scattering |
| `--show` | off | Open an interactive preview window after export |

---

### `gallery/patterns/shrunk.py`

Debug export of inset (shrunk) local Voronoi boundaries used as cut outlines.

```bash
.venv/bin/python gallery/patterns/shrunk.py gallery/shapes/CurvyLamp.stl --gap 2 --show
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `10.0` | Minimum center-to-center spacing between seeds (mm) |
| `--gap` | `2.0` | Minimum distance between pebbles (mm); each cell shrinks by half this amount |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--margin` | `10.0` | Added to typical seed spacing for the partner search sphere (mm) |
| `--perpendicular-half-length` | `50.0` | Bisector half-length used when clipping local Voronoi cells (mm) |
| `--marker-radius` | `1.5` | Radius of seed marker spheres (mm) |
| `--outer-line-radius` | `0.18` | Radius of the original Voronoi outline tubes (mm) |
| `--shrunk-line-radius` | `0.28` | Radius of the shrunk boundary tubes (mm) |
| `--no-outer` | off | Hide the original Voronoi tessellation outlines |
| `--seed` | `42` | Random seed for point scattering |
| `--show` | off | Open an interactive preview window after export |

---

### `gallery/patterns/twinkle.py`

Tiny cylindrical bores aimed at a shared axial light source. As an observer
moves around the shade, different holes line up with the light.

```bash
.venv/bin/python gallery/patterns/twinkle.py gallery/shapes/CurvyLamp.stl \
  --min-spacing 6 --hole-radius 0.8 --light-source-offset 30
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `6.0` | Minimum center-to-center spacing between holes (mm) |
| `--hole-radius` | `0.8` | Radius of each cylindrical bore (mm) |
| `--light-source-offset` | `30.0` | Distance below the top of the shade where the light sits on the central axis (mm) |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--seed` | `42` | Random seed for point scattering |
| `--cylinder-sections` | `16` | Number of radial segments used for each cylinder cutter |

---

### `gallery/patterns/reflector.py`

Square through-cuts oriented so light from an axial point source reflects off
one face of each hole and exits horizontally.

```bash
.venv/bin/python gallery/patterns/reflector.py gallery/shapes/CurvyLamp.stl \
  --min-spacing 8 --hole-size 1.5 --light-source-offset 30 --add-light-debug
```

| Argument | Default | Description |
| --- | --- | --- |
| `mesh` | required | Target mesh STL |
| `--min-spacing` | `8.0` | Minimum center-to-center spacing between holes (mm) |
| `--hole-size` | `1.5` | Side length of each square bore (mm) |
| `--light-source-offset` | `30.0` | Distance below the top of the shade where the light sits on the central axis (mm) |
| `--spacing-metric` | `euclidean` | `surface` or `euclidean` |
| `--bottom-border` | `5.0` | Unpatterned margin at the base (mm) |
| `--top-border` | `20.0` | Unpatterned margin at the top (mm) |
| `--seed` | `42` | Random seed for point scattering |
| `--light-marker-diameter` | `40.0` | Diameter of the red light-source debug sphere (mm) |
| `--add-light-debug` | off | Also export a perforated mesh copy with a red light-source sphere |
