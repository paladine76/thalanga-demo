#!/usr/bin/env python3
"""
Shared 3D scene: extent, vertical exaggeration, camera, and the flight between
the regional view and the blind panel.

Both pages draw the same ground in the same coordinate system, and both need to
say honestly how much they have stretched the vertical. This module owns that
arithmetic so neither page can state one factor while drawing another.

------------------------------------------------------------------------------
EXAGGERATION IS COMPUTED, NEVER ASSERTED
------------------------------------------------------------------------------
Plotly's `aspectratio` sets the drawn length of each axis, so the exaggeration
actually applied is

    (z_length / z_range) / (x_length / x_range)

`aspect_for()` derives all three lengths from one scale factor, which makes
that ratio exactly the number requested. `applied_exaggeration()` recovers it
from a finished layout, and `verify()` checks the two agree. The pipeline's own
`figures_3d._layout` hardcodes x=2.0, y=1.0, z=0.45 and labels every figure
"x3" while applying about 1.1 -- that is the failure this guards against.

------------------------------------------------------------------------------
TWO SCENES, TWO FACTORS
------------------------------------------------------------------------------
The regional view spans 6,392 m east-west against 1,151 m of relief and needs
stretching to read at all. The blind panel is 548 m by 328 m by 960 m deep --
taller than it is wide -- and needs none.

Flying between them without changing the factor would show the panel distorted
on arrival; changing it abruptly would make the orebody visibly deform at the
moment of transition. `flight()` interpolates both camera and exaggeration, so
the body relaxes into true proportion as the viewer arrives. That second of
animation is the clearest statement in the demo that the regional view was
stretched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import palette

# The regional view draws relief at this fraction of the site width. Chosen so
# the terrain reads as terrain; stated in the title either way.
RELIEF_FRACTION = 0.35

# The panel is already tall relative to its footprint, so it is drawn true.
PANEL_EXAGGERATION = 1.0

# Eye vector before zoom, inherited from figures_3d.py so the demo opens on the
# view the team already recognises.
BASE_EYE = (1.6, -1.7, 0.9)
REGIONAL_ZOOM = 1.8        # beyond about 2.0 the easting labels leave the canvas
PANEL_ZOOM = 1.2        # framed on the trimmed extent, which is roughly cubic


# --------------------------------------------------------------------------- #
# Extent
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Extent:
    """The bounding box of everything a scene must contain, in metres."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def around(cls, x, y, z, margin: float = 0.0) -> "Extent":
        return cls(float(np.min(x)) - margin, float(np.max(x)) + margin,
                   float(np.min(y)) - margin, float(np.max(y)) + margin,
                   float(np.min(z)), float(np.max(z)))

    @classmethod
    def from_box(cls, box: dict, zmin: float, zmax: float,
                 margin: float = 0.0) -> "Extent":
        return cls(box["x_min"] - margin, box["x_max"] + margin,
                   box["y_min"] - margin, box["y_max"] + margin, zmin, zmax)

    def union(self, other: "Extent") -> "Extent":
        return Extent(min(self.xmin, other.xmin), max(self.xmax, other.xmax),
                      min(self.ymin, other.ymin), max(self.ymax, other.ymax),
                      min(self.zmin, other.zmin), max(self.zmax, other.zmax))

    @property
    def dx(self) -> float:
        return max(self.xmax - self.xmin, 1e-9)

    @property
    def dy(self) -> float:
        return max(self.ymax - self.ymin, 1e-9)

    @property
    def dz(self) -> float:
        return max(self.zmax - self.zmin, 1e-9)

    @property
    def centre(self) -> tuple[float, float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2,
                (self.zmin + self.zmax) / 2)

    def describe(self) -> str:
        return (f"{self.dx:,.0f} m E x {self.dy:,.0f} m N x "
                f"{self.dz:,.0f} m vertical")


def auto_exaggeration(extent: Extent,
                      relief_fraction: float = RELIEF_FRACTION) -> float:
    """The factor that draws relief at `relief_fraction` of the site width."""
    return relief_fraction * max(extent.dx, extent.dy) / extent.dz


# --------------------------------------------------------------------------- #
# Aspect and camera
# --------------------------------------------------------------------------- #

def aspect_for(extent: Extent, exaggeration: float) -> dict:
    """Axis lengths with x and y true to each other and z stretched.

    Deriving all three from one scale factor is what makes the applied
    exaggeration equal the requested one.
    """
    s = max(extent.dx, extent.dy)
    return dict(x=extent.dx / s, y=extent.dy / s,
                z=(extent.dz / s) * exaggeration)


def applied_exaggeration(aspect: dict, extent: Extent) -> float:
    """Recover the exaggeration a finished layout actually draws."""
    return (aspect["z"] / extent.dz) / (aspect["x"] / extent.dx)


def verify(aspect: dict, extent: Extent, claimed: float,
           tol: float = 1e-6) -> None:
    """Raise if a layout would draw a different factor than it states."""
    got = applied_exaggeration(aspect, extent)
    if abs(got - claimed) > tol * max(1.0, claimed):
        raise ValueError(
            f"scene would draw {got:.4f}x while claiming {claimed:.4f}x")


def camera(zoom: float = REGIONAL_ZOOM, centre=(0.0, 0.0, 0.0)) -> dict:
    """Eye vector scaled by zoom, looking at a point in scene coordinates."""
    return dict(eye=dict(x=BASE_EYE[0] / zoom, y=BASE_EYE[1] / zoom,
                         z=BASE_EYE[2] / zoom),
                center=dict(x=centre[0], y=centre[1], z=centre[2]))


def to_scene_coords(point, extent: Extent, aspect: dict) -> tuple[float, float, float]:
    """Map a point in metres to plotly's scene coordinates.

    Plotly places the scene box at the origin with the side lengths given by
    `aspectratio`, so a data value maps to its offset from the axis midpoint,
    expressed as a fraction of the axis range, times that side length.

    This is what lets the camera centre on the blind panel rather than on the
    middle of the DEM.
    """
    cx, cy, cz = extent.centre
    return (
        (point[0] - cx) / extent.dx * aspect["x"],
        (point[1] - cy) / extent.dy * aspect["y"],
        (point[2] - cz) / extent.dz * aspect["z"],
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def scene_layout(extent: Extent, exaggeration: float,
                 zoom: float = REGIONAL_ZOOM, centre=(0.0, 0.0, 0.0)) -> dict:
    """The plotly `scene` block, with the aspect verified against the claim."""
    aspect = aspect_for(extent, exaggeration)
    verify(aspect, extent, exaggeration)
    def axis(label: str) -> dict:
        return dict(
            title=dict(text=label, font=dict(size=12, color=palette.MUTED)),
            showbackground=True, backgroundcolor=palette.PANEL_BG,
            gridcolor=palette.RULE, zeroline=False, color=palette.MUTED,
        )

    return dict(
        xaxis=axis("Easting (m, MGA94 Z55)"),
        yaxis=axis("Northing (m)"),
        zaxis=axis("Elevation (m RL)"),
        aspectmode="manual",
        aspectratio=aspect,
        camera=camera(zoom, centre),
    )


def figure_layout(title: str, subtitle: str, extent: Extent,
                  exaggeration: float, zoom: float = REGIONAL_ZOOM,
                  height: int = 820, centre=(0.0, 0.0, 0.0)) -> dict:
    """Page-consistent figure layout. The subtitle always carries the factor."""
    return dict(
        title=dict(
            text=(f"{title}<br><sub>{subtitle}; vertical exaggeration "
                  f"x{exaggeration:.1f} — not true scale</sub>"),
            x=0.5, xanchor="center",
            font=dict(size=15, color=palette.INK, family=palette.FONT_SANS),
        ),
        scene=scene_layout(extent, exaggeration, zoom, centre),
        paper_bgcolor=palette.PAPER,
        font=dict(family=palette.FONT_SANS, color=palette.INK, size=12),
        margin=dict(l=0, r=0, t=96, b=0),
        # Below the three button rows the regional figure stacks at 0.99, 0.93
        # and 0.87, so the legend clears them when both are on screen.
        legend=dict(yanchor="top", y=0.80, xanchor="left", x=0.0,
                    itemsizing="constant", font=dict(size=13),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=palette.RULE, borderwidth=1),
        height=height,
    )


# --------------------------------------------------------------------------- #
# The flight
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Flight:
    """Where the camera is going, for the shell script to animate toward.

    The starting point is wherever the viewer has rotated to, so it is read in
    the browser at click time. Only the destination is decided here.
    """

    centre: tuple[float, float, float]      # scene coordinates of the panel
    eye: tuple[float, float, float]
    aspect_from: dict
    aspect_to: dict
    exaggeration_from: float
    exaggeration_to: float
    frames: int
    duration_ms: int

    def as_dict(self) -> dict:
        return dict(
            centre=dict(zip("xyz", self.centre)),
            eye=dict(zip("xyz", self.eye)),
            aspectFrom=self.aspect_from, aspectTo=self.aspect_to,
            exaggerationFrom=self.exaggeration_from,
            exaggerationTo=self.exaggeration_to,
            frames=self.frames, durationMs=self.duration_ms,
        )


def flight(regional: Extent, panel_box: dict, regional_exaggeration: float,
           panel_exaggeration: float = PANEL_EXAGGERATION,
           zoom: float = 3.2, frames: int = 36,
           duration_ms: int = 1100) -> Flight:
    """Plan the move from the regional view into the blind panel.

    The camera ends centred on the panel at close range, with the vertical
    relaxed toward true scale. Holding the eye direction constant means the
    viewer keeps their bearings: the ground does not spin, it approaches.
    """
    aspect_from = aspect_for(regional, regional_exaggeration)
    aspect_to = aspect_for(regional, panel_exaggeration)
    box_centre = ((panel_box["x_min"] + panel_box["x_max"]) / 2,
                  (panel_box["y_min"] + panel_box["y_max"]) / 2,
                  regional.centre[2])
    centre = to_scene_coords(box_centre, regional, aspect_to)
    return Flight(
        centre=centre,
        eye=(BASE_EYE[0] / zoom + centre[0],
             BASE_EYE[1] / zoom + centre[1],
             BASE_EYE[2] / zoom + centre[2]),
        aspect_from=aspect_from, aspect_to=aspect_to,
        exaggeration_from=regional_exaggeration,
        exaggeration_to=panel_exaggeration,
        frames=frames, duration_ms=duration_ms,
    )


def camera_to_hash(cam: dict) -> str:
    """Encode a camera for the URL, so it survives the jump between pages.

    A hash rather than storage or a fetch: it is the only channel that works
    identically when the pages are opened from the filesystem.
    """
    e, c = cam["eye"], cam.get("center", {"x": 0, "y": 0, "z": 0})
    return ("#eye=" + ",".join(f"{e[k]:.4f}" for k in "xyz")
            + "&center=" + ",".join(f"{c[k]:.4f}" for k in "xyz"))


def main(argv=None) -> int:
    """Print both scenes and confirm each draws what it claims."""
    regional = Extent(368_676, 374_130, 7_749_583, 7_752_485, -741, 411)
    panel_box = {"x_min": 372_049.79, "x_max": 372_597.87,
                 "y_min": 7_750_115.94, "y_max": 7_750_443.92}
    panel = Extent.from_box(panel_box, -634.5, 325.5)

    for name, ext, exg in (("regional", regional, auto_exaggeration(regional)),
                           ("blind panel", panel, PANEL_EXAGGERATION)):
        aspect = aspect_for(ext, exg)
        verify(aspect, ext, exg)
        print(f"{name:<12} {ext.describe()}")
        print(f"{'':<12} aspect x={aspect['x']:.3f} y={aspect['y']:.3f} "
              f"z={aspect['z']:.3f}")
        print(f"{'':<12} requested x{exg:.2f}  applied "
              f"x{applied_exaggeration(aspect, ext):.2f}\n")

    f = flight(regional, panel_box, auto_exaggeration(regional))
    print(f"flight  centre {tuple(round(v, 3) for v in f.centre)}  "
          f"eye {tuple(round(v, 3) for v in f.eye)}")
    print(f"        exaggeration x{f.exaggeration_from:.2f} -> "
          f"x{f.exaggeration_to:.2f} over {f.duration_ms} ms")

    print("\nthe failure this guards against:")
    bad = dict(x=2.0, y=1.0, z=0.45)
    print(f"  figures_3d._layout on these extents claims x3.00, applies "
          f"x{applied_exaggeration(bad, regional):.2f}")
    try:
        verify(bad, regional, 3.0)
    except ValueError as exc:
        print(f"  verify() rejects it: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
