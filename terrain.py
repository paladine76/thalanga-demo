#!/usr/bin/env python3
"""
The ground surface: load a gridded DEM, crop it to the area in view, cache it.

The DEM reaches this repo already gridded, inside the plotly HTML that
`dxf_surface_3d.py` writes. Re-deriving it from the 168 MB DXF on every build
would cost 40 seconds to reproduce a grid that has not changed, so the surface
is read back out of that file once and cached as a small npz.

    thalanga_3d.html   ->   terrain.npz   ->   cropped grid for the scene
         4.6 MB                 0.6 MB              124 x 74 cells

------------------------------------------------------------------------------
WHY THE CROP MATTERS
------------------------------------------------------------------------------
The DEM covers 25.9 km east-west by 15.9 km north-south. The drilling occupies
5.5 km by 2.9 km inside it, about 4% of that footprint. Drawn at full extent
the deposit is a smudge and the terrain is the subject, which inverts the point
of the figure.

Cropping costs resolution and the figure should say so. The source grid is
52 m cells, so the cropped window holds roughly 124 by 74 of them. That is
coarse for a 5 km view and it is the honest limit of what the circulated DEM
carries. Regenerating from the DXF over the drilling window alone would give
about 11 m cells; `--grid` in `dxf_surface_3d.py` is where that happens, not
here.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import scene


# --------------------------------------------------------------------------- #
# Reading a plotly HTML
# --------------------------------------------------------------------------- #

def _to_array(v) -> np.ndarray:
    """Decode a plotly value that may be a base64 typed array."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=v.get("dtype", "f8"))
        shape = v.get("shape")
        if shape:
            a = a.reshape([int(s) for s in str(shape).split(",")])
        return a
    return np.asarray(v)


def _extract_figure(path: Path) -> dict:
    """Pull {'data', 'layout'} out of a plotly `write_html` file.

    The embedded plotly.js bundle quotes 'Plotly.newPlot(' inside its own error
    messages, so the real call is matched by its div UUID argument rather than
    by the function name.
    """
    html = path.read_text(encoding="utf-8")
    hits = list(re.finditer(r'Plotly\.newPlot\(\s*"[0-9a-fA-F-]{36}"', html))
    if not hits:
        raise SystemExit(f"{path}: no plotly figure found")
    dec = json.JSONDecoder()
    i = html.index("(", hits[-1].start()) + 1
    args = []
    while len(args) < 3:
        while html[i] in " \t\r\n,":
            i += 1
        val, i = dec.raw_decode(html, i)
        args.append(val)
    return {"data": args[1], "layout": args[2]}


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #

@dataclass
class Terrain:
    """A gridded ground surface. `z` is (ny, nx), indexed [row, col] = [y, x]."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def cell_size(self) -> tuple[float, float]:
        return (float(np.diff(self.x).mean()), float(np.diff(self.y).mean()))

    @property
    def extent(self) -> scene.Extent:
        return scene.Extent(float(self.x.min()), float(self.x.max()),
                            float(self.y.min()), float(self.y.max()),
                            float(np.nanmin(self.z)), float(np.nanmax(self.z)))

    @property
    def gaps(self) -> int:
        return int(np.isnan(self.z).sum())

    def crop(self, xlim: tuple[float, float],
             ylim: tuple[float, float]) -> "Terrain":
        """Clip to an easting/northing window, keeping whole cells."""
        xs = (self.x >= xlim[0]) & (self.x <= xlim[1])
        ys = (self.y >= ylim[0]) & (self.y <= ylim[1])
        if xs.sum() < 2 or ys.sum() < 2:
            raise SystemExit(
                f"crop window {xlim} x {ylim} does not overlap the DEM "
                f"({self.x.min():.0f}–{self.x.max():.0f} x "
                f"{self.y.min():.0f}–{self.y.max():.0f})")
        return Terrain(self.x[xs], self.y[ys], self.z[np.ix_(ys, xs)],
                       self.source)

    def crop_to(self, extent: scene.Extent, margin: float = 0.0) -> "Terrain":
        return self.crop((extent.xmin - margin, extent.xmax + margin),
                         (extent.ymin - margin, extent.ymax + margin))

    def describe(self) -> str:
        cx, cy = self.cell_size
        return (f"{self.shape[1]} x {self.shape[0]} cells at "
                f"{cx:.0f} x {cy:.0f} m, {self.extent.describe()}"
                + (f", {self.gaps:,} empty" if self.gaps else ""))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, x=self.x, y=self.y,
                            z=self.z.astype(np.float32), source=self.source)
        return path


def from_plotly_html(path: str | Path) -> Terrain:
    """Read the surface trace back out of a figure written by dxf_surface_3d."""
    path = Path(path)
    fig = _extract_figure(path)
    surfaces = [t for t in fig["data"] if t.get("type") == "surface"]
    if not surfaces:
        raise SystemExit(f"{path}: no surface trace in the figure")
    s = surfaces[0]
    return Terrain(
        x=_to_array(s["x"]).astype(float).ravel(),
        y=_to_array(s["y"]).astype(float).ravel(),
        z=_to_array(s["z"]).astype(float),
        source=path.name,
    )


def from_npz(path: str | Path) -> Terrain:
    d = np.load(path)
    return Terrain(x=d["x"], y=d["y"], z=d["z"].astype(float),
                   source=str(d["source"]) if "source" in d else str(path))


def load(path: str | Path, cache: str | Path | None = None) -> Terrain:
    """Load a DEM from either form, caching the HTML case for later builds.

    A cache older than its source is ignored rather than trusted, so a
    regenerated DEM is picked up without anyone remembering to delete anything.
    """
    path = Path(path)
    if path.suffix == ".npz":
        return from_npz(path)

    if cache:
        cache = Path(cache)
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            t = from_npz(cache)
            print(f"  terrain      {t.describe()}  (cached)")
            return t

    t = from_plotly_html(path)
    if cache:
        t.save(cache)
        print(f"  terrain      cached to {cache}")
    return t


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect or cache the ground surface.")
    p.add_argument("dem", help="thalanga_3d.html from dxf_surface_3d, or a .npz")
    p.add_argument("--cache", default=None, help="write/read a .npz cache here")
    p.add_argument("--crop", nargs=4, type=float, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="report the grid that survives this window")
    a = p.parse_args(argv)

    t = load(a.dem, a.cache)
    print(f"source       {t.source}")
    print(f"full grid    {t.describe()}")

    if a.crop:
        c = t.crop((a.crop[0], a.crop[1]), (a.crop[2], a.crop[3]))
        print(f"cropped      {c.describe()}")
        kept = c.z.size / t.z.size
        print(f"             {kept * 100:.1f}% of the cells, "
              f"{c.extent.dx / max(c.shape[1] - 1, 1):.0f} m across the view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
