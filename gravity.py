#!/usr/bin/env python3
"""
The 1982 ground gravity survey, draped over the terrain inside the scene.

    Gravity_1982_Grid_Point_data.csv
        46,068 cells of 18 gridded products at 20 m, GDA94 / MGA zone 55

The CSV is 18 Geosoft grids of the 1982 survey as reprocessed by McInnes
(2021), "Thalanga gravity data processing", one row per grid cell: GridE,
GridN, then one column per grid, named after its file. The regional page
offers three of them as a Gravity layer.

    python gravity.py data/Gravity_1982_Grid_Point_data.csv

prints the products and, with --box, what survives a given scene.

------------------------------------------------------------------------------
WHAT THE THREE PRODUCTS ARE
------------------------------------------------------------------------------
All three come from the 90 m stream of the reprocessed 1982 data:

    Residual    Thal_EW_2nd_DS_0090p0           filtered gravity
    Band pass   Thal_EW_2nd_DS_0090p0_Bp        a further band pass
    Tilt        Thal_EW_2nd_DS_0090p0_Bp_Tilt   tilt angle, degrees

The grids carry no units. The first two are read as mGal, the unit of the
memo's survey values; the tilt is an angle by definition. "0090p0" is read
as a 90.0 m filter wavelength. These readings come from the file names and
the memo, not from a statement in the files.

------------------------------------------------------------------------------
THE SURVEY NEVER MOVES THE SCENE
------------------------------------------------------------------------------
The survey covers the deposit: nine tenths of its cells fall inside the
scene and lie over all but 75 of the 77,220 drill positions. `drape()` keeps
the cells inside the scene box and lifts each one a few metres above the
ground surface, never above the top of the scene, so the layer cannot change
the scene's extent.

The 2021 survey, a strip west of the deposit, is not drawn here. Where the
two overlap they correlate at 0.06, and their mGal grids differ in spread by
a factor of four to ten, so they are not merged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The products the page offers, in switch order: key, column, label, unit,
# and whether the colour scale is a fixed angle range.
PRODUCTS = (
    ("residual", "Thal_EW_2nd_DS_0090p0", "Residual", "mGal", False),
    ("bandpass", "Thal_EW_2nd_DS_0090p0_Bp", "Band pass", "mGal", False),
    ("tilt", "Thal_EW_2nd_DS_0090p0_Bp_Tilt", "Tilt", "°", True),
)

# Height of the drape above the ground surface, in metres. Enough to sit
# clear of the contour lines; small against the 1,151 m of the scene.
LIFT = 4.0

# mGal products are coloured symmetrically about zero, out to the 98th
# percentile of the magnitude drawn, so white means no residual.
PERCENTILE = 98


@dataclass
class Product:
    key: str
    column: str
    label: str
    unit: str
    values: np.ndarray        # (ny, nx), NaN where the survey has no cell
    limit: float              # colour scale runs -limit to +limit
    inside: int               # cells inside the scene
    total: int                # cells in the whole survey

    def note(self) -> str:
        return (f"1982 survey: {self.inside:,} of {self.total:,} grid cells "
                f"fall inside this scene.")


@dataclass
class Drape:
    """The in-scene part of the survey on a regular grid over the terrain."""

    x: np.ndarray             # (nx,) eastings of cell centres
    y: np.ndarray             # (ny,) northings
    z: np.ndarray             # (ny, nx) ground + LIFT, NaN outside coverage
    products: list[Product]


def load(path: str | Path) -> pd.DataFrame:
    """Read the CSV and check the columns the page relies on."""
    df = pd.read_csv(path)
    need = ["GridE", "GridN"] + [p[1] for p in PRODUCTS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column {', '.join(missing)}")
    step = np.diff(np.unique(df["GridE"]))
    if len(step) == 0 or not np.allclose(step, step.min()):
        raise ValueError(f"{path}: GridE is not on a regular spacing")
    return df


def drape(df: pd.DataFrame, extent, ground) -> Drape:
    """Grid the in-scene cells and lift them onto the terrain.

    `extent` is the regional scene.Extent; `ground` is the cropped
    terrain.Terrain the scene was built from, so the drape follows the same
    surface the wireframe draws.
    """
    from scipy.interpolate import RegularGridInterpolator

    d = float(np.diff(np.unique(df["GridE"])).min())
    inside = (df["GridE"].between(extent.xmin, extent.xmax)
              & df["GridN"].between(extent.ymin, extent.ymax))
    sub = df[inside]
    x = np.arange(sub["GridE"].min(), sub["GridE"].max() + d / 2, d)
    y = np.arange(sub["GridN"].min(), sub["GridN"].max() + d / 2, d)
    col = np.rint((sub["GridE"] - x[0]) / d).astype(int).to_numpy()
    row = np.rint((sub["GridN"] - y[0]) / d).astype(int).to_numpy()

    products = []
    for key, column, label, unit, angle in PRODUCTS:
        v = np.full((len(y), len(x)), np.nan)
        v[row, col] = sub[column].to_numpy(dtype=float)
        drawn = v[np.isfinite(v)]
        limit = 90.0 if angle else float(np.percentile(np.abs(drawn), PERCENTILE))
        products.append(Product(key, column, label, unit, v, limit,
                                inside=int(np.isfinite(v).sum()),
                                total=int(df[column].notna().sum())))

    surface = RegularGridInterpolator((ground.y, ground.x), ground.z,
                                      bounds_error=False, fill_value=np.nan)
    XX, YY = np.meshgrid(x, y)
    zg = surface(np.column_stack([np.clip(YY.ravel(), ground.y.min(), ground.y.max()),
                                  np.clip(XX.ravel(), ground.x.min(), ground.x.max())]))
    # The page carries z to one decimal, so the cap is the scene top rounded
    # down to one decimal; capping at the exact top and rounding afterwards
    # lifts the highest cell 5 cm above the scene and widens it.
    z = np.minimum(zg.reshape(XX.shape) + LIFT, np.floor(extent.zmax * 10) / 10)
    covered = np.logical_or.reduce([np.isfinite(p.values) for p in products])
    z[~covered] = np.nan
    return Drape(x=x, y=y, z=z, products=products)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Inspect the 1982 gravity grids.")
    p.add_argument("csv")
    p.add_argument("--box", nargs=4, type=float, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="report what falls inside this scene footprint")
    a = p.parse_args(argv)

    df = load(a.csv)
    print(f"{len(df):,} cells, E {df.GridE.min():,}-{df.GridE.max():,}, "
          f"N {df.GridN.min():,}-{df.GridN.max():,}")
    for key, column, label, unit, _ in PRODUCTS:
        v = df[column].dropna()
        print(f"  {label:<16}{column:<34}{len(v):>7,} cells  "
              f"{v.min():+.4g} to {v.max():+.4g} {unit}")
    if a.box:
        x0, x1, y0, y1 = a.box
        inside = df.GridE.between(x0, x1) & df.GridN.between(y0, y1)
        print(f"\ninside the box: {inside.sum():,} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
