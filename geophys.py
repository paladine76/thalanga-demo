#!/usr/bin/env python3
"""
The dipole-dipole IP and resistivity inversion, clipped to the regional scene.

    Dipole-Dipole_IP-Res_Inversion_Point_data.csv
        8,648 inversion points on 10 survey lines, GDA94 / MGA zone 55

Each row is one cell of an inverted section: a station along a line, an
elevation, and the two inverted properties. The regional page draws them as
vertical curtains beneath the lines, one property at a time.

    python geophys.py data/Dipole-Dipole_IP-Res_Inversion_Point_data.csv

prints the survey and, with --box, what survives a given scene.

------------------------------------------------------------------------------
THE SURVEY NEVER MOVES THE SCENE
------------------------------------------------------------------------------
The regional scene has no fixed axis ranges, so plotly fits it to whatever is
drawn. Two lines (1000ez and 16250ez) lie west of the drilling, and adding
them would widen the scene and shrink everything already in it. `clip()`
keeps only the points inside the scene box, which is the union of the
drilling and the ground surface. 6,026 of the 8,648 points are inside it.

------------------------------------------------------------------------------
WHAT IS NOT KNOWN ABOUT THIS FILE
------------------------------------------------------------------------------
The data is unpublished (McInnes D, EPM 12969). The unit of InvIP is not
stated in the file and is shown without one. ResErr and IPErr are not
documented and are not used.

LogRes is log10(InvRes) rounded to two decimals, so the page colours by
log10(InvRes) computed here. `load()` checks the two agree to within 0.01, so
a file whose LogRes means something else stops the build.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("Line", "Stn", "GridE", "GridN", "Elev", "InvRes", "LogRes", "InvIP")

# Colour limits are the 2nd and 98th percentiles of the points drawn, so a few
# extreme cells do not push the rest of the sections into one shade.
PERCENTILES = (2, 98)

# Resistivity colour-bar ticks are taken from this ladder, within the limits.
RES_TICKS = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)


@dataclass
class IPSurvey:
    """The inversion points inside one scene, and what the page says about them."""

    points: pd.DataFrame          # the points drawn, in file order
    total: int                    # points in the file
    lines: tuple[str, ...]        # every line in the file, in file order
    outside: tuple[str, ...]      # lines with no point inside the scene

    @property
    def chargeability(self) -> np.ndarray:
        return self.points["InvIP"].to_numpy(dtype=float)

    @property
    def log_resistivity(self) -> np.ndarray:
        return np.log10(self.points["InvRes"].to_numpy(dtype=float))

    def chargeability_limits(self) -> tuple[float, float]:
        lo, hi = np.percentile(self.chargeability, PERCENTILES)
        return float(lo), float(hi)

    def resistivity_limits(self) -> tuple[float, float]:
        """In log10 ohm-metres."""
        lo, hi = np.percentile(self.log_resistivity, PERCENTILES)
        return float(lo), float(hi)

    def resistivity_ticks(self) -> tuple[list[float], list[str]]:
        lo, hi = self.resistivity_limits()
        vals = [v for v in RES_TICKS if lo <= np.log10(v) <= hi]
        return [float(np.log10(v)) for v in vals], [f"{v:,}" for v in vals]

    def note(self) -> str:
        """The sentence under the layer switch."""
        s = (f"{len(self.points):,} of {self.total:,} inversion points fall "
             f"inside this scene.")
        if self.outside:
            names = (self.outside[0] if len(self.outside) == 1 else
                     ", ".join(self.outside[:-1]) + " and " + self.outside[-1])
            s += (f" Line{'s' if len(self.outside) > 1 else ''} {names} "
                  f"lie{'s' if len(self.outside) == 1 else ''} outside it.")
        return s


def load(path: str | Path) -> pd.DataFrame:
    """Read the inversion points and check the columns the page relies on."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column {', '.join(missing)}; "
                         f"it has {', '.join(df.columns)}")
    df["Line"] = df["Line"].astype(str)
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError(f"{path}: empty cells in {', '.join(REQUIRED)}")
    if (df["InvRes"] <= 0).any():
        raise ValueError(f"{path}: {(df['InvRes'] <= 0).sum()} resistivities "
                         "at or below zero cannot be drawn on a log scale")
    gap = np.abs(np.log10(df["InvRes"]) - df["LogRes"]).max()
    if gap > 0.01:
        raise ValueError(f"{path}: LogRes differs from log10(InvRes) by up to "
                         f"{gap:.3f}, so it is not the same quantity")
    return df


def clip(df: pd.DataFrame, extent) -> IPSurvey:
    """The points inside a scene.Extent, edges included."""
    inside = (df["GridE"].between(extent.xmin, extent.xmax)
              & df["GridN"].between(extent.ymin, extent.ymax)
              & df["Elev"].between(extent.zmin, extent.zmax))
    pts = df[inside].reset_index(drop=True)
    lines = tuple(dict.fromkeys(df["Line"]))
    kept = set(pts["Line"])
    return IPSurvey(points=pts, total=len(df), lines=lines,
                    outside=tuple(l for l in lines if l not in kept))


def main(argv=None) -> int:
    import argparse

    import scene as scn

    p = argparse.ArgumentParser(description="Inspect the IP-resistivity points.")
    p.add_argument("csv")
    p.add_argument("--box", nargs=6, type=float, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="report what falls inside this scene box")
    a = p.parse_args(argv)

    df = load(a.csv)
    print(f"{len(df):,} points on {df['Line'].nunique()} lines")
    for line, g in df.groupby("Line", sort=False):
        print(f"  {line:<9}{len(g):>6,}  E {g.GridE.min():,.0f}-{g.GridE.max():,.0f}"
              f"  N {g.GridN.min():,.0f}-{g.GridN.max():,.0f}"
              f"  RL {g.Elev.min():,.0f}-{g.Elev.max():,.0f}")
    if a.box:
        s = clip(df, scn.Extent(*a.box))
        lo, hi = s.chargeability_limits()
        rlo, rhi = s.resistivity_limits()
        print(f"\n{s.note()}")
        print(f"chargeability limits {lo:.2f}-{hi:.2f}")
        print(f"resistivity limits   {10 ** rlo:,.1f}-{10 ** rhi:,.0f} ohm.m,"
              f" ticks {', '.join(s.resistivity_ticks()[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
