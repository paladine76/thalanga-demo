#!/usr/bin/env python3
"""
The deposit and the blind panel, as one rotatable scene.

Acts 1 and 2 share a figure rather than being two figures shown in turn. The
ground, the samples and the camera are the same in both; only the colouring
changes. Switching acts therefore keeps the viewer's bearings, and the move
into the panel starts from wherever they have rotated to.

    act 1   topography over 77,220 assay positions, one metal at a time,
            shaded pale to dark by grade
    act 2   the same positions recoloured: informing data in grey, the blind
            panel in pink, and the panel drawn as a volume you can click

------------------------------------------------------------------------------
THE BOX IS THE DOOR
------------------------------------------------------------------------------
The blind panel is drawn twice: a translucent Mesh3d volume and a line
wireframe over it. The wireframe reads better, the mesh is the click target --
plotly fires click events on a mesh face, and thin lines are nearly impossible
to hit in a rotated 3D scene.

Clicking it starts the flight into the panel page. The same move is on a
button, because a viewer who has not realised the box is live will look for one.

------------------------------------------------------------------------------
WHAT THE SURFACE DOES
------------------------------------------------------------------------------
A solid ground hides every sample beneath it, so the figure opens as wireframe
and the Filled button puts the ground back. The wireframe is drawn in one grey
rather than coloured by elevation: the terrain ramp's greens sit on top of the
zinc ramp's greens, and two green things in one scene read as one thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

import palette
import scene as scn
import terrain as terrain_mod

# Act 2 shows the shape of what was withheld, not every informing sample. The
# full set is 64,235 positions and drawing them all costs 5 MB to make a grey
# cloud slightly denser. The count drawn is stated in the legend.
MAX_INFORMING = 40_000


# --------------------------------------------------------------------------- #
# Trace bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class Traces:
    """Which trace index holds what, so nothing downstream counts by hand."""

    dem: int = 0
    metals: dict[str, int] = field(default_factory=dict)
    informing: int = -1
    panel: int = -1
    box_mesh: int = -1
    box_lines: int = -1

    @property
    def metal_indices(self) -> list[int]:
        return list(self.metals.values())

    @property
    def act2_indices(self) -> list[int]:
        return [self.informing, self.panel, self.box_mesh, self.box_lines]

    def as_dict(self) -> dict:
        return dict(dem=self.dem, metals=self.metals,
                    informing=self.informing, panel=self.panel,
                    boxMesh=self.box_mesh, boxLines=self.box_lines)


# --------------------------------------------------------------------------- #
# The ground
# --------------------------------------------------------------------------- #

def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> plotly restyle paths, e.g. {'contours.z.show': True}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def _surface_modes(t: terrain_mod.Terrain, contours: int = 20) -> tuple[dict, dict]:
    """Attributes for the filled and wireframe states of the ground.

    Both are built from one place so the initial trace and the toggle cannot
    describe different things. Wireframe is plotly's `hidesurface`: the filled
    quads are dropped and the contour lines left, which is a transparent
    surface rather than a separately drawn mesh.
    """
    zmin, zmax = float(np.nanmin(t.z)), float(np.nanmax(t.z))
    step = max((zmax - zmin) / max(contours, 1), 1e-9)

    def mode(wire: bool) -> dict:
        return {
            "hidesurface": wire,
            "contours": {"z": {
                "show": True, "start": zmin, "end": zmax, "size": step,
                "usecolormap": False,
                "color": palette.GROUND if wire else "rgba(0,0,0,0.30)",
                "width": 2 if wire else 1,
            }},
        }

    return mode(False), mode(True)


def dem_trace(t: terrain_mod.Terrain, wireframe: bool = True) -> tuple[go.Surface, dict, dict]:
    filled, wire = _surface_modes(t)
    zmin, zmax = float(np.nanmin(t.z)), float(np.nanmax(t.z))
    surface = go.Surface(
        x=t.x, y=t.y, z=t.z.astype(np.float32),
        colorscale=palette.land_ramp(),
        cmin=zmin, cmax=zmax,
        name="ground",
        showlegend=False,
        # The wireframe is one grey, so an elevation ramp beside it would
        # describe a colouring that is not on screen. It follows the fill mode.
        showscale=not wireframe,
        colorbar=dict(title="Elevation<br>(m RL)", thickness=13, len=0.42,
                      x=1.01, y=0.95, yanchor="top",
                      tickfont=dict(size=10)),
        hovertemplate="%{x:.0f}E %{y:.0f}N %{z:.0f}mRL<extra>ground</extra>",
        lighting=dict(ambient=0.74, diffuse=0.78, specular=0.06, roughness=0.9),
        lightposition=dict(x=-1e5, y=-1e5, z=1e5),
        **(wire if wireframe else filled),
    )
    return surface, filled, wire


# --------------------------------------------------------------------------- #
# Act 1: grade
# --------------------------------------------------------------------------- #

def metal_cloud(df, element: str, elements, visible: bool,
                size: float = 2.2, opacity: float = 0.75) -> go.Scatter3d:
    """Every assay position, shaded pale to dark by this metal's grade.

    Grade spans five orders of magnitude, so colour runs on log10. Zero and
    trace assays clamp to the floor rather than being dropped: barren ground is
    a measurement, and a hole that stops being drawn looks like a hole that was
    never drilled.
    """
    lo, hi = palette.grade_bounds()
    ticks, tick_text = palette.grade_ticks()
    grade = df[f"{element}_pct"].to_numpy(dtype=float)
    cvals = np.log10(np.clip(grade, palette.GRADE_FLOOR, palette.GRADE_CEILING))
    custom = np.column_stack(
        [df[f"{e}_pct"].to_numpy(dtype=np.float32) for e in elements])
    hover = "<br>".join(
        [(f"<b>{e}</b>" if e == element else e) + f" %{{customdata[{i}]:.3f}}%"
         for i, e in enumerate(elements)]
        + ["%{x:.0f}E %{y:.0f}N %{z:.0f}mRL"]) + "<extra></extra>"

    return go.Scatter3d(
        x=df["X"].to_numpy(dtype=float),
        y=df["Y"].to_numpy(dtype=float),
        z=df["Z"].to_numpy(dtype=np.float32),
        mode="markers",
        marker=dict(
            size=size, color=cvals.astype(np.float32),
            colorscale=palette.grade_ramp(element), cmin=lo, cmax=hi,
            opacity=opacity,
            # Scatter3d markers carry a white outline by default. Below about
            # 5 px it covers most of the marker and a dense cloud renders as a
            # white mass instead of its own colour.
            line=dict(width=0),
            showscale=True,
            colorbar=dict(title=f"{element} %", thickness=13, len=0.42,
                          x=1.01, y=0.45, yanchor="top",
                          tickvals=ticks, ticktext=tick_text,
                          tickfont=dict(size=10)),
        ),
        name=element, showlegend=False, visible=visible,
        customdata=custom, hovertemplate=hover,
    )


# --------------------------------------------------------------------------- #
# Act 2: the blind test
# --------------------------------------------------------------------------- #

def split_clouds(df, max_informing: int = MAX_INFORMING,
                 seed: int = 0, size: float = 1.8) -> list[go.Scatter3d]:
    """Informing data in grey, the blind panel in its own colour.

    The panel is drawn complete; it is the subject. The informing data is
    thinned, because its job here is to show the shape of the ground that was
    used, and the legend says how many of it are drawn.
    """
    rng = np.random.default_rng(seed)
    out = []
    for label, colour, opacity in (("informing", palette.INFORMING, 0.45),
                                   ("blind panel", palette.BLIND_PANEL, 0.85)):
        sub = df[df["split"] == ("holdout" if label == "blind panel" else "train")]
        total = len(sub)
        if label == "informing" and total > max_informing:
            sub = sub.iloc[rng.choice(total, max_informing, replace=False)]
        name = (f"{label} ({total:,})" if len(sub) == total
                else f"{label} ({len(sub):,} of {total:,} drawn)")
        out.append(go.Scatter3d(
            x=sub["X"].to_numpy(dtype=float),
            y=sub["Y"].to_numpy(dtype=float),
            z=sub["Z"].to_numpy(dtype=np.float32),
            mode="markers",
            marker=dict(size=size, color=colour, opacity=opacity,
                        line=dict(width=0)),
            name=name, showlegend=True, visible=False,
            hovertemplate=f"{label}<br>%{{x:.0f}}E %{{y:.0f}}N %{{z:.0f}}mRL"
                          "<extra></extra>",
        ))
    return out


def box_traces(box: dict, zmin: float, zmax: float) -> list:
    """The panel as a clickable volume plus a wireframe over it.

    Testing note: plotly's gl3d resolves its picking buffer from pointer
    movement, so a synthetic click that jumps straight to a position and
    presses fires nothing. Move the pointer there first, then press and
    release with a short gap. A real user does this without trying.
    """
    x0, x1 = box["x_min"], box["x_max"]
    y0, y1 = box["y_min"], box["y_max"]
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [zmin] * 4 + [zmax] * 4
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    mesh = go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
        color=palette.BLIND_PANEL, opacity=0.12, flatshading=True,
        hovertemplate="blind panel<br>click to see the estimate<extra></extra>",
        name="panel volume", showlegend=False, visible=False,
    )
    # Bottom ring, top ring, then the four verticals; None breaks the line.
    lx = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None,
          x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
    ly = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None,
          y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
    lz = [zmin] * 5 + [None] + [zmax] * 5 + [None] + \
         [zmin, zmax, None, zmin, zmax, None, zmin, zmax, None, zmin, zmax]
    lines = go.Scatter3d(
        x=lx, y=ly, z=lz, mode="lines",
        line=dict(color=palette.BLIND_PANEL, width=4),
        name="panel outline", showlegend=False, visible=False,
        hoverinfo="skip",
    )
    return [mesh, lines]


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #

def _buttons(label_pairs, x: float, y: float, active: int = 0) -> dict:
    return dict(type="buttons", direction="right",
                x=x, y=y, xanchor="left", yanchor="top",
                showactive=True, active=active,
                pad=dict(l=4, r=4, t=2, b=2),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=palette.RULE, borderwidth=1,
                font=dict(size=12, family=palette.FONT_SANS),
                buttons=[dict(label=l, method="restyle", args=a)
                         for l, a in label_pairs])


def controls(idx: Traces, filled: dict, wire: dict,
             metals: list[str]) -> list[dict]:
    """Act, fill and metal switches.

    These exist so the figure stands up on its own during development. In the
    assembled site the shell drives the same restyles from its own stepper and
    hides this row.
    """
    flat = flatten
    metal_idx = idx.metal_indices
    act_idx = metal_idx + idx.act2_indices

    def act(deposit: bool, metal: int = 0):
        vis = [deposit and i == metal for i in range(len(metal_idx))] \
              + [not deposit] * 4
        return [{"visible": vis}, act_idx]

    return [
        _buttons([("Deposit", act(True)), ("Blind panel", act(False))],
                 x=0.0, y=0.99),
        _buttons([("Wireframe", [{k: [v] for k, v in flat(wire).items()}
                                 | {"showscale": [False]}, [idx.dem]]),
                  ("Filled", [{k: [v] for k, v in flat(filled).items()}
                              | {"showscale": [True]}, [idx.dem]])],
                 x=0.0, y=0.93),
        _buttons([(m, [{"visible": [i == n for i in range(len(metal_idx))]},
                       metal_idx]) for n, m in enumerate(metals)],
                 x=0.0, y=0.87),
    ]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build(data, ground: terrain_mod.Terrain, margin: float = 500.0,
          metals=None, max_informing: int = MAX_INFORMING,
          with_controls: bool = True) -> dict:
    """Both acts in one figure.

    Returns everything a page needs to drive it: the figure, which trace index
    holds what, the scene extent and exaggeration, and the two surface states
    already flattened into restyle paths.
    """
    metals = metals or list(data.sims) or ["Zn", "Pb", "Cu"]
    df = data.assays

    sample_extent = scn.Extent.around(df["X"], df["Y"], df["Z"])
    ground = ground.crop_to(sample_extent, margin)
    extent = sample_extent.union(ground.extent)
    exaggeration = scn.auto_exaggeration(extent)

    surface, filled, wire = dem_trace(ground, wireframe=True)
    idx = Traces(dem=0)
    traces = [surface]

    for n, m in enumerate(metals):
        idx.metals[m] = len(traces)
        traces.append(metal_cloud(df, m, metals, visible=(n == 0)))

    idx.informing, idx.panel = len(traces), len(traces) + 1
    traces += split_clouds(df, max_informing)
    idx.box_mesh, idx.box_lines = len(traces), len(traces) + 1
    traces += box_traces(data.box, extent.zmin, extent.zmax)

    fig = go.Figure(traces)
    fig.update_layout(**scn.figure_layout(
        title="Thalanga — the deposit and the blind panel",
        subtitle=(f"{len(df):,} assay positions; colour is grade on a log "
                  f"scale, one metal at a time"),
        extent=extent, exaggeration=exaggeration, zoom=scn.REGIONAL_ZOOM,
    ))
    if with_controls:
        fig.update_layout(updatemenus=controls(idx, filled, wire, metals))
    return {"figure": fig, "traces": idx, "extent": extent,
            "exaggeration": exaggeration, "metals": metals,
            "filled": flatten(filled), "wire": flatten(wire)}


def main(argv=None) -> int:
    import argparse
    import json
    import dataio

    p = argparse.ArgumentParser(description="Build the regional scene.")
    p.add_argument("--assays", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--sims", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--dem-cache", default=None)
    p.add_argument("--margin", type=float, default=500.0)
    p.add_argument("--max-informing", type=int, default=MAX_INFORMING)
    p.add_argument("-o", "--out", default="regional.html")
    a = p.parse_args(argv)

    data = dataio.load(a.assays, a.baseline, a.sims, a.manifest)
    ground = terrain_mod.load(a.dem, a.dem_cache)
    r = build(data, ground, a.margin, max_informing=a.max_informing)
    fig, idx, extent, exg = (r["figure"], r["traces"], r["extent"],
                             r["exaggeration"])
    flight = scn.flight(extent, data.box, exg)
    print(f"\nscene        {extent.describe()}")
    print(f"exaggeration x{exg:.2f} (verified)")
    print(f"traces       {json.dumps(idx.as_dict())}")
    print(f"flight       centre {tuple(round(v, 3) for v in flight.centre)}, "
          f"x{flight.exaggeration_from:.2f} -> x{flight.exaggeration_to:.2f}")

    fig.write_html(a.out, include_plotlyjs=True, full_html=True)
    import os
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
