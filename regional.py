#!/usr/bin/env python3
"""
The deposit and the blind panel, as one rotatable scene.

Acts 1 and 2 share a figure rather than being two figures shown in turn. The
ground, the samples and the camera are the same in both; only the colouring
changes. Switching acts therefore keeps the viewer's bearings, and the move
into the panel starts from wherever they have rotated to.

    act 1   topography over 77,220 assay positions, coloured by one target
            metal's grade, by logged zone, or by one associated element;
            the IP survey or the gravity survey can be drawn over any of them
    act 2   the same positions recoloured: informing data in grey, the blind
            panel in pink, and the panel drawn as a volume you can click

------------------------------------------------------------------------------
ONE DRILLING TRACE, RECOLOURED IN THE BROWSER
------------------------------------------------------------------------------
Act 1 draws the drilling once. Switching between Zn, Pb and Cu, or to Zone or
an associated element, restyles that one trace's colours, sizes, symbols and
hover; the positions are never sent twice. An earlier build carried one copy
of the 77,220 positions per metal.

`build()` returns the colouring for every view under `layers`, and the shell
applies them. The figure itself opens on the first metal, so it renders
correctly before any script runs.

The associated colour bar sits on an invisible one-point trace, because the
drilling trace is then coloured point by point and plotly draws a colour bar
only for a numeric colour array.

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

import base64
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

import geophys
import gravity as grav_mod
import palette
import scene as scn
import terrain as terrain_mod

# Act 1 marker styling, shared by every view of the drilling.
DRILL_SIZE = 2.2
DRILL_OPACITY = 0.75

# Hover labels are white. Plotly otherwise fills a label with the point's own
# colour, which hides the element squares of a similar hue.
HOVER_INK = "#1F2328"
HOVER_NEUTRAL = palette.INFORMING

# Act 2 shows the shape of what was withheld, not every informing sample. The
# full set is 64,235 positions and drawing them all costs 5 MB to make a grey
# cloud slightly denser. The count drawn is stated in the legend.
MAX_INFORMING = 40_000


# --------------------------------------------------------------------------- #
# Trace bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class Traces:
    """Which trace index holds what, so nothing downstream counts by hand.

    `ip` and `gravity` are -1 when that survey was not supplied.
    """

    dem: int = 0
    drill: int = -1
    informing: int = -1
    panel: int = -1
    box_mesh: int = -1
    box_lines: int = -1
    assoc_bar: int = -1
    ip: int = -1
    gravity: int = -1

    @property
    def act2_indices(self) -> list[int]:
        return [self.informing, self.panel, self.box_mesh, self.box_lines]

    def as_dict(self) -> dict:
        return dict(dem=self.dem, drill=self.drill,
                    informing=self.informing, panel=self.panel,
                    boxMesh=self.box_mesh, boxLines=self.box_lines,
                    assocBar=self.assoc_bar, ip=self.ip,
                    gravity=self.gravity)


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


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

def _lower_bar(title: str, **extra) -> dict:
    """The colour bar in the lower slot, shared by the metals and Associated."""
    return dict(title=dict(text=title), thickness=13, len=0.42,
                x=1.01, y=0.45, yanchor="top", tickfont=dict(size=10), **extra)


def grade_values(df, element: str) -> np.ndarray:
    """log10 grade, clamped, as the colour value for one metal.

    Grade spans five orders of magnitude, so colour runs on log10. Zero and
    trace assays clamp to the floor rather than being dropped: barren ground is
    a measurement, and a hole that stops being drawn looks like a hole that was
    never drilled.
    """
    grade = df[f"{element}_pct"].to_numpy(dtype=float)
    return np.log10(np.clip(grade, palette.GRADE_FLOOR,
                            palette.GRADE_CEILING)).astype(np.float32)


def target_hover(element: str, elements) -> str:
    """Hover for one metal: all three grades, the chosen one in bold.

    `%{text}` after the last grade is where the shell writes the associated
    elements present at that point. Until it does, the text is empty.
    """
    rows = [(f"<b>{e}</b>" if e == element else e) + f" %{{customdata[{i}]:.3f}}%"
            for i, e in enumerate(elements)]
    rows[-1] += "%{text}"
    return "<br>".join(rows + ["%{x:.0f}E %{y:.0f}N %{z:.0f}mRL"]) \
        + "<extra></extra>"


def target_states(df, elements) -> dict:
    """What the shell restyles the drilling trace to for each metal."""
    lo, hi = palette.grade_bounds()
    ticks, tick_text = palette.grade_ticks()
    return {e: {
        "color": _b64(grade_values(df, e), "<f4"),
        "colorscale": palette.grade_ramp(e),
        "cmin": lo, "cmax": hi,
        "colorbar": _lower_bar(f"{e} %", tickvals=ticks, ticktext=tick_text),
        "hovertemplate": target_hover(e, elements),
        "border": palette.ELEMENT[e],
    } for e in elements}


def drill_trace(df, elements, states: dict, visible: bool = True) -> go.Scatter3d:
    """Every assay position, opened on the first metal."""
    first = states[elements[0]]
    return go.Scatter3d(
        x=df["X"].to_numpy(dtype=float),
        y=df["Y"].to_numpy(dtype=float),
        z=df["Z"].to_numpy(dtype=np.float32),
        mode="markers",
        marker=dict(
            size=DRILL_SIZE, color=grade_values(df, elements[0]),
            colorscale=first["colorscale"], cmin=first["cmin"],
            cmax=first["cmax"], opacity=DRILL_OPACITY,
            # Scatter3d markers carry a white outline by default. Below about
            # 5 px it covers most of the marker and a dense cloud renders as a
            # white mass instead of its own colour.
            line=dict(width=0),
            showscale=True, colorbar=first["colorbar"],
        ),
        name="drilling", showlegend=False, visible=visible,
        customdata=np.column_stack(
            [df[f"{e}_pct"].to_numpy(dtype=np.float32) for e in elements]),
        hovertemplate=first["hovertemplate"],
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=first["border"],
                        font=dict(color=HOVER_INK)),
    )


# --------------------------------------------------------------------------- #
# Act 1: zone and associated elements, applied by the shell
# --------------------------------------------------------------------------- #

def zone_state(data) -> dict:
    """Zone code per drawn interval, and the key the page lists."""
    codes = data.zone_codes()
    n = np.bincount(codes, minlength=len(palette.ZONE))
    return {
        "codes": _b64(codes, "u1"),
        "classes": [{"label": label, "color": colour, "size": size,
                     "n": int(n[i])}
                    for i, (_, label, colour, size) in enumerate(palette.ZONE)],
    }


def _nice_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 values whose log10 falls inside [lo, hi]."""
    return [v for v in (m * 10.0 ** k for k in range(-4, 7) for m in (1, 2, 5))
            if lo <= np.log10(v) <= hi]


def associated_state(data) -> dict:
    """Per element: which intervals were assayed, and what was found.

    `mask` marks the assayed intervals, bit i for interval i. `vals` holds one
    float per assayed interval in order: the value where measured, and minus
    the detection limit where censored. `limit` marks rule 3's at-limit rows
    and is omitted when an element has none.

    Colour limits are the 2nd and 98th percentiles of the measured values, on
    log10, so a few extreme assays do not wash out the rest. A presence-only
    element (rule 6) gets none and is drawn in its plain hue.
    """
    out = []
    for sym, info in data.associated.items():
        c = data.associated_columns(sym)
        cen = c["censored"].astype(bool)
        assayed = ~np.isnan(c["value"]) | cen
        stored = np.where(cen, -c["dl"], c["value"])[assayed]
        present = c["value"][~np.isnan(c["value"])]
        if info.presence_only:
            lo = hi = None
            ticks = []
        else:
            lo, hi = (float(v) for v in np.log10(np.percentile(present, [2, 98])))
            ticks = _nice_ticks(lo, hi)
        limit = c["at_limit"].astype(bool)
        out.append({
            "sym": sym, "unit": info.unit,
            "color": palette.ASSOCIATED_COLOUR[sym],
            "ramp": palette.associated_ramp(sym),
            "lo": lo, "hi": hi, "pa": info.presence_only,
            "tickvals": [float(np.log10(t)) for t in ticks],
            "ticktext": [f"{t:g}" for t in ticks],
            "nDrawn": info.drawn, "nPresent": int(len(present)),
            "censShare": round(info.share, 3),
            "mask": _b64(np.packbits(assayed, bitorder="little"), "u1"),
            "vals": _b64(stored, "<f4"),
            "limit": (_b64(np.packbits(limit, bitorder="little"), "u1")
                      if limit.any() else None),
        })
    return {"elements": out, "notAssayed": palette.NOT_ASSAYED,
            "belowDL": palette.BELOW_DL, "size": palette.ASSOCIATED_SIZE,
            "notAssayedSize": palette.NOT_ASSAYED_SIZE,
            "border": HOVER_NEUTRAL}


def associated_bar(extent: scn.Extent) -> go.Scatter3d:
    """An invisible point at the scene centre that carries Associated's bar.

    It sits at the centre so it can never widen the scene.
    """
    cx, cy, cz = extent.centre
    return go.Scatter3d(
        x=[cx], y=[cy], z=[cz], mode="markers", hoverinfo="skip",
        marker=dict(size=1, opacity=0, color=[0],
                    colorscale=palette.associated_ramp(next(iter(
                        palette.ASSOCIATED_COLOUR))),
                    cmin=0, cmax=1, showscale=True, colorbar=_lower_bar("")),
        name="associated scale", showlegend=False, visible=False,
    )


# --------------------------------------------------------------------------- #
# The IP survey
# --------------------------------------------------------------------------- #

def _r(a, d: int) -> list[float]:
    """Python rounding, as the approved preview used, to keep the page small."""
    return [round(float(v), d) for v in a]


def ip_layer(survey: geophys.IPSurvey) -> tuple[go.Scatter3d, dict]:
    """The curtains as one trace, and the two properties the shell switches.

    The trace opens on chargeability. Its colour bar takes the upper slot,
    where the elevation bar sits, and the shell hides the elevation bar while
    the survey is on.
    """
    pts = survey.points
    c_lo, c_hi = survey.chargeability_limits()
    r_lo, r_hi = survey.resistivity_limits()
    tickvals, ticktext = survey.resistivity_ticks()
    chg, lres = _r(survey.chargeability, 2), _r(survey.log_resistivity, 3)
    props = [
        {"color": chg, "cmin": c_lo, "cmax": c_hi, "reversescale": False,
         "title": "Chargeability<br>(dark = high)",
         "tickvals": None, "ticktext": None},
        {"color": lres, "cmin": r_lo, "cmax": r_hi, "reversescale": True,
         "title": "Resistivity, Ω·m<br>(dark = conductive)",
         "tickvals": tickvals, "ticktext": ticktext},
    ]
    trace = go.Scatter3d(
        x=_r(pts["GridE"], 1), y=_r(pts["GridN"], 1), z=_r(pts["Elev"], 1),
        mode="markers",
        customdata=[[ln, int(st), float(f"{rs:.3g}"), round(float(c), 2)]
                    for ln, st, rs, c in zip(pts["Line"], pts["Stn"],
                                             pts["InvRes"], pts["InvIP"])],
        hovertemplate=("<b>line %{customdata[0]}</b> · station %{customdata[1]}"
                       "<br>%{z:.0f} m RL<br>resistivity %{customdata[2]:,} Ω·m"
                       "<br>chargeability %{customdata[3]:.1f}"
                       "<extra>IP-Res</extra>"),
        marker=dict(**palette.IP_MARKER, line=dict(width=0),
                    color=chg, colorscale=palette.IP_RAMP,
                    cmin=c_lo, cmax=c_hi, reversescale=False, showscale=True,
                    colorbar=dict(title=dict(text=props[0]["title"]),
                                  thickness=13, len=0.42, x=1.01, y=0.95,
                                  yanchor="top", tickfont=dict(size=10))),
        name="IP-Res", showlegend=False, visible=False,
    )
    return trace, {"props": props, "note": survey.note()}


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
# The gravity survey
# --------------------------------------------------------------------------- #

def _grid(a: np.ndarray, d: int) -> list:
    """A 2D array as nested lists, rounded, with None for no data."""
    return [[None if not np.isfinite(v) else round(float(v), d) for v in row]
            for row in a]


def gravity_layer(drp: grav_mod.Drape) -> tuple[go.Surface, dict]:
    """The survey as one draped surface, and the products the shell switches.

    The surface is lit flat so its colours read as the colour bar says;
    shading would darken slopes and make a hillside look like a low. It opens
    on the first product. Its colour bar takes the upper slot, which the
    elevation bar and the IP bar also use, and the shell shows one at a time.
    """
    props = []
    for p in drp.products:
        mgal = p.unit == "mGal"
        props.append({
            "label": p.label, "color": _grid(p.values, 4 if mgal else 1),
            "cmin": -p.limit, "cmax": p.limit,
            "title": (f"Gravity, {p.label.lower()}<br>({p.unit}, violet = low)"
                      if mgal else "Gravity tilt<br>(degrees)"),
            # Plotly's surface hover exposes x, y, z and per-point arrays
            # such as customdata, but not surfacecolor, so the shell points
            # customdata at the same array as the colours.
            "hover": (f"{p.label.lower()} %{{customdata:.3f}} {p.unit}" if mgal
                      else "tilt %{customdata:.0f}°"),
            "note": p.note(),
        })
    first = props[0]
    trace = go.Surface(
        x=_r(drp.x, 1), y=_r(drp.y, 1), z=_grid(drp.z, 1),
        surfacecolor=first["color"], colorscale=palette.GRAVITY_RAMP,
        cmin=first["cmin"], cmax=first["cmax"], showscale=True,
        colorbar=dict(title=dict(text=first["title"]), thickness=13, len=0.42,
                      x=1.01, y=0.95, yanchor="top", tickfont=dict(size=10)),
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
        hovertemplate=("%{x:.0f}E %{y:.0f}N<br>" + first["hover"]
                       + "<extra>gravity 1982</extra>"),
        name="gravity 1982", showlegend=False, visible=False,
    )
    return trace, {"props": props}


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


def controls(idx: Traces, filled: dict, wire: dict, df,
             metals: list[str], states: dict) -> list[dict]:
    """Act, fill and metal switches.

    These exist so the figure stands up on its own during development. In the
    assembled site the shell drives the same restyles from its own controls
    and this row is not built. Zone, Associated and the IP survey are shell
    only.
    """
    flat = flatten
    act_idx = [idx.drill] + idx.act2_indices

    def act(deposit: bool):
        return [{"visible": [deposit] + [not deposit] * 4}, act_idx]

    def metal(m: str):
        s = states[m]
        return [{"marker.color": [grade_values(df, m)],
                 "marker.colorscale": [s["colorscale"]],
                 "marker.colorbar.title.text": [s["colorbar"]["title"]["text"]],
                 "hovertemplate": [s["hovertemplate"]],
                 "hoverlabel.bordercolor": [s["border"]]}, [idx.drill]]

    return [
        _buttons([("Deposit", act(True)), ("Blind panel", act(False))],
                 x=0.0, y=0.99),
        _buttons([("Wireframe", [{k: [v] for k, v in flat(wire).items()}
                                 | {"showscale": [False]}, [idx.dem]]),
                  ("Filled", [{k: [v] for k, v in flat(filled).items()}
                              | {"showscale": [True]}, [idx.dem]])],
                 x=0.0, y=0.93),
        _buttons([(m, metal(m)) for m in metals], x=0.0, y=0.87),
    ]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build(data, ground: terrain_mod.Terrain, margin: float = 500.0,
          metals=None, max_informing: int = MAX_INFORMING,
          with_controls: bool = True, ip=None, gravity=None) -> dict:
    """Both acts in one figure.

    Returns everything a page needs to drive it: the figure, which trace index
    holds what, the scene extent and exaggeration, the two surface states
    already flattened into restyle paths, and under `layers` the colouring of
    every drilling view plus the IP survey.

    `ip` is the survey as `geophys.load` returns it, or None. It is clipped to
    the scene here, after the scene is fixed, so it can never change it.
    `gravity` is the grid table as `gravity.load` returns it, or None; it is
    draped over the same cropped ground, inside the same scene.
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

    states = target_states(df, metals)
    idx.drill = len(traces)
    traces.append(drill_trace(df, metals, states))

    idx.informing, idx.panel = len(traces), len(traces) + 1
    traces += split_clouds(df, max_informing)
    idx.box_mesh, idx.box_lines = len(traces), len(traces) + 1
    traces += box_traces(data.box, extent.zmin, extent.zmax)

    layers = {"targets": states}
    if data.zones:
        layers["zones"] = zone_state(data)
    if data.associated:
        layers["associated"] = associated_state(data)
        idx.assoc_bar = len(traces)
        traces.append(associated_bar(extent))
    if gravity is not None:
        drp = grav_mod.drape(gravity, extent, ground)
        trace, layers["gravity"] = gravity_layer(drp)
        idx.gravity = len(traces)
        traces.append(trace)

    survey = None
    if ip is not None:
        survey = geophys.clip(ip, extent)
        if len(survey.points):
            trace, layers["ip"] = ip_layer(survey)
            idx.ip = len(traces)
            traces.append(trace)

    fig = go.Figure(traces)
    fig.update_layout(**scn.figure_layout(
        title="Thalanga — the deposit and the blind panel",
        subtitle=(f"{len(df):,} assay positions; colour is grade on a log "
                  f"scale, one metal at a time"),
        extent=extent, exaggeration=exaggeration, zoom=scn.REGIONAL_ZOOM,
    ))
    if with_controls:
        fig.update_layout(updatemenus=controls(idx, filled, wire, df,
                                               metals, states))
    return {"figure": fig, "traces": idx, "extent": extent,
            "exaggeration": exaggeration, "metals": metals,
            "filled": flatten(filled), "wire": flatten(wire),
            "layers": layers, "survey": survey}


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
    p.add_argument("--ip", default=None, help="IP-resistivity inversion csv")
    p.add_argument("--gravity", default=None, help="1982 gravity grid csv")
    p.add_argument("-o", "--out", default="regional.html")
    a = p.parse_args(argv)

    data = dataio.load(a.assays, a.baseline, a.sims, a.manifest)
    ground = terrain_mod.load(a.dem, a.dem_cache)
    r = build(data, ground, a.margin, max_informing=a.max_informing,
              ip=geophys.load(a.ip) if a.ip else None,
              gravity=grav_mod.load(a.gravity) if a.gravity else None)
    if r["survey"] is not None:
        print(f"ip           {r['survey'].note()}")
    if "gravity" in r["layers"]:
        print(f"gravity      {r['layers']['gravity']['props'][0]['note']}")
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
