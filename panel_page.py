#!/usr/bin/env python3
"""
The blind panel page: one view, switched between measured, kriged, simulated
and diffusion.

A single 3D scene with four buttons, rather than four scenes side by side.
Comparing panels means moving your eye between them and holding a picture in
memory; switching one panel means the points stay where they are and only the
colour changes, which is the comparison a geologist is actually making.

    Metal       Cu · Pb · Zn
    Showing     Measured · Kriged · Simulated · Diffusion
    Realisation the median of all runs, or a single one (the two ensembles)
    Cutoff      dims everything below it

The readout follows the selection. Against the measured view the performance
cells read em-dash: the answer sheet has no error to report, and showing it a
score of 100 % would invite the reader to compare four numbers of which one is
meaningless.

------------------------------------------------------------------------------
WHY EACH ENSEMBLE OPENS ON A SINGLE REALISATION
------------------------------------------------------------------------------
Both ensembles land on run 1 and keep their median one slider step away.

No pointwise summary answers the question this page asks. Proportion above a
cutoff is a question about a distribution, and collapsing a hundred fields to
one number per position throws the distribution away before the question is
put. At a 4 % Zn cutoff, against 2,462 measured positions:

    kriged                      149    6 %
    SGS  pointwise median       160    6 %
    SGS  pointwise mean       1,641   67 %
    SGS  single draw, typical 2,197   89 %
    NDP  pointwise median     1,000   41 %
    NDP  pointwise mean       3,104  126 %
    NDP  single draw, typical 2,286   93 %

The mean fills in every low that any realisation had and over-reports; the
median is dragged down wherever the realisations disagree and under-reports.
A realisation is the one object built to reproduce the histogram, so it is
what the page opens on.

The gap between the two medians, 160 against 1,000, is mostly a measure of how
much each method's draws disagree with one another rather than of accuracy.
SGS draws range from 889 to 6,508 positions above the cutoff and NDP draws
from 2,123 to 2,485, so the SGS median collapses and the NDP median survives.
That same tight agreement is why the NDP P10-P90 band misses the measured
proportion at every cutoff while the wider SGS band contains it.

------------------------------------------------------------------------------
BUILT ON PLOTLY, NOT THE TRIPTYCH RENDERER
------------------------------------------------------------------------------
The earlier page used a bespoke WebGL renderer drawing three viewports from one
set of buffers. One viewport with a switch is a different program, and plotly
already draws the regional page. Sharing the stack means the camera handed over
from the deposit view needs no conversion, and there is one rendering path to
keep working rather than two.

Interaction is all client side. The page holds the grade arrays once and
recolours a single trace; nothing is recomputed in Python and no state round
trips. The cutoff dims points rather than removing them, so a slider step is
one restyle of one array instead of rebuilding the trace.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

import fonts
import palette
import scene as scn

# Six grade classes, blue through dark red, carried over from the circulated
# triptych so a reader who has seen that page reads this one the same way.
RAMP = ["#2c5aa0", "#4aa3df", "#e3b52e", "#f0661c", "#b8121a", "#5e0606"]

# Five edges per metal. Cu and Pb run at lower tenor than Zn and share a ladder.
DEFAULT_CLASSES = {
    "Cu": [0.1, 0.25, 0.5, 1.0, 2.0],
    "Pb": [0.1, 0.25, 0.5, 1.0, 2.0],
    "Zn": [0.5, 1.0, 2.0, 4.0, 8.0],
}

# `kind` tells the page whether the realisation slider applies. `key` for an
# ensemble is the method name in `Thalanga.ensembles`, so adding a third
# estimator here needs no other change in this module.
VIEWS = [
    {"key": "actual", "kind": "static", "label": "Measured",
     "tag": "Measured assays"},
    {"key": "ok", "kind": "static", "label": "Kriged",
     "tag": "Ordinary kriging"},
    {"key": "sgs", "kind": "sim", "label": "Simulated",
     "tag": "Sequential Gaussian simulation"},
    {"key": "ndp", "kind": "sim", "label": "Diffusion",
     "tag": "Neural Diffusion Process"},
]

DEFAULT_RUNS = 6
MARKER_SIZE = 2.6

# The lede counts the views in words, because "3 ways" in running prose reads
# as a typo on a page whose every other small number is spelled out.
COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

TITLE = "The Estimate Against the Ground"
LEDE = ("The same {n:,} held-out positions, {k} ways. Measured is what the core "
        "assayed. The others were built from the drilling outside this block "
        "and never saw an assay inside it. The simulated views open on one "
        "realisation of a hundred; raise the cutoff and watch the kriged view "
        "empty first.")


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def pick_runs(n_available: int, want: int) -> list[int]:
    """Evenly spaced realisation indices, so the slider spans the spread."""
    want = max(1, min(want, n_available))
    return [int(round(i)) for i in np.linspace(0, n_available - 1, want)]


ROBUST_Q = 0.003   # fraction trimmed from each end when framing the scene


def robust_extent(xyz: np.ndarray, q: float = ROBUST_Q) -> tuple[scn.Extent, int]:
    """Frame the bulk of the panel, not its outliers.

    The block is defined to 635 m below surface but the drilling barely passes
    the collar elevation: 98.7 % of positions sit in the top 400 m. Framing on
    raw min/max leaves most of the view empty and the ore a band across the
    top. Trimming a few tenths of a percent from each end fixes the framing.

    Positions outside the frame are not deleted, only off-screen, and they
    still count in every number the readout reports. The count is returned so
    the page can say how many.
    """
    lo = np.quantile(xyz, q, axis=0)
    hi = np.quantile(xyz, 1 - q, axis=0)
    outside = int((((xyz < lo) | (xyz > hi)).any(axis=1)).sum())
    return scn.Extent(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), outside


def panel_layout(data, elements) -> tuple[dict, int]:
    """Scene layout for the panel alone, drawn at true vertical scale.

    The block is 548 m by 328 m and 960 m deep, taller than it is wide, so it
    needs no exaggeration. This is also where the flight lands, and arriving at
    true scale is the point of the move.
    """
    xyz = data.panel[["X", "Y", "Z"]].to_numpy(dtype=float)
    extent, outside = robust_extent(xyz)
    layout = scn.figure_layout(
        title="", subtitle="", extent=extent,
        exaggeration=scn.PANEL_EXAGGERATION, zoom=scn.PANEL_ZOOM, height=560)
    layout.pop("title")
    layout["margin"] = dict(l=0, r=0, t=0, b=0)
    layout["showlegend"] = False
    layout["paper_bgcolor"] = "#ffffff"
    for axis, (lo, hi) in zip(("xaxis", "yaxis", "zaxis"),
                              ((extent.xmin, extent.xmax),
                               (extent.ymin, extent.ymax),
                               (extent.zmin, extent.zmax))):
        layout["scene"][axis]["range"] = [lo, hi]
    return layout, outside


def views_for(data) -> list[dict]:
    """The views this dataset can actually draw.

    An ensemble that was not loaded is dropped rather than shipped as a dead
    button, so building without the diffusion cube gives the three-view page
    back unchanged.
    """
    return [v for v in VIEWS
            if v["kind"] != "sim" or v["key"] in data.ensembles]


def build_payload(data, elements=("Cu", "Pb", "Zn"), runs: int = DEFAULT_RUNS,
                  classes=None, default_element: str = "Zn") -> dict:
    classes = classes or DEFAULT_CLASSES
    xyz = data.panel[["X", "Y", "Z"]].to_numpy(dtype=float)
    views = views_for(data)
    layout, outside = panel_layout(data, elements)

    # The two static views are columns of the panel frame; the ensembles are
    # cubes. Keeping them apart in the payload means the page never has to ask
    # which kind of thing a key refers to -- the view says so.
    static = {
        "actual": {e: _b64(data.measured(e)) for e in elements},
        "ok": {e: _b64(data.estimate(e, "ok")) for e in elements},
    }

    ensembles, chosen = {}, {}
    for v in views:
        if v["kind"] != "sim":
            continue
        m = v["key"]
        picks = pick_runs(data.n_draws(m), runs)
        chosen[m] = picks
        ensembles[m] = {
            # The median over every draw the pipeline produced, not over the
            # handful shipped: the central estimate should not move because
            # the page carries six realisations instead of a hundred.
            "centre": {e: _b64(data.centre(m, e)) for e in elements},
            "runs": {e: [_b64(data.realisations(m, e)[:, r]) for r in picks]
                     for e in elements},
        }

    return {
        "n": int(len(xyz)),
        "elements": list(elements),
        "defaultElement": default_element,
        "views": views,
        "classes": {e: list(classes[e]) for e in elements},
        "ramp": RAMP,
        "nreal": {m: len(p) for m, p in chosen.items()},
        "ofDraws": {m: int(data.n_draws(m)) for m in chosen},
        "markerSize": MARKER_SIZE,
        # Full coordinates, not origin-relative: plotly labels its own axes and
        # the regional page uses the same numbers, so the two agree on sight.
        "xyz": _b64(xyz.ravel()),
        "static": static,
        "ensembles": ensembles,
        "layout": json.loads(_to_json(layout)),
        "offFrame": outside,
    }


def _to_json(obj) -> str:
    import plotly.utils
    return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def render(template: str | Path, payload: dict, embed_plotly: bool = True) -> str:
    html = Path(template).read_text(encoding="utf-8")
    import plotly.offline

    js = plotly.offline.get_plotlyjs() if embed_plotly else ""
    if not embed_plotly:
        html = html.replace(
            "<script>__PLOTLYJS__</script>",
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>')

    for token, value in (
        ("__TITLE__", TITLE),
        ("__LEDE__", LEDE.format(
            n=payload["n"],
            k=COUNT_WORDS.get(len(payload["views"]), len(payload["views"])))),
        ("__N__", f"{payload['n']:,}"),
        ("__CSS_VARS__", palette.css_variables()),
        ("__FONTS__", fonts.css()),
    ):
        html = html.replace(token, value)
    html = html.replace("__PAYLOAD__", _to_json(payload))
    if embed_plotly:
        html = html.replace("__PLOTLYJS__", js)
    return html


def main(argv=None) -> int:
    import argparse
    import os

    import dataio

    p = argparse.ArgumentParser(description="Build the blind-panel page.")
    p.add_argument("--assays", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--sims", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ndp", default=None, help="diffusion realisations npz")
    p.add_argument("--template", default="templates/panel.html")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--element", default="Zn", help="metal selected on load")
    p.add_argument("--cdn", action="store_true",
                   help="link plotly.js instead of embedding it")
    p.add_argument("-o", "--out", default="dist/panel.html")
    a = p.parse_args(argv)

    data = dataio.load(a.assays, a.baseline, a.sims, a.manifest, a.ndp)
    payload = build_payload(data, runs=a.runs, default_element=a.element)
    for m, k in payload["nreal"].items():
        print(f"payload      {m}: {k} of {payload['ofDraws'][m]} draws "
              f"({pick_runs(payload['ofDraws'][m], a.runs)})")
    print(f"             {payload['n']:,} positions, "
          f"{len(payload['views'])} views")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a.template, payload, embed_plotly=not a.cdn),
                   encoding="utf-8")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
