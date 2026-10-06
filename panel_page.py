#!/usr/bin/env python3
"""
The blind panel page: one view, switched between measured, kriged and simulated.

A single 3D scene with three buttons, rather than three scenes side by side.
Comparing panels means moving your eye between them and holding a picture in
memory; switching one panel means the points stay where they are and only the
colour changes, which is the comparison a geologist is actually making.

    Metal       Cu · Pb · Zn
    Showing     Measured · Kriged · Simulated
    Realisation mean of all runs, or a single one (simulated only)
    Cutoff      dims everything below it

The readout follows the selection. Against the measured view the performance
cells read em-dash: the answer sheet has no error to report, and showing it a
score of 100 % would invite the reader to compare three numbers of which one is
meaningless.

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

VIEWS = [
    {"key": "actual", "label": "Measured", "tag": "Measured assays"},
    {"key": "ok", "label": "Kriged", "tag": "Ordinary kriging"},
    {"key": "sim", "label": "Simulated", "tag": "Sequential Gaussian simulation"},
]

DEFAULT_RUNS = 6
MARKER_SIZE = 2.6

TITLE = "The Estimate Against the Ground"
LEDE = ("The same {n:,} held-out positions, three ways. Measured is what the "
        "core assayed. Kriged and simulated were built from the drilling "
        "outside this block and never saw an assay inside it. Switch between "
        "them and raise the cutoff; the kriged view empties first.")


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


def build_payload(data, elements=("Cu", "Pb", "Zn"), runs: int = DEFAULT_RUNS,
                  classes=None, default_element: str = "Zn") -> dict:
    classes = classes or DEFAULT_CLASSES
    xyz = data.panel[["X", "Y", "Z"]].to_numpy(dtype=float)
    chosen = pick_runs(data.n_realisations, runs)

    layout, outside = panel_layout(data, elements)
    grade, run_blobs = {}, {}
    for e in elements:
        grade[e] = {
            "actual": _b64(data.measured(e)),
            "ok": _b64(data.estimate(e, "ok")),
            # The E-type uses all 100 realisations even though only a few
            # ship: a mean of a hundred is the smoother, and smoothing is
            # what this view is about.
            "etype": _b64(data.sims[e].mean(axis=1)),
        }
        run_blobs[e] = [_b64(data.sims[e][:, r]) for r in chosen]

    return {
        "n": int(len(xyz)),
        "elements": list(elements),
        "defaultElement": default_element,
        "views": VIEWS,
        "classes": {e: list(classes[e]) for e in elements},
        "ramp": RAMP,
        "nreal": len(chosen),
        "markerSize": MARKER_SIZE,
        # Full coordinates, not origin-relative: plotly labels its own axes and
        # the regional page uses the same numbers, so the two agree on sight.
        "xyz": _b64(xyz.ravel()),
        "grade": grade,
        "runs": run_blobs,
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
        ("__LEDE__", LEDE.format(n=payload["n"])),
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
    p.add_argument("--template", default="templates/panel.html")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--element", default="Zn", help="metal selected on load")
    p.add_argument("--cdn", action="store_true",
                   help="link plotly.js instead of embedding it")
    p.add_argument("-o", "--out", default="dist/panel.html")
    a = p.parse_args(argv)

    data = dataio.load(a.assays, a.baseline, a.sims, a.manifest)
    payload = build_payload(data, runs=a.runs, default_element=a.element)
    print(f"payload      {payload['n']:,} positions, "
          f"{payload['nreal']} of {data.n_realisations} realisations "
          f"({pick_runs(data.n_realisations, a.runs)})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a.template, payload, embed_plotly=not a.cdn),
                   encoding="utf-8")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
