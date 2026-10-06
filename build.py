#!/usr/bin/env python3
"""
Build the whole demo into dist/.

    python build.py

Put the five files the pipeline produces into data/ and that is the whole
command. Each one is found by shape, so names do not have to match:

    data/thalanga_modelling.parquet            *modelling*.parquet
    data/baseline_predictions.parquet          *baseline*.parquet
    data/sgs_realizations.npz                  *.npz
    data/thalanga_modelling_manifest.json      *manifest*.json
    data/thalanga_3d.html                      *.html

Any of them can still be pointed at directly with --assays, --dem and so on,
and --data moves the whole folder.

Two pages come out, and nothing else is needed to view them:

    dist/index.html   the deposit and the blind panel, one scene, two acts
    dist/panel.html   inside the panel: measured, kriged, simulated

They link to each other by relative path and carry their own data and
plotly.js, so the pair works from a filesystem, from a shared drive, or from
any static host. Nothing is fetched at run time.

------------------------------------------------------------------------------
WHAT THE SHELL IS GIVEN
------------------------------------------------------------------------------
`regional.py` produces the figure; the shell drives it. The payload carries the
figure itself, the trace map so the page never counts indices by hand, both
surface states already flattened into restyle paths, and the flight plan.

The flight's destination is fixed at build time and its origin is not: the
camera starts wherever the viewer has rotated to, so only the target can be
decided here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dataio
import fonts
import palette
import panel_page
import regional
import scene as scn
import terrain as terrain_mod

TITLE = "A Blind Test on Thalanga"
LEDE = ("{n:,} assay intervals through a polymetallic deposit in north "
        "Queensland. A block of ground was set aside before any modelling "
        "began; the estimate was built from everything outside it. Look at "
        "what was drilled, then go inside the block and see how close the "
        "estimate came.")


SCALE = ("{n:,} assay positions over {extent} · vertical exaggeration "
         "x{e:.1f}, not true scale · colour is grade on a log scale")


# Found by shape rather than by exact name, so a file keeps whatever the
# pipeline called it. Each pattern has to be unambiguous inside data/.
INPUTS = {
    "assays": ("*modelling*.parquet", "the assay table with X, Y, Z and grades"),
    "baseline": ("*baseline*.parquet", "ordinary kriging and the simpler estimates"),
    "sims": ("*.npz", "the conditional simulations"),
    "manifest": ("*manifest*.json", "the blind-panel extent"),
    "dem": ("*.html", "the ground surface, from dxf_surface_3d.py"),
}


def find_inputs(folder: Path, override: dict) -> dict:
    """Locate the five pipeline files, or say precisely what is missing.

    Passing five paths on the command line is error-prone and, with line
    continuations, does not survive being pasted into a Windows prompt. A
    folder with the files in it is the thing a person can actually check.
    """
    found, problems = {}, []
    for key, (pattern, what) in INPUTS.items():
        if override.get(key):
            found[key] = Path(override[key])
            if not found[key].exists():
                problems.append(f"  --{key} {found[key]} does not exist")
            continue
        hits = sorted(p for p in folder.glob(pattern) if p.is_file())
        if not hits:
            problems.append(f"  missing {pattern:<24} {what}")
        elif len(hits) > 1:
            names = ", ".join(h.name for h in hits)
            problems.append(f"  {pattern} matches {len(hits)} files ({names});"
                            f" pass --{key} to choose")
        else:
            found[key] = hits[0]

    if problems:
        raise SystemExit(
            f"\ncannot build from {folder.resolve()}\n"
            + "\n".join(problems)
            + "\n\nPut the five pipeline files in that folder, or pass each "
              "path with its own flag.\n")
    return found


def _to_json(obj) -> str:
    import plotly.utils
    return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder)


def build_shell(result: dict, data, template: Path, embed_plotly: bool) -> str:
    """Fill the regional page template.

    The figure's own title is dropped: the page already has a heading, and two
    titles stacked above one scene reads as a mistake. The exaggeration it
    carried moves into a caption under the scene, because a stretched vertical
    has to stay stated wherever the figure is shown.
    """
    fig = result["figure"]
    fig.update_layout(title=None, margin=dict(l=0, r=0, t=8, b=0))
    flight = scn.flight(result["extent"], data.box, result["exaggeration"])
    payload = {
        "figure": json.loads(_to_json(fig)),
        "traces": result["traces"].as_dict(),
        "metals": result["metals"],
        "filled": result["filled"],
        "wire": result["wire"],
        "flight": flight.as_dict(),
    }

    html = template.read_text(encoding="utf-8")
    for token, value in (
        ("__TITLE__", TITLE),
        ("__LEDE__", LEDE.format(n=len(data.assays))),
        ("__LADDER__", data.counts.as_sentence()),
        ("__SCALE__", SCALE.format(
            n=len(data.assays), e=result["exaggeration"],
            extent=result["extent"].describe())),
        ("__CSS_VARS__", palette.css_variables()),
        ("__FONTS__", fonts.css()),
    ):
        html = html.replace(token, value)
    html = html.replace("__PAYLOAD__", _to_json(payload))

    if embed_plotly:
        import plotly.offline
        html = html.replace("__PLOTLYJS__", plotly.offline.get_plotlyjs())
    else:
        html = html.replace("<script>__PLOTLYJS__</script>",
                            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js">'
                            "</script>")
    return html


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="data",
                   help="folder holding the five pipeline files")
    p.add_argument("--assays", default=None, help="override discovery")
    p.add_argument("--baseline", default=None, help="override discovery")
    p.add_argument("--sims", default=None, help="override discovery")
    p.add_argument("--manifest", default=None, help="override discovery")
    p.add_argument("--dem", default=None, help="override discovery")
    p.add_argument("--dem-cache", default="dist/.terrain.npz",
                   help="npz cache so later builds skip re-reading the DEM")
    p.add_argument("--out", default="dist")
    p.add_argument("--templates", default="templates")
    p.add_argument("--margin", type=float, default=500.0,
                   help="metres of ground kept around the drilling")
    p.add_argument("--max-informing", type=int, default=regional.MAX_INFORMING,
                   help="informing positions drawn in the blind-panel act")
    p.add_argument("--runs", type=int, default=panel_page.DEFAULT_RUNS,
                   help="realisations carried into the panel page")
    p.add_argument("--element", default="Zn", help="metal selected on load")
    p.add_argument("--cdn", action="store_true",
                   help="link plotly.js instead of embedding it; much smaller "
                        "pages that need internet to open")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    templates = Path(args.templates)

    paths = find_inputs(Path(args.data),
                        {k: getattr(args, k) for k in INPUTS})
    print(f"inputs from {Path(args.data).resolve()}")
    for key in INPUTS:
        print(f"  {key:<9} {paths[key].name}")
    print()

    data = dataio.load(paths["assays"], paths["baseline"], paths["sims"],
                       paths["manifest"])
    Path(args.dem_cache).parent.mkdir(parents=True, exist_ok=True)
    ground = terrain_mod.load(paths["dem"], args.dem_cache)

    print("\nregional page")
    result = regional.build(data, ground, args.margin,
                            max_informing=args.max_informing,
                            with_controls=False)
    print(f"  scene        {result['extent'].describe()}")
    print(f"  exaggeration x{result['exaggeration']:.2f} (verified)")
    index = out / "index.html"
    index.write_text(
        build_shell(result, data, templates / "shell.html", not args.cdn),
        encoding="utf-8")
    print(f"  wrote {index} ({index.stat().st_size / 1e6:.1f} MB)")

    print("\npanel page")
    payload = panel_page.build_payload(data, runs=args.runs,
                                       default_element=args.element)
    print(f"  positions    {payload['n']:,}, {payload['nreal']} of "
          f"{data.n_realisations} realisations")
    print(f"  off-frame    {payload['offFrame']:,} (drawn off-screen, still counted)")
    panel = out / "panel.html"
    panel.write_text(
        panel_page.render(templates / "panel.html", payload,
                          embed_plotly=not args.cdn),
        encoding="utf-8")
    print(f"  wrote {panel} ({panel.stat().st_size / 1e6:.1f} MB)")

    total = sum(f.stat().st_size for f in (index, panel)) / 1e6
    print(f"\nbuilt in {time.time() - t_start:.1f}s, {total:.1f} MB total")
    print(f"open {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
