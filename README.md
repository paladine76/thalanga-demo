# Thalanga blind-panel demo

An interactive demonstration that a grade estimate can be tested against ground
it never saw. Two static pages: the deposit at regional scale, and the blind
panel at its own scale.

The demonstration answers one question a resource geologist asks of any
estimate. **We withheld a block of drilled ground, estimated it from the rest,
and then opened the envelope. How close was it?**

---

## The three acts

**1 · The deposit.** Topography over 77,220 assay intervals. One metal at a
time, shaded light to dark by grade. This establishes what the drilling
actually found before any modelling is mentioned.

**2 · The blind panel.** The same scene, recoloured. A prism of ground 548 m by
328 m and 960 m deep is set aside and never used. The panel is a volume rather
than a surface patch, which a plan view cannot convey and a rotatable 3D scene
can.

**3 · The estimate against the ground.** Clicking the panel flies the camera
into it and opens the second page. One scene holds the same 12,967 positions,
switched between measured, kriged and simulated. The readout follows the
selection and reads em-dash against the measured view, which has no error to
report. At a 6.4 % Zn cutoff the measured ground holds 1,780 positions, kriging
finds 6, and a single realisation finds 1,347.

---

## Tech stack

Everything is generated at build time by Python and shipped as static HTML.
There is no server, no JavaScript build step and no framework.

| layer | choice | why |
|---|---|---|
| build | Python 3.11+ | matches the modelling pipeline |
| data | numpy, pandas, pyarrow | reads the pipeline's parquet and npz directly |
| regional 3D | plotly 7.x (`Surface`, `Scatter3d`, `Mesh3d`) | rotatable scene with no custom rendering |
| panel 3D | plotly 7.x (`Scatter3d`) | same stack as the regional page, so the camera hands over without conversion |
| colour ramps | matplotlib colormaps | sampled to plotly colourscales; not used for plotting |
| typography | IBM Plex, subset and inlined as data URIs | no webfont request, so the pages render identically offline |
| verification | Playwright + Chromium | dev only; renders both pages headless and checks them |

Output is two self-contained `.html` files. They open by double-click over
`file://` and work equally from any static host. Nothing is fetched at runtime,
so the demo works offline and survives being emailed.

Both pages are laid out to one viewport: a fixed rail carrying the heading,
the controls and the numbers, and the scene filling whatever height is left.
Nothing sits below the fold, because a scene the viewer has to scroll to find
is a scene they stop rotating. Below 54 rem wide the layout stacks and
scrolling returns, which is the phone case rather than the demo case.

**Why two pages rather than one.** The regional scene spans 6.4 km by 3.8 km.
The blind panel occupies about one twelfth of that footprint, so the panel's
positions drawn inside the regional scene would be a smear, and a single page
would carry both payloads whichever one you were looking at. Separate pages also make the
transition a plain link, which avoids the cross-origin restrictions that stop a
`file://` page from scripting into an embedded one.

---

## Repo structure

```
thalanga-demo/
├── README.md
├── requirements.txt
├── data/               the five pipeline files go here (not tracked)
│   └── README.md       says which file goes where
├── assets/
│   └── fonts/          10 woff2 subsets, 148 kB; inlined at build time
├── build.py            entry point; writes dist/
├── dataio.py           loads and aligns the assay and estimate tables
├── terrain.py          loads, crops and caches the ground surface
├── palette.py          one palette for both pages
├── fonts.py            reads assets/fonts/ and emits @font-face data URIs
├── scene.py            shared scale, exaggeration, camera, flight
├── scorecard.py        grade-tonnage, head grade, calibration, swath
├── regional.py         acts 1 and 2: DEM, grades, blind panel, box
├── panel_page.py       act 3: payload and page for the blind panel
├── templates/
│   ├── shell.html      regional page: stepper, flight, narrative
│   └── panel.html      panel page: scene, controls, readout
└── dist/               generated output
    ├── index.html
    └── panel.html
```

Each module does one thing and is runnable on its own, so a number can be
checked without building the whole site. `python dataio.py --assays ...` prints
the count ladder and the summary statistics, `python scorecard.py ...` prints
the grade–tonnage table, `python terrain.py ... --crop ...` reports what
survives the crop, `python palette.py` runs the colour-vision check,
`python fonts.py` reports the subset sizes, and `python scene.py` confirms each
scene draws the exaggeration it states.

`scorecard.py` is the one module the build does not import. It computes the
fuller set of comparisons described under **Scoring** below; the panel page
currently derives its four live numbers in the browser instead, because they
have to follow the cutoff slider. Nothing is broken by that, but it is the
loose end to know about.

---

## Data inputs

The five pipeline artifacts stay outside the repo, in `data/`.

| file | holds |
|---|---|
| `thalanga_modelling.parquet` | 77,222 assay intervals, coordinates, split, PCA |
| `thalanga_3d.html` | the gridded ground surface, read back out of the figure |
| `baseline_predictions.parquet` | 12,985 blind-panel rows: OK, IDW, local and global mean |
| `sgs_realizations.npz` | 12,985 × 100 conditional simulations per element |
| `thalanga_modelling_manifest.json` | blind-panel extent, run provenance |

The ground surface is the odd one out. It arrives as the plotly figure
`dxf_surface_3d.py` already writes, because that is where the gridded DEM
exists; re-deriving it from the 168 MB DXF on every build would take 40 seconds
to reproduce a grid that has not changed. `terrain.py` reads it once and caches
a 197 kB npz.

That grid is 52 m cells, which is the resolution of the circulated DEM rather
than of the 917,568-face TIN behind it. Across the 6.4 km regional view the
crop leaves 124 by 74 cells. Regenerating over the drilling window alone with
`dxf_surface_3d.py --grid 500` would give about 11 m cells, and is worth doing
before the demo is shown.

### Alignment contract

None of these files share a primary key, and `(Hole, X, Y, Z)` is **not
unique** — hole `SE796F1920-3` has 13 intervals that desurvey to one point, so
a key join inflates 12,985 rows to 15,169.

`dataio.py` relies on row order instead and verifies it position by position
against hole name and all three coordinates. A re-export written in a different
order raises `AlignmentError` and names the first row that disagrees. The build
stops rather than producing figures that look right and are not.

**If you re-export, add `From` and `To`.** They make the join a key rather than
an assumption, and `dataio.py` will use them when present.

### The count ladder

Three sample counts appear across the demo. All are correct for what they
describe, and the demo states them rather than leaving a reader to wonder.

| count | meaning |
|---|---|
| 12,985 | assay intervals inside the blind panel |
| 12,973 | distinct positions after co-located intervals collapse |
| 12,967 | positions that also received an estimate; 6 fell outside the kriging neighbourhood |

---

## Conventions

### Vocabulary

The audience is geologists. Machine-learning register is avoided throughout,
in code identifiers as well as on screen.

| not used | used |
|---|---|
| train / holdout split | informing data / blind panel |
| model, prediction | estimate (kriging), realisation (simulation) |
| ground truth | measured assays |
| R², RMSE, conditional bias | proportion above cutoff, head grade, calibration |
| feature, target | variable, grade |

### Scoring

Every number on the panel page is one a resource geologist reads without
translation. Four of them sit under the scene and follow both the view and the
cutoff slider:

- **Positions above cutoff**, out of 12,967.
- **Ground found, against measured.** The same count as a percentage of what
  the assays found at that cutoff. At 6.4 % Zn kriging reports 0 %.
- **Metal accounted for.** Mean grade over the panel against the measured mean.
  Kriging carries 38 % of the metal, a single realisation 82 %.
- **Highest value**, against the highest measured assay. Measured peaks at
  49.70 % Zn, kriging at 7.46 %, realisation 1 at 88.86 %.

Against the measured view the two comparison cells read em-dash. The answer
sheet has no error to report, and a score of 100 % there would invite a reader
to compare three numbers of which one is meaningless.

`scorecard.py` computes the fuller set offline, for a slide or a hand-out:
grade–tonnage curves with a P10–P90 band across all 100 realisations, head
grade recovered (measured 2.86 % Zn against kriged 1.09 %), calibration — where
the simulations give a 30–50 % chance of exceeding the cutoff, 30.1 % of those
positions really are above it — and a swath plot binned down elevation.

R² and RMSE are deliberately absent. Scored against individual assays, a
conditional mean always looks poor because it carries none of the nugget
variance, and the resulting negative numbers describe the estimator's purpose
rather than its failure.

### Vertical scale

Horizontal extents dwarf relief, so true scale renders terrain as a flat sheet
and the orebody as a smear. Both pages exaggerate the vertical and **state the
factor applied**, computed from the aspect ratio actually in force rather than
asserted in the title.

The two scenes need different factors. The regional scene is 6,392 m by 3,802 m
by 1,151 m and reads at ×1.94. Its vertical range is set by the drilling, which
reaches −741 m RL, not by the terrain, which spans only 112 m within the crop.
The blind panel is 548 m by 328 m by 960 m,
taller than it is wide, and needs none. The camera flight interpolates between
them, so the panel relaxes into true proportion as the viewer arrives.

### Palette

One palette across both pages, from the Okabe-Ito set, distinguishable under
the common forms of colour blindness.

| meaning | colour |
|---|---|
| Zn | `#009E73` |
| Pb | `#0072B2` |
| Cu | `#D55E00` |
| blind panel | `#CC79A7` |
| informing data | neutral grey |

Element hues are reserved for elements. The informing data is grey rather than
the pipeline's `#4C72B0`, which sits close enough to Pb that a viewer moving
between acts carries the wrong association.

Within one metal, grade is shown by lightness on a log scale from 0.001 % to
30 %, shared across all three metals so a given shade means the same grade
whichever is on screen. Grades above the ceiling take the darkest shade and the
colourbar marks it `30+`.

---

## Build

Put the five pipeline files into `data/`, then:

```bash
pip install -r requirements.txt
python build.py
```

That is the whole command. Each input is found by shape rather than by exact
name, so `data/` can hold whatever the pipeline called them:

| pattern | file |
|---|---|
| `*modelling*.parquet` | assay intervals, coordinates, split |
| `*baseline*.parquet` | ordinary kriging, IDW, the mean baselines |
| `*.npz` | the conditional simulations |
| `*manifest*.json` | the blind-panel extent |
| `*.html` | the ground surface from `dxf_surface_3d.py` |

A missing or ambiguous file stops the build and says which one. Any input can
still be pointed at directly with `--assays`, `--dem` and so on, and `--data`
moves the whole folder.

Then open `dist/index.html`. Both pages are already built in the archive, so
this is only needed when the pipeline produces new data.

### Payload budget

The regional page carries the DEM, 77,220 positions per metal and the
blind-panel act, 17.6 MB with plotly embedded. The panel page carries 12,967 positions and six realisations across three
metals, around 7.1 MB with plotly embedded. Neither page loads the other's data.

The full simulation cube is 20.8 MB of float32 and is never shipped. Summary
curves are computed at build time and arrive as a few kilobytes.

---

## Status

| module | state |
|---|---|
| `dataio.py` | done; reproduces the circulated figures exactly |
| `terrain.py` | done; caches the DEM, crop verified |
| `palette.py` | done; colour-vision check passes, closest pair ΔE 13.9 |
| `fonts.py` | done; 148 kB of woff2 subsets, no external request |
| `scene.py` | done; camera mapping verified against rendered pixels |
| `scorecard.py` | done; all four comparisons, 4.6 kB payload — not imported by the build |
| `regional.py` | done; both acts in one figure, box click verified |
| `panel_page.py` + `templates/panel.html` | done; one scene, three views, fits one viewport |
| `templates/shell.html` | done; act switch, flight, camera handover, fits one viewport |
| `build.py` | done; both pages in about 5 s from a warm DEM cache |

Both pages were rendered headless at 1920×1080, 1680×1050, 1512×982, 1440×900,
1366×768 and 1280×720. Neither scrolls at any of them, including with the notes
expanded.

What remains: narrative copy reviewed by a geologist, a pass over the regional
page's 17.6 MB, and a decision on whether `scorecard.py`'s curves belong in the
demo or stay a hand-out.

### Known risk

The regional page is 17.6 MB, nearly all of it plotly plus 77,220 positions per
metal. It opens fine from disk and from a static host, but it is slow enough
over a hotel connection to be worth testing before a live showing. Building
with `--cdn` drops it to a few hundred kilobytes at the cost of needing
internet, which is the wrong trade for a demo that may be shown on site.
