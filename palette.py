#!/usr/bin/env python3
"""
One palette for both pages, and the checks that keep it honest.

Colours are defined here once and emitted two ways: as plotly colourscales for
the 3D scenes, and as CSS custom properties for the page chrome. Defining them
twice is how a demo ends up with two slightly different greens for zinc.

Run this module to print the palette and the colour-vision check.

------------------------------------------------------------------------------
WHY THESE HUES
------------------------------------------------------------------------------
The three metals use Okabe-Ito, a set chosen to stay separable under the common
forms of colour blindness. `check()` measures that rather than asserting it,
simulating deuteranopia and protanopia and reporting the perceptual distance
between every pair.

The pipeline's own split colours are not reused. `figures.py` draws training
data in #4C72B0, which sits close to the Pb blue; a viewer moving from the
deposit view to the blind-panel view would carry the wrong association across.
Informing data is therefore grey, which also reads correctly: it is context,
and the panel is the subject.

------------------------------------------------------------------------------
GRADE WITHIN ONE METAL
------------------------------------------------------------------------------
Grade is carried by lightness within the metal's own hue, never by a second
hue. The ramp runs pale to dark on log10 of grade, floored at 0.001 % and
clamped at 30 %, shared across all three metals so one shade means one grade
whichever is on screen.

The pale end stops short of white. A ramp that reaches white loses its lowest
class against the page, and barren ground is still a measurement.

------------------------------------------------------------------------------
THE DESCRIPTIVE LAYERS
------------------------------------------------------------------------------
Zones, associated elements and the IP survey are drawn on the same scene as
the metals, so their colours are defined here too.

Zones use Okabe-Ito hues the metals do not use, plus a dark brown and black.
"Not logged" is a light grey, because it is most of the drilling and is the
absence of a log rather than a class.

The ten associated elements keep the hues of the pipeline's 06c figure, so a
reader who knows that figure reads these the same way. Each gets a ramp from a
pale tint through its own hue to a darker shade of it.

The IP ramp runs pale lemon to near-black. It is shown with the metals hidden
or beside them, so it has to stay apart from all three metal hues and from the
panel colour and the greys. `check_layers()` measures that.

The gravity ramp diverges from white at zero, violet for lows and slate for
highs. Every diverging ramp tried came within ΔE 10 of some colour among the
metals, zones and greys; this one keeps the most distance from the metals and
the panel colour, at worst ΔE 8.1 against Zn, and `check_layers()` reports it.
"""
from __future__ import annotations

import colorsys

import numpy as np

# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #

# Okabe-Ito. Reserved for metals and used for nothing else.
ELEMENT = {
    "Zn": "#009E73",   # bluish green
    "Pb": "#0072B2",   # blue
    "Cu": "#D55E00",   # vermillion
    "Ag": "#CC79A7",   # reddish purple, if ever needed
    "Au": "#E69F00",   # orange
}

# The blind test. Reddish purple rather than the pipeline's #C44E52, which sits
# too close to the Cu vermillion once red sensitivity is reduced -- see check().
BLIND_PANEL = "#CC79A7"
INFORMING = "#8A8F98"      # grey: context, not a category competing with metals
GROUND = "#5A6068"         # the terrain wireframe

# Page chrome, shared by the templates.
PAPER = "#FBFBFA"
INK = "#1A1C1E"
MUTED = "#6B7177"
RULE = "#D8DAD5"
PANEL_BG = "#F2F3F0"

FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', monospace"

# Grade colour scale, shared across metals.
GRADE_FLOOR = 0.001        # %, zero and trace assays clamp here
GRADE_CEILING = 30.0       # %, higher grades take the darkest shade

# Logged zones: table value, label, colour, marker size. Mineralised is drawn
# larger because it is the class a reader looks for first.
ZONE = (
    ("MINERALISED", "Mineralised", "#E69F00", 2.8),
    ("FOOTWALL", "Footwall", "#56B4E9", 2.2),
    ("HANGINGWALL", "Hanging wall", "#7A5C2E", 2.2),
    ("INTRUSIVE", "Intrusive", "#1A1A1A", 2.2),
    ("UNKNOWN", "Not logged", "#C9CDD2", 1.8),
)

# Associated elements, hues from the pipeline's 06c figure.
ASSOCIATED_COLOUR = {
    "Ag": "#4e79a7", "Au": "#f28e2b", "Fe": "#76b7b2", "Ba": "#edc948",
    "S": "#59a14f", "As": "#b07aa1", "Mn": "#17becf", "Sb": "#ff9da7",
    "Bi": "#9c755f", "Cd": "#e15759",
}
NOT_ASSAYED = "#E1E4E8"    # drawn small, as a circle
BELOW_DL = "#A7ADB4"       # drawn as an x, so shape separates it as well
ASSOCIATED_SIZE = 2.2
NOT_ASSAYED_SIZE = 1.6

# IP survey, shared by both properties. Dark is the anomalous end: high
# chargeability, or low resistivity (the resistivity view reverses the scale).
IP_RAMP = [[0.0, "#FAF3B5"], [0.333, "#BFB25A"], [0.667, "#5E5830"],
           [1.0, "#141410"]]
IP_MARKER = dict(size=2.0, symbol="square", opacity=0.6)

# Gravity, a diverging ramp centred on white at zero: violet for lows, slate
# for highs. Lows get the stronger hue because the gravity memo (McInnes,
# 2021) places the Thalanga ore systems in localised lows. Gravity and the IP
# survey are never on screen together, so this ramp is checked against the
# metals and the panel colour, not against the IP ramp.
GRAVITY_RAMP = [[0.0, "#3D2463"], [0.25, "#A196C6"], [0.5, "#F7F7F7"],
                [0.75, "#8C9AA6"], [1.0, "#2F3B45"]]


# --------------------------------------------------------------------------- #
# Ramps
# --------------------------------------------------------------------------- #

def grade_ramp(element: str, n: int = 24,
               light: float = 0.90, dark: float = 0.15) -> list:
    """Pale-to-dark plotly colourscale in one metal's hue.

    Hue is held fixed and lightness swept, so the ramp reads as one metal at
    varying tenor. Saturation eases off toward the pale end, where a fully
    saturated light colour looks washed out rather than light.
    """
    hex_colour = ELEMENT.get(element, INFORMING)
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    out = []
    for i in range(n):
        t = i / (n - 1)
        rr, gg, bb = colorsys.hls_to_rgb(
            h, light + (dark - light) * t, min(1.0, s * (0.5 + 0.5 * t)))
        out.append([t, f"rgb({int(rr * 255)},{int(gg * 255)},{int(bb * 255)})"])
    return out


def associated_ramp(element: str, n: int = 24) -> list:
    """Pale tint, then the element's hue at 0.6 of the way, then a darker shade.

    The pale end is 15 % of the hue mixed into white and the dark end is the
    hue at 55 % brightness. Mixing is in plain sRGB, as plotly interpolates.
    """
    c = np.array([int(ASSOCIATED_COLOUR[element][i:i + 2], 16)
                  for i in (1, 3, 5)], float)
    light, dark = 0.85 * 255 + 0.15 * c, 0.55 * c
    out = []
    for u in np.linspace(0, 1, n):
        rgb = (light + (c - light) * (u / 0.6) if u < 0.6
               else c + (dark - c) * ((u - 0.6) / 0.4))
        out.append([round(float(u), 4),
                    "#%02X%02X%02X" % tuple(int(round(x)) for x in rgb)])
    return out


def land_ramp(name: str = "terrain", lo: float = 0.25, n: int = 32) -> list:
    """Terrain colourscale with the water band removed.

    matplotlib's `terrain` reserves its lowest sixth for below-sea-level blue.
    The Thalanga DEM bottoms out at 299 m RL, so that band would render dry
    ground as though it were flooded.
    """
    from matplotlib import colormaps
    cmap = colormaps[name]
    return [[i / (n - 1),
             "rgb({},{},{})".format(*(int(v * 255) for v in
                                      cmap(lo + (i / (n - 1)) * (1 - lo))[:3]))]
            for i in range(n)]


def grade_bounds() -> tuple[float, float]:
    """(cmin, cmax) in log10 space for every grade-coloured trace."""
    return float(np.log10(GRADE_FLOOR)), float(np.log10(GRADE_CEILING))


def grade_ticks() -> tuple[list[float], list[str]]:
    """Colourbar ticks at powers of ten, plus a marked clamp at the ceiling."""
    lo, hi = grade_bounds()
    vals, text = [], []
    e = int(np.floor(lo))
    while e <= hi:
        if e >= lo:
            vals.append(float(e))
            v = 10.0 ** e
            text.append(f"{v:g}" if v >= 0.01 else f"{v:.3f}".rstrip("0"))
        e += 1
    vals.append(hi)
    text.append(f"{GRADE_CEILING:g}+")
    return vals, text


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

def css_variables() -> str:
    """The same colours as custom properties, for the page templates."""
    pairs = {
        "paper": PAPER, "ink": INK, "muted": MUTED, "rule": RULE,
        "panel-bg": PANEL_BG, "blind-panel": BLIND_PANEL,
        "informing": INFORMING, "ground": GROUND,
        **{f"metal-{e.lower()}": v for e, v in ELEMENT.items() if e in ("Zn", "Pb", "Cu")},
    }
    body = "\n".join(f"  --{k}: {v};" for k, v in pairs.items())
    return (":root {\n" + body
            + f"\n  --font-sans: {FONT_SANS};"
            + f"\n  --font-mono: {FONT_MONO};\n}}")


# --------------------------------------------------------------------------- #
# Colour-vision check
# --------------------------------------------------------------------------- #

def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ rgb / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _simulate(rgb_linear, kind: str):
    """Viénot-Brettel-Mollon dichromat simulation in linear RGB."""
    if kind == "normal":
        return rgb_linear
    mats = {
        "deutan": np.array([[0.625, 0.375, 0.0],
                            [0.700, 0.300, 0.0],
                            [0.0, 0.300, 0.700]]),
        "protan": np.array([[0.1115, 0.8885, 0.0],
                            [0.1115, 0.8885, 0.0],
                            [0.0, 0.0, 1.0]]),
    }
    return mats[kind] @ rgb_linear


def _delta_e(a: str, b: str, kind: str) -> float:
    def lab(h):
        rgb = _srgb_to_linear([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        return _linear_to_lab(_simulate(rgb, kind))
    return float(np.linalg.norm(lab(a) - lab(b)))


def check(floor: float = 10.0) -> bool:
    """Report pairwise separation for every colour that carries meaning.

    A CIE76 distance below about 10 is where two colours start being confused
    at the small marker sizes this demo uses. The check covers the three metals
    and the blind panel together, because the panel colour has to stay distinct
    from all of them -- a reader meets it immediately after the metals.
    """
    names = {"Zn": ELEMENT["Zn"], "Pb": ELEMENT["Pb"], "Cu": ELEMENT["Cu"],
             "panel": BLIND_PANEL, "informing": INFORMING}
    keys = list(names)
    worst, ok = (None, 1e9), True
    print(f"{'pair':<20}{'normal':>9}{'deutan':>9}{'protan':>9}")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ds = [_delta_e(names[a], names[b], k)
                  for k in ("normal", "deutan", "protan")]
            flag = "" if min(ds) >= floor else "   <- too close"
            if min(ds) < worst[1]:
                worst = (f"{a}/{b}", min(ds))
            ok &= min(ds) >= floor
            print(f"{a + '/' + b:<20}" + "".join(f"{d:>9.1f}" for d in ds) + flag)
    print(f"\nclosest pair under any vision: {worst[0]} at dE {worst[1]:.1f} "
          f"(floor {floor:g})")
    return ok


def _samples(scale: list, n: int = 24) -> list[str]:
    """A colourscale resampled at n even steps, interpolated in sRGB."""
    pos = np.array([s[0] for s in scale], float)
    rgb = np.array([[int(s[1][i:i + 2], 16) for i in (1, 3, 5)] for s in scale],
                   float)
    return ["#%02X%02X%02X" % tuple(int(round(np.interp(u, pos, rgb[:, j])))
                                    for j in range(3))
            for u in np.linspace(0, 1, n)]


def _closest(group_a: dict, group_b: dict) -> tuple[float, str, str, int]:
    """Worst separation between two groups, and how many pairs fall under 10."""
    worst, under = (1e9, "", ""), 0
    for na, ca in group_a.items():
        for nb, cb in group_b.items():
            d = min(_delta_e(ca, cb, k) for k in ("normal", "deutan", "protan"))
            under += d < 10.0
            if d < worst[0]:
                worst = (d, na, nb)
    return (*worst, under)


def check_layers(floor: float = 10.0) -> bool:
    """Separation of the descriptive layers, under all three kinds of vision.

    Ramps are compared at all 24 steps, not only at their anchors, because the
    in-between shades are what most points are drawn in.
    """
    metals = {e: ELEMENT[e] for e in ("Zn", "Pb", "Cu")}
    others = {"panel": BLIND_PANEL, "informing": INFORMING, "ground": GROUND}
    ip = {f"IP step {i}": c for i, c in enumerate(_samples(IP_RAMP))}
    zones = {label: colour for _, label, colour, _ in ZONE}
    ramps = {f"{e} step {i}": s[1] for e in ASSOCIATED_COLOUR
             for i, s in enumerate(associated_ramp(e))}
    greys = {"not assayed": NOT_ASSAYED, "below detection": BELOW_DL}

    grav = {f"gravity step {i}": c for i, c in enumerate(_samples(GRAVITY_RAMP))}
    rows = [("IP ramp / metals", _closest(ip, metals)),
            ("IP ramp / panel and greys", _closest(ip, others)),
            ("gravity ramp / metals, panel", _closest(grav, {**metals, "panel": BLIND_PANEL}))]
    labels = list(zones)
    pairs = [_closest({a: zones[a]}, {b: zones[b]})
             for i, a in enumerate(labels) for b in labels[i + 1:]]
    worst = min(pairs, key=lambda p: p[0])
    rows.append(("zones / each other", (*worst[:3], sum(p[3] for p in pairs))))
    rows.append(("associated ramps / greys", _closest(ramps, greys)))

    ok = True
    print(f"{'layer pair':<28}{'worst dE':>9}  {'closest':<36}{'pairs < 10':>11}")
    for name, (d, a, b, under) in rows:
        flag = "" if d >= floor else "   <- too close"
        ok &= d >= floor
        print(f"{name:<28}{d:>9.1f}  {a + ' / ' + b:<36}{under:>11}{flag}")
    return ok


def main(argv=None) -> int:
    print("metals")
    for e in ("Zn", "Pb", "Cu"):
        print(f"  {e:<4} {ELEMENT[e]}")
    print(f"\nblind panel  {BLIND_PANEL}\ninforming    {INFORMING}\n"
          f"ground       {GROUND}")
    lo, hi = grade_bounds()
    vals, text = grade_ticks()
    print(f"\ngrade scale  log10 {lo:g} to {hi:.3g}  "
          f"({GRADE_FLOOR:g} % to {GRADE_CEILING:g} %)")
    print(f"  ticks      {', '.join(text)}")
    print(f"\nZn ramp, pale to dark:")
    for stop in grade_ramp("Zn", n=6):
        print(f"  {stop[0]:.1f}  {stop[1]}")
    print("\ncolour-vision check")
    good = check()
    print("\n" + ("palette passes" if good else "PALETTE FAILS: see pairs above"))
    print("\ndescriptive layers")
    layers = check_layers()
    print("\n" + ("layers pass" if layers else "LAYERS FAIL: see pairs above"))
    return 0 if good and layers else 1


if __name__ == "__main__":
    raise SystemExit(main())
