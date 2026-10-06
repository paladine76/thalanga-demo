#!/usr/bin/env python3
"""
IBM Plex, inlined, so the pages do not reach the network to look right.

Both pages originally linked Google Fonts. That works at a desk and fails on a
plane, at a mine site, or behind an egress policy, and the failure is silent:
the page renders in a system font and nobody knows it was meant to look like
anything else. For a demo that travels as two files, a webfont that only
sometimes arrives is worse than no webfont.

The faces are vendored under assets/fonts/ and emitted as @font-face rules with
the woff2 inlined as data URIs.

------------------------------------------------------------------------------
WHY THESE TEN FILES
------------------------------------------------------------------------------
Five faces are used: Plex Sans at 400, 500 and 600, Plex Mono at 400 and 500.
IBM ships each one split by script, and the pages need two of those splits.

    Latin1   ASCII, Latin-1 Supplement, the dashes and quotes      ~20 kB
    Pi       the symbols: arrows, >=, the mathematical operators    ~7 kB

Latin1 alone would drop the arrows on the navigation links and the >= in the
grade-class legend to a fallback font, which looks like a mistake rather than a
choice. Together the ten files are 148 kB, about 200 kB once base64-encoded --
under 1% of the regional page and about 3% of the panel page.

The complete faces would be 288 kB before encoding and carry Cyrillic and Greek
that nothing here uses.
"""
from __future__ import annotations

import base64
from pathlib import Path

ASSETS = Path(__file__).parent / "assets" / "fonts"

# Lifted verbatim from the unicode-range declarations IBM ships with each
# subset. Narrowing them by hand would mean the browser downloads a face and
# then cannot render a character it actually contains.
RANGES = {
    "Latin1": (
        "U+0020-007E, U+00A0-00FF, U+0131, U+0152-0153, U+02C6, U+02DA, "
        "U+02DC, U+2013-2014, U+2018-201A, U+201C-201E, U+2020-2022, U+2026, "
        "U+2030, U+2039-203A, U+2044, U+20AC, U+2122, U+2212, U+FB01-FB02"
    ),
    "Pi": (
        "U+03C0, U+0E3F, U+2000-200D, U+2010-2012, U+2015, U+2028-2029, "
        "U+202F, U+2032-2033, U+203E, U+205F, U+2070, U+2074-2079, "
        "U+2080-2089, U+2113, U+2116, U+2126, U+212E, U+2150-2151, "
        "U+2153-215E, U+2190-2199, U+21A9-21AA, U+21B0-21B3, U+21B6-21B7, "
        "U+21BA-21BB, U+21C4, U+21C6, U+2202, U+2206, U+220F, U+2211, "
        "U+2215, U+2219-221A, U+221E, U+222B, U+2236, U+2248, U+2260, "
        "U+2264-2265, U+2400-2421, U+2500-259F, U+25CA, U+2713, U+274C, "
        "U+2B0E-2B11, U+3000, U+FEFF, U+FFFD"
    ),
}

# (family, css family name, weight, file stem)
FACES = [
    ("Sans", "IBM Plex Sans", 400, "IBMPlexSans-Regular"),
    ("Sans", "IBM Plex Sans", 500, "IBMPlexSans-Medium"),
    ("Sans", "IBM Plex Sans", 600, "IBMPlexSans-SemiBold"),
    ("Mono", "IBM Plex Mono", 400, "IBMPlexMono-Regular"),
    ("Mono", "IBM Plex Mono", 500, "IBMPlexMono-Medium"),
]


class MissingFaces(RuntimeError):
    """The vendored woff2 files are not where they should be."""


def face_files() -> list[tuple[str, int, str, Path]]:
    """Every (css family, weight, subset, path) the pages need."""
    out = []
    for _fam, css_family, weight, stem in FACES:
        for subset in RANGES:
            out.append((css_family, weight, subset,
                        ASSETS / f"{stem}-{subset}.woff2"))
    return out


def css(local_first: bool = True) -> str:
    """@font-face rules with the faces inlined as data URIs.

    `local()` is listed ahead of the data URI so a machine that already has
    IBM Plex installed uses its own copy and skips decoding ours entirely.
    """
    missing = [p for *_rest, p in face_files() if not p.exists()]
    if missing:
        raise MissingFaces(
            f"{len(missing)} vendored face(s) not found, first is {missing[0]}. "
            "Run `python fonts.py --check` to see what is expected."
        )

    rules = []
    for css_family, weight, subset, path in face_files():
        b64 = base64.b64encode(path.read_bytes()).decode()
        local = (f'local("{css_family}"), local("{css_family.replace(" ", "")}"), '
                 if local_first else "")
        rules.append(
            "@font-face {\n"
            f"  font-family: '{css_family}';\n"
            "  font-style: normal;\n"
            f"  font-weight: {weight};\n"
            "  font-display: swap;\n"
            f"  src: {local}url(data:font/woff2;base64,{b64}) format('woff2');\n"
            f"  unicode-range: {RANGES[subset]};\n"
            "}"
        )
    return "\n".join(rules)


def size_report() -> str:
    rows, total = [], 0
    for css_family, weight, subset, path in face_files():
        n = path.stat().st_size if path.exists() else 0
        total += n
        rows.append(f"  {path.name:<36}{n / 1024:>6.0f} kB"
                    + ("" if path.exists() else "   MISSING"))
    rows.append(f"  {'':<36}{'-' * 9}")
    rows.append(f"  {'raw':<36}{total / 1024:>6.0f} kB")
    rows.append(f"  {'inlined (base64)':<36}{total * 4 / 3 / 1024:>6.0f} kB")
    return "\n".join(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Inline IBM Plex into CSS.")
    p.add_argument("--check", action="store_true",
                   help="report the vendored faces and exit")
    p.add_argument("-o", "--out", default=None, help="write the CSS here")
    a = p.parse_args(argv)

    print(f"assets: {ASSETS}")
    print(size_report())
    if a.check:
        return 0 if all(p.exists() for *_r, p in face_files()) else 1

    sheet = css()
    print(f"\n{sheet.count('@font-face')} @font-face rules, "
          f"{len(sheet) / 1024:.0f} kB of CSS")
    if a.out:
        Path(a.out).write_text(sheet, encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
