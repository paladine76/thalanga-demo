#!/usr/bin/env python3
"""
Load the Thalanga artifacts and align them into one evaluation set.

Five files come out of the modelling pipeline, and most of them carry no
shared primary key. This module joins them, proves the join is sound, and
refuses to continue if it is not -- a silent misalignment here would corrupt
every figure downstream while still rendering perfectly.

    thalanga_modelling.parquet      77,222 assay intervals, all splits
    baseline_predictions.parquet    12,985 blind-panel rows: OK, IDW, means
    sgs_realizations.npz            12,985 x 100 conditional simulations
    ndp_realizations_all.npz        12,985 x 100 diffusion draws, plus `row`
    manifest.json                   blind-panel extent and run provenance

------------------------------------------------------------------------------
WHY THE JOIN NEEDS CARE
------------------------------------------------------------------------------
(Hole, X, Y, Z) is NOT unique. Hole SE796F1920-3 has 13 downhole intervals that
all desurvey to the same collar position, each with its own assay -- 12.0, 18.7,
14.7 % Zn and so on. Merging on those four columns inflates 12,985 rows to
15,169 and every statistic computed afterwards is wrong.

The baseline and SGS exports do not carry From/To, so there is no key to merge
on. What they do have is identical row order, inherited from the pipeline that
wrote them. This module relies on that order and then verifies it, position by
position, against Hole and all three coordinates. If a future run writes the
rows sorted differently the check fails loudly instead of quietly producing
plausible nonsense.

The NDP export is the exception and does not need that trust. It carries a
`row` array holding each sample's position in the 77,222-row modelling table,
so its join is checked against a key rather than against an assumption.
`load_assays` keeps that number as `source_row` for exactly this purpose. A
reordered NDP export still lands on the right samples; a differently filtered
one is caught by name rather than by a coordinate that happens to drift.

------------------------------------------------------------------------------
THE COUNT LADDER
------------------------------------------------------------------------------
Three different sample counts appear across the existing figures, and all three
are correct for what they describe:

    12,985   intervals inside the blind panel
    12,973   distinct positions -- the 13 co-located intervals above collapse
             to one; this is the manifest's n_eval_eligible and the
             eval_eligible flag
    12,967   positions that also received an estimate; 6 fell outside the
             kriging neighbourhood and came back null

`CountLadder` carries these so a page can state them rather than leaving a
reader to wonder why the number changed between screens.

------------------------------------------------------------------------------
THE DESCRIPTIVE COLUMNS
------------------------------------------------------------------------------
The regional page also draws what was logged and assayed beside the three
target metals: the geological zone, and ten associated elements. These come
from the same modelling table and are joined back on `source_row`, so they
need no alignment check of their own.

The associated elements go through the below-detection rule in censor.py
before anything else sees them. The rule is applied to the whole file, all
77,222 rows, because the rule works per file. Applying it after the two rows
without a position are dropped gives the same flags for all ten elements.

A cell that was never assayed stays NaN in every column. A censored cell has
NaN as its value and its limit in `<El>_dl`. A cell at the limit (rule 3's
exception) keeps its value and is marked in `<El>_at_limit`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import censor

ELEMENTS = ("Zn", "Pb", "Cu")

# Logged zones, in the order a page lists them. A value outside this set
# stops the load, because a page would otherwise draw it in no colour at all.
ZONES = ("MINERALISED", "FOOTWALL", "HANGINGWALL", "INTRUSIVE", "UNKNOWN")

# Associated elements: symbol, column in the modelling table, unit. The order
# is the dropdown's order on the regional page.
ASSOCIATED = (
    ("Ag", "AG_PPM_LAB", "ppm"), ("Au", "AU_PPM_LAB", "ppm"),
    ("Fe", "FE_PCT_LAB", "%"), ("Ba", "BA_PPM_LAB", "ppm"),
    ("S", "S_PPM_LAB", "ppm"), ("As", "AS_PPM_LAB", "ppm"),
    ("Mn", "MN_PPM_LAB", "ppm"), ("Sb", "SB_PPM_LAB", "ppm"),
    ("Bi", "BI_PPM_LAB", "ppm"), ("Cd", "CD_PPM_LAB", "ppm"),
)

# Estimators present in baseline_predictions.parquet, in the order a geologist
# would compare them: the method under test first, then the simpler things it
# has to beat. `mean` is a single number applied everywhere -- the floor any
# estimate must clear to have earned its keep.
BASELINE_METHODS = ("ok", "idw", "localmean", "mean")

COORDS = ("X", "Y", "Z")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CountLadder:
    """Why the sample count differs between screens."""

    intervals: int
    positions: int
    estimated: int

    @property
    def co_located(self) -> int:
        return self.intervals - self.positions

    @property
    def unestimated(self) -> int:
        return self.positions - self.estimated

    def as_sentence(self) -> str:
        return (
            f"{self.intervals:,} assay intervals fall inside the blind panel. "
            f"{self.co_located} of them repeat a position already counted — "
            f"intervals from one hole that desurvey to the same point — leaving "
            f"{self.positions:,} distinct positions. "
            f"{self.unestimated} of those fell outside the "
            f"kriging neighbourhood, so {self.estimated:,} positions carry both "
            f"a measured assay and an estimate."
        )


@dataclass(frozen=True)
class Associated:
    """What the rule found in one associated-element column of the file."""

    sym: str
    column: str
    unit: str
    assayed: int              # non-missing assays in the whole file
    share: float              # censored share of those
    drawn: int                # assayed intervals that have a position
    presence_only: bool       # rule 6
    at_limit: int             # rows flagged by rule 3's exception
    seen_dl: tuple[float, ...]


@dataclass
class Thalanga:
    """Everything the figures need, already aligned.

    `assays` is every interval in the deposit, for the regional view. It also
    carries `zone` and, per associated element, `<El>_value`, `<El>_censored`,
    `<El>_dl` and `<El>_at_limit`.
    `panel` is the blind-panel evaluation set: measured assays, each estimator,
    and the simulations, one row per position, in a single frame.
    `ensembles` maps method -> element -> (n_positions, n_draws) of simulated
    grade. "sgs" is always present; "ndp" joins it when the diffusion cube is
    supplied.
    `associated` maps symbol -> Associated, in dropdown order, for the
    elements the file carries. `zones` holds the logged zones found.
    """

    assays: pd.DataFrame
    panel: pd.DataFrame
    ensembles: dict[str, dict[str, np.ndarray]]
    box: dict[str, float]
    manifest: dict
    counts: CountLadder
    associated: dict[str, Associated] = field(default_factory=dict)
    zones: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Medians are the central estimate the pages draw, and each one is a
        # sort over 100 draws at 12,967 positions. Computing them on demand and
        # keeping them costs a few milliseconds once instead of once per
        # redraw.
        self._centres: dict[tuple[str, str], np.ndarray] = {}

    # -- ensembles ----------------------------------------------------------
    @property
    def sims(self) -> dict[str, np.ndarray]:
        """The SGS cube, under its original name, for callers that predate NDP."""
        return self.ensembles["sgs"]

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.ensembles)

    @property
    def n_realisations(self) -> int:
        return self.n_draws("sgs")

    def n_draws(self, method: str = "sgs") -> int:
        return next(iter(self.ensembles[method].values())).shape[1]

    def realisations(self, method: str, element: str) -> np.ndarray:
        return self.ensembles[method][element].astype(float)

    def centre(self, method: str, element: str) -> np.ndarray:
        """The median across every draw.

        The median rather than the mean. Averaging realisations rebuilds the
        smoother that simulation exists to avoid: the SGS mean reports 234 % of
        the ground that was actually there at a 2 % Zn cutoff, because averaging
        a hundred rough fields fills in every low that any of them had. The
        median is a grade some realisation actually produced.
        """
        key = (method, element)
        if key not in self._centres:
            self._centres[key] = np.median(self.ensembles[method][element], axis=1)
        return self._centres[key]

    # -- single series ------------------------------------------------------
    def measured(self, element: str) -> np.ndarray:
        return self.panel[f"{element}_pct"].to_numpy(dtype=float)

    def estimate(self, element: str, method: str = "ok") -> np.ndarray:
        return self.panel[f"{method}_{element}_pct"].to_numpy(dtype=float)

    def realisation(self, element: str, run: int, method: str = "sgs") -> np.ndarray:
        return self.ensembles[method][element][:, run].astype(float)

    # -- descriptive columns, over every drawn interval -----------------------
    def zone_codes(self) -> np.ndarray:
        """Each interval's zone as its index in ZONES, uint8."""
        return self.assays["zone"].map({z: i for i, z in enumerate(ZONES)}) \
            .to_numpy(dtype=np.uint8)

    def associated_columns(self, sym: str) -> dict[str, np.ndarray]:
        """value, censored, dl and at_limit for one element, as arrays."""
        return {k: self.assays[f"{sym}_{k}"].to_numpy()
                for k in ("value", "censored", "dl", "at_limit")}


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

class AlignmentError(RuntimeError):
    """The files no longer line up row for row."""


def _verify_alignment(name: str, left: pd.DataFrame, right, tol: float = 1e-6) -> None:
    """Confirm two sources describe the same positions in the same order.

    `right` may be a DataFrame or a mapping of coordinate arrays, so the same
    check covers both the baseline parquet and the npz.
    """
    get = (lambda c: np.asarray(right[c], dtype=float)) if not hasattr(right, "columns") \
        else (lambda c: right[c].to_numpy(dtype=float))

    n_right = len(get("X"))
    if len(left) != n_right:
        raise AlignmentError(
            f"{name}: {n_right:,} rows against {len(left):,} blind-panel rows. "
            "The exports no longer cover the same samples."
        )

    for c in COORDS:
        delta = np.abs(left[c].to_numpy(dtype=float) - get(c))
        bad = int((delta > tol).sum())
        if bad:
            first = int(np.argmax(delta > tol))
            raise AlignmentError(
                f"{name}: {bad:,} rows disagree on {c}, first at row {first} "
                f"({left[c].iloc[first]:.3f} vs {get(c)[first]:.3f}). Row order "
                "differs from the modelling table. Re-export with Hole, From "
                "and To so the sources can be joined on a key instead."
            )

    if hasattr(right, "columns") and "Hole" in right.columns:
        mismatch = (left["Hole"].to_numpy() != right["Hole"].to_numpy())
        if mismatch.any():
            first = int(np.argmax(mismatch))
            raise AlignmentError(
                f"{name}: hole names differ from row {first} "
                f"({left['Hole'].iloc[first]!r} vs {right['Hole'].iloc[first]!r})."
            )


def _verify_row_key(name: str, left: pd.DataFrame, rows: np.ndarray) -> None:
    """Check a keyed join: every sample named, in the same order.

    Stronger than the coordinate check above, because it compares identities
    rather than numbers that merely agree. Two different samples can share a
    position; they cannot share a row in the modelling table.
    """
    want = left["source_row"].to_numpy(dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)

    if len(rows) != len(want):
        raise AlignmentError(
            f"{name}: {len(rows):,} rows against {len(want):,} blind-panel rows. "
            "The exports no longer cover the same samples."
        )

    bad = rows != want
    if bad.any():
        first = int(np.argmax(bad))
        missing = np.setdiff1d(want, rows)
        extra = np.setdiff1d(rows, want)
        detail = (f" {len(missing):,} blind-panel samples are absent and "
                  f"{len(extra):,} rows belong to samples outside it."
                  if len(missing) or len(extra)
                  else " The same samples are present but in a different order.")
        raise AlignmentError(
            f"{name}: row keys diverge at position {first} "
            f"(expected {want[first]}, found {rows[first]}).{detail}"
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_assays(path: str | Path, elements=ELEMENTS) -> pd.DataFrame:
    """Every assay interval with a desurveyed position.

    Rows without coordinates cannot be drawn or estimated. Dropping only those
    reproduces the sample counts in the existing figures exactly, so the
    regional view stays comparable to what has already been circulated.
    """
    cols = ["Hole", "From", "To", *COORDS, "split", "eval_eligible",
            *[f"{e}_pct" for e in elements]]
    df = pd.read_parquet(path, columns=cols)
    # Kept before any row is dropped: this is the number the NDP export uses to
    # name its samples, so it has to count from the file as written, not from
    # whatever survives the filter below.
    df["source_row"] = np.arange(len(df), dtype=np.int64)
    before = len(df)
    df = df.dropna(subset=list(COORDS)).reset_index(drop=True)
    if before != len(df):
        print(f"  dropped {before - len(df)} interval(s) without a position")
    return df


def load_descriptive(path: str | Path, associated=ASSOCIATED
                     ) -> tuple[pd.DataFrame, dict[str, Associated], tuple[str, ...]]:
    """Zone and the associated elements, one row per row of the file.

    The frame is indexed by `source_row`, so `load` joins it onto the assays
    without trusting row order. A column the file does not carry is left out
    and named in the output; the page then offers one fewer element.
    """
    import pyarrow.parquet as pq

    have = set(pq.read_schema(path).names)
    wanted = [a for a in associated if a[1] in have]
    lacking = [a[1] for a in associated if a[1] not in have]
    if lacking:
        print(f"  not in the file, left out: {', '.join(lacking)}")
    cols = (["zone"] if "zone" in have else []) + [c for _, c, _ in wanted]
    raw = pd.read_parquet(path, columns=cols)
    raw.index = pd.RangeIndex(len(raw), name="source_row")

    out, info = pd.DataFrame(index=raw.index), {}
    zones: tuple[str, ...] = ()
    if "zone" in raw:
        unknown = sorted(set(raw["zone"].dropna()) - set(ZONES))
        if unknown or raw["zone"].isna().any():
            raise ValueError(
                f"zone column holds values outside {ZONES}: "
                f"{unknown or ''}{' and nulls' if raw['zone'].isna().any() else ''}")
        out["zone"] = raw["zone"].astype(str)
        zones = tuple(z for z in ZONES if (out["zone"] == z).any())
    else:
        print("  not in the file, left out: zone")

    for sym, col, unit in wanted:
        c = censor.apply(raw[col])
        out[f"{sym}_value"] = c.value.to_numpy(dtype=float)
        out[f"{sym}_censored"] = c.censored.to_numpy(dtype=bool)
        out[f"{sym}_dl"] = c.dl.to_numpy(dtype=float)
        out[f"{sym}_at_limit"] = c.at_limit.to_numpy(dtype=bool)
        info[sym] = Associated(
            sym=sym, column=col, unit=unit, assayed=c.assayed, share=c.share,
            drawn=0, presence_only=c.presence_only, at_limit=int(c.at_limit.sum()),
            seen_dl=tuple(c.seen_dl))
    return out, info, zones


def load_panel(
    assays: pd.DataFrame,
    baseline_path: str | Path,
    sims_path: str | Path,
    ndp_path: str | Path | None = None,
    elements=ELEMENTS,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], CountLadder]:
    """Assemble the blind-panel evaluation set.

    Returns the per-position frame, the ensembles keyed by method and element,
    and the count ladder explaining how many positions survive to evaluation.
    """
    panel = assays[assays["split"] == "holdout"].reset_index(drop=True)
    baseline = pd.read_parquet(baseline_path)
    sims_npz = np.load(sims_path)

    _verify_alignment("baseline_predictions", panel, baseline)
    _verify_alignment("sgs_realizations", panel, sims_npz)

    merged = pd.concat(
        [panel, baseline.drop(columns=[c for c in ("Hole", "split", *COORDS)
                                       if c in baseline.columns])],
        axis=1,
    )

    # A position is evaluable when it is one of the distinct locations the
    # pipeline flagged AND the estimator returned a value there.
    estimated = merged[f"ok_{elements[0]}_pct"].notna().to_numpy()
    keep = merged["eval_eligible"].to_numpy(dtype=bool) & estimated

    counts = CountLadder(
        intervals=len(panel),
        positions=int(merged["eval_eligible"].sum()),
        estimated=int(keep.sum()),
    )

    ensembles = {"sgs": {e: np.asarray(sims_npz[e])[keep] for e in elements}}

    if ndp_path is not None:
        ndp_npz = np.load(ndp_path)
        # The key check if the export carries one, the coordinate check if not.
        # Both are run against the full blind panel, before `keep` narrows it,
        # so a disagreement is reported at the row number the export itself
        # would show.
        if "row" in getattr(ndp_npz, "files", ()):
            _verify_row_key("ndp_realizations", panel, ndp_npz["row"])
        else:
            _verify_alignment("ndp_realizations", panel, ndp_npz)
        missing = [e for e in elements if e not in ndp_npz.files]
        if missing:
            raise AlignmentError(
                f"ndp_realizations: no array for {', '.join(missing)}. "
                f"It carries {', '.join(sorted(ndp_npz.files))}."
            )
        ensembles["ndp"] = {e: np.asarray(ndp_npz[e])[keep] for e in elements}

        shapes = {m: next(iter(c.values())).shape[1] for m, c in ensembles.items()}
        if len(set(shapes.values())) > 1:
            print(f"  note: draw counts differ between methods -- {shapes}")

    return merged[keep].reset_index(drop=True), ensembles, counts


def load(
    assays_path: str | Path,
    baseline_path: str | Path,
    sims_path: str | Path,
    manifest_path: str | Path,
    ndp_path: str | Path | None = None,
    elements=ELEMENTS,
) -> Thalanga:
    """Load and align every artifact. Raises AlignmentError if they disagree."""
    print("loading Thalanga artifacts")
    manifest = load_manifest(manifest_path)
    assays = load_assays(assays_path, elements)
    print(f"  assays        {len(assays):,} intervals with a position")

    panel, ensembles, counts = load_panel(
        assays, baseline_path, sims_path, ndp_path, elements)
    print(f"  blind panel   {counts.intervals:,} intervals -> "
          f"{counts.positions:,} positions -> {counts.estimated:,} estimated")
    for method, cube in ensembles.items():
        print(f"  {method + ' draws':<13} {counts.estimated:,} x "
              f"{next(iter(cube.values())).shape[1]} per element")

    # Joined after the panel is cut, so the panel frame carries only what the
    # panel page reads.
    desc, associated, zones = load_descriptive(assays_path)
    assays = assays.join(desc, on="source_row")
    # The page states how many points it draws, which can be fewer than the
    # file holds: the rows without a position are not drawn.
    from dataclasses import replace
    associated = {
        s: replace(a, drawn=int((assays[f"{s}_value"].notna()
                                 | assays[f"{s}_censored"]).sum()))
        for s, a in associated.items()}
    if zones:
        n = assays["zone"].value_counts()
        print("  zones         " + ", ".join(f"{z.lower()} {n.get(z, 0):,}"
                                              for z in zones))
    if associated:
        pa = [a.sym for a in associated.values() if a.presence_only]
        print(f"  associated    {len(associated)} elements"
              + (f"; {' and '.join(pa)} mostly below detection, presence only"
                 if pa else ""))

    return Thalanga(
        assays=assays,
        panel=panel,
        ensembles=ensembles,
        box=manifest["box"],
        manifest=manifest,
        counts=counts,
        associated=associated,
        zones=zones,
    )


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

def _self_check(data: Thalanga) -> None:
    """Report the figures a reader can verify against the circulated plots.

    The existing 'Smoothed Against Simulated' page states measured Zn max
    49.70 / sd 5.80, kriged 7.46 / 0.98 and run 1 at 88.86 / 4.51. Reproducing
    those here is the evidence that this loader selected the same samples the
    pipeline did.
    """
    print("\n" + data.counts.as_sentence())
    print(f"\n{'series':<22}{'n':>9}{'max':>9}{'sd':>8}{'mean':>8}")
    meas = data.measured("Zn")
    print(f"{'measured Zn':<22}{len(meas):>9,}{meas.max():>9.2f}"
          f"{meas.std(ddof=1):>8.2f}{meas.mean():>8.3f}")
    ok = data.estimate("Zn", "ok")
    print(f"{'kriged Zn':<22}{len(ok):>9,}{ok.max():>9.2f}"
          f"{ok.std(ddof=1):>8.2f}{ok.mean():>8.3f}")
    r1 = data.realisation("Zn", 0)
    print(f"{'simulated Zn, run 1':<22}{len(r1):>9,}{r1.max():>9.2f}"
          f"{r1.std(ddof=1):>8.2f}{r1.mean():>8.3f}")
    for method in data.methods:
        c = data.centre(method, "Zn")
        print(f"{method + ' median Zn':<22}{len(c):>9,}{c.max():>9.2f}"
              f"{c.std(ddof=1):>8.2f}{c.mean():>8.3f}")
    print("\nexpected from the circulated page: "
          "measured 49.70 / 5.80, kriged 7.46 / 0.98, run 1 88.86 / 4.51")

    # The claim the panel page rests on, restated as a number rather than an
    # impression: a central estimate that collapses, against one that does not.
    meas4 = int((meas >= 4.0).sum())
    print(f"\nat a 4 % Zn cutoff the measured ground holds {meas4:,} positions")
    print(f"  {'kriged':<14}{int((ok >= 4.0).sum()):>8,}")
    for method in data.methods:
        n = int((data.centre(method, "Zn") >= 4.0).sum())
        print(f"  {method + ' median':<14}{n:>8,}")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--assays", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--sims", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ndp", default=None, help="diffusion realisations npz")
    a = p.parse_args(argv)

    data = load(a.assays, a.baseline, a.sims, a.manifest, a.ndp)
    _self_check(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
