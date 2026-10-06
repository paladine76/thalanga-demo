#!/usr/bin/env python3
"""
Load the Thalanga artifacts and align them into one evaluation set.

Four files come out of the modelling pipeline, and nothing in them carries a
shared primary key. This module joins them, proves the join is sound, and
refuses to continue if it is not -- a silent misalignment here would corrupt
every figure downstream while still rendering perfectly.

    thalanga_modelling.parquet      77,222 assay intervals, all splits
    baseline_predictions.parquet    12,985 blind-panel rows: OK, IDW, means
    sgs_realizations.npz            12,985 x 100 conditional simulations
    manifest.json                   blind-panel extent and run provenance

------------------------------------------------------------------------------
WHY THE JOIN NEEDS CARE
------------------------------------------------------------------------------
(Hole, X, Y, Z) is NOT unique. Hole SE796F1920-3 has 13 downhole intervals that
all desurvey to the same collar position, each with its own assay -- 12.0, 18.7,
14.7 % Zn and so on. Merging on those four columns inflates 12,985 rows to
15,169 and every statistic computed afterwards is wrong.

The prediction exports do not carry From/To, so there is no key to merge on.
What they do have is identical row order, inherited from the pipeline that
wrote them. This module relies on that order and then verifies it, position by
position, against Hole and all three coordinates. If a future run writes the
rows sorted differently the check fails loudly instead of quietly producing
plausible nonsense.

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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ELEMENTS = ("Zn", "Pb", "Cu")

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


@dataclass
class Thalanga:
    """Everything the figures need, already aligned.

    `assays` is every interval in the deposit, for the regional view.
    `panel` is the blind-panel evaluation set: measured assays, each estimator,
    and the simulations, one row per position, in a single frame.
    `sims` maps element -> (n_positions, n_realisations) of simulated grade.
    """

    assays: pd.DataFrame
    panel: pd.DataFrame
    sims: dict[str, np.ndarray]
    box: dict[str, float]
    manifest: dict
    counts: CountLadder

    @property
    def n_realisations(self) -> int:
        return next(iter(self.sims.values())).shape[1]

    def measured(self, element: str) -> np.ndarray:
        return self.panel[f"{element}_pct"].to_numpy(dtype=float)

    def estimate(self, element: str, method: str = "ok") -> np.ndarray:
        return self.panel[f"{method}_{element}_pct"].to_numpy(dtype=float)

    def realisation(self, element: str, run: int) -> np.ndarray:
        return self.sims[element][:, run].astype(float)


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
    before = len(df)
    df = df.dropna(subset=list(COORDS)).reset_index(drop=True)
    if before != len(df):
        print(f"  dropped {before - len(df)} interval(s) without a position")
    return df


def load_panel(
    assays: pd.DataFrame,
    baseline_path: str | Path,
    sims_path: str | Path,
    elements=ELEMENTS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], CountLadder]:
    """Assemble the blind-panel evaluation set.

    Returns the per-position frame, the simulation cube per element, and the
    count ladder explaining how many positions survive to evaluation.
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

    sims = {e: np.asarray(sims_npz[e])[keep] for e in elements}
    return merged[keep].reset_index(drop=True), sims, counts


def load(
    assays_path: str | Path,
    baseline_path: str | Path,
    sims_path: str | Path,
    manifest_path: str | Path,
    elements=ELEMENTS,
) -> Thalanga:
    """Load and align every artifact. Raises AlignmentError if they disagree."""
    print("loading Thalanga artifacts")
    manifest = load_manifest(manifest_path)
    assays = load_assays(assays_path, elements)
    print(f"  assays        {len(assays):,} intervals with a position")

    panel, sims, counts = load_panel(assays, baseline_path, sims_path, elements)
    print(f"  blind panel   {counts.intervals:,} intervals -> "
          f"{counts.positions:,} positions -> {counts.estimated:,} estimated")
    print(f"  simulations   {counts.estimated:,} x "
          f"{next(iter(sims.values())).shape[1]} realisations per element")

    return Thalanga(
        assays=assays,
        panel=panel,
        sims=sims,
        box=manifest["box"],
        manifest=manifest,
        counts=counts,
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
    print("\nexpected from the circulated page: "
          "measured 49.70 / 5.80, kriged 7.46 / 0.98, run 1 88.86 / 4.51")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--assays", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--sims", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args(argv)

    data = load(a.assays, a.baseline, a.sims, a.manifest)
    _self_check(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
