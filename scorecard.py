#!/usr/bin/env python3
"""
How close the estimate came, in figures a resource geologist reads directly.

Four comparisons, each answering a question someone would actually ask of a
blind test. None of them require a statistics background to interpret.

    grade_tonnage   How much ground sits above a cutoff, and at what grade?
    average_grade   Does the estimate account for the metal that is there?
    calibration     When the simulations give a chance, does it hold up?
    elevation_swath Where down the hole is the estimate reliable?

------------------------------------------------------------------------------
WHY NOT R-SQUARED
------------------------------------------------------------------------------
Kriging produces a conditional mean. An assay interval carries the full nugget
and short-range variance that a conditional mean is defined to leave out, so
scoring one against the other returns a poor number no matter how good the
estimate is. On this panel every method lands between -0.1 and -0.6, which
describes what kriging is for rather than how it performed.

The comparisons here avoid that trap by asking about the quantities a mine plan
actually depends on: how much ore, at what grade, and with what confidence.

------------------------------------------------------------------------------
TWO TRUTHS, NOT A WINNER
------------------------------------------------------------------------------
Kriging is locally careful and globally wrong. It minimises error position by
position, and in doing so collapses the grade distribution -- above 8 % Zn it
reports that nothing exists where one sample in nine assayed higher. Simulation
reproduces the distribution and carries honest uncertainty, but any single
realisation is locally noisy by design.

The scorecard is built to show both, because a geologist asked to choose
between them has been given a false choice. Kriging answers "what grade here",
simulation answers "how much above a cutoff, and how sure are we".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# Round numbers a geologist would choose, not quantiles of this particular
# dataset. Rows where nothing was measured above the cutoff are dropped, so
# the ladder self-truncates per element.
DEFAULT_CUTOFFS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0)

# Probability bands for the calibration check. Wider at the ends, where few
# positions land, so every band holds enough samples to mean something.
PROBABILITY_BANDS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0))

LOW, HIGH = 10, 90  # the simulation envelope reported everywhere


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #

@dataclass
class GradeTonnageRow:
    cutoff: float
    measured_pct: float          # % of positions above the cutoff
    measured_grade: float        # mean grade of those positions
    kriged_pct: float
    kriged_grade: float
    sim_pct_p10: float
    sim_pct_p50: float
    sim_pct_p90: float
    sim_grade_p50: float


def grade_tonnage(measured, kriged, sims, cutoffs=DEFAULT_CUTOFFS) -> list[GradeTonnageRow]:
    """Proportion of the panel above each cutoff, and the grade of that portion.

    The standard resource comparison. Proportion of positions stands in for
    tonnage: every assay interval here is the same nominal length, so the
    sample count is proportional to tonnage without needing a density model.

    The simulation columns are a spread across realisations, not one number.
    Each realisation gives its own proportion above the cutoff, and the P10-P90
    of those is the honest statement of what the ground might hold.
    """
    rows = []
    for c in cutoffs:
        above_m = measured >= c
        if not above_m.any():
            break  # nothing measured this high; the ladder ends here
        above_k = kriged >= c
        per_realisation = (sims >= c).mean(axis=0) * 100.0
        sim_grades = [sims[sims[:, i] >= c, i].mean() if (sims[:, i] >= c).any() else 0.0
                      for i in range(sims.shape[1])]
        rows.append(GradeTonnageRow(
            cutoff=float(c),
            measured_pct=float(above_m.mean() * 100),
            measured_grade=float(measured[above_m].mean()),
            kriged_pct=float(above_k.mean() * 100),
            kriged_grade=float(kriged[above_k].mean()) if above_k.any() else 0.0,
            sim_pct_p10=float(np.percentile(per_realisation, LOW)),
            sim_pct_p50=float(np.percentile(per_realisation, 50)),
            sim_pct_p90=float(np.percentile(per_realisation, HIGH)),
            sim_grade_p50=float(np.median(sim_grades)),
        ))
    return rows


@dataclass
class AverageGrade:
    measured: float
    kriged: float
    simulated_p50: float
    simulated_p10: float
    simulated_p90: float

    @property
    def kriged_share(self) -> float:
        """Fraction of the measured metal the kriged estimate accounts for."""
        return self.kriged / self.measured if self.measured else float("nan")

    @property
    def simulated_share(self) -> float:
        return self.simulated_p50 / self.measured if self.measured else float("nan")


def average_grade(measured, kriged, sims) -> AverageGrade:
    """Mean grade across the whole panel, measured against estimated.

    Equal-length intervals again, so the mean grade is proportional to contained
    metal. This is the bluntest question in the scorecard: did the estimate
    account for the metal that is in the ground?
    """
    per_realisation = sims.mean(axis=0)
    return AverageGrade(
        measured=float(measured.mean()),
        kriged=float(kriged.mean()),
        simulated_p50=float(np.median(per_realisation)),
        simulated_p10=float(np.percentile(per_realisation, LOW)),
        simulated_p90=float(np.percentile(per_realisation, HIGH)),
    )


@dataclass
class CalibrationRow:
    low: float
    high: float
    positions: int
    actual_pct: float            # % of those positions genuinely above the cutoff


def calibration(measured, sims, cutoff: float,
                bands=PROBABILITY_BANDS) -> list[CalibrationRow]:
    """Does a simulated probability mean what it says?

    For each position, the share of realisations exceeding the cutoff is a
    stated chance. Grouping positions by that chance and checking how many
    really were above the cutoff tests whether the number can be trusted. A
    well-behaved set of realisations puts roughly 30 % of the 30 % band above
    the cutoff.

    This is the one place the simulations make a claim that can be checked
    position by position rather than in aggregate.
    """
    chance = (sims >= cutoff).mean(axis=1)
    truth = measured >= cutoff
    rows = []
    for lo, hi in bands:
        inside = (chance >= lo) & (chance < hi if hi < 1.0 else chance <= 1.0)
        n = int(inside.sum())
        if n == 0:
            continue
        rows.append(CalibrationRow(
            low=float(lo), high=float(hi), positions=n,
            actual_pct=float(truth[inside].mean() * 100),
        ))
    return rows


@dataclass
class SwathRow:
    elevation: float
    positions: int
    measured: float
    kriged: float
    simulated: float


def elevation_swath(z, measured, kriged, sims, bin_height: float = 50.0,
                    min_positions: int = 150) -> list[SwathRow]:
    """Mean grade by depth slice, measured against estimated.

    Run down elevation rather than along strike. The panel is only 548 m wide
    east to west, which gives two usable bins, but it is 960 m deep, which
    gives ten. This is where kriging shows its strength: it tracks the vertical
    grade profile closely even while collapsing the distribution within each
    slice.
    """
    sim_mean = sims.mean(axis=1)
    edges = np.arange(np.floor(z.min()), np.ceil(z.max()) + bin_height, bin_height)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (z >= lo) & (z < hi)
        n = int(inside.sum())
        if n < min_positions:
            continue
        rows.append(SwathRow(
            elevation=float((lo + hi) / 2),
            positions=n,
            measured=float(measured[inside].mean()),
            kriged=float(kriged[inside].mean()),
            simulated=float(sim_mean[inside].mean()),
        ))
    return rows


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

@dataclass
class ElementScorecard:
    element: str
    positions: int
    cutoff_for_calibration: float
    measured_max: float
    kriged_max: float
    simulated_max: float
    grade_tonnage: list[GradeTonnageRow] = field(default_factory=list)
    average: AverageGrade | None = None
    calibration: list[CalibrationRow] = field(default_factory=list)
    swath: list[SwathRow] = field(default_factory=list)

    def headline(self) -> str:
        """The smoothing result in one sentence, for the page."""
        return (
            f"The highest kriged estimate anywhere in the panel is "
            f"{self.kriged_max:.1f} % {self.element}. The highest measured sample "
            f"is {self.measured_max:.1f} %."
        )


def score_element(data, element: str, calibration_cutoff: float | None = None,
                  cutoffs=DEFAULT_CUTOFFS) -> ElementScorecard:
    """Every comparison for one element, from a loaded Thalanga dataset."""
    measured = data.measured(element)
    kriged = data.estimate(element, "ok")
    sims = data.sims[element].astype(float)
    z = data.panel["Z"].to_numpy(dtype=float)

    if calibration_cutoff is None:
        # A cutoff with enough ground above it to test, and far enough up the
        # distribution to matter. The upper quartile of measured grade puts
        # roughly a quarter of the panel above it for every element.
        calibration_cutoff = float(np.round(np.quantile(measured, 0.75), 1)) or 1.0

    return ElementScorecard(
        element=element,
        positions=len(measured),
        cutoff_for_calibration=calibration_cutoff,
        measured_max=float(measured.max()),
        kriged_max=float(kriged.max()),
        simulated_max=float(sims.max()),
        grade_tonnage=grade_tonnage(measured, kriged, sims, cutoffs),
        average=average_grade(measured, kriged, sims),
        calibration=calibration(measured, sims, calibration_cutoff),
        swath=elevation_swath(z, measured, kriged, sims),
    )


def score_all(data, elements=None, **kw) -> dict[str, ElementScorecard]:
    elements = elements or list(data.sims)
    return {e: score_element(data, e, **kw) for e in elements}


def to_json(cards: dict[str, ElementScorecard], path: str | Path) -> Path:
    """Write the scorecard for the page to read. A few kilobytes.

    The simulation cube is 20.8 MB and never ships. Everything the page needs
    about the realisations is reduced to these summaries at build time.
    """
    payload = {e: asdict(c) for e, c in cards.items()}
    for e, c in cards.items():
        payload[e]["average"]["kriged_share"] = c.average.kriged_share
        payload[e]["average"]["simulated_share"] = c.average.simulated_share
        payload[e]["headline"] = c.headline()
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1))
    return path


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_card(c: ElementScorecard) -> None:
    print(f"\n{'=' * 78}\n{c.element} — blind panel, {c.positions:,} positions\n{'=' * 78}")
    print(c.headline())

    print(f"\ngrade and tonnage above cutoff")
    print(f"{'cutoff':>7}  {'measured':>18}  {'kriged':>18}  {'simulated P10-P90':>22}")
    for r in c.grade_tonnage:
        sim = f"{r.sim_pct_p10:5.1f}–{r.sim_pct_p90:<5.1f}% @{r.sim_grade_p50:5.2f}"
        print(f"{r.cutoff:>6.1f}%  {r.measured_pct:>8.1f}% @{r.measured_grade:>6.2f}  "
              f"{r.kriged_pct:>8.1f}% @{r.kriged_grade:>6.2f}  {sim:>22}")

    a = c.average
    print(f"\naverage grade across the panel")
    print(f"  measured    {a.measured:6.2f} %")
    print(f"  kriged      {a.kriged:6.2f} %   accounts for {a.kriged_share * 100:.0f}% "
          f"of the measured metal")
    print(f"  simulated   {a.simulated_p50:6.2f} %   P10-P90 "
          f"{a.simulated_p10:.2f}–{a.simulated_p90:.2f},  "
          f"accounts for {a.simulated_share * 100:.0f}%")

    print(f"\ndoes a simulated chance hold up?  (cutoff {c.cutoff_for_calibration:g} %)")
    print(f"{'simulations say':>18}  {'positions':>10}  {'really above':>13}")
    for r in c.calibration:
        band = f"{r.low * 100:.0f}–{r.high * 100:.0f}%"
        print(f"{band:>18}  {r.positions:>10,}  {r.actual_pct:>12.1f}%")

    print(f"\nmean grade by elevation slice")
    print(f"{'m RL':>8}  {'positions':>10}  {'measured':>9}  {'kriged':>8}  {'simulated':>10}")
    for r in c.swath:
        print(f"{r.elevation:>8.0f}  {r.positions:>10,}  {r.measured:>8.2f}%  "
              f"{r.kriged:>7.2f}%  {r.simulated:>9.2f}%")


def main(argv=None) -> int:
    import argparse
    import dataio

    p = argparse.ArgumentParser(description="Blind-panel scorecard.")
    p.add_argument("--assays", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--sims", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--elements", nargs="+", default=list(dataio.ELEMENTS))
    p.add_argument("--json", default=None, help="also write the page payload here")
    a = p.parse_args(argv)

    data = dataio.load(a.assays, a.baseline, a.sims, a.manifest)
    cards = score_all(data, a.elements)
    for e in a.elements:
        print_card(cards[e])
    if a.json:
        out = to_json(cards, a.json)
        print(f"\nwrote {out} ({out.stat().st_size / 1e3:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
