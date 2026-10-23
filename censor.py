#!/usr/bin/env python3
"""
Below-detection rule for assay columns, applied per element and per file.

The rule is below_detection_rule_2026-10-23.md. `censor()` below is that
document's reference implementation, copied without change. The two helpers
after it add what the document asks for and the reference does not return:
the "at limit" flag of rule 3's exception, and the presence/absence call of
rule 6.

    python censor.py data/thalanga_modelling.parquet

prints the result for every associated element the regional page draws.

------------------------------------------------------------------------------
WHICH FILE, AND WHY THE WHOLE FILE IS ONE UNIT
------------------------------------------------------------------------------
The rule says to derive limits per file and never to pool files. This module
is applied to thalanga_modelling.parquet as one file. Splitting it further, by
laboratory job, is not possible: the table carries 101 job numbers and 74,830
intervals carry none.

The document asks, before rule 3 is applied, which file and column a mass
point comes from and whether the table has been through a step that replaced
negatives by half the limit. In this file:

    SB_PPM_LAB   2.5 on 54 % of assays, beside -5 codes   censored, DL 5
    BI_PPM_LAB   2.5 on 51 % of assays, beside -5 codes   censored, DL 5
    AS_PPM_LAB   5 on 24 %, equal to the -5 code          at limit, kept

The 2.5 values are already present in the raw ThalangaML_Assays.csv (Sb 5,766
rows, Bi 5,294, alongside -5 codes), so the modelling pipeline did not write
them. Whether the laboratory or an earlier database step did is not known.
Either way the rule sends them to the same place: censored at DL 5.

------------------------------------------------------------------------------
ONE KNOWN DEFECT THE RULE DOES NOT CATCH
------------------------------------------------------------------------------
FE_PCT_LAB has 32 negative values in 3 holes. In TH717 from 314 m to 332 m
they fall by exactly 0.939 per metre to -27.332, which is a filled-down series
and not a set of detection codes. Rule 1 reads each one as below detection
with that limit. They are left as the rule leaves them, because correcting
source data is not this module's job; the count is reported by `main()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reference implementation, verbatim from below_detection_rule_2026-10-23.md
# --------------------------------------------------------------------------- #

ROUND_DL = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])


def censor(col: pd.Series, mass_share: float = 0.20):
    """censored flag, detection limit, value for one element column of one file."""
    v = col.astype(float)
    ok = v.notna()
    censored = pd.Series(False, index=v.index)
    dl = pd.Series(np.nan, index=v.index)
    # rule 1
    neg = ok & (v < 0)
    censored[neg], dl[neg] = True, -v[neg]
    # rule 2
    zero = ok & (v == 0)
    pos_min = v[ok & (v > 0)].min()
    censored[zero], dl[zero] = True, pos_min
    # rule 3
    pos = v[ok & (v > 0)]
    if len(pos):
        counts = pos.value_counts()
        seen_dl = set(np.round(-v[neg].unique(), 9)) if neg.any() else set()
        for val, n in counts.items():
            if n / ok.sum() < mass_share:
                continue
            if any(abs(val - d) < 1e-9 for d in seen_dl):
                continue                      # reported at the limit: a measurement
            half_of_seen = any(abs(val - d / 2) < 1e-9 for d in seen_dl)
            half_of_round = np.any(np.abs(val - ROUND_DL / 2) < 1e-9)
            if half_of_seen or half_of_round:
                m = ok & (v == val)
                censored[m], dl[m] = True, 2 * val
    value = v.where(~censored)
    return censored, dl, value


# --------------------------------------------------------------------------- #
# Additions: rule 3's "at limit" flag and rule 6
# --------------------------------------------------------------------------- #

# Rule 6 says "mostly censored". More than half of the non-missing assays is
# the plain reading of "mostly".
PRESENCE_SHARE = 0.5


def at_limit(col: pd.Series, mass_share: float = 0.20) -> pd.Series:
    """Rule 3's exception: a mass point equal to a limit seen as a negative code.

    The reference skips these with `continue` and returns them as ordinary
    measurements. The document also says to flag them, and this is that flag.
    It covers the mass point only. A value that merely equals a seen limit on
    a few rows is an ordinary measurement and is not flagged.
    """
    v = col.astype(float)
    ok = v.notna()
    flag = pd.Series(False, index=v.index)
    neg = ok & (v < 0)
    if not neg.any():
        return flag
    seen_dl = set(np.round(-v[neg].unique(), 9))
    pos = v[ok & (v > 0)]
    for val, n in pos.value_counts().items():
        if n / ok.sum() >= mass_share and any(abs(val - d) < 1e-9 for d in seen_dl):
            flag |= ok & (v == val)
    return flag


@dataclass
class Censored:
    """One element column after the rule, with what the page needs to say."""

    censored: pd.Series       # bool, below detection
    dl: pd.Series             # the limit where censored, NaN elsewhere
    value: pd.Series          # the measurement where not censored, NaN elsewhere
    at_limit: pd.Series       # bool, rule 3's exception
    assayed: int              # non-missing assays
    share: float              # censored share of the non-missing assays
    seen_dl: list[float]      # limits seen as negative codes, ascending

    @property
    def presence_only(self) -> bool:
        """Rule 6: mostly censored, so a presence/absence map and no grade scale."""
        return self.share > PRESENCE_SHARE


def apply(col: pd.Series, mass_share: float = 0.20) -> Censored:
    """The rule and both additions for one column. Text columns are coerced."""
    v = pd.to_numeric(col, errors="coerce")
    cen, dl, value = censor(v, mass_share)
    ok = v.notna()
    n = int(ok.sum())
    neg = v[ok & (v < 0)]
    return Censored(
        censored=cen, dl=dl, value=value,
        at_limit=at_limit(v, mass_share),
        assayed=n,
        share=float(cen[ok].mean()) if n else 0.0,
        seen_dl=sorted(float(d) for d in np.round(-neg.unique(), 9)),
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

# The ten associated elements the regional page draws, as named in the file.
REPORT_COLUMNS = ("AG_PPM_LAB", "AU_PPM_LAB", "FE_PCT_LAB", "BA_PPM_LAB",
                  "S_PPM_LAB", "AS_PPM_LAB", "MN_PPM_LAB", "SB_PPM_LAB",
                  "BI_PPM_LAB", "CD_PPM_LAB")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Apply the below-detection rule.")
    p.add_argument("assays", help="thalanga_modelling.parquet")
    p.add_argument("--columns", nargs="+", default=list(REPORT_COLUMNS))
    a = p.parse_args(argv)

    cols = {c.split("_")[0].title(): c for c in a.columns}
    df = pd.read_parquet(a.assays, columns=list(cols.values()))
    print(f"{'el':<4}{'assayed':>9}{'censored':>10}{'at limit':>10}  reading")
    for sym, col in cols.items():
        c = apply(df[col])
        neg = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
        read = "presence/absence" if c.presence_only else "graded"
        print(f"{sym:<4}{c.assayed:>9,}{c.share:>9.1%}{int(c.at_limit.sum()):>10,}"
              f"  {read}; {neg:,} negative codes, limits {c.seen_dl or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
