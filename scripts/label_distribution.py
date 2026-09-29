"""Calibration gate: are all four outcome states populated?

Labels are derived exclusively from the information state, never from the
message log (Section III-F).  The thresholds below are PROVISIONAL -- they are
what the calibration phase exists to set, and they are frozen before the study
dataset is generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cistress.config import CONDITIONS
from cistress.simulate import simulate

CALIBRATION_SEEDS = range(0, 300)

STABLE_BY_TICK = 450      # completed this early -> never really disturbed
DEPRESSED_FRACTION = 0.60  # below this share of the required set -> broken down

STATES = ("stable", "adaptive", "stressed", "broken_down")


def label(run, n_required: int) -> str:
    integ = np.asarray(run.integration)
    hit = np.flatnonzero(integ >= n_required)
    completed_at = int(hit[0]) if hit.size else None
    final_fraction = integ[-1] / n_required

    # A destroyed required item makes the correct option unidentifiable for the
    # rest of the run, whatever the group does afterwards.
    if run.destroyed_required or final_fraction < DEPRESSED_FRACTION:
        return "broken_down"
    if completed_at is None:
        return "stressed"
    return "stable" if completed_at <= STABLE_BY_TICK else "adaptive"


def main() -> None:
    print("%-4s %-22s %s" % ("", "", "  ".join("%-11s" % s for s in STATES)))
    rows = {}
    for name, cfg in CONDITIONS.items():
        runs = [simulate(cfg, s, name) for s in CALIBRATION_SEEDS]
        labels = [label(r, cfg.n_required) for r in runs]
        counts = {s: labels.count(s) / len(labels) for s in STATES}
        rows[name] = counts
        desc = "%s, %s" % ("human+AI" if cfg.ai_present else "human-only",
                           "stressed" if cfg.stressed else "calm")
        print("%-4s %-22s %s" % (name, desc,
              "  ".join("%-11.3f" % counts[s] for s in STATES)))

    pooled = {s: np.mean([rows[n][s] for n in rows]) for s in STATES}
    print("\n%-27s %s" % ("POOLED", "  ".join("%-11.3f" % pooled[s] for s in STATES)))
    print("\nmajority-class baseline: %.3f" % max(pooled.values()))
    print("smallest class:          %.3f" % min(pooled.values()))


if __name__ == "__main__":
    main()
