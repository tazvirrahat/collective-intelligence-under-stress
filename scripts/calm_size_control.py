"""Supplementary check: is the calm-condition AI effect the AI, or group size?

C3 differs from C1 in two ways at once: it has an AI, and it has a thirteenth
participant.  The design has a size control only under stress (C5).  This
adds a calm one (C6, 13 humans) and an error-free AI (C3b) to separate the two.

Seeds are drawn from SEED_START upward, clear of both the calibration range
(0-299) and the study range (10000-14999).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cistress.config import CONDITIONS
from cistress.simulate import simulate
from label_distribution import STATES, label

SEED_START = 20_000
RUNS = 1_000

CASES = {
    "C1   12 humans, calm": CONDITIONS["C1"],
    "C6   13 humans, calm": replace(CONDITIONS["C1"], n_humans=13),
    "C3   12 humans + AI, calm": CONDITIONS["C3"],
    "C3b  12 humans + AI, 0% error": replace(CONDITIONS["C3"], ai_error_rate=0.0),
}


def main() -> None:
    print("%-30s %s" % ("", "  ".join("%-11s" % s for s in STATES)))
    for name, cfg in CASES.items():
        labels = [label(simulate(cfg, seed), cfg.n_required)
                  for seed in range(SEED_START, SEED_START + RUNS)]
        print("%-30s %s" % (name, "  ".join("%-11.3f" % (labels.count(s) / RUNS) for s in STATES)))


if __name__ == "__main__":
    main()
