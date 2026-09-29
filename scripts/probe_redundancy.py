"""Calibration probe: how much redundancy has the group built when disruption arrives?

Only information still held by exactly one agent can be destroyed by the
perturbation.  This measures the size of that pool at the perturbation tick.
"""

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataclasses import replace
from cistress.config import CONDITIONS
from cistress.simulate import simulate

SEEDS = range(200)
print("At the perturbation tick (55), 12 agents:\n")
print("ipa items  cond   unique items left  as %% of all  removed agent's   %% of sends that")
print("                     (redundancy gap)              unshared items    were own original")
for ipa in (1, 2, 3, 5):
    for name in ("C1", "C2"):
        cfg = replace(CONDITIONS[name], n_humans=12, items_per_agent=ipa,
                      n_required=max(2, round(12 * ipa * 0.67)))
        runs = [simulate(cfg, s) for s in SEEDS]
        uniq = np.mean([r.unique_items_at_perturbation for r in runs])
        unsh = np.mean([r.removed_originals_unshared for r in runs])
        own = np.mean([r.own_item_sends / max(r.total_sends, 1) for r in runs])
        print("%3d %5d  %-5s %14.2f %13.1f%% %16.2f %17.1f%%" % (
            ipa, cfg.n_items, name, uniq, 100 * uniq / cfg.n_items, unsh, 100 * own))
