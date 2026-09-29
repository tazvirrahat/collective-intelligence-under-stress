"""Calibration probe: does the AI end up solving the task itself?

Integration is the count on the best-informed agent's desk.  Nothing in the
model stops that agent from being the AI -- and because the AI is load-immune
it keeps absorbing at full rate while stressed humans do not.  If the AI is
frequently the best-informed agent, then "the group integrated the
information" is really "the AI did", which is not collective intelligence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cistress.config import CONDITIONS
from cistress.simulate import simulate

SEEDS = range(500)

print("%-22s %10s %10s %10s %12s" % (
    "", "AI leads", "integration", "humans only", "gap"))
for name in ("C3", "C4"):
    cfg = CONDITIONS[name]
    runs = [simulate(cfg, s, name) for s in SEEDS]
    lead = np.mean([r.leader_is_ai for r in runs])
    both = np.mean([r.integration[-1] for r in runs])
    human = np.mean([r.human_integration[-1] for r in runs])
    desc = "%s %s" % (name, "calm" if not cfg.stressed else "stressed")
    print("%-22s %9.1f%% %10.2f %10.2f %12.2f" % (desc, 100 * lead, both, human, both - human))
