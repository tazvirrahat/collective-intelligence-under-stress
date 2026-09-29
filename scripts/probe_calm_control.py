"""Calibration probe: is the AI's effect in calm conditions actually the AI?

C3 differs from C1 in two ways at once -- it has an AI, and it has one more
participant.  The design has no calm size control, so this probe adds one
(C6) and also runs an error-free AI (C3b) to separate the two causes.
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace
from cistress.config import CONDITIONS
from cistress.simulate import simulate
from label_distribution import label, STATES

SEEDS = range(1000)
cases = {
    "C1  12 humans, calm":        CONDITIONS["C1"],
    "C6  13 humans, calm (new)":  replace(CONDITIONS["C1"], n_humans=13),
    "C3  12 humans + AI, calm":   CONDITIONS["C3"],
    "C3b 12 + AI, 0%% error":      replace(CONDITIONS["C3"], ai_error_rate=0.0),
}
print("%-28s %s" % ("", "  ".join("%-10s" % s for s in STATES)))
for name, cfg in cases.items():
    runs = [simulate(cfg, s) for s in SEEDS]
    lb = [label(r, cfg.n_required) for r in runs]
    print("%-28s %s   (%d agents, %d items)" % (
        name, "  ".join("%-10.3f" % (lb.count(s)/len(lb)) for s in STATES),
        cfg.n_agents, cfg.n_items))
