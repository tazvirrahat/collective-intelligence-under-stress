"""Calibration probe: does stress produce the fragility the study depends on?

Runs C1 (human-only, calm) against C2 (human-only, stressed) and plots the two
readings of the information state -- the level ("odometer") and its smoothed
rate of change ("speedometer") -- around the perturbation.

These are calibration runs.  Seeds here are reserved and excluded from the
study dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cistress.config import CONDITIONS
from cistress.simulate import simulate

CALIBRATION_SEEDS = range(0, 200)
RATE_WINDOW = 20


def arrival_rate(integration: list[int], window: int = RATE_WINDOW) -> np.ndarray:
    """Smoothed first difference: how fast the fullest desk is filling up."""
    arr = np.asarray(integration, dtype=float)
    out = np.zeros_like(arr)
    out[window:] = (arr[window:] - arr[:-window]) / window
    return out


def main() -> None:
    cfg_any = CONDITIONS["C1"]
    summary = {}
    traces = {}

    for name in ("C1", "C2"):
        cfg = CONDITIONS[name]
        runs = [simulate(cfg, seed, name) for seed in CALIBRATION_SEEDS]
        integ = np.array([r.integration for r in runs], dtype=float)
        rates = np.array([arrival_rate(r.integration) for r in runs])
        traces[name] = (integ.mean(axis=0), rates.mean(axis=0))

        final = integ[:, -1]
        destroyed = np.array([len(r.destroyed_required) for r in runs])
        at_perturb = integ[:, cfg.perturbation_tick - 1]
        summary[name] = {
            "mean final integration": final.mean(),
            "reached all %d" % cfg.n_required: float((final >= cfg.n_required).mean()),
            "integration at perturbation": at_perturb.mean(),
            "runs losing a required item": float((destroyed > 0).mean()),
            "mean messages/run": float(np.mean([len(r.messages) for r in runs])),
        }

    label = {"C1": "C1 calm", "C2": "C2 stressed"}
    colour = {"C1": "#2a6f9e", "C2": "#c2593a"}

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name in ("C1", "C2"):
        lvl, rate = traces[name]
        axes[0].plot(lvl, color=colour[name], label=label[name], lw=1.8)
        axes[1].plot(rate, color=colour[name], label=label[name], lw=1.8)

    for ax in axes:
        ax.axvline(cfg_any.perturbation_tick, color="#888", ls="--", lw=1)
        ax.axvspan(cfg_any.ramp_start, cfg_any.ramp_end, color="#c2593a", alpha=0.07)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("integration\n(required items, fullest desk)")
    axes[0].set_title("Information state, mean of %d runs per condition" % len(CALIBRATION_SEEDS))
    axes[1].set_ylabel("arrival rate\n(smoothed, items/tick)")
    axes[1].set_xlabel("tick   (dashed = perturbation, shaded = stress ramp)")

    fig.tight_layout()
    out = Path("results/calm_vs_stressed.png")
    fig.savefig(out, dpi=140)

    for name, stats in summary.items():
        print("\n%s" % label[name])
        for key, value in stats.items():
            print("  %-32s %.3f" % (key, value))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
