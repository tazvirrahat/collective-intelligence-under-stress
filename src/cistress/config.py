"""Simulation parameters.

Every value here is provisional and fixed by the calibration phase before the
study dataset is generated (see Section III-G).  Nothing in this file may be
revised after the pre-registration tag is cut.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fake items emitted by an unreliable AI agent are drawn from a reserved range
# so they can never collide with a genuine item id.  Agents cannot tell the
# difference -- items carry no content -- but integration never counts them.
FAKE_ITEM_BASE = 100_000


@dataclass(frozen=True)
class Config:
    # --- group and task -------------------------------------------------
    n_humans: int = 12
    items_per_agent: int = 3
    n_required: int = 16

    # --- run length -----------------------------------------------------
    ticks: int = 800
    perturbation_tick: int = 55

    # --- stress schedule ------------------------------------------------
    # Load is the single controlled stressor.  It suppresses transmission and
    # reception alike, so the two effects compound; the peak is held well below
    # 1.0 to keep the compounded swing around 5x rather than 60x.
    load_calm: float = 0.10
    load_peak: float = 0.45
    ramp_start: int = 10
    ramp_end: int = 50

    # --- transmission ---------------------------------------------------
    base_transmit_rate: float = 0.30

    # --- trust ----------------------------------------------------------
    trust_init: float = 0.50
    trust_increment: float = 0.05
    # Asymmetric: trust is slower to lose than to gain.  Symmetric updates put the
    # bistable tipping point at 50% acceptance, which is unreachable under load,
    # so trust drained to the floor in every condition and no group functioned.
    trust_decrement: float = 0.02
    trust_min: float = 0.05   # never 0: a dead route must stay recoverable
    trust_max: float = 1.00

    # --- artificial participant -----------------------------------------
    ai_present: bool = False
    ai_error_rate: float = 0.15
    ai_trust_weight: float = 1.00  # 1.0 = neutral; sweep for over/under-reliance

    # --- condition ------------------------------------------------------
    stressed: bool = False

    @property
    def n_agents(self) -> int:
        return self.n_humans + (1 if self.ai_present else 0)

    @property
    def n_items(self) -> int:
        return self.n_agents * self.items_per_agent


CONDITIONS = {
    "C1": Config(ai_present=False, stressed=False),
    "C2": Config(ai_present=False, stressed=True),
    "C3": Config(ai_present=True, stressed=False),
    "C4": Config(ai_present=True, stressed=True),
    # size control: one extra human in place of the AI, so C4 and C5 hold both
    # agent count and item count constant and differ only in what the extra
    # participant is.
    "C5": Config(ai_present=False, n_humans=13, stressed=True),
}
