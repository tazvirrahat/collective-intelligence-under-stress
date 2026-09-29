"""Single-run engine.

One run produces two artifacts that are kept strictly separate:

  * ``messages``  -- the communication record.  Predictive features are drawn
    from this and nothing else.
  * ``integration`` -- the information state over time.  Outcome labels are
    drawn from this and nothing else.

Keeping them apart is the central methodological commitment of the study
(Section III-F): features derived from the same source as labels would make the
prediction task circular.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import FAKE_ITEM_BASE, Config


@dataclass
class RunResult:
    seed: int
    condition: str
    integration: list[int]           # per tick: required items on the fullest desk
    human_integration: list[int]     # same, but ignoring the AI: what the people achieved
    messages: list[dict]             # the communication record
    required: set[int]
    removed_agent: int | None
    destroyed_required: set[int]     # required items lost with the removed agent
    n_agents: int
    # redundancy actually built by the time disruption arrives
    unique_items_at_perturbation: int = 0      # items still held by exactly one agent
    removed_originals_unshared: int = 0        # of the removed agent's own items
    own_item_sends: int = 0                    # sends that were the sender's own original
    leader_is_ai: bool = False                 # was the AI the best-informed agent at the end?
    total_sends: int = 0


def load_at(tick: int, cfg: Config, is_ai: bool) -> float:
    """Load for one agent at one tick.

    The AI is load-immune: computational participants do not experience social
    or cognitive stress, so its load is held at the calm value throughout.
    """
    if is_ai or not cfg.stressed:
        return cfg.load_calm
    if tick <= cfg.ramp_start:
        return cfg.load_calm
    if tick >= cfg.ramp_end:
        return cfg.load_peak
    frac = (tick - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start)
    return cfg.load_calm + frac * (cfg.load_peak - cfg.load_calm)


def simulate(cfg: Config, seed: int, condition: str = "") -> RunResult:
    rng = random.Random(seed)
    n = cfg.n_agents
    ai_index = n - 1 if cfg.ai_present else None
    is_ai = [i == ai_index for i in range(n)]

    # --- distribute unique items -------------------------------------------
    # Every item has exactly one holder at initialisation.  Redundancy is not
    # given; it accumulates through transmission over the course of the run,
    # which is what makes a group's resilience to agent loss a product of its
    # own prior communication.
    pool = list(range(cfg.n_items))
    rng.shuffle(pool)
    k = cfg.items_per_agent
    held: list[set[int]] = [set(pool[i * k:(i + 1) * k]) for i in range(n)]
    originals: list[set[int]] = [set(h) for h in held]
    required = set(rng.sample(range(cfg.n_items), cfg.n_required))

    trust = [[cfg.trust_init] * n for _ in range(n)]

    alive = [True] * n
    removed: int | None = None
    destroyed: set[int] = set()
    unique_at_pert = 0
    removed_unshared = 0
    own_sends = 0
    all_sends = 0

    integration: list[int] = []
    human_integration: list[int] = []
    messages: list[dict] = []

    for tick in range(cfg.ticks):
        if tick == cfg.perturbation_tick:
            # How much redundancy had the group actually built by now?
            counts: dict[int, int] = {}
            for i in range(n):
                for it in held[i]:
                    counts[it] = counts.get(it, 0) + 1
            unique_at_pert = sum(1 for c in counts.values() if c == 1)
            # The removed agent is always human, so that AI presence is held
            # constant across the perturbation.
            candidates = [i for i in range(n) if alive[i] and not is_ai[i]]
            removed = rng.choice(candidates)
            alive[removed] = False
            removed_unshared = sum(1 for it in originals[removed] if counts.get(it, 0) == 1)
            reachable: set[int] = set()
            for i in range(n):
                if alive[i]:
                    reachable |= held[i]
            destroyed = required - reachable

        order = [i for i in range(n) if alive[i]]
        rng.shuffle(order)

        for s in order:
            load_s = load_at(tick, cfg, is_ai[s])
            if rng.random() >= cfg.base_transmit_rate * (1.0 - load_s):
                continue

            # --- choose an item ---------------------------------------------
            if is_ai[s] and rng.random() < cfg.ai_error_rate:
                # A well-formed identifier the AI does not actually hold.  The
                # recipient cannot detect this; only the analyst can.
                item = FAKE_ITEM_BASE + rng.randrange(cfg.n_items)
                genuine = False
            else:
                if not held[s]:
                    continue
                item = rng.choice(sorted(held[s]))
                genuine = item < FAKE_ITEM_BASE

            # --- choose a recipient -----------------------------------------
            # Weighted by the sender's trust, over every other agent including
            # one that has been removed: nobody is notified of the removal, so
            # senders keep routing to an absent partner until repeated failures
            # push their trust down and traffic redistributes on its own.
            others = [j for j in range(n) if j != s]
            recipient = rng.choices(others, weights=[trust[s][j] for j in others], k=1)[0]

            accepted = False
            if alive[recipient]:
                load_r = load_at(tick, cfg, is_ai[recipient])
                p_accept = trust[recipient][s] * (1.0 - load_r)
                if is_ai[s]:
                    p_accept *= cfg.ai_trust_weight
                if rng.random() < p_accept:
                    accepted = True
                    held[recipient].add(item)

            if accepted:
                trust[s][recipient] = min(cfg.trust_max, trust[s][recipient] + cfg.trust_increment)
            else:
                trust[s][recipient] = max(cfg.trust_min, trust[s][recipient] - cfg.trust_decrement)

            all_sends += 1
            own_sends += (item in originals[s])
            messages.append({
                "tick": tick,
                "sender": s,
                "recipient": recipient,
                "item": item,
                "accepted": accepted,
                "sender_is_ai": is_ai[s],
                "item_genuine": genuine,
            })

        scores = {i: len(held[i] & required) for i in range(n) if alive[i]}
        integration.append(max(scores.values()))
        human_integration.append(max(
            (v for i, v in scores.items() if not is_ai[i]), default=0))

    return RunResult(
        seed=seed,
        condition=condition,
        integration=integration,
        human_integration=human_integration,
        messages=messages,
        required=required,
        removed_agent=removed,
        destroyed_required=destroyed,
        n_agents=n,
        unique_items_at_perturbation=unique_at_pert,
        removed_originals_unshared=removed_unshared,
        own_item_sends=own_sends,
        total_sends=all_sends,
        leader_is_ai=bool(ai_index is not None and alive[ai_index]
                          and scores[ai_index] == max(scores.values())),
    )
