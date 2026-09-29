"""Study dataset: one row per run, written to results/dataset.csv.

Labels are taken from label_distribution.label, which reads only the
information state. Predictors are taken only from the message log, and only
from ticks before the perturbation, so they do not contain the outcome.

Seeds start at STUDY_SEED_START and do not overlap the calibration range
label_distribution.py uses (0-299).
"""

from __future__ import annotations

import csv
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cistress.config import CONDITIONS
from cistress.simulate import simulate
from label_distribution import STATES, label

# Reserved calibration seeds are range(0, 300). Keep the study set clear of them.
STUDY_SEED_START = 10_000
RUNS_PER_CONDITION = 1_000
ROLLING_WINDOW = 10
OUT_PATH = ROOT / "results" / "dataset.csv"
OVERVIEW_PATH = ROOT / "results" / "label_overview.csv"

DESCRIPTIONS = {
    "C1": "calm, humans",
    "C2": "stressed, humans",
    "C3": "calm, humans + AI",
    "C4": "stressed, humans + AI",
    "C5": "stressed, extra human",
}

FEATURE_COLUMNS = (
    "n_messages",
    "mean_messages_per_tick",
    "message_rate_slope",
    "gini_sends",
    "silent_agents",
    "activity_ratio",
    "n_components",
    "density",
    "degree_centralisation",
    "reciprocity",
    "mean_reciprocation_delay",
    "rolling_variance",
    "lag1_autocorr",
    "ai_accept_rate",
    "ai_traffic_ratio",
)


def gini(counts: np.ndarray) -> float:
    arr = np.sort(np.asarray(counts, dtype=float))
    total = arr.sum()
    if total == 0:
        return 0.0
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * total)) - (n + 1) / n)


def lag1_autocorr(series: np.ndarray) -> float:
    if series.size < 3 or np.std(series) == 0:
        return 0.0
    return float(np.corrcoef(series[:-1], series[1:])[0, 1])


def features_from_messages(
    messages: list[dict], n_agents: int, window_end: int, ai_present: bool,
) -> dict:
    """Predictors from the pre-perturbation message log. No item identity, no integration."""
    window = [m for m in messages if m["tick"] < window_end]
    ai_index = n_agents - 1 if ai_present else None
    ticks = np.arange(window_end)
    per_tick = np.zeros(window_end, dtype=float)
    sends = np.zeros(n_agents, dtype=float)

    undirected: set[tuple[int, int]] = set()
    by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    ai_sent = 0
    ai_accepted = 0
    ai_to_human = 0
    human_to_human = 0

    for index, message in enumerate(window):
        per_tick[message["tick"]] += 1
        sends[message["sender"]] += 1
        pair = (message["sender"], message["recipient"])
        by_pair[pair].append(index)
        a, b = pair
        if a > b:
            a, b = b, a
        undirected.add((a, b))

        if message["sender_is_ai"]:
            ai_sent += 1
            ai_accepted += int(message["accepted"])
            ai_to_human += 1
        elif message["recipient"] != ai_index:
            human_to_human += 1

    if window_end >= 2 and per_tick.sum() > 0:
        slope = float(np.polyfit(ticks, per_tick, 1)[0])
    else:
        slope = 0.0

    positive = sends[sends > 0]
    if positive.size and sends.min() > 0:
        activity_ratio = float(sends.max() / sends.min())
    elif positive.size:
        activity_ratio = float("nan")
    else:
        activity_ratio = 0.0

    parent = list(range(n_agents))

    def find(agent: int) -> int:
        while parent[agent] != agent:
            parent[agent] = parent[parent[agent]]
            agent = parent[agent]
        return agent

    for a, b in undirected:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    n_components = len({find(agent) for agent in range(n_agents)})

    possible = n_agents * (n_agents - 1) / 2
    density = len(undirected) / possible if possible else 0.0

    degree = np.zeros(n_agents, dtype=float)
    for a, b in undirected:
        degree[a] += 1
        degree[b] += 1
    if n_agents > 2:
        spread = float(np.sum(degree.max() - degree))
        degree_centralisation = spread / ((n_agents - 1) * (n_agents - 2))
    else:
        degree_centralisation = 0.0

    reciprocated = 0
    delays: list[float] = []
    for index, message in enumerate(window):
        later = by_pair[(message["recipient"], message["sender"])]
        position = bisect_right(later, index)
        if position < len(later):
            reciprocated += 1
            reply = window[later[position]]
            delays.append(reply["tick"] - message["tick"])

    if window_end > ROLLING_WINDOW:
        rolling = np.array([
            per_tick[end - ROLLING_WINDOW:end].var()
            for end in range(ROLLING_WINDOW, window_end + 1)
        ])
        rolling_variance = float(rolling.mean())
    else:
        rolling_variance = 0.0

    return {
        "n_messages": len(window),
        "mean_messages_per_tick": len(window) / window_end if window_end else 0.0,
        "message_rate_slope": slope,
        "gini_sends": gini(sends),
        "silent_agents": int(np.sum(sends == 0)),
        "activity_ratio": activity_ratio,
        "n_components": n_components,
        "density": density,
        "degree_centralisation": degree_centralisation,
        "reciprocity": reciprocated / len(window) if window else 0.0,
        "mean_reciprocation_delay": float(np.mean(delays)) if delays else 0.0,
        "rolling_variance": rolling_variance,
        "lag1_autocorr": lag1_autocorr(per_tick),
        "ai_accept_rate": ai_accepted / ai_sent if ai_sent else 0.0,
        "ai_traffic_ratio": ai_to_human / human_to_human if human_to_human else 0.0,
    }


def write_overview(counts: dict, runs_per: int) -> None:
    """One row per condition: share of runs in each outcome state."""
    OVERVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ("condition", "description", *STATES)
    with OVERVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in counts:
            writer.writerow({
                "condition": name,
                "description": DESCRIPTIONS[name],
                **{state: "%.3f" % (counts[name][state] / runs_per) for state in STATES},
            })


def main() -> None:
    rows = []
    counts = {name: {state: 0 for state in STATES} for name in CONDITIONS}

    for condition_index, (name, cfg) in enumerate(CONDITIONS.items()):
        for offset in range(RUNS_PER_CONDITION):
            seed = STUDY_SEED_START + condition_index * RUNS_PER_CONDITION + offset
            run = simulate(cfg, seed, name)
            state = label(run, cfg.n_required)
            counts[name][state] += 1
            row = {
                "seed": seed,
                "condition": name,
                "label": state,
            }
            row.update(features_from_messages(
                run.messages, run.n_agents, cfg.perturbation_tick, cfg.ai_present))
            rows.append(row)
        print("finished %s (%d runs)" % (name, RUNS_PER_CONDITION))

    integer_features = {"n_messages", "silent_agents", "n_components"}
    formatted = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if key in FEATURE_COLUMNS and key not in integer_features:
                out[key] = "" if value != value else "%.4f" % value
            else:
                out[key] = value
        formatted.append(out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "condition", "label", *FEATURE_COLUMNS))
        writer.writeheader()
        writer.writerows(formatted)

    print("\n%-4s %s" % ("", "  ".join("%-12s" % state for state in STATES)))
    for name in CONDITIONS:
        total = RUNS_PER_CONDITION
        print("%-4s %s" % (
            name,
            "  ".join("%-12.3f" % (counts[name][state] / total) for state in STATES),
        ))
    write_overview(counts, RUNS_PER_CONDITION)
    print("\nwrote %d rows to %s" % (len(rows), OUT_PATH))
    print("wrote the condition summary to %s" % OVERVIEW_PATH)
    last_seed = STUDY_SEED_START + len(CONDITIONS) * RUNS_PER_CONDITION - 1
    print("seeds %d-%d, disjoint from calibration seeds 0-299" % (STUDY_SEED_START, last_seed))


if __name__ == "__main__":
    main()
