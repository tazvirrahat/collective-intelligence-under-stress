# Collective intelligence under stress

A small agent-based model of a group solving a hidden-profile task. Information starts split across agents. They pass items to each other under a trust rule. Stress is a load parameter that makes human agents less likely to send and less likely to accept. Partway through the run one human agent is removed, and whatever that agent still held alone is gone.

The question for later modelling is whether the message log, read before that removal, says anything about how the run ends.

## Layout

| Path | What it is |
|---|---|
| `src/cistress/config.py` | Parameters and the five conditions |
| `src/cistress/simulate.py` | One run |
| `scripts/label_distribution.py` | Calibration check. Prints label shares for seeds 0–299 and does not write a file |
| `scripts/generate_dataset.py` | Writes the study dataset |
| `results/dataset.csv` | 5,000 runs, one row each |
| `results/label_overview.csv` | Those runs counted by condition and label |
| `DATASET.md` | What each column means |

## Run

From the repo root, with numpy installed:

```
python scripts/label_distribution.py
python scripts/generate_dataset.py
```

Seeds 0–299 are the calibration set used by `label_distribution.py`. The study file uses seeds 10000–14999, so the two sets do not overlap. Re-running `generate_dataset.py` overwrites both CSVs.

## Conditions

| | Agents | Stress |
|---|---|---|
| C1 | 12 humans | calm, load stays at 0.10 |
| C2 | 12 humans | load rises from 0.10 to 0.32 |
| C3 | 12 humans and one AI | calm |
| C4 | 12 humans and one AI | stressed |
| C5 | 13 humans | stressed |

C5 is the size control for C4. Both have 13 agents. They differ in whether the extra participant is an AI.

The AI does not feel load. On 15% of its sends it emits an item id it does not hold. Those fakes are never counted as integration.

## Labels and columns

Each run gets one label from the information state, and a set of predictors from the message log before tick 55. The two are kept apart so a classifier is not trained on the same numbers that define the answer. The rules and every column are written out in `DATASET.md`.
