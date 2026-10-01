# Collective intelligence under stress

A small agent-based model of a group solving a hidden-profile task. Information starts split across agents. They pass items to each other under a trust rule. Stress is a load parameter that makes human agents less likely to send and less likely to accept. Partway through the run one human agent is removed, and whatever that agent still held alone is gone.

The question is whether the early message log says anything about how the run ends. The short answer from the current dataset: it mostly tells you whether the group was stressed, and very little beyond that. Every breakdown comes from the removed agent having been the only holder of a required item. `main.tex` is the paper, and `results/model_results.txt` has the full numbers.

## Layout

| Path | What it is |
|---|---|
| `src/cistress/config.py` | Parameters and the five conditions |
| `src/cistress/simulate.py` | One run |
| `scripts/label_distribution.py` | Calibration check. Prints label shares for seeds 0–299 and does not write a file |
| `scripts/generate_dataset.py` | Writes the study dataset and the files beside it |
| `scripts/train_models.py` | Baselines, classifiers, and every analysis in the paper's results section |
| `scripts/calm_size_control.py` | Supplementary check: is the calm AI effect the AI, or the thirteenth member? |
| `results/dataset.csv` | 5,000 runs, one row each: label and predictors |
| `results/horizons.csv` | The same predictors over shorter windows |
| `results/outcomes.csv` | Information-state numbers per run. Not predictors |
| `results/label_overview.csv` | Runs counted by condition and label |
| `results/model_results.txt`, `results/horizon.png` | Output of `train_models.py` |
| `DATASET.md` | What each column means |
| `main.tex` | The paper |

## Run

From the repo root, with numpy, pandas, scikit-learn and matplotlib installed:

```
python scripts/label_distribution.py
python scripts/generate_dataset.py
python scripts/train_models.py
python scripts/calm_size_control.py
```

Seeds 0–299 are the calibration set used by `label_distribution.py`. The study files use seeds 10000–14999, and `calm_size_control.py` uses 20000–20999, so none of the sets overlap. Re-running `generate_dataset.py` overwrites the CSVs in `results/`. Generation takes about three minutes, and training about two.

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

Each run gets one label from the information state, and a set of predictors from the message log over ticks 0–199. The two are kept apart so a classifier is not trained on the same numbers that define the answer. The rules and every column are written out in `DATASET.md`.
