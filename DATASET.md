# Dataset columns

`results/dataset.csv` is one row per run: 1,000 runs in each of C1–C5, 5,000 rows in total. Seeds run from 10000 through 14999, with C1 on 10000–10999, C2 on 11000–11999, and so on. The same seed with the same condition reproduces the same run.

`generate_dataset.py` writes three other files next to it:

- `results/label_overview.csv` is the study file counted up: for each condition, the share of runs in each label. Open that one when you want the summary table.
- `results/horizons.csv` has the same predictors computed over shorter windows, for the prediction-horizon analysis. Four rows per run, one for each `window` value (55, 100, 150, 200). The rows with `window` = 200 are identical to `dataset.csv`.
- `results/outcomes.csv` holds information-state numbers for each run: when it completed, final integration, integration at tick 199, how many required items were destroyed, and how many items were held by a single agent at the removal. These are what the labels are built from, so they must never be used as predictors. `train_models.py` uses them only to recompute labels at other thresholds and for the "information-state ceiling" comparison.

Float columns are written to 4 decimal places. A blank cell means the quantity was undefined for that run.

## Where the two halves come from

A run keeps two records.

- Integration is how many of the 12 required items sit on the best-informed living agent's desk at each tick. The label is computed only from this, plus whether the removed agent took any required item with them.
- The message log is who sent what to whom. Every predictor below is computed only from messages with `tick < 200`. One human agent is removed at tick 55, so the window covers the ticks before the removal and 145 ticks of the group's response. Only 5 of the 5,000 runs finished the task inside that window, so the outcome is almost always still open when the window closes.

Before any predictor is computed, each message is cut down to five fields: tick, sender, recipient, whether it was accepted, and whether the sender was the AI. The feature code never sees which item was sent or whether it was a fake.

Every broken-down run in this dataset is broken down because a required item was lost at the removal. The 60% rule below never decided a label at these settings.

## Identifiers

**seed.** Random seed for that run.

**condition.** C1–C5. Who is in the group, and whether human load rises. See the table in `README.md`.

## Label

Checked in this order.

**broken_down.** A required item was held only by the agent who got removed, so the correct option can no longer be assembled. Or the run finishes with the best desk holding under 60% of the required set (fewer than 7.2 of the 12 items; in practice that is 7 or fewer).

**stressed.** The group never got all 12 required items onto one desk, but it did not lose a required item and it finished at or above that 60% line. Partial progress, no recovery to a full set.

**stable.** Some desk reached all 12 required items at tick 450 or earlier, and the run is not broken down. Finished early.

**adaptive.** Some desk reached all 12, but the first time that happened was after tick 450. Finished late, which is the recovery the longer run was meant to leave room for.

## Predictors

All of these are from the message log for ticks 0–199.

Three columns carry no information at these settings, and `train_models.py` drops them. `silent_agents` is 0 and `n_components` is 1 in every run. `mean_messages_per_tick` is `n_messages` divided by a fixed number, so it repeats it.

**n_messages.** How many messages were sent in that window.

**mean_messages_per_tick.** `n_messages / 200` (divided by the `window` value in `horizons.csv`).

**message_rate_slope.** Slope of a straight line fit to the number of messages per tick. Negative means the group was sending less as the window went on.

**gini_sends.** Gini coefficient of how many messages each agent sent. 0 means every agent sent the same number. Higher means a few agents did most of the sending.

**silent_agents.** How many agents sent nothing in the window.

**activity_ratio.** Sends by the busiest agent divided by sends by the quietest. Blank if any agent sent nothing, because that ratio would be a division by zero. 0 if nobody sent at all.

**n_components.** Connected components in the undirected graph of who communicated with whom. A single message in either direction links two agents. An agent with no messages is its own component. 1 means the group was one piece.

**density.** Share of possible undirected pairs that exchanged at least one message. 1 means every pair talked.

**degree_centralisation.** Freeman degree centralisation of that same undirected graph. 0 means every agent had the same number of partners. Higher means the links pile onto one agent.

**reciprocity.** Share of messages that are later answered in the opposite direction. "Later" follows the order of the log, so a reply on the same tick counts only if it was recorded after the original message.

**mean_reciprocation_delay.** For messages that did get a reply, the average number of ticks until the first reply. 0 if nothing was reciprocated.

**rolling_variance.** Take the message count in each tick, compute the variance inside every 10-tick window, then average those variances. High means the sending rate was jumpy.

**lag1_autocorr.** Correlation between the per-tick message count and the same series shifted by one tick. Near 1 means a busy tick tends to follow a busy tick. Recorded as 0 when the series is too flat to correlate.

**ai_accept_rate.** Of the messages the AI sent in the window, the share the recipient accepted. 0 when the AI sent nothing, which includes every human-only run (C1, C2, C5).

**ai_traffic_ratio.** Messages from the AI to a human, divided by messages from one human to another human. Messages from a human to the AI are left out of the denominator. 0 when there is no human-to-human traffic, and on human-only runs.
