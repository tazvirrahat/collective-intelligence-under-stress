# Dataset columns

`results/dataset.csv` is one row per run: 1,000 runs in each of C1–C5, 5,000 rows in total. Seeds run from 10000 through 14999, with C1 on 10000–10999, C2 on 11000–11999, and so on. The same seed with the same condition reproduces the same run.

`results/label_overview.csv` is not a second dataset. It is the study file counted up: for each condition, the share of runs in each label. Open that one when you want the summary table.

Float columns in `dataset.csv` are written to 4 decimal places. A blank cell means the quantity was undefined for that run.

## Where the two halves come from

A run keeps two records.

- Integration is how many of the 12 required items sit on the best-informed living agent's desk at each tick. The label is computed only from this, plus whether the removed agent took any required item with them.
- The message log is who sent what to whom. Every predictor below is computed only from messages with `tick < 55`. Tick 55 is when one human agent is removed, so these numbers are from before the disruption. None of them use item identity or the integration curve.

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

All of these are from the pre-removal message log (ticks 0–54).

**n_messages.** How many messages were sent in that window.

**mean_messages_per_tick.** `n_messages / 55`.

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
