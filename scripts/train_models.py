"""Classifiers and evaluation for Sections III-H and III-I.

Reads the files written by generate_dataset.py and writes
results/model_results.txt and results/horizon.png.

Two comparisons are added to the planned baselines.  A condition-only model
shows how much of the score comes from recognising which condition a run was
in.  An information-state ceiling fits the same forest to quantities the
predictors are not allowed to see (integration and redundancy at the end of
the feature window), which bounds what any message-log model could reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_dataset import FEATURE_COLUMNS, FEATURE_WINDOW, HORIZONS
from label_distribution import DEPRESSED_FRACTION, STABLE_BY_TICK, STATES

RESULTS = ROOT / "results"
SPLIT_SEED = 0
TEST_SIZE = 0.20
FOLDS = 5

# Exact rescaling of n_messages at a fixed window length.
DUPLICATE_FEATURES = ("mean_messages_per_tick",)

STABLE_GRID = (350, 450, 550)
DEPRESSED_GRID = (0.60, 0.70, 0.80)

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def logistic() -> object:
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def cv_f1(model, X, y) -> tuple[float, float]:
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=folds, scoring="f1_macro")
    return float(scores.mean()), float(scores.std())


def relabel(outcomes: pd.DataFrame, stable_by: int, depressed: float) -> np.ndarray:
    completed = pd.to_numeric(outcomes["completed_at"], errors="coerce")
    fraction = outcomes["final_integration"] / outcomes["n_required"]
    broken = (outcomes["n_destroyed_required"] > 0) | (fraction < depressed)
    labels = np.where(completed.isna(), "stressed",
                      np.where(completed <= stable_by, "stable", "adaptive"))
    return np.where(broken, "broken_down", labels).astype(object)


def main() -> None:
    data = pd.read_csv(RESULTS / "dataset.csv")
    horizons = pd.read_csv(RESULTS / "horizons.csv")
    outcomes = pd.read_csv(RESULTS / "outcomes.csv", keep_default_na=False,
                           na_values={"completed_at": [""]})
    assert (data["seed"].to_numpy() == outcomes["seed"].to_numpy()).all()

    y = data["label"].to_numpy(dtype=object)
    assert (relabel(outcomes, STABLE_BY_TICK, DEPRESSED_FRACTION) == y).all()

    constant = [c for c in FEATURE_COLUMNS if data[c].nunique(dropna=False) <= 1]
    features = [c for c in FEATURE_COLUMNS if c not in constant and c not in DUPLICATE_FEATURES]
    X = data[features].fillna(-1).to_numpy(float)
    conditions = pd.get_dummies(data["condition"]).to_numpy(float)

    say("Feature window: ticks 0-%d.  Runs: %d." % (FEATURE_WINDOW - 1, len(data)))
    say("Dropped as constant: %s" % (", ".join(constant) or "none"))
    say("Dropped as duplicate: %s" % ", ".join(DUPLICATE_FEATURES))
    say("Model features (%d): %s" % (len(features), ", ".join(features)))

    completed = pd.to_numeric(outcomes["completed_at"], errors="coerce")
    say("Runs that completed inside the feature window: %d" % int((completed < FEATURE_WINDOW).sum()))
    broken = data["label"] == "broken_down"
    say("Broken-down runs caused by a destroyed required item: %d of %d"
        % (int((outcomes["n_destroyed_required"][broken] > 0).sum()), int(broken.sum())))

    index = np.arange(len(data))
    train, test = train_test_split(index, test_size=TEST_SIZE, stratify=y,
                                   random_state=SPLIT_SEED)

    # --- main comparison ----------------------------------------------------
    say("\n== Four-class results (train CV macro-F1, then held-out test)")
    say("%-28s %14s %9s %9s" % ("model", "CV macro-F1", "test F1", "test acc"))
    volume = [features.index("n_messages")]
    info_cols = ["integration_at_window", "unique_items_at_perturbation",
                 "removed_originals_unshared", "n_destroyed_required"]
    info = outcomes[info_cols].to_numpy(float)
    models = {
        "majority class": (DummyClassifier(strategy="most_frequent"), X),
        "volume only (logistic)": (logistic(), X[:, volume]),
        "condition only (logistic)": (logistic(), conditions),
        "logistic regression": (logistic(), X),
        "random forest": (forest(), X),
        "random forest + condition": (forest(), np.hstack([X, conditions])),
        "info-state ceiling (forest)": (forest(), info),
        "  ceiling, no destroyed flag": (forest(), info[:, :3]),
    }
    fitted = {}
    for name, (model, matrix) in models.items():
        mean, sd = cv_f1(model, matrix[train], y[train])
        model.fit(matrix[train], y[train])
        pred = model.predict(matrix[test])
        fitted[name] = (model, matrix, pred)
        say("%-28s %8.3f+/-%.3f %9.3f %9.3f" % (
            name, mean, sd, f1_score(y[test], pred, average="macro"),
            accuracy_score(y[test], pred)))

    for name in ("logistic regression", "random forest"):
        pred = fitted[name][2]
        say("\n-- %s, held-out test, per class" % name)
        p, r, f, s = precision_recall_fscore_support(y[test], pred, labels=list(STATES),
                                                     zero_division=0)
        say("%-12s %9s %7s %6s %8s" % ("", "precision", "recall", "F1", "support"))
        for i, state in enumerate(STATES):
            say("%-12s %9.3f %7.3f %6.3f %8d" % (state, p[i], r[i], f[i], s[i]))
        say("confusion matrix (rows true, columns predicted, order %s)" % ", ".join(STATES))
        for row in confusion_matrix(y[test], pred, labels=list(STATES)):
            say("  " + " ".join("%5d" % v for v in row))

    # --- within each condition ------------------------------------------
    say("\n== Within one condition (5-fold CV macro-F1; labels with >= 20 runs)")
    say("%-4s %8s %8s %8s   labels" % ("", "forest", "logistic", "chance"))
    for cond in sorted(data["condition"].unique()):
        mask = (data["condition"] == cond).to_numpy()
        present = data.loc[mask, "label"].value_counts()
        keep = mask & data["label"].isin(present[present >= 20].index).to_numpy()
        rf_mean, _ = cv_f1(forest(), X[keep], y[keep])
        lr_mean, _ = cv_f1(logistic(), X[keep], y[keep])
        chance, _ = cv_f1(DummyClassifier(strategy="stratified", random_state=0), X[keep], y[keep])
        say("%-4s %8.3f %8.3f %8.3f   %s" % (cond, rf_mean, lr_mean, chance,
                                           ", ".join(sorted(present[present >= 20].index))))

    # --- adaptive vs broken down --------------------------------------------
    say("\n== Adaptive versus broken down only")
    pair = np.isin(y, ["adaptive", "broken_down"])
    pair_train = np.intersect1d(train, np.flatnonzero(pair))
    pair_test = np.intersect1d(test, np.flatnonzero(pair))
    say("%-28s %14s %9s" % ("model", "CV macro-F1", "test F1"))
    for name, model, matrix in (
        ("majority class", DummyClassifier(strategy="most_frequent"), X),
        ("condition only (logistic)", logistic(), conditions),
        ("logistic regression", logistic(), X),
        ("random forest", forest(), X),
        ("info-state ceiling (forest)", forest(), info),
    ):
        mean, sd = cv_f1(model, matrix[pair_train], y[pair_train])
        model.fit(matrix[pair_train], y[pair_train])
        pred = model.predict(matrix[pair_test])
        say("%-28s %8.3f+/-%.3f %9.3f" % (name, mean, sd,
                                        f1_score(y[pair_test], pred, average="macro")))

    # --- feature importance ------------------------------------------------
    say("\n== Permutation importance, random forest, held-out test (drop in macro-F1)")
    rf_model = fitted["random forest"][0]
    imp = permutation_importance(rf_model, X[test], y[test], scoring="f1_macro",
                                 n_repeats=20, random_state=0, n_jobs=-1)
    for i in np.argsort(imp.importances_mean)[::-1]:
        say("%-26s %7.4f +/- %.4f" % (features[i], imp.importances_mean[i], imp.importances_std[i]))

    # --- prediction horizon ------------------------------------------------
    say("\n== Prediction horizon (train-set CV macro-F1)")
    say("%-8s %8s %8s %10s" % ("window", "forest", "logistic", "condition"))
    train_seeds = set(data["seed"].to_numpy()[train])
    cond_mean, _ = cv_f1(logistic(), conditions[train], y[train])
    horizon_scores = []
    for window in HORIZONS:
        h = horizons[horizons["window"] == window].set_index("seed").loc[data["seed"]]
        hx = h[features].fillna(-1).to_numpy(float)
        mask = data["seed"].isin(train_seeds).to_numpy()
        rf_mean, _ = cv_f1(forest(), hx[mask], y[mask])
        lr_mean, _ = cv_f1(logistic(), hx[mask], y[mask])
        horizon_scores.append((window, rf_mean, lr_mean))
        say("%-8d %8.3f %8.3f %10.3f" % (window, rf_mean, lr_mean, cond_mean))

    # --- threshold sensitivity --------------------------------------------
    say("\n== Threshold sensitivity (train-set CV macro-F1)")
    say("%-10s %-10s %8s %10s   label shares (%s)" % (
        "stable by", "depressed", "forest", "condition", ", ".join(STATES)))
    for stable_by in STABLE_GRID:
        for depressed in DEPRESSED_GRID:
            yy = relabel(outcomes, stable_by, depressed)
            rf_mean, _ = cv_f1(forest(), X[train], yy[train])
            c_mean, _ = cv_f1(logistic(), conditions[train], yy[train])
            shares = [np.mean(yy == s) for s in STATES]
            say("%-10d %-10.2f %8.3f %10.3f   %s" % (
                stable_by, depressed, rf_mean, c_mean, " ".join("%.3f" % v for v in shares)))

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    windows = [w for w, _, _ in horizon_scores]
    ax.plot(windows, [s for _, s, _ in horizon_scores], "o-", color="#2a6f9e", label="random forest")
    ax.plot(windows, [s for _, _, s in horizon_scores], "s-", color="#c2593a", label="logistic regression")
    ax.axhline(cond_mean, color="#555", ls="--", lw=1, label="condition only")
    ax.axvline(55, color="#aaa", ls=":", lw=1)
    ax.set_xlabel("feature window, ticks (dotted = perturbation)")
    ax.set_ylabel("CV macro-F1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "horizon.png", dpi=200)

    (RESULTS / "model_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote %s and %s" % (RESULTS / "model_results.txt", RESULTS / "horizon.png"))


if __name__ == "__main__":
    main()
