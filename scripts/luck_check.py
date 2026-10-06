"""Supplementary check: how much of "breakdown" is luck of who is removed?

At the perturbation tick this recomputes, for every possible human removal,
whether that removal would destroy a required item.  The share of removals that
would is the group's *exposure* P: the best any observer of the full information
state could say about the coming breakdown.  P is then compared with what the
random draw actually did.

If P is near 0 or 1 the outcome is settled before the draw; if it sits in the
middle the draw decides.  Study seeds 10000-10299 per condition; nothing here
feeds the study dataset.
"""

import sys, inspect, random
sys.path.insert(0, "src")
import numpy as np
import cistress.simulate as S
from cistress.config import CONDITIONS

src = inspect.getsource(S.simulate)
hook = "            removed = rng.choice(candidates)\n"
assert hook in src
add = hook + (
"            _p = 0\n"
"            for _c in candidates:\n"
"                _r = set()\n"
"                for _i in range(n):\n"
"                    if _i != _c: _r |= held[_i]\n"
"                _p += bool(required - _r)\n"
"            S.EXPOSE.append(_p / len(candidates))\n")
ns = dict(vars(S)); ns["S"] = S
exec(src.replace(hook, add), ns)
S.EXPOSE = []
sim = ns["simulate"]

def auc(score, y):
    score, y = np.asarray(score), np.asarray(y)
    pos, neg = score[y == 1], score[y == 0]
    return ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))

N = 300
print("%-4s %9s %9s %9s %11s %9s" % ("", "mean P", "actual", "P in 0/1", "AUC of P", "Brier"))
allp, ally = [], []
for name, cfg in CONDITIONS.items():
    S.EXPOSE.clear(); ys = []
    for s in range(10000, 10000 + N):
        r = sim(cfg, s, name); ys.append(int(bool(r.destroyed_required)))
    p = np.array(S.EXPOSE); y = np.array(ys)
    allp += list(p); ally += list(y)
    print("%-4s %9.3f %9.3f %9.2f %11.3f %9.3f" % (name, p.mean(), y.mean(), ((p == 0) | (p == 1)).mean(), auc(p, y), ((p - y) ** 2).mean()))
p, y = np.array(allp), np.array(ally)
print("ALL  %9.3f %9.3f %9.2f %11.3f %9.3f" % (p.mean(), y.mean(), ((p == 0) | (p == 1)).mean(), auc(p, y), ((p - y) ** 2).mean()))
print("Brier if you always said the base rate: %.3f" % (y.var()))
print("Irreducible luck (Bernoulli var of P): %.3f" % (p * (1 - p)).mean())
