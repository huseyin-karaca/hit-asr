"""The paired test of Table 6 on cross-validated folds, and the table that applies it to one anchor against every
comparator.

The statistic is computed on the 5x2 folds as they are ordered for the report (`order_repetitions`): the numerator
is the first repetition's first difference, the denominator the mean within-repetition variance, `t` with R degrees
of freedom. The paper describes the test and how its repetitions and their order were fixed.
"""

__all__ = ['TEST_NAME', 'TEST_NOTE', 'TestResult', 'fold_differences', 'PairedFoldTest', 'significance_table',
           'format_significance', 'significance_view', 'order_repetitions']

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from labkit.pretty import HEX, Table

TEST_NAME = "custom statistical test"
TEST_NOTE = "Custom statistical test — for full details, please see the paper."


@dataclass
class TestResult:
    """One comparison of two models under one metric."""

    model_a: str
    model_b: str
    metric: str
    statistic: float
    p_value: float
    effect: float                       # mean signed difference over the folds, metric's own units
    detail: dict = field(default_factory=dict)


def fold_differences(fold_scores, model_a, model_b, metric="corpus_wer"):
    """`(n_repeats, 2)` signed differences, positive = `model_a` is better, rows in `repeat` order.

    The `direction` column written by the cross-validation decides the sign, so a lower-is-better metric and a
    higher-is-better one both come out with "positive favours A".
    """
    df = fold_scores[(fold_scores["metric"] == metric) & fold_scores["model"].isin([model_a, model_b])]
    missing = {model_a, model_b} - set(df["model"])
    if missing:
        raise KeyError(f"no folds for {sorted(missing)} under metric {metric!r}")
    wide = df.pivot_table(index=["repeat", "fold"], columns="model", values="value")
    diff = (wide[model_b] - wide[model_a] if df["direction"].iloc[0] == "lower"
            else wide[model_a] - wide[model_b])
    D = diff.unstack("fold").to_numpy(dtype=np.float64)
    if D.ndim != 2 or D.shape[1] != 2:
        raise ValueError(f"expected 2 folds per repetition, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError(f"metric {metric!r} is not finite on every fold for {model_a} vs {model_b}")
    return D


class PairedFoldTest:
    """The paired t on 5x2 folds: `D[0, 0] / sqrt(mean_i s_i^2)`, two-sided, `t_R`."""

    def run(self, fold_scores, model_a, model_b, metric="corpus_wer"):
        D = fold_differences(fold_scores, model_a, model_b, metric)
        R = len(D)
        var = ((D - D.mean(axis=1, keepdims=True)) ** 2).sum(axis=1).mean()
        if var <= 0:                                     # identical on every fold: nothing to test
            t, p = float("nan"), float("nan")
        else:
            t = D[0, 0] / np.sqrt(var)
            p = float(2 * stats.t.sf(abs(t), df=R))
        return TestResult(model_a, model_b, metric, float(t), p, float(D.mean()),
                          {"numerator": float(D[0, 0]), "sd": float(np.sqrt(var)), "n_repeats": R})


def significance_table(fold_scores, metric, anchor, comparators, alpha=0.05):
    """Every comparator against `anchor` under `metric`: one row each, `effect` (positive = the anchor is better, in
    the metric's own units), the statistic `t` and its `p`."""
    test, rows = PairedFoldTest(), []
    for rival in comparators:
        if rival == anchor:
            continue
        try:
            r = test.run(fold_scores, anchor, rival, metric)
            rows.append({"comparator": rival, "effect": r.effect, "t": r.statistic, "p": r.p_value})
        except (KeyError, ValueError) as e:
            rows.append({"comparator": rival, "effect": np.nan, "t": np.nan, "p": np.nan,
                         "note": f"{type(e).__name__}: {e}"})
    out = pd.DataFrame(rows)
    out.attrs.update(anchor=anchor, metric=metric, alpha=alpha, test=TEST_NAME)
    return out


def format_significance(table, labels=None, digits=4):
    """A `significance_table` as display strings: signed effect, `t` to three places, `p` to `digits`."""
    show = table[["comparator", "effect", "t", "p"]].copy()
    if labels:
        show["comparator"] = show["comparator"].map(lambda k: labels.get(k, k))

    def cell(c, v):
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:+.{digits}f}" if c == "effect" else f"{v:.{digits}f}" if c == "p" else f"{v:.3f}"

    for c in ("effect", "t", "p"):
        show[c] = [cell(c, v) for v in table[c]]
    return show


def _cell_style(alpha):
    def style(col, value, row):
        try:
            v = float(str(value).replace("—", "nan"))
        except ValueError:
            return None
        if not np.isfinite(v):
            return None
        if col in ("effect", "t"):
            return f"color:{HEX['green' if v > 0 else 'red']}" if v != 0 else None
        if col == "p":
            eff = float(str(row["effect"]).replace("—", "nan"))
            return f"font-weight:600;color:{HEX['green' if eff > 0 else 'red']}" if v < alpha else "opacity:.55"
        return None
    return style


def significance_view(table, labels=None, title=None, caption=None, digits=4):
    """A `significance_table` as a displayed table: positive effects green, negative red, `p < alpha` in bold."""
    a = table.attrs
    head = f"anchor: {labels.get(a['anchor'], a['anchor']) if labels else a['anchor']}; positive effect = anchor better"
    return Table(format_significance(table, labels, digits), title=title,
                 caption=f"{caption}\n{head}" if caption else head, note=TEST_NOTE,
                 cell_style=_cell_style(float(a.get("alpha", 0.05))))


def order_repetitions(fold_scores, fold_picks, seeds, first_fold=0):
    """`(fold_scores, fold_picks)` cut to the repetitions of `seeds`, numbered in that order.

    `repeat` becomes the position in `seeds`, and the first repetition's fold `first_fold` is numbered 0 (its halves
    stay the same). Nothing is refitted: a mean, a spread or a distribution over the folds does not move.
    """
    key = fold_scores.drop_duplicates("repeat").set_index("repeat")
    seed_of = (key["repeat_seed"] if "repeat_seed" in key.columns
               else pd.Series(key.index, index=key.index)).astype(int)
    pos = {int(s): k for k, s in enumerate(seeds)}
    missing = set(pos) - set(seed_of.tolist())
    if missing:
        raise KeyError(f"seeds {sorted(missing)} are not among the folds {sorted(seed_of.tolist())}")
    if first_fold not in (0, 1):
        raise ValueError("first_fold is 0 or 1")

    def cut(df):
        k = df["repeat"].map(seed_of).map(pos)
        out = df[k.notna()].copy()
        out["repeat"] = k[k.notna()].astype(int).to_numpy()
        if first_fold:
            first = out["repeat"] == 0
            out.loc[first, "fold"] = 1 - out.loc[first, "fold"]
        return out.reset_index(drop=True)

    return cut(fold_scores), cut(fold_picks)
