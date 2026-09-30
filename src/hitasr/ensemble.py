"""Oracle arithmetic and diversity statistics for a group of experts on a set of rows."""

__all__ = ['ensemble_stats', 'win_rates', 'best_single_model', 'pair_diversity_table', 'error_correlation_matrix']

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def ensemble_stats(E, n, bsm):
    """Oracle arithmetic for one group on one set of rows. A flat dict of scalars."""
    total = n.sum()
    best = E.min(axis=1)
    is_best = E == best[:, None]
    n_best = is_best.sum(axis=1)
    gain = E[:, bsm] - best
    ranked = np.sort(gain)[::-1]

    def top_share(frac):
        if gain.sum() <= 0:
            return float("nan")
        return ranked[:max(1, int(round(len(ranked) * frac)))].sum() / gain.sum()

    wer = E.sum(axis=0) / total
    oracle, bsm_wer = best.sum() / total, E[:, bsm].sum() / total
    return {"oracle_wer": oracle, "bsm_wer": bsm_wer, "gap_abs": bsm_wer - oracle,
            "gap_rel": (bsm_wer - oracle) / bsm_wer if bsm_wer else float("nan"),
            "random_wer": float(wer.mean()), "worst_wer": float(wer.max()),
            "best_member_wer": float(wer.min()),
            "all_tie_rate": float((n_best == E.shape[1]).mean()),
            "decisive_rate": float((n_best == 1).mean()),
            "gain_utts": float((gain > 0).mean()),
            "gain_top1pct": top_share(0.01), "gain_top5pct": top_share(0.05),
            "gain_top10pct": top_share(0.10), "n_utts": int(len(n))}


def win_rates(E):
    """`(win_rate, sole_win_rate)`, each `(k,)`. Ties share the win equally."""
    is_best = E == E.min(axis=1, keepdims=True)
    return ((is_best / is_best.sum(axis=1, keepdims=True)).mean(axis=0),
            (is_best & (is_best.sum(axis=1, keepdims=True) == 1)).mean(axis=0))


def best_single_model(store, members, splits=None):
    """Index of the lowest-WER member, chosen on `splits` (default: the spec's fit splits)."""
    return int(store.corpus_wer(members, splits or store.spec.fit_splits).to_numpy().argmin())


def _pair_stats(err_a, err_b, n):
    wer_a, wer_b = err_a / np.maximum(n, 1), err_b / np.maximum(n, 1)
    ok_a, ok_b = err_a == 0, err_b == 0
    n11 = float((ok_a & ok_b).mean()); n00 = float((~ok_a & ~ok_b).mean())
    n10 = float((ok_a & ~ok_b).mean()); n01 = float((~ok_a & ok_b).mean())
    ad, bc = n11 * n00, n01 * n10
    observed = n11 + n00
    expected = ok_a.mean() * ok_b.mean() + (1 - ok_a.mean()) * (1 - ok_b.mean())
    return {"pearson_r": float(pearsonr(wer_a, wer_b)[0]) if wer_a.std() > 0 and wer_b.std() > 0 else float("nan"),
            "spearman_r": float(spearmanr(wer_a, wer_b)[0]) if wer_a.std() > 0 and wer_b.std() > 0 else float("nan"),
            "disagreement": n10 + n01, "double_fault": n00, "both_clean": n11,
            "q_statistic": (ad - bc) / (ad + bc) if ad + bc > 0 else float("nan"),
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else float("nan")}


def pair_diversity_table(store, splits=None, models=None, rows=None):
    """One row per expert pair: error correlation, disagreement, the pair's own oracle gap."""
    models = list(models or store.experts)
    E, n = store.E(models, splits), store.words(splits)
    if rows is not None:
        E, n = E[rows], n[rows]
    wer = E.sum(axis=0) / n.sum()
    out = []
    for i, j in itertools.combinations(range(len(models)), 2):
        row = {"model_a": models[i], "model_b": models[j], "wer_a": wer[i], "wer_b": wer[j]}
        row.update(_pair_stats(E[:, i], E[:, j], n))
        pair_oracle = np.minimum(E[:, i], E[:, j]).sum() / n.sum()
        row["oracle_wer"] = pair_oracle
        row["gap_abs"] = min(wer[i], wer[j]) - pair_oracle
        out.append(row)
    return pd.DataFrame(out).sort_values("gap_abs", ascending=False, ignore_index=True)


def error_correlation_matrix(pairs, metric="pearson_r"):
    models = sorted(set(pairs["model_a"]) | set(pairs["model_b"]))
    M = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    if metric in ("disagreement", "double_fault"):
        np.fill_diagonal(M.to_numpy(), 0.0)
    for _, r in pairs.iterrows():
        M.loc[r["model_a"], r["model_b"]] = M.loc[r["model_b"], r["model_a"]] = r[metric]
    return M
