"""Routing under 5x2 cross-validation: the metrics a routing decision is scored by, one fold of every arm, and
the result tables (Table 3, the routing behaviour, the cost). The splits themselves are `labkit.cv`."""

__all__ = ['ACC_TOLERANCES', 'METRICS', 'CONTEXT_METRICS', 'TABLE_METRICS', 'TABLE_LABELS', 'picked_errors',
           'achievable_choice', 'achievable_errors', 'selection_shares', 'format_shares', 'reference_choices', 'fit_arm',
           'score_arm', 'run_cv', 'fold_arms', 'fold_report', 'cost_table', 'selection_distribution', 'results_table',
           'format_results', 'to_latex_rows']

import time

import numpy as np
import pandas as pd

from hitasr.eda import fold_brief
from labkit.cv import FiveByTwoSplit, fold_rng
from labkit.pretty import paint, show_table


def picked_errors(E, choice):
    return E[np.arange(len(E)), choice]


def _corpus_wer(E, nref, picked, bsm):
    return float(picked.sum() / nref.sum())


def _mean_utt_wer(E, nref, picked, bsm):
    return float((picked / np.maximum(nref, 1)).mean())


def _routing_error(E, nref, picked, bsm):
    return float((picked > E.min(axis=1)).mean())


def _regret_per_word(E, nref, picked, bsm):
    return float((picked - E.min(axis=1)).sum() / nref.sum())


def _gap_closed_abs(E, nref, picked, bsm):
    total = nref.sum()
    return float(E[:, bsm].sum() / total - picked.sum() / total)


def achievable_choice(E, bsm, margin=0.0):
    """The oracle that keeps the BSM unless another member saves MORE than `margin` words."""
    best = E.argmin(axis=1)
    gain = E[:, bsm] - E[np.arange(len(E)), best]
    return np.where(gain > margin, best, bsm)


def achievable_errors(E, bsm, margin=0.0):
    return E[np.arange(len(E)), achievable_choice(E, bsm, margin)]


def _best_mask(E):
    is_best = E == E.min(axis=1, keepdims=True)
    return is_best, is_best.sum(axis=1)


def _acc_strict(E, nref, picked, bsm):
    _, n_best = _best_mask(E)
    decisive = n_best == 1
    if not decisive.any():
        return float("nan")
    return float((picked[decisive] <= E[decisive].min(axis=1)).mean())


def _tie_rate(E, nref, picked, bsm):
    _, n_best = _best_mask(E)
    return float((n_best == E.shape[1]).mean())


def _decisive_rate(E, nref, picked, bsm):
    _, n_best = _best_mask(E)
    return float((n_best == 1).mean())


def _acc_at(n):
    def metric(E, nref, picked, bsm):
        return float((picked <= E.min(axis=1) + n).mean())
    metric.__name__ = f"_acc_tol{n}"
    return metric


def _gap_closed_at(n, macro=False):
    def metric(E, nref, picked, bsm):
        if macro:
            utt = np.maximum(nref, 1)
            ref = float((E[:, bsm] / utt).mean())
            target = float((achievable_errors(E, bsm, n) / utt).mean())
            got = float((picked / utt).mean())
        else:
            total = nref.sum()
            ref = E[:, bsm].sum() / total
            target = achievable_errors(E, bsm, n).sum() / total
            got = picked.sum() / total
        gap = ref - target
        return float("nan") if gap <= 0 else float((ref - got) / gap)
    metric.__name__ = f"_gap_closed_{'mean_utt' if macro else 'corpus'}_tol{n}"
    return metric


ACC_TOLERANCES = (0, 1, 2)

METRICS = {
    "corpus_wer": (_corpus_wer, "lower"),
    "mean_utt_wer": (_mean_utt_wer, "lower"),
    "routing_error": (_routing_error, "lower"),
    "regret_per_word": (_regret_per_word, "lower"),
    "gap_closed_abs": (_gap_closed_abs, "higher"),
    "acc_strict": (_acc_strict, "higher"),
    "tie_rate": (_tie_rate, "higher"),
    "decisive_rate": (_decisive_rate, "higher"),
    **{f"acc_tol{n}": (_acc_at(n), "higher") for n in ACC_TOLERANCES},
    **{f"gap_closed_tol{n}": (_gap_closed_at(n), "higher") for n in ACC_TOLERANCES},
    **{f"gap_closed_mean_utt_tol{n}": (_gap_closed_at(n, macro=True), "higher") for n in ACC_TOLERANCES},
}
CONTEXT_METRICS = ("tie_rate", "decisive_rate")
TABLE_METRICS = ("corpus_wer", "mean_utt_wer",
                 *[f"gap_closed_tol{n}" for n in ACC_TOLERANCES],
                 *[f"gap_closed_mean_utt_tol{n}" for n in ACC_TOLERANCES],
                 *[f"acc_tol{n}" for n in ACC_TOLERANCES])
TABLE_LABELS = {"corpus_wer": "WER", "mean_utt_wer": "uWER",
                **{f"gap_closed_tol{n}": f"GC@{n}" for n in ACC_TOLERANCES},
                **{f"gap_closed_mean_utt_tol{n}": f"uGC@{n}" for n in ACC_TOLERANCES},
                **{f"acc_tol{n}": f"Acc@{n}" for n in ACC_TOLERANCES}}


def selection_shares(choice, k):
    """`(k,)` share of utterances routed to each member; all-NaN for a fusion arm (`-1`s)."""
    c = np.asarray(choice, dtype=int)
    if c.size == 0 or (c < 0).any():
        return np.full(k, np.nan)
    return np.bincount(c, minlength=k)[:k] / len(c)


def format_shares(shares, digits=2):
    shares = np.asarray(shares, dtype=np.float64)
    if shares.size == 0 or not np.isfinite(shares).all():
        return "—"
    return "-".join(f"{s:.{digits}f}" for s in shares)

def reference_choices(E, bsm, rng=None, members=None, tolerances=()):
    arms = {"oracle": E.argmin(axis=1), "bsm": np.full(len(E), bsm), "worst": E.argmax(axis=1)}
    if rng is not None:
        arms["random"] = rng.integers(0, E.shape[1], len(E))
    for n in tolerances:
        if n > 0:
            arms[f"oracle_tol{n}"] = achievable_choice(E, bsm, margin=n)
    if members is not None:
        for i, m in enumerate(members):
            if m in arms:
                raise KeyError(f"member {m!r} collides with a reference arm name")
            arms[m] = np.full(len(E), i)
    return arms


def fit_arm(arm, store, rows_tr, rows_ev, timings=None):
    """Fit one arm on a fold and read its evaluation-half output. Returns `(kind, out)`."""
    t0 = time.perf_counter()
    arm.fit(store, rows_tr)
    t1 = time.perf_counter()
    out = arm.predict(store, rows_ev)
    t2 = time.perf_counter()
    if timings is not None:
        timings["fit_seconds"] = t1 - t0
        timings["predict_seconds"] = t2 - t1
        timings["predict_us_per_utt"] = (t2 - t1) * 1e6 / max(len(rows_ev), 1)
    return getattr(arm, "produces", "choice"), np.asarray(out)


def score_arm(E_ev, kind, out):
    """`(picked, choice)` — `choice` is None for a fusion arm."""
    if kind == "errors":
        return np.asarray(out, dtype=np.float64), None
    choice = np.asarray(out, dtype=int)
    return picked_errors(E_ev, choice), choice


def run_cv(store, estimators, scheme=None, rows=None, references=True, verbose=True,
           random_seed=20260821, fold_eda=False, member_arms=True, tolerances=ACC_TOLERANCES,
           fold_table=True):
    """Fit and score every estimator on every fold. Returns `(fold_scores, fold_picks)`."""
    scheme = scheme or FiveByTwoSplit()
    members = tuple(store.members)
    row_ids = np.arange(store.n) if rows is None else np.asarray(rows, dtype=int)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("`rows` contains duplicates")
    E_all, nref_all = store.E[row_ids], store.nref[row_ids]
    tag = "+".join(members)
    if verbose:
        print(f"{tag}\n  {len(row_ids):,} of {store.n:,} pooled utterances, {scheme.n_folds} folds "
              f"({scheme.name}), {len(estimators)} estimator(s)")
    scores, picks = [], []
    for repeat, fold, tr, ev in scheme.split(len(row_ids)):
        seed = scheme.repeat_seed(repeat) if hasattr(scheme, "repeat_seed") else repeat
        if fold_eda:
            fold_brief(E_all, nref_all, members, tr, ev, seed=seed, fold=fold)
        s, p = fold_arms(store, estimators, row_ids, E_all, nref_all, tr, ev, members=members, tag=tag,
                         repeat=repeat, fold=fold, references=references,
                         rng=fold_rng(random_seed, seed, fold), verbose=verbose,
                         extra={"repeat_seed": int(seed)}, member_arms=member_arms,
                         tolerances=tolerances, fold_table=fold_table)
        scores += s
        picks += p
    return pd.DataFrame(scores), pd.concat(picks, ignore_index=True)


def fold_arms(store, estimators, row_ids, E_all, nref_all, tr, ev, members, tag, repeat, fold,
              references=True, rng=None, verbose=True, extra=None, member_arms=True,
              tolerances=(), fold_table=True):
    """Fit and score every arm on ONE fold. Returns `(score_rows, pick_frames)`."""
    assert not np.intersect1d(tr, ev).size, "train and eval folds overlap"
    E_ev, nref_ev = E_all[ev], nref_all[ev]
    bsm = int((E_all[tr].sum(0) / nref_all[tr].sum()).argmin())
    arms = {}
    if references:
        free = {"fit_seconds": 0.0, "predict_seconds": 0.0, "predict_us_per_utt": 0.0}
        refs = reference_choices(E_ev, bsm, rng, members=members if member_arms else None,
                                 tolerances=tolerances)
        for name, choice in refs.items():
            arms[name] = (picked_errors(E_ev, choice), choice, dict(free), np.nan)
    for name, factory in estimators.items():
        timings = {}
        t0 = time.perf_counter()
        arm = factory()
        kind, out = fit_arm(arm, store, row_ids[tr], row_ids[ev], timings=timings)
        timings["fit_seconds"] = time.perf_counter() - t0 - timings["predict_seconds"]
        picked, choice = score_arm(E_ev, kind, out)
        soft = np.nan
        if hasattr(arm, "expected_wer") and kind == "choice":
            try:
                soft = float(arm.expected_wer(store, row_ids[ev]))
            except NotImplementedError:      # a choice without a distribution (CN-MBR in `hyp` mode)
                pass
        n_params = arm.n_params() if hasattr(arm, "n_params") else None
        arms[name] = (picked, choice, {**timings, **({"n_params": n_params} if n_params else {})}, soft)

    score_rows, pick_frames = [], []
    for name, (picked, choice, timings, soft) in arms.items():
        base = {"members": tag, "model": name, "repeat": repeat, "fold": fold, "n_train": len(tr),
                "n_eval": len(ev), "bsm": members[bsm], **timings, **(extra or {})}
        for metric, (fn, direction) in METRICS.items():
            score_rows.append({**base, "metric": metric, "value": fn(E_ev, nref_ev, picked, bsm),
                               "direction": direction})
        if np.isfinite(soft):
            score_rows.append({**base, "metric": "soft_expected_wer", "value": soft, "direction": "lower"})
        pick_frames.append(pd.DataFrame({"members": tag, "model": name, "repeat": repeat, "fold": fold,
                                         "row": row_ids[ev], "errors": picked,
                                         "choice": -1 if choice is None else choice, **(extra or {})}))
    if verbose:
        seed = (extra or {}).get("repeat_seed")
        label = f"r{repeat}" if seed is None else f"seed={seed}"
        if fold_table:
            show_table(fold_report(score_rows, pick_frames, members),
                       title=f"[{label} f{fold}]  train {len(tr):,} / eval {len(ev):,}  bsm={members[bsm]}",
                       best="auto", best_exclude=lambda r: "oracle" in str(r["arm"]),
                       highlight=lambda r: str(r["arm"]).startswith("hit_asr"), mono=("dist",))
        else:
            body = "  ".join(f"{n} {v[0].sum() / nref_ev.sum():.4f}" for n, v in arms.items())
            print(f"  {label:<12} f{fold}  bsm={members[bsm]:<20} {body}")
    return score_rows, pick_frames


def fold_report(score_rows, pick_frames, members, digits=4):
    df = pd.DataFrame(score_rows)
    wide = (df[df["metric"].isin(TABLE_METRICS)]
            .pivot_table(index="model", columns="metric", values="value", sort=False)
            .reindex(columns=list(TABLE_METRICS)))
    dist = {p["model"].iloc[0]: format_shares(selection_shares(p["choice"], len(members))) for p in pick_frames}
    wide.insert(0, "arm", wide.index)
    wide["dist"] = [dist.get(m, "—") for m in wide.index]
    for m in TABLE_METRICS:
        if m in wide.columns and m.startswith(("gap_closed", "acc_")):
            wide[m] = (100 * wide[m]).round(1)
        elif m in wide.columns:
            wide[m] = wide[m].round(digits)
    return wide.rename(columns=TABLE_LABELS).reset_index(drop=True)

def cost_table(fold_scores, results=None, per=1000):
    """Wall-clock per arm: search, training, inference. One row per model."""
    folds = fold_scores.drop_duplicates(subset=["model", "repeat", "fold"])
    g = folds.groupby("model")
    out = pd.DataFrame({"n_folds": g.size(), "fit_s_mean": g["fit_seconds"].mean(),
                        "fit_s_sd": g["fit_seconds"].std(), "fit_s_total": g["fit_seconds"].sum(),
                        "infer_us_per_utt": g["predict_us_per_utt"].mean()})
    out[f"infer_s_per_{per}"] = out["infer_us_per_utt"] * per / 1e6
    if "n_params" in folds.columns:
        out["n_params"] = g["n_params"].first()
    out["n_eval"] = g["n_eval"].mean()
    if results is not None:
        search = pd.DataFrame([{"model": name, "search_s": float(getattr(r, "seconds", np.nan)),
                                "trials": int(getattr(r, "n_trials", 0))}
                               for name, r in results.items()]).set_index("model")
        out = out.join(search)
        for c in ("search_s", "trials"):
            out[c] = out[c].fillna(0.0)
    return out.reset_index().sort_values("fit_s_total", ignore_index=True)


def selection_distribution(fold_picks, members, fold_scores=None):
    """How often each arm chose each member: shares, entropy, the dominant member, the switch rate."""
    members = list(members)
    picks = fold_picks.copy()
    if fold_scores is not None and "bsm" in fold_scores.columns:
        bsm = fold_scores.drop_duplicates(subset=["model", "repeat", "fold"])[["model", "repeat", "fold", "bsm"]]
        picks = picks.merge(bsm, on=["model", "repeat", "fold"], how="left")
        picks["bsm_idx"] = picks["bsm"].map({m: i for i, m in enumerate(members)})
    rows = []
    for name, grp in picks.groupby("model", sort=False):
        fused = int((grp["choice"] < 0).sum())
        row = {"model": name, "n_picks": len(grp), "n_fused": fused}
        routed = grp[grp["choice"] >= 0]
        if len(routed) == 0:
            row.update({f"share_{m}": np.nan for m in members})
            row.update(entropy=np.nan, entropy_max=np.log2(len(members)), share_top=np.nan,
                       top_member="— fused —", share_sd_mean=np.nan, switch_rate=np.nan)
            rows.append(row)
            continue
        counts = routed["choice"].value_counts(normalize=True)
        shares = np.array([counts.get(i, 0.0) for i in range(len(members))])
        for m, s in zip(members, shares):
            row[f"share_{m}"] = float(s)
        nz = shares[shares > 0]
        row["entropy"] = float(-(nz * np.log2(nz)).sum()) + 0.0
        row["entropy_max"] = float(np.log2(len(members)))
        top = int(shares.argmax())
        row["share_top"], row["top_member"] = float(shares[top]), members[top]
        per_fold = (routed.groupby(["repeat", "fold"])["choice"].value_counts(normalize=True)
                    .unstack(fill_value=0.0).reindex(columns=range(len(members)), fill_value=0.0))
        row["share_sd_mean"] = float(per_fold.std(axis=0).mean())
        row["switch_rate"] = (float((routed["choice"] != routed["bsm_idx"]).mean())
                              if "bsm_idx" in routed.columns else np.nan)
        rows.append(row)
    cols = (["model", "n_picks", "n_fused"] + [f"share_{m}" for m in members]
            + ["top_member", "share_top", "entropy", "entropy_max", "share_sd_mean", "switch_rate"])
    return pd.DataFrame(rows)[cols].sort_values("model", ignore_index=True)


def results_table(fold_scores, fold_picks, members, order=None, labels=None, metrics=TABLE_METRICS):
    """The main-results table, one row per arm, numeric: mean and sd over folds, plus shares."""
    members = list(members)
    have = list(dict.fromkeys(fold_scores["model"]))
    order = [m for m in (order or []) if m in have] + [m for m in have if m not in (order or [])]
    df = fold_scores[fold_scores["metric"].isin(metrics)]
    mean = df.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    sd = df.pivot_table(index="model", columns="metric", values="value", aggfunc="std")
    rows = []
    for m in order:
        row = {"model": m, "label": (labels or {}).get(m, m)}
        for metric in metrics:
            row[metric] = float(mean.loc[m, metric]) if metric in mean.columns else np.nan
            row[f"{metric}_sd"] = float(sd.loc[m, metric]) if metric in sd.columns else np.nan
        picks = fold_picks[fold_picks["model"] == m]
        fused = int((picks["choice"] < 0).sum()) if len(picks) else 0
        shares = selection_shares(picks["choice"], len(members)) if len(picks) else np.full(len(members), np.nan)
        for name, s in zip(members, shares):
            row[f"share_{name}"] = float(s)
        row["n_fused"] = fused
        if fused > 0:
            for metric in metrics:
                if metric.startswith("acc_"):
                    row[metric] = row[f"{metric}_sd"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["members"] = members
    out.attrs["n_folds"] = int(fold_scores.drop_duplicates(["repeat", "fold"]).shape[0])
    return out


def format_results(table, metrics=TABLE_METRICS, digits=4, sd=True, pct=None):
    """`results_table` as manuscript strings: `mean ± sd`, `—` where undefined."""
    if pct is None:
        pct = tuple(m for m in metrics if m.startswith(("gap_closed", "acc_")))
    members = table.attrs.get("members") or [c[len("share_"):] for c in table.columns if c.startswith("share_")]

    def cell(v, s, as_pct):
        if not np.isfinite(v):
            return "—"
        if as_pct:
            main = f"{100 * v:.1f}"
            return f"{main} ± {100 * s:.1f}" if sd and np.isfinite(s) else main
        main = f"{v:.{digits}f}"
        return f"{main} ± {s:.{digits}f}" if sd and np.isfinite(s) else main

    out = pd.DataFrame({"arm": table["label"] if "label" in table.columns else table["model"]})
    for metric in metrics:
        out[TABLE_LABELS.get(metric, metric)] = [cell(v, s, metric in pct)
                                                 for v, s in zip(table[metric], table[f"{metric}_sd"])]
    out["dist"] = [format_shares([r[f"share_{m}"] for m in members]) for _, r in table.iterrows()]
    return out


def to_latex_rows(formatted, sep=" & ", end=r" \\"):
    lines = []
    for _, r in formatted.iterrows():
        cells = [str(v).replace("±", r"$\pm$").replace("—", "---") for v in r.values]
        lines.append(sep.join(cells) + end)
    return "\n".join(lines)
