__all__ = ['gain_profile', 'selection_accuracy', 'base_model_report', 'eda', 'partition_sizes', 'fold_brief', 'win_rates_tolerant', 'feature_dims', 'dataset_table']

import numpy as np
import pandas as pd

from hitasr.ensemble import (ensemble_stats, error_correlation_matrix,
                             pair_diversity_table, win_rates)


def gain_profile(E, nref, bsm, quantiles=5):
    """Where the achievable gain lives, by reference length. One row per quantile."""
    gain = E[:, bsm] - E.min(axis=1)
    if gain.sum() <= 0:
        return pd.DataFrame()
    ranks = np.argsort(np.argsort(nref))
    bins = np.minimum((ranks * quantiles) // len(nref), quantiles - 1)
    rows = []
    for b in range(quantiles):
        m = bins == b
        rows.append({"length_bin": b, "n_utts": int(m.sum()), "mean_nref": float(nref[m].mean()),
                     "share_of_gain": float(gain[m].sum() / gain.sum()),
                     "share_of_words": float(nref[m].sum() / nref.sum()),
                     "contested_rate": float((E[m].min(1) < E[m].max(1)).mean())})
    out = pd.DataFrame(rows)
    out["gain_per_word"] = out["share_of_gain"] / out["share_of_words"].clip(1e-12)
    return out


def selection_accuracy(E, choice):
    """`(strict, lenient)` accuracy of one per-utterance choice vector."""
    picked = E[np.arange(len(E)), np.asarray(choice, dtype=int)]
    best = E.min(axis=1)
    lenient = float((picked <= best).mean())
    decisive = (E == best[:, None]).sum(axis=1) == 1
    strict = float((picked[decisive] <= best[decisive]).mean()) if decisive.any() else float("nan")
    return strict, lenient


def base_model_report(E, nref, members, bsm=None):
    """Per-member statistics for the experts INSIDE the ensemble. One row each."""
    members = list(members)
    share, sole = win_rates(E)
    rows = []
    for i, m in enumerate(members):
        strict, lenient = selection_accuracy(E, np.full(len(E), i))
        rows.append({"model": m, "corpus_wer": float(E[:, i].sum() / nref.sum()),
                     "mean_utt_wer": float((E[:, i] / np.maximum(nref, 1)).mean()),
                     "clean_rate": float((E[:, i] == 0).mean()),
                     "win_rate": float(share[i]), "sole_win_rate": float(sole[i]),
                     "acc_strict": strict, "acc_lenient": lenient,
                     "is_bsm": (bsm is not None and i == int(bsm))})
    return pd.DataFrame(rows).sort_values("corpus_wer", ignore_index=True)


def eda(store, members, splits=None, rows=None, quantiles=5, verbose=True):
    """The pre-flight briefing for one member set. Returns a dict of frames."""
    members = tuple(members)
    E, nref = store.E(members), store.nref
    if splits is not None:
        mask = store.mask(splits)
        E, nref = E[mask], nref[mask]
    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        E, nref = E[rows], nref[rows]
    bsm = int((E.sum(0) / nref.sum()).argmin())
    stats = ensemble_stats(E, nref, bsm)
    best = E.min(axis=1)
    trivial = pd.DataFrame([{"all_members_perfect": float((E == 0).all(axis=1).mean()),
                             "all_members_tied": float((E == best[:, None]).all(axis=1).mean()),
                             "bsm_already_optimal": float((E[:, bsm] == best).mean()),
                             "gainable": float((E[:, bsm] > best).mean())}])
    pairs = pair_diversity_table(store, splits, list(members), rows=rows)
    out = {"stats": pd.DataFrame([{"members": "+".join(members), "bsm_model": members[bsm], **stats}]),
           "per_model": base_model_report(E, nref, members, bsm), "trivial": trivial,
           "gain_profile": gain_profile(E, nref, bsm, quantiles), "pairs": pairs,
           "error_correlation": error_correlation_matrix(pairs, "pearson_r")}
    if verbose:
        s = out["stats"].iloc[0]
        print(f"=== {'+'.join(members)}   {len(nref):,} utterances, {nref.sum():,.0f} reference words")
        print(f"    bsm {s['bsm_model']} {s['bsm_wer']:.4f}   oracle {s['oracle_wer']:.4f}"
              f"   gap {s['gap_abs']:.4f} ({s['gap_rel']:.1%} relative)   random {s['random_wer']:.4f}")
        t = trivial.iloc[0]
        print(f"    unroutable: {t['all_members_perfect']:.1%} all perfect, "
              f"{t['all_members_tied']:.1%} all tied, {t['bsm_already_optimal']:.1%} bsm already optimal "
              f"-> only {t['gainable']:.1%} of rows are winnable")
        print(f"    prize concentration: {s['gain_top10pct']:.0%} of the gain is in the worst 10% "
              f"of utterances" + ("   (>0.85 means no router will find it)" if s["gain_top10pct"] > 0.85 else ""))
        print("\n" + out["per_model"].round(4).to_string(index=False))
        if len(out["gain_profile"]):
            print("\nwhere the gain sits, by reference length:")
            print(out["gain_profile"].round(3).to_string(index=False))
        print("\npairwise error correlation:")
        print(out["error_correlation"].round(3).to_string())
    return out


def partition_sizes(n, parts):
    """`{name: row indices}` -> sizes, plus the disjoint-or-nested check every p-value rests on."""
    named = {k: np.asarray(v, dtype=int) for k, v in parts.items()}
    sets = {k: set(v.tolist()) for k, v in named.items()}
    keys = list(named)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = sets[a] & sets[b]
            if shared and not (sets[a] <= sets[b] or sets[b] <= sets[a]):
                raise ValueError(f"partitions {a!r} and {b!r} overlap without either containing "
                                 f"the other: {len(shared)} shared row(s)")
    rows = []
    for k, v in named.items():
        parents = [o for o in keys if o != k and sets[k] < sets[o]]
        within = min(parents, key=lambda o: len(sets[o])) if parents else ""
        rows.append({"partition": k, "n_utts": len(v), "share": len(v) / max(n, 1), "within": within})
    return pd.DataFrame(rows)


def fold_brief(E, nref, members, train_idx, eval_idx, seed=None, fold=None, verbose=True):
    """The per-fold briefing `run_cv(fold_eda=True)` prints before fitting."""
    tr, ev = np.asarray(train_idx, dtype=int), np.asarray(eval_idx, dtype=int)
    E_tr, E_ev, n_tr, n_ev = E[tr], E[ev], nref[tr], nref[ev]
    bsm = int((E_tr.sum(0) / n_tr.sum()).argmin())
    stats = ensemble_stats(E_ev, n_ev, bsm)
    out = {"bsm": members[bsm], "stats": pd.DataFrame([stats]),
           "per_model": base_model_report(E_ev, n_ev, members, bsm)}
    if verbose:
        label = "fold" if seed is None else f"seed={seed}"
        print(f"  [{label} f{fold}]  train {len(tr):,} / eval {len(ev):,} utts   bsm={members[bsm]}  "
              f"bsm_wer {stats['bsm_wer']:.4f}  oracle {stats['oracle_wer']:.4f}  "
              f"gap {stats['gap_abs']:.4f} ({stats['gap_rel']:.1%})  winnable {stats['gain_utts']:.1%}")
    return out


def win_rates_tolerant(E, tolerances=(0, 1, 2), members=None):
    """Per-member acceptability rate at each word margin."""
    E = np.asarray(E, dtype=np.float64)
    k = E.shape[1]
    names = list(members) if members is not None else [f"m{i}" for i in range(k)]
    best = E.min(axis=1, keepdims=True)
    share, sole = win_rates(E)
    out = {"model": names, "win_rate_shared": share, "sole_win_rate": sole}
    context = {}
    for n in tolerances:
        ok = E <= best + n
        out[f"win_rate_tol{n}"] = ok.mean(axis=0)
        context[n] = float(ok.sum(axis=1).mean())
    df = pd.DataFrame(out)
    df.attrs["n_acceptable_mean"] = context
    return df


def feature_dims(store, members):
    """One row per member: encoder width, frame rate, mean frames per utterance."""
    rows = []
    for m in members:
        d = store.pooled(m).shape[1] if m in store._pooled else np.nan
        nf = store.n_frames.get(m)
        rows.append({"member": m, "d": d, "frame_rate_hz": store.frame_rate_hz.get(m),
                     "mean_frames": float(nf.mean()) if nf is not None else np.nan,
                     "max_frames": int(nf.max()) if nf is not None else np.nan})
    return pd.DataFrame(rows)


def dataset_table(store, partitions=None):
    """The dataset-statistics rows of the manuscript (Table 1), one corpus."""
    n = int(store.n)
    rows = [("Total clips", n)]
    if partitions is not None:
        tune, ev = len(partitions["tune"]), len(partitions["eval"])
        rows += [("Hyperparameter hold-out", int(tune)), ("Evaluation rows (5x2)", int(ev)),
                 ("Train size (per fold)", int(ev // 2)), ("Test size (per fold)", int(ev - ev // 2))]
    secs = store.seconds()
    if secs is not None:
        rows += [("Hours", float(secs.sum() / 3600)), ("Mean clip length (s)", float(secs.mean()))]
    rows.append(("Reference words", int(store.nref.sum())))
    return pd.DataFrame({"statistic": [r[0] for r in rows],
                         store.spec.name: pd.Series([r[1] for r in rows], dtype=object)})