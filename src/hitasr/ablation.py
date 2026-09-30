"""The ablation of HIT-ASR's design switches, read off its hold-out exploration, and the pool-size study.

**The switches.** The exploration (`results/<corpus>/exploration_hit_asr.json`) drew configurations of the router at
random from a space in which every architectural switch is a hyperparameter — the cross-expert fusion, the sharing of
Stage 1, the training objective, the input normalisation — and fitted each once on the hold-out. Grouping its trials
by the level of one switch shows what that switch does across everything else: how often a level collapses onto the
best single expert, how much of the gap to the oracle its typical and its best configurations close, and what it
costs in parameters and fitting time. Nothing is refitted and nothing is filtered.

**The pool size.** `extend_pool` grows the trio to K experts, adding each time the expert that lowers the oracle
most on the hold-out's fit rows; `holdout_score` scores a router's choices on the hold-out's validation rows the
way the exploration does.
"""

__all__ = ['SWITCHES', 'COLLAPSE', 'trials_frame', 'switch_table', 'refit_spread', 'level_stability', 'main_rows',
           'extend_pool', 'holdout_score', 'pool_study', 'pool_table', 'Ablation']

import numpy as np
import pandas as pd

SWITCHES = ("fusion", "share_stage1", "loss_preset", "input_norm")
COLLAPSE = 0.01          # a configuration closing at most this share of the gap stayed on the best single expert


def trials_frame(record):
    """The exploration's trials as a frame: one row per trial, the parameters as columns, `top_share` the largest
    share of the validation clips routed to one expert."""
    rows = []
    for t in record["trials"]:
        rows.append({**{k: v for k, v in t.items() if k not in ("params", "shares")}, **t["params"],
                     "top_share": max(t["shares"]) if t["shares"] else np.nan})
    return pd.DataFrame(rows)


def switch_table(trials, switch, seed=42, collapse=COLLAPSE):
    """Per level of `switch`, over the trials of `seed` (seed 42 holds every configuration drawn): how many, the
    share that collapsed, the median / upper-quartile / best gap closed, the median share of the busiest expert,
    and the median parameters (M) and fitting seconds."""
    t = trials[(trials["seed"] == seed) & trials["gap_closed"].notna()]
    g = t.groupby(switch, sort=True)
    out = pd.DataFrame({"n": g.size(), "collapsed": g["gap_closed"].apply(lambda x: float((x <= collapse).mean())),
                        "gap_median": g["gap_closed"].median(), "gap_q75": g["gap_closed"].quantile(0.75),
                        "gap_best": g["gap_closed"].max(), "top_share": g["top_share"].median(),
                        "params_M": g["n_params"].median() / 1e6, "fit_s": g["fit_seconds"].median()})
    out.index = out.index.map(str)
    return out.rename_axis("level").reset_index().assign(switch=switch)[["switch", "level", *out.columns]]


def refit_spread(trials, collapse=COLLAPSE):
    """How much one configuration's outcome moves between seeds: over the configurations refitted at least once,
    the median standard deviation of its gap closed, and the share whose collapsed/trained status changes."""
    by = trials[trials["gap_closed"].notna()].groupby("config")
    multi = [g for _, g in by if g["seed"].nunique() > 1]
    if not multi:
        return {"n_configs": 0}
    sd = [float(g["gap_closed"].std()) for g in multi]
    flips = [g["gap_closed"].le(collapse).nunique() > 1 for g in multi]
    return {"n_configs": len(multi), "gap_sd_median": float(np.median(sd)), "status_flips": float(np.mean(flips))}


def level_stability(trials, switch):
    """The median gap closed of each level of `switch` at every seed, over the configurations fitted at all three
    seeds — whether a level's standing holds when the same configurations are refitted."""
    t = trials[trials["gap_closed"].notna()]
    seeds = sorted(t["seed"].unique())
    full = t.groupby("config")["seed"].nunique()
    t = t[t["config"].isin(full[full == len(seeds)].index)]
    return t.pivot_table(index=switch, columns="seed", values="gap_closed", aggfunc="median").rename_axis(None, axis=1)


def main_rows(record, arms=("bsm", "mlp_pool_hard_ce", "adastt_ce", "mlp_pool", "hit_asr", "oracle"), p="p"):
    """Rows of a main record's Table 3 (mean ± sd over its ten folds) for `arms`, with the busiest expert's share and
    the paired test against HIT-ASR from its Table 6 (a control arm, not a row of Table 6: from `table6_controls`)
    — the ablation rows that live on the main folds."""
    t3 = {r["model"]: r for r in record["tables"]["table3"]}
    beh = {r["model"]: r for r in record["tables"].get("behaviour", [])}
    controls = record["tables"].get("table6_controls", {}).get("corpus_wer", {}).get("rows", [])   # their own family
    tests = {r["comparator"]: r for r in controls + record["tables"]["table6"]["corpus_wer"]["rows"]}
    out = []
    for a in arms:
        if a not in t3:
            continue
        r = t3[a]
        out.append({"arm": a, "label": record.get("labels", {}).get(a, r.get("label", a)), "corpus_wer": r["corpus_wer"],
                    "corpus_wer_sd": r.get("corpus_wer_sd"), "gap_closed": r.get("gap_closed_tol0"),
                    "acc": r.get("acc_tol0"), "top_share": beh.get(a, {}).get("share_top"),
                    "p_vs_hit_asr": tests.get(a, {}).get(p) if a != "hit_asr" else None})
    return pd.DataFrame(out)


def extend_pool(E, nref, rows, trio, candidates, K):
    """`trio` grown to `K` experts: each step adds the candidate that lowers the oracle's corpus WER on `rows` most.
    `E` is `{expert: (N,) errors}`; ties go to the candidate listed first."""
    pool = list(trio)
    rows = np.asarray(rows, dtype=int)
    words = nref[rows].sum()
    while len(pool) < K:
        best = min((c for c in candidates if c not in pool),
                   key=lambda c: np.minimum.reduce([E[m][rows] for m in pool + [c]]).sum() / words)
        pool.append(best)
    return tuple(pool)


def holdout_score(Emat, nref, fit_rows, val_rows, choice):
    """A router's `choice` (index into the columns of `Emat`, one per validation row) scored as the exploration
    scores it: corpus WER, the share of the gap between the best single expert (best on the fit rows) and the
    oracle it closes, selection accuracy (ties count), and the share of the busiest expert."""
    fit_rows, val_rows = np.asarray(fit_rows, dtype=int), np.asarray(val_rows, dtype=int)
    E, n = Emat[val_rows], nref[val_rows].sum()
    choice = np.asarray(choice, dtype=int)
    picked = E[np.arange(len(val_rows)), choice]
    bsm = int(np.argmin(Emat[fit_rows].sum(0)))
    wer, bsm_wer, oracle = picked.sum() / n, E[:, bsm].sum() / n, E.min(1).sum() / n
    shares = np.bincount(choice, minlength=Emat.shape[1]) / len(choice)
    return {"corpus_wer": float(wer), "bsm_wer": float(bsm_wer), "oracle_wer": float(oracle),
            "gap_closed": float((bsm_wer - wer) / (bsm_wer - oracle)) if bsm_wer > oracle else np.nan,
            "acc": float((picked <= E.min(1)).mean()), "top_share": float(shares.max())}


def pool_study(corpus, main, cfg, device="cuda"):
    """HIT-ASR and MLP-pool with their main configurations on K = `cfg.ks` real experts, on the hold-out: the trio grown
    by `extend_pool`, each router fitted on the hold-out's fit rows and scored on its validation rows, once per model
    seed. `main` is the corpus's main record. Returns the record `pool_size.json` holds."""
    import time

    from hitasr.arms import MODELS
    from hitasr.core import DATASETS, use_dataset
    from hitasr.frames import FrameSet
    from hitasr.labels import LabelStore
    from hitasr.store import RouterStore
    from labkit.cv import random_partition
    from labkit.env import empty_cache

    use_dataset(corpus, verbose=False)
    trio, mc = tuple(main["members"]), main["config"]
    labels = LabelStore(experts=tuple(dict.fromkeys(trio + tuple(cfg.candidates)))).load()
    part = random_partition(labels.n, mc["holdout_frac"], mc["inner_val_frac"], mc["partition_seed"])
    errs = {m: labels.E([m])[:, 0] for m in labels.experts}
    pools = {k: extend_pool(errs, labels.nref, part["fit"], trio, [m for m in cfg.candidates if m in errs], k)
             for k in cfg.ks}
    rows = []
    for k, pool in pools.items():
        store = RouterStore(labels, pool).open(device=None)          # frames memory-mapped; batches go to the GPU
        for arm in ("hit_asr", "mlp_pool"):
            for seed in cfg.k_seeds:
                a = MODELS[arm].build(main["params"][arm], store, seed=seed,
                                      ctx={"epochs": mc["epochs"], "device": device, "nthread": mc.get("nthread", 4)})
                t0 = time.perf_counter()
                a.fit(store, part["fit"])
                fit_s = time.perf_counter() - t0
                s = holdout_score(store.E, store.nref, part["fit"], part["val"], a.predict(store, part["val"]))
                rows.append({"K": k, "arm": arm, "seed": seed, **s, "fit_s": fit_s, "n_params": a.n_params()})
                print(f"  {corpus} K={k} {arm:9s} seed {seed}: gap closed {s['gap_closed']:.3f}  "
                      f"wer {s['corpus_wer']:.4f}  ({fit_s:.0f} s)")
        del store
        empty_cache(report=False)
    if cfg.evict_frames:
        for m in labels.experts:
            FrameSet(DATASETS[corpus], m).evict()
    return {"corpus": corpus, "trio": list(trio), "pools": {str(k): list(p) for k, p in pools.items()},
            "config": {"ks": list(cfg.ks), "k_seeds": list(cfg.k_seeds), "candidates": list(cfg.candidates)},
            "n_fit": int(len(part["fit"])), "n_val": int(len(part["val"])), "rows": rows}


def pool_table(record, labels):
    """A pool-size record as a table: per K and router, the mean gap closed (sd over seeds), WER, selection accuracy,
    the busiest expert's share, and the router's size and fitting time."""
    r = pd.DataFrame(record["rows"])
    g = r.groupby(["K", "arm"], sort=True)
    t = pd.DataFrame({"gap_closed": g["gap_closed"].mean(), "gap_sd": g["gap_closed"].std(),
                      "corpus_wer": g["corpus_wer"].mean(), "acc": g["acc"].mean(), "top_share": g["top_share"].mean(),
                      "bsm_wer": g["bsm_wer"].first(), "oracle_wer": g["oracle_wer"].first(),
                      "params_M": g["n_params"].mean() / 1e6, "fit_s": g["fit_s"].mean()}).reset_index()
    t["arm"] = t["arm"].map(labels)
    return t


class Ablation:
    """The ablation notebook: HIT-ASR's design switches read off its hold-out exploration, the main table's rows that
    isolate the temporal encoder and the objective, and the pool size.

    Level 1 reads every record from the Hub; level 2 refits the pool-size study (a GPU, ~50 min)."""

    def __init__(self, cfg, level=1, device="cuda"):
        from hitasr.core import use_dataset
        from hitasr.experiment import LABELS
        from hitasr.records import load_record
        if level not in (1, 2):
            raise ValueError("level is 1 or 2")
        self.cfg, self.level, self.device, self.labels = cfg, level, device, LABELS
        corpora = tuple(dict.fromkeys(cfg.corpora + cfg.k_corpora))
        for c in corpora:
            use_dataset(c, verbose=False)
        self.main = {c: load_record(f"main_{c}", c) for c in corpora}
        self.exploration = {c: load_record("exploration_hit_asr", c) for c in cfg.corpora}
        for c in cfg.corpora:
            assert tuple(self.exploration[c]["members"]) == tuple(self.main[c]["members"]), c
        self.trials = {c: trials_frame(self.exploration[c]) for c in cfg.corpora}
        self.out = {}
        for c in cfg.corpora:
            print(f"{c}: {', '.join(LABELS[m] for m in self.main[c]['members'])} — {len(self.trials[c])} exploration "
                  "trials; main configuration " + ", ".join(f"{k}={self.main[c]['params']['hit_asr'][k]}"
                                                            for k in SWITCHES))

    def switches(self):
        """Per corpus and switch level, over every configuration drawn at seed 42: the share that collapsed onto the
        best single expert, the gap closed (median, upper quartile, best), the busiest expert's share, parameters and
        fitting time. ◀ marks the level the main search selected."""
        from labkit.pretty import Table
        out = []
        for c in self.cfg.corpora:
            chosen = {k: str(self.main[c]["params"]["hit_asr"][k]) for k in SWITCHES}
            t = pd.concat([switch_table(self.trials[c], s, collapse=self.cfg.collapse) for s in SWITCHES],
                          ignore_index=True)
            t["main"] = np.where(t["level"] == t["switch"].map(chosen), "◀", "")
            self.out.setdefault("switches", {})[c] = t
            out.append(Table(t.round(3), title=f"{c} — HIT-ASR's switches on the hold-out", group="switch",
                             highlight=lambda r: r["main"] == "◀"))
        return out

    def noise(self):
        """How much one configuration's gap closed moves when it is refitted with another seed."""
        from labkit.pretty import Table
        self.out["noise"] = noise = {c: refit_spread(self.trials[c], collapse=self.cfg.collapse)
                                     for c in self.cfg.corpora}
        tables = [Table(pd.DataFrame(noise).T.round(3), title="Refit noise: sd of one configuration's gap closed "
                        "across seeds", index=True)]
        tables += [Table(level_stability(self.trials[c], "fusion").round(3).reset_index(),
                         title=f"{c} — fusion: median gap closed per seed, configurations fitted at all three seeds")
                   for c in self.cfg.corpora]
        return tables

    def across(self):
        """Every switch level, averaged over the corpora."""
        from labkit.pretty import Table
        a = pd.concat([t.assign(corpus=c) for c, t in self.out["switches"].items()], ignore_index=True)
        s = (a.groupby(["switch", "level"], sort=False)
             .agg(collapsed=("collapsed", "mean"), gap_median=("gap_median", "mean"), gap_best=("gap_best", "mean"),
                  params_M=("params_M", "mean"), chosen_by_main=("main", lambda x: int((x == "◀").sum())))
             .reset_index())
        self.out["across"] = s
        return Table(s.round(3), title="Every switch level, averaged over the corpora", group="switch")

    def main_rows(self):
        """Rows of each corpus's Table 3 (mean ± sd over its ten folds) with the test of each row against HIT-ASR:
        the temporal encoder against mean pooling (MLP-pool), and the hard-CE controls."""
        from labkit.pretty import Table
        from labkit.significance import TEST_NOTE
        out = []
        for c in self.cfg.corpora:
            t = main_rows(self.main[c])
            self.out.setdefault("main_rows", {})[c] = t
            out.append(Table(t.round(4), title=f"{c} — on the ten folds of Table 3", note=TEST_NOTE,
                             highlight=lambda r: r["arm"] == "hit_asr"))
        return out

    def pool_size(self):
        """HIT-ASR and MLP-pool on K = 3, 5 and 10 real experts, on the hold-out."""
        from hitasr.models.registry import EXPERT_LABELS
        from hitasr.records import load_record
        from labkit.pretty import Table
        out = []
        for c in self.cfg.k_corpora:
            rec = (pool_study(c, self.main[c], self.cfg, self.device) if self.level >= 2
                   else load_record("pool_size", c))
            t = pool_table(rec, self.labels)
            self.out.setdefault("pool_size", {})[c] = t
            out.append(Table(t.round(4), title=f"{c} — K experts on the hold-out, mean over seeds "
                             f"{tuple(rec['config']['k_seeds'])}", group="K",
                             caption="pools: " + "; ".join(f"K={k}: {', '.join(EXPERT_LABELS.get(m, m) for m in p)}"
                                                           for k, p in rec["pools"].items())))
        return out

    def save(self, path="ablation.json"):
        import json
        from dataclasses import asdict
        rec = lambda df: json.loads(df.to_json(orient="records"))                         # noqa: E731
        o = self.out
        record = {"config": asdict(self.cfg), "level": self.level,
                  "switches": {c: rec(t) for c, t in o.get("switches", {}).items()}, "noise": o.get("noise"),
                  "across": rec(o["across"]) if "across" in o else None,
                  "main_rows": {c: rec(t) for c, t in o.get("main_rows", {}).items()},
                  "pool_size": {c: rec(t) for c, t in o.get("pool_size", {}).items()}}
        with open(path, "w") as fh:
            json.dump(record, fh, indent=1, default=str)
        print(f"wrote {path}")
        return path
