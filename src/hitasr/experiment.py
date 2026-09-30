"""The main experiment on one corpus, as `notebooks/main_<corpus>` runs it: the hyperparameter search on a hold-out,
5x2 cross-validation on the remaining rows, and the paper's tables.

    exp = MainExperiment(MAIN["ami_sdm"], level=1)
    exp.spaces()                  # Table 2
    exp.load()                    # the trio's labels (levels 2-3: and its frames)
    exp.search()                  # every arm's hyperparameters, on the hold-out
    exp.dataset(); exp.selected() # Table 1, and what the search selected
    exp.cross_validate()          # every arm on the same ten folds
    exp.results(); exp.routing(); exp.tests()      # Table 3, the routing behaviour, Table 6
    exp.save()                    # main_<corpus>.json

**Levels.** 1 reads every search and every fold from the Hub and fits nothing (a missing one stops the run); 2 refits
every fold from the published labels and frames (`retune=True`: the search as well); 3 is level 2 on labels and frames
you rebuilt yourself (`notebooks/extract`, then `HITASR_HUB`). Nothing is ever written to the Hub.
"""

__all__ = ['MainExperiment', 'LABELS', 'table_rows']

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import hitasr.rover  # noqa: F401  (registers the fusion arms)
from hitasr.arms import MODELS, check_spaces
from hitasr.configs import ARMS, BASELINE_ARMS, CONTROL_ARMS, SELECTION_ARMS, SPACES
from hitasr.core import PUBLIC_REPO, REPO_ID, use_dataset
from hitasr.crossval import (ACC_TOLERANCES, METRICS, cost_table, format_results, results_table,
                             selection_distribution)
from hitasr.eda import dataset_table
from hitasr.labels import LabelStore
from hitasr.models.registry import EXPERT_LABELS
from hitasr.rover import check_counters
from hitasr.runlog import RunLog
from hitasr.store import RouterStore
from hitasr.tuning import hpt
from labkit.cv import RepetitionSplit
from labkit.env import installed_commit, set_determinism
from labkit.hub import set_read_only
from labkit.pretty import Table
from labkit.search import space_source
from labkit.significance import TEST_NAME, order_repetitions, significance_table, significance_view

LABELS = {**EXPERT_LABELS, **{a: MODELS[a].label for a in MODELS},
          "bsm": "Best single expert", "random": "Random", "worst": "Worst single expert", "oracle": "Oracle",
          "oracle_tol1": "Oracle (1-word tolerance)", "oracle_tol2": "Oracle (2-word tolerance)"}
METRIC_TITLES = {"corpus_wer": "corpus WER", "mean_utt_wer": "mean utterance WER"}
SHOWN_METRICS = ("corpus_wer", "mean_utt_wer", "gap_closed_tol0", "acc_tol0")   # Table 3 as displayed (all are recorded)


def table_rows(members):
    """Table 3's rows in order: `(key, family)`."""
    return ([(m, "expert") for m in members] + [("bsm", "expert"), ("random", "reference")]
            + [(a, "fusion") for a in BASELINE_ARMS]
            + [(a, "pooled") for a in ("mlp_pool", "adastt_ce")]
            + [("hit_asr", "ours")]
            + [(a, "oracle") for a in ("oracle", "oracle_tol1", "oracle_tol2")])


class MainExperiment:
    """One corpus's main experiment at one reproduction level. See the module docstring for the order of calls."""

    def __init__(self, cfg, level=1, retune=False, device="cuda"):
        if level not in (1, 2, 3):
            raise ValueError("level is 1, 2 or 3")
        if level == 3 and REPO_ID == PUBLIC_REPO:
            raise RuntimeError("level 3 reads your own extraction: set HITASR_HUB to the repo `extract` wrote, "
                               "then restart")
        self.cfg, self.level, self.retune, self.device = cfg, level, bool(retune), device
        self.spec = use_dataset(cfg.dataset, verbose=False)
        set_determinism(cfg.model_seed)
        set_read_only(True)
        check_spaces({a: SPACES[a] for a in ARMS})
        refit_search, self.refit_folds = level >= 2 and retune, level >= 2
        self._search_cache = dict(sync=not refit_search, readonly=True, strict=not refit_search)
        self._fold_cache = dict(sync=not self.refit_folds, readonly=True, refresh=self.refit_folds,
                                strict=not self.refit_folds)
        self.ctx = {"epochs": cfg.epochs, "device": device, "nthread": cfg.nthread}
        # local mirrors of the searches and folds read from (or, at levels 2-3, computed for) this run
        self.workdir = Path(os.environ.get("HITASR_CACHE", Path.home() / ".cache" / "hitasr")) / "runs" / f"level{level}"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rows = table_rows(cfg.members)
        self.tables = {}
        print(f"{cfg.title}: level {level} — search {'refitted' if refit_search else 'read from the Hub'}, "
              f"folds {'refitted' if self.refit_folds else 'read from the Hub'}\n"
              f"experts  {', '.join(LABELS.get(m, m) for m in cfg.members)}\n"
              f"arms     {', '.join(LABELS.get(a, a) for a in ARMS)}")

    # ------------------------------------------------------------------ Table 2 --

    def spaces(self):
        """Table 2: the free parameters of every arm and their choices."""
        rows = []
        for arm in ARMS:
            free = [(n, p) for n, p in SPACES[arm].items() if p.frozen is None]
            for n, p in free:
                rows.append({"arm": f"{LABELS.get(arm, arm)}  ({len(free)} free of {len(SPACES[arm])})", "parameter": n,
                             "choices": "{" + ", ".join(str(c) for c in p.choices) + "}"})
        return Table(pd.DataFrame(rows), title="Table 2 — hyperparameter spaces",
                     caption=f"{self.cfg.search_budget} random draws per arm; parameters not listed are fixed "
                             "(hitasr.configs.SPACES)", group="arm", mono=("choices",), wrap=("choices",))

    # --------------------------------------------------------------------- data --

    def load(self):
        """The trio's labels — transcripts and WER counters — and, when anything is refitted, its frames. The stored
        WER counters are recomputed from the stored transcripts on a sample of rows."""
        members = tuple(self.cfg.members)
        self.labels = LabelStore(experts=members).load()
        self.store = RouterStore(self.labels, members).open(device=self.device if self.refit_folds else None,
                                                            frames=self.refit_folds)
        counters = check_counters(self.store, n=400, seed=self.cfg.model_seed)
        assert float(counters["match"].mean()) == 1.0, "the stored WER counters do not reproduce"
        return self.store

    # ------------------------------------------------------------------- search --

    def search(self):
        """Every arm's hyperparameters, chosen on a random 25 % hold-out: `search_budget` random draws from its
        space, each fitted on 70 % of the hold-out and scored on the rest."""
        c = self.cfg
        self.run = hpt(self.store, ARMS, trials=c.search_budget, spaces=SPACES, study_seed=c.study_seed,
                       model_seed=c.model_seed, versions=c.arm_versions, pruner="none", sampler="random",
                       objectives=("corpus_wer",), holdout_frac=c.holdout_frac, inner_val_frac=c.inner_val_frac,
                       partition_seed=c.partition_seed, ctx=self.ctx, save=False, dirname=str(self.workdir),
                       verbose=not self._search_cache["strict"], **self._search_cache)
        if self._search_cache["strict"]:                   # read, not run: the partition, and what was read
            self.run.tuner.report()
            print("  " + "; ".join(f"{LABELS.get(a, a)} {r.n_trials} trials" for a, r in self.run.results.items())
                  + " — read from the Hub")
        return self.run

    def dataset(self):
        """Table 1: the corpus, the hold-out and the folds."""
        self.tables["table1"] = t = dataset_table(self.labels, self.run.tuner.rows)
        return Table(t, title=f"Table 1 — {self.cfg.title}",
                     caption="train / test sizes are those of one 5x2 fold; the hold-out is the search partition")

    def selected(self):
        """What the search selected per arm, with its WER on the hold-out's scoring rows."""
        self.tables["selected"] = t = pd.DataFrame([
            {"arm": a, "label": LABELS.get(a, a), "trials": r.n_trials, "hold-out WER": r.value,
             "selected configuration": json.dumps(r.params, sort_keys=True, default=str)}
            for a, r in self.run.results.items()])
        return Table(t.drop(columns="arm"), title="Selected configurations",
                     caption=f"corpus WER on the search's scoring rows after {self.cfg.search_budget} trials per arm "
                             "— a hold-out number, not a test number",
                     best={"hold-out WER": "min"}, mono=("selected configuration",), wrap=("selected configuration",))

    # --------------------------------------------------------------- 5x2 folds --

    def cross_validate(self):
        """Each repetition shuffles the evaluation rows with its seed and halves them; every arm is fitted on one half
        and scored on the other, then the other way round. Returns `(fold_scores, fold_picks)` of the reported
        repetitions, in the order of `cv_seeds`."""
        c, run = self.cfg, self.run
        log = RunLog(f"main_{c.dataset}", dirname=self.workdir, spec=self.spec, **self._fold_cache)
        key = {"partition": run.partition(), "n_eval": int(len(run.eval_rows)), "random_seed": c.cv_random_seed,
               "member_arms": True, "tolerances": list(ACC_TOLERANCES), "metrics": sorted(METRICS),
               "epochs": c.epochs}

        def one_repetition(seed, estimators):
            return run.evaluate(estimators=estimators, scheme=RepetitionSplit(seed), random_seed=c.cv_random_seed,
                                member_arms=True, tolerances=ACC_TOLERANCES, fold_table=True)

        scores, picks = log.per_arm("cv5x2", c.cv_seeds, run.arms, one_repetition, key=key, arm_keys=run.arm_keys())
        self.fold_scores, self.fold_picks = order_repetitions(scores, picks, c.cv_seeds, c.first_fold)
        self.n_folds = int(self.fold_scores.drop_duplicates(["repeat", "fold"]).shape[0])
        self.bsm = self.fold_scores["bsm"].mode().iloc[0]
        print(f"{len(c.cv_seeds)} repetitions x 2 folds x {self.fold_scores['model'].nunique()} arms")
        return self.fold_scores, self.fold_picks

    # ------------------------------------------------------------------- tables --

    def results(self):
        """Table 3: every arm, mean ± sd over the ten folds."""
        order = [k for k, _ in self.rows]
        t3 = results_table(self.fold_scores, self.fold_picks, self.cfg.members, order=order, labels=LABELS)
        t3.insert(2, "family", t3["model"].map(dict(self.rows)).fillna("other"))
        self.tables["table3"] = t3
        shown = t3[t3["model"].isin(order)].copy()
        shown.loc[shown["model"] == "bsm", "label"] = f"Best single expert ({LABELS.get(self.bsm, self.bsm)})"
        full, fmt = format_results(shown), format_results(shown, metrics=SHOWN_METRICS)
        for f in (full, fmt):
            f.insert(1, "family", shown["family"].values)
        self.tables["table3_formatted"] = full
        return Table(fmt, title=f"Table 3 — main results, {self.cfg.title}",
                     caption=f"mean ± sd over {self.n_folds} folds; WER and uWER (mean utterance WER) as fractions, "
                             "GC (share of the gap between the best single expert and the oracle closed) and Acc "
                             "(selection accuracy) in %; "
                             f"dist = share of clips routed to {' / '.join(LABELS.get(m, m) for m in self.cfg.members)}",
                     group="family", best="auto", best_exclude=lambda r: r["family"] == "oracle",
                     highlight=lambda r: r["family"] == "ours", mono=("dist",))

    def routing(self):
        """How each router distributes the clips over the trio: shares, entropy (bits), switch rate away from the best
        single expert."""
        beh = selection_distribution(self.fold_picks, list(self.cfg.members), fold_scores=self.fold_scores)
        self.tables["behaviour"] = beh
        keep = [k for k, f in self.rows if f in ("reference", "pooled", "ours", "oracle")]
        t = beh[beh["model"].isin(keep) & (beh["n_fused"] == 0)].copy()
        t["model"] = pd.Categorical(t["model"], keep, ordered=True)
        t = t.sort_values("model")
        out = pd.DataFrame({"router": t["model"].map(LABELS).astype(str)})
        for m in self.cfg.members:
            out[LABELS.get(m, m)] = (100 * t[f"share_{m}"]).round(1).to_numpy()
        out["entropy"] = t["entropy"].round(3).to_numpy()
        out["switch rate"] = (100 * t["switch_rate"]).round(1).to_numpy()
        return Table(out, title=f"Routing behaviour — {self.cfg.title}",
                     caption=f"share of clips (%) routed to each expert over the {self.n_folds} folds; entropy in bits "
                             f"(max {beh['entropy_max'].iloc[0]:.2f}); switch rate = % of clips not routed to the best "
                             "single expert", highlight=lambda r: r["router"] == "HIT-ASR")

    def tests(self):
        """Table 6: HIT-ASR against every other system on the identical folds."""
        comparators = [k for k, _ in self.rows]
        self.table6 = {m: significance_table(self.fold_scores, m, "hit_asr", comparators, alpha=self.cfg.alpha)
                       for m in METRIC_TITLES}
        self.table6_controls = {m: significance_table(self.fold_scores, m, "hit_asr", list(CONTROL_ARMS),
                                                      alpha=self.cfg.alpha) for m in METRIC_TITLES}
        both = pd.concat([t.assign(metric=METRIC_TITLES[m]) for m, t in self.table6.items()], ignore_index=True)
        both.attrs = dict(self.table6["corpus_wer"].attrs)
        view = significance_view(both, labels=LABELS, title=f"Table 6 — HIT-ASR against every system, {self.cfg.title}",
                                 caption=f"on the same {self.n_folds} folds; effect in the metric's own units")
        view.df.insert(0, "metric", both["metric"].values)
        view.style["group"] = "metric"
        return view

    # ------------------------------------------------------------------- record --

    def record(self):
        """Everything above in one JSON-able dict: the configuration, the selected hyperparameters, the search spaces
        as source, every table."""
        c, run = self.cfg, self.run
        rec = lambda df: json.loads(df.to_json(orient="records"))                         # noqa: E731
        tests = lambda tabs: {m: {"anchor": t.attrs["anchor"], "test": TEST_NAME, "rows": rec(t)}  # noqa: E731
                              for m, t in tabs.items()}
        return {
            "notebook": f"main_{c.dataset}", "hitasr_commit": installed_commit(), "level": self.level,
            "retune": self.retune, "device": self.device, "config": asdict(c), "members": list(c.members),
            "corpus": {"name": self.spec.name, "source": self.spec.source, "base": self.spec.base_config,
                       "license": self.spec.license},
            "n_eval_rows": int(len(run.eval_rows)), "n_train_per_fold": int(len(run.eval_rows) // 2),
            "seeds": {"seeds": list(c.cv_seeds), "first_fold": int(c.first_fold)}, "bsm": self.bsm,
            "study": run.name, "studies": run.study_paths(), "versions": dict(c.arm_versions),
            "search_spaces": {a: space_source(SPACES[a], varname=a) for a in ARMS},
            "params": {a: r.params for a, r in run.results.items()},
            "arms": {"selection": list(SELECTION_ARMS), "baseline": list(BASELINE_ARMS),
                     "control": list(CONTROL_ARMS)},
            "rows": self.rows, "labels": LABELS,
            "tables": {"table1": rec(self.tables["table1"]), "table3": rec(self.tables["table3"]),
                       "table3_formatted": rec(self.tables["table3_formatted"]),
                       "behaviour": rec(self.tables["behaviour"]), "table6": tests(self.table6),
                       "table6_controls": tests(self.table6_controls), "selected": rec(self.tables["selected"]),
                       "cost": rec(cost_table(self.fold_scores, results=run.results))},
        }

    def save(self, path=None):
        """Write the record (`main_<corpus>.json` in the working directory)."""
        path = path or f"main_{self.cfg.dataset}.json"
        with open(path, "w") as fh:
            json.dump(self.record(), fh, indent=2, default=str)
        print(f"wrote {path}")
        return path


def main(argv=None):
    """`python -m hitasr.experiment <corpus> [--level N] [--retune] [--device D]`: a main notebook as a script, the
    tables printed as text and the record written to the working directory."""
    import argparse

    from hitasr.configs import MAIN
    ap = argparse.ArgumentParser(description=main.__doc__.split(":")[0])
    ap.add_argument("corpus", choices=sorted(MAIN))
    ap.add_argument("--level", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    exp = MainExperiment(MAIN[a.corpus], level=a.level, retune=a.retune, device=a.device)
    print(exp.spaces())
    exp.load()
    exp.search()
    for table in (exp.dataset(), exp.selected()):
        print(table)
    exp.cross_validate()
    for table in (exp.results(), exp.routing(), exp.tests()):
        print(table)
    exp.save()


if __name__ == "__main__":
    main()
