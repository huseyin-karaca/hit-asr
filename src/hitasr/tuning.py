"""The hyperparameter search of the paper: the hold-out partition, one Optuna study per arm with a fixed budget of
random draws, resumable on the Hub, and the content-addressed names the studies and the fold caches are found by."""

__all__ = ['OBJECTIVES', 'SAMPLERS', 'PARTITION_FIELDS', 'trial_metrics_line', 'Tuner', 'StudyStore', 'TuningResult',
           'hpt_identity', 'partition_identity', 'arm_identity', 'arm_study_name', 'cv_arm_key', 'HPTRun',
           'save_tuning', 'hpt', 'hpt_status', 'load_hpt_json']

import hashlib
import json
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hitasr.arms import MODELS, check_spaces
from labkit.search import space_source
from hitasr.crossval import (CONTEXT_METRICS, METRICS, TABLE_LABELS, TABLE_METRICS, fit_arm, format_shares, run_cv,
                             score_arm, selection_shares)
from labkit.cv import FiveByTwoSplit, random_partition
from hitasr.hub import HitHub
from labkit.runlog import CacheMiss
from labkit.studies import StudyStore as _StudyStore
from labkit.studies import _sign, best_trial, completed, sync_ledgers
from labkit.pretty import metric_token, paint, show_table, signed, state

try:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
except ImportError:                                            # pragma: no cover
    optuna = TrialState = None


def _require_optuna():
    if optuna is None:
        raise ImportError("tuning needs Optuna — `pip install optuna`")


OBJECTIVES = {
    "corpus_wer": ("minimize", "sum(errors) / sum(words) on the scoring rows"),
    "mean_utt_wer": ("minimize", "mean per-utterance WER on the scoring rows"),
    "fit_seconds": ("minimize", "wall-clock of the fit"),
}

SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


def trial_metrics_line(attrs):
    parts = []
    for m in TABLE_METRICS:
        v = attrs.get(m)
        if v is None or not np.isfinite(v):
            continue
        lab = TABLE_LABELS.get(m, m)
        parts.append(metric_token(lab, f"{100 * v:.1f}%" if m.startswith(("gap_closed", "acc_")) else f"{v:.4f}"))
    if attrs.get("selection_dist"):
        parts.append(metric_token("dist", attrs["selection_dist"]))
    return "  ".join(parts)


class Tuner:
    """The partition, the reference points, and an Optuna objective per arm.

    Parameters
    ----------
    store : a `RouterStore` (frames opened if any frame arm is to be tuned).
    holdout_frac, inner_val_frac, partition_seed : the hold-out cut.
    model_seed, study_seed : every fit; the sampler.
    inner_folds : 1 = the fixed `fit`/`val` cut; k = k-fold over the whole
        hold-out, a trial scored on every tuning row by a model that did not
        fit it. The trial then also carries `fold_wers`, `fold_bsm` and
        `fold_gaps` (each fold against its own fit-row BSM and oracle), and
        every trial carries `gap_closed_oof`, the gap against the fold-wise
        constant policy pooled out of fold.
    objectives : keys of `OBJECTIVES`; the first is primary.
    ctx : what builders need that is not a hyperparameter — `epochs`, `device`,
        `nthread`.
    """

    def __init__(self, store, holdout_frac=0.25, inner_val_frac=0.30, partition_seed=20260901,
                 model_seed=42, study_seed=7, inner_folds=1, objectives=("corpus_wer",),
                 sampler="random", pruner="median", ctx=None):
        self.store, self.members = store, tuple(store.members)
        self.model_seed, self.study_seed = model_seed, study_seed
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(SAMPLERS)}")
        self.sampler, self.pruner = sampler, pruner
        self.objectives = tuple(objectives)
        for o in self.objectives:
            if o not in OBJECTIVES:
                raise KeyError(f"unknown objective {o!r}; have {sorted(OBJECTIVES)}")
        self.inner_folds = int(inner_folds)
        self.ctx = dict(ctx or {})
        self.E, self.nref = store.E, store.nref
        self.rows = random_partition(store.n, holdout_frac, inner_val_frac, partition_seed)
        self.partition_seed, self.holdout_frac, self.inner_val_frac = partition_seed, holdout_frac, inner_val_frac
        fit = self.rows["fit"]
        self.score_rows = self.rows["val"] if self.inner_folds == 1 else self.rows["tune"]
        bsm = int((self.E[fit].sum(0) / self.nref[fit].sum()).argmin())
        self.bsm, self.bsm_model = bsm, self.members[bsm]
        sr = self.score_rows
        self.bsm_wer = self.corpus_wer(sr, np.full(len(sr), bsm))
        self.oracle_wer = self.corpus_wer(sr, self.E[sr].argmin(1))

    def __repr__(self):
        return (f"Tuner({'+'.join(self.members)}, tune {len(self.rows['tune']):,} / "
                f"eval {len(self.rows['eval']):,}, inner_folds={self.inner_folds}, "
                f"objectives={'+'.join(self.objectives)})")

    def directions(self):
        return [OBJECTIVES[o][0] for o in self.objectives]

    def inner_splits(self):
        if self.inner_folds == 1:
            return [(self.rows["fit"], self.rows["val"])]
        tune = self.rows["tune"]
        order = np.random.default_rng(self.partition_seed + 2).permutation(len(tune))
        chunks = np.array_split(order, self.inner_folds)
        return [(np.sort(tune[np.concatenate([c for j, c in enumerate(chunks) if j != i])]),
                 np.sort(tune[np.sort(chunk)])) for i, chunk in enumerate(chunks)]

    def corpus_wer(self, rows, choice):
        return float(self.E[rows, np.asarray(choice, dtype=int)].sum() / self.nref[rows].sum())

    def gap_closed(self, wer):
        gap = self.bsm_wer - self.oracle_wer
        return float("nan") if gap <= 0 else (self.bsm_wer - wer) / gap

    def report(self, verbose=True):
        sizes = pd.DataFrame([{"partition": k, "n_utts": len(v), "share": len(v) / self.store.n}
                              for k, v in self.rows.items()])
        if verbose:
            show_table(sizes.round(4), title=f"{self.store.n:,} pooled utterances",
                       caption="the tuning partition — what a trial may fit, what it is scored on, and what it never sees")
            print(f"  on the tuning-score rows: bsm({self.bsm_model}) {paint(f'{self.bsm_wer:.4f}', 'bold')}   "
                  f"oracle {paint(f'{self.oracle_wer:.4f}', 'bold')}   gap {self.bsm_wer - self.oracle_wer:.4f}")
            print(f"  the {len(self.rows['eval']):,} evaluation rows are never seen by a trial")
        return sizes

    def _progress(self, name, n_trials):
        def callback(study, trial):
            done = completed(study)
            pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            secs = ((trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start else float("nan"))
            s = _sign(OBJECTIVES[self.objectives[0]][0])
            best_val = min((t.values[0] for t in done), key=lambda x: s * x) if done else None
            if trial.state == TrialState.COMPLETE and trial.values:
                gap = trial.user_attrs.get("gap_closed")
                what = paint(f"{self.objectives[0]} {trial.values[0]:.4f}", "bold")
                if gap is not None and np.isfinite(gap):
                    what += "  " + signed(f"gap {gap:+.1%}", gap)
                improved = best_val is not None and trial.values[0] == best_val
            else:
                what, improved = state("pruned"), False
            best_txt = (paint(f"best {best_val:.4f} *", "green", "bold") if improved
                        else f"best {best_val:.4f}  " if best_val is not None else "best --")
            print(f"  {paint(name, 'dim')} {paint(f'{trial.number + 1:4d}/{n_trials}', 'bold')}  {what}  "
                  f"{best_txt}  pruned {pruned:3d}  {paint(f'{secs:6.1f}s', 'dim')}", flush=True)
            if trial.state == TrialState.COMPLETE and trial.values:
                print("      " + trial_metrics_line(trial.user_attrs), flush=True)
        return callback

    def objective(self, spec, space, pruner="median"):
        """An Optuna objective over this tuner's hold-out for one arm."""
        seed = self.model_seed
        splits = self.inner_splits()

        def objective(trial):
            p = {n: prm.suggest(trial) for n, prm in space.items()}
            errs, refs, Es, choices, per_fold = [], [], [], [], []
            fold_bsm, fold_ref, fold_orc, ref_errs, orc_errs = [], [], [], [], []   # the fold's own constant policies
            t_fit = t_pred = 0.0
            t_trial = time.perf_counter()
            for step, (fit_rows, val_rows) in enumerate(splits):
                arm = spec.build(p, self.store, seed=seed, ctx=self.ctx)
                if spec.prunable and pruner == "median" and len(splits) == 1:
                    def report(epoch, val_wer):
                        trial.report(val_wer, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    t0 = time.perf_counter()
                    arm.fit(self.store, fit_rows, callback=report)
                    t1 = time.perf_counter()
                    out = np.asarray(arm.predict(self.store, val_rows))
                    t_fit += t1 - t0; t_pred += time.perf_counter() - t1
                    kind = getattr(arm, "produces", "choice")
                else:
                    timings = {}
                    kind, out = fit_arm(arm, self.store, fit_rows, val_rows, timings=timings)
                    t_fit += timings["fit_seconds"]; t_pred += timings["predict_seconds"]
                picked, choice = score_arm(self.E[val_rows], kind, out)
                choices.append(np.full(len(picked), -1) if choice is None else choice)
                errs.append(picked); refs.append(self.nref[val_rows]); Es.append(self.E[val_rows])
                per_fold.append(float(picked.sum() / self.nref[val_rows].sum()))
                # the fold's reference points: the best single member OF ITS FIT ROWS scored on its scoring rows
                # (the constant policy under the same protocol as the trial), and the oracle on those rows
                b = int((self.E[fit_rows].sum(0) / self.nref[fit_rows].sum()).argmin())
                fold_bsm.append(b)
                ref_errs.append(self.E[val_rows, b]); orc_errs.append(self.E[val_rows].min(1))
                fold_ref.append(float(ref_errs[-1].sum() / self.nref[val_rows].sum()))
                fold_orc.append(float(orc_errs[-1].sum() / self.nref[val_rows].sum()))
                if len(splits) > 1 and pruner == "median":
                    trial.report(float(np.concatenate(errs).sum() / np.concatenate(refs).sum()), step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if hasattr(arm, "n_params") and arm.n_params():
                    trial.set_user_attr("n_params", int(arm.n_params()))
            picked, refs_all, E_all = np.concatenate(errs), np.concatenate(refs), np.concatenate(Es, axis=0)
            wer = float(picked.sum() / refs_all.sum())
            ref_oof = float(np.concatenate(ref_errs).sum() / refs_all.sum())     # the per-fold BSM policy, out of fold
            orc_oof = float(np.concatenate(orc_errs).sum() / refs_all.sum())
            if len(per_fold) > 1:
                trial.set_user_attr("fold_wers", per_fold)
                trial.set_user_attr("fold_wer_sd", float(np.std(per_fold, ddof=1)))
                trial.set_user_attr("fold_bsm", fold_bsm)
                trial.set_user_attr("fold_gaps", [(r - w) / (r - o) if r > o else float("nan")
                                                  for w, r, o in zip(per_fold, fold_ref, fold_orc)])
            shares = selection_shares(np.concatenate(choices), self.E.shape[1])
            trial.set_user_attr("selection_dist", format_shares(shares))
            trial.set_user_attr("members", "+".join(self.members))     # a study name carries no trio; the trial does
            scores = {}
            for name, (fn, _d) in METRICS.items():
                if name in CONTEXT_METRICS:
                    continue
                try:
                    scores[name] = float(fn(E_all, refs_all, picked, self.bsm))
                except Exception:                              # noqa: BLE001
                    scores[name] = float("nan")
            scores.update(trial_seconds=time.perf_counter() - t_trial, fit_seconds=t_fit,
                          predict_seconds=t_pred, gap_closed=self.gap_closed(wer),
                          # the same gap against the fold-wise constant policy: identical to gap_closed on the
                          # fixed fit/val cut, the honest reference under k-fold
                          gap_closed_oof=(ref_oof - wer) / (ref_oof - orc_oof) if ref_oof > orc_oof else float("nan"))
            for k, v in scores.items():
                trial.set_user_attr(k, float(v))
            values = [float(scores[o]) for o in self.objectives]
            if any(np.isnan(v) for v in values):
                raise optuna.TrialPruned(f"{spec.name} cannot report {self.objectives}")
            return values[0] if len(values) == 1 else tuple(values)
        return objective

    def make_pruner(self, kind=None, warmup=3):
        kind = self.pruner if kind is None else kind
        if not isinstance(kind, str):
            return kind
        if kind == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=warmup)
        if kind == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"unknown pruner {kind!r}")

    def study(self, spec, space, n_trials, seed=None, name="study", verbose=True, pruner=None, store=None,
              warm_start=None):
        """One study over `space`, run up to `n_trials` **in total** (resumed if `store` holds some).

        `warm_start` is a list of parameter dicts queued ahead of any fresh
        sampling; configurations the study already holds are not queued twice.
        They count against `n_trials`. This is how `13_hparam_explore` runs one
        configuration pool in every context.
        """
        _require_optuna()
        kind = self.pruner if pruner is None else pruner
        storage = store.url() if isinstance(store, StudyStore) else store
        sampler = SAMPLERS[self.sampler](seed if seed is not None else self.study_seed)
        kw = dict(study_name=name, sampler=sampler, storage=storage, load_if_exists=storage is not None,
                  pruner=self.make_pruner(kind))
        multi = len(self.objectives) > 1
        study = (optuna.create_study(directions=self.directions(), **kw) if multi
                 else optuna.create_study(direction=self.directions()[0], **kw))
        study.set_user_attr("objectives", list(self.objectives))
        if warm_start:
            seen = {json.dumps(t.params, sort_keys=True, default=str) for t in study.trials}
            queued = 0
            for params in warm_start:
                free = {k: v for k, v in params.items() if k in space and space[k].frozen is None}
                if json.dumps(free, sort_keys=True, default=str) not in seen:
                    study.enqueue_trial(free, skip_if_exists=True)
                    queued += 1
            if verbose and queued:
                print(f"  {name}: queued {queued} configuration(s)")
        already = sum(1 for t in study.trials if t.state != TrialState.WAITING)
        todo = max(0, n_trials - already)
        if verbose and already:
            print(f"  {name}: resuming — {already} trial(s) already stored, {todo} to go")
        if todo:
            callbacks = [self._progress(name, n_trials)] if verbose else []
            if isinstance(store, StudyStore):
                callbacks.append(store.callback())
            study.optimize(self.objective(spec, space, pruner=kind if isinstance(kind, str) else "median"),
                           n_trials=todo, gc_after_trial=True, callbacks=callbacks or None)
            if isinstance(store, StudyStore):
                store.push(message=f"{name}: {len(study.trials)} trials", verbose=verbose)
        if verbose:
            done = completed(study)
            best = f"{best_trial(study).values[0]:.4f}" if done else "--"
            print(f"  {paint(name, 'dim')}: {len(done)} complete, "
                  f"{sum(1 for t in study.trials if t.state == TrialState.PRUNED)} pruned, "
                  f"best {self.objectives[0]} {paint(best, 'green', 'bold')} "
                  f"(bsm {self.bsm_wer:.4f}, oracle {self.oracle_wer:.4f})")
        return study

class StudyStore(_StudyStore):
    """`labkit.studies.StudyStore` for one corpus: the ledger lives under `studies/ledger/<corpus>/<name>/` of the
    corpus's records repo."""

    def __init__(self, name, dirname=None, repo_id=None, spec=None, sync=True, sync_every=5, readonly=False):
        self.repo_id, self.spec = repo_id, spec
        super().__init__(name, dirname, group=getattr(spec, "name", None) or "shared", sync=sync,
                         sync_every=sync_every, readonly=readonly)

    def hub(self):
        return HitHub(self.repo_id, spec=self.spec)


@dataclass
class TuningResult:
    name: str
    params: dict
    value: float
    gap_closed: float
    space: dict
    study: object = None
    n_trials: int = 0
    seconds: float = 0.0
    trial_number: int = None

    def as_row(self):
        return {"model": self.name, "value": self.value, "gap_closed": self.gap_closed,
                "n_trials": self.n_trials, "seconds": self.seconds,
                "params": json.dumps(self.params, default=str, sort_keys=True)}


def _digest(payload):
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


PARTITION_FIELDS = ("dataset", "members", "n_rows", "holdout_frac", "partition_seed")


def hpt_identity(tuner, models, spaces=None):
    """Everything that makes one tuning call a different search from another (budget excluded)."""
    return {"dataset": tuner.store.spec.name, "members": list(tuner.members), "models": list(models),
            "objectives": list(tuner.objectives), "n_rows": int(tuner.store.n),
            "holdout_frac": float(tuner.holdout_frac), "inner_val_frac": float(tuner.inner_val_frac),
            "inner_folds": int(tuner.inner_folds), "partition_seed": int(tuner.partition_seed),
            "sampler": str(tuner.sampler), "ctx": {k: v for k, v in tuner.ctx.items() if k != "device"},
            "spaces": {k: sorted((n, repr(p)) for n, p in v.items()) for k, v in (spaces or {}).items()}}


def partition_identity(identity):
    return {k: identity[k] for k in PARTITION_FIELDS if k in identity}


def arm_identity(identity, arm, version=None):
    out = {k: v for k, v in identity.items() if k not in ("models", "spaces")}
    out.update(arm=arm, space=(identity.get("spaces") or {}).get(arm, []))
    if version is not None:
        out["version"] = str(version)
    return out


def arm_study_name(spec, arm, identity, version=None):
    return f"hpt_{spec.name}_{arm}_{_digest(arm_identity(identity, arm, version))}"


def cv_arm_key(identity, arm, params, version=None):
    key = {"arm": arm, "params": json.loads(json.dumps(params, sort_keys=True, default=str))}
    if version is not None:
        key["version"] = str(version)
    return key


@dataclass
class HPTRun:
    """One tuning run: what won, where it is stored, what it may be scored on."""
    name: str
    tuner: object
    results: dict
    studies: object = None
    path: str = None
    identity: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)

    def __repr__(self):
        best = min(self.results.values(), key=lambda r: r.value, default=None)
        won = f"best {best.name} {best.value:.4f}" if best else "no arms"
        return f"HPTRun({self.name!r}, {len(self.results)} arm(s), {won}, eval on {len(self.eval_rows):,} rows)"

    @property
    def arms(self):
        """`{name: factory}` — the tuned configurations, ready for `run_cv`."""
        t = self.tuner
        return {a: (lambda a=a, r=r: MODELS[a].build(r.params, t.store, seed=t.model_seed, ctx=t.ctx))
                for a, r in self.results.items()}

    @property
    def eval_rows(self):
        return self.tuner.rows["eval"]

    def table(self):
        return pd.DataFrame([r.as_row() for r in self.results.values()]).sort_values("value", ignore_index=True)

    def partition(self):
        return partition_identity(self.identity)

    def arm_keys(self):
        return {a: cv_arm_key(self.identity, a, r.params, self.versions.get(a)) for a, r in self.results.items()}

    def study_paths(self):
        return {a: {"db": self.studies[a].path_in_repo if self.studies else None,
                    "study": f"{self.names.get(a, a)}_final"} for a in self.results}

    def evaluate(self, estimators=None, scheme=None, seeds=None, verbose=True, **kw):
        """5x2cv over the held-out rows. Returns `(fold_scores, fold_picks)`."""
        if scheme is None:
            scheme = FiveByTwoSplit(seeds=seeds) if seeds else FiveByTwoSplit()
        arms = dict(self.arms if estimators is None else estimators)
        rows = self.eval_rows
        if verbose:
            print(f"{scheme.__class__.__name__} over {len(rows):,} of {self.tuner.store.n:,} rows — "
                  f"the {len(self.tuner.rows['tune']):,} the tuner saw are excluded.\narms: {', '.join(arms)}\n")
        return run_cv(self.tuner.store, arms, scheme=scheme, rows=rows, verbose=verbose, **kw)


def _tuning_json_path(spec, name):
    return spec.results_path(f"tuning/{name}")


def save_tuning(results, path, tuner, extra=None):
    payload = {"members": list(tuner.members), "partition_seed": tuner.partition_seed,
               "bsm_wer": tuner.bsm_wer, "oracle_wer": tuner.oracle_wer,
               "results": {a: {"params": r.params, "value": r.value, "gap_closed": r.gap_closed,
                               "n_trials": r.n_trials, "seconds": r.seconds,
                               "space": space_source(r.space, varname=a)} for a, r in results.items()},
               **(extra or {})}
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    return payload


def hpt(store, models, name=None, objectives=("corpus_wer",), sampler="random", trials=10,
        holdout_frac=0.25, inner_val_frac=0.30, inner_folds=1, partition_seed=20260901,
        model_seed=42, study_seed=7, pruner="median", spaces=None, studies=True, sync=True,
        sync_every=5, dirname=None, save=True, versions=None, ctx=None, verbose=True, readonly=False, strict=False):
    """Tune `models` on `store` and return an `HPTRun`. `trials` random draws per arm, resumable.

    `sync` reads (and unless `readonly` extends) each arm's study on the Hub; `strict` runs no trial at all — an
    arm whose study holds fewer than `trials` trials raises `CacheMiss` — which is how a reproduction reads the
    published search without re-running it.
    """
    _require_optuna()
    models = tuple(models)
    versions = {k: str(v) for k, v in (versions or {}).items()}
    tuner = Tuner(store, holdout_frac=holdout_frac, inner_val_frac=inner_val_frac,
                  partition_seed=partition_seed, model_seed=model_seed, study_seed=study_seed,
                  inner_folds=inner_folds, objectives=objectives, sampler=sampler, pruner=pruner, ctx=ctx)
    spec = store.spec
    spaces = dict(spaces or {})
    check_spaces(spaces)
    identity = hpt_identity(tuner, models, spaces)
    names = {a: (arm_study_name(spec, a, identity, versions.get(a)) if name is None
                 else f"{name}_{a}" + (f"_{versions[a]}" if a in versions else "")) for a in models}
    name = name or f"hpt_{spec.name}_{_digest(identity)}"
    path = Path(dirname or ".") / f"{name}.json"
    if verbose:
        print(f"{paint(name, 'bold')}\n  corpus   {spec.name}   {store.n:,} rows   members {'+'.join(tuner.members)}")
        print(f"  arms     {', '.join(models)}   objectives {'+'.join(tuner.objectives)}   "
              f"{sampler} x {trials} trials per arm")
        for a in models:
            print(f"  study    {names[a]}")
        print()
        tuner.report()

    results, dbs = {}, None
    if studies:
        dbs = {a: StudyStore(names[a], dirname=dirname, spec=spec, sync=sync, sync_every=sync_every, readonly=readonly)
               for a in models if not MODELS[a].placeholder}
        sync_ledgers(dbs.values(), verbose=verbose)          # every arm's ledger: one listing, one parallel fetch
    for a in models:
        ms = MODELS[a]
        if ms.placeholder:
            if verbose:
                print(f"  {a}: placeholder arm, skipped")
            continue
        space = spaces.get(a) or ms.space()
        db = None
        if studies:
            db = dbs[a].pull(verbose=verbose, fetch=False)
            if db.sync and db.stored_trials(f"{names[a]}_final") < trials:
                db.pull(verbose=False)       # about to search: one fresh look, so a parallel session's trials count
        if strict:
            have = db.stored_trials(f"{names[a]}_final") if db is not None else 0
            if have < trials:
                raise CacheMiss(f"{a}: the search {names[a]} holds {have} of {trials} trials, and this run reads the "
                                "search only (strict) — run it with the search refitted, or check the spaces")
        t0 = time.perf_counter()
        study = tuner.study(ms, space, trials, name=f"{names[a]}_final", verbose=verbose, store=db)
        done = completed(study)
        if not done:
            raise RuntimeError(f"{a}: no trial completed")
        best = best_trial(study)
        p = {n: prm.frozen for n, prm in space.items() if prm.frozen is not None}
        p.update(best.params)
        # the search's cost is what its trials took — recorded by each trial, so the same number whether the study
        # ran now or was resumed off the ledger (the wall-clock of a resumed call is the time to load it)
        searched = sum(float(t.user_attrs.get("trial_seconds", 0.0)) for t in study.trials
                       if t.state in (TrialState.COMPLETE, TrialState.PRUNED))
        results[a] = TuningResult(name=a, params=p, value=float(best.values[0]),
                                  gap_closed=float(best.user_attrs.get("gap_closed", float("nan"))),
                                  space=space, study=study, n_trials=len(study.trials),
                                  seconds=searched or time.perf_counter() - t0, trial_number=best.number)
    if save:
        save_tuning(results, path, tuner, extra={"hpt": identity, "study": name, "versions": versions,
                                                 "studies": {a: {"db": names[a], "study": f"{names[a]}_final"}
                                                             for a in results}})
        if studies and sync and not readonly:
            try:                                           # an unchanged record makes no commit
                HitHub(spec=spec).push_file(path, _tuning_json_path(spec, name),
                                            f"{name}: tuned configurations", verbose=False, skip_identical=True)
            except Exception as e:                             # noqa: BLE001
                print(f"  (could not mirror {path.name}: {str(e).splitlines()[0][:120]})")
    return HPTRun(name=name, tuner=tuner, results=results, studies=dbs, path=str(path) if save else None,
                  identity=identity, names=names, versions=versions)


def hpt_status(store, models, trials, spaces=None, name=None, sampler="random",
               objectives=("corpus_wer",), holdout_frac=0.25, inner_val_frac=0.30, inner_folds=1,
               partition_seed=20260901, dirname=None, sync=True, versions=None, ctx=None, verbose=True):
    """What a matching `hpt(...)` call would find already done. Returns a frame; fits nothing."""
    _require_optuna()
    models = tuple(models)
    versions = {k: str(v) for k, v in (versions or {}).items()}
    tuner = Tuner(store, holdout_frac=holdout_frac, inner_val_frac=inner_val_frac,
                  partition_seed=partition_seed, inner_folds=inner_folds, objectives=objectives,
                  sampler=sampler, ctx=ctx)
    spec = store.spec
    identity = hpt_identity(tuner, models, spaces)
    names = {a: (arm_study_name(spec, a, identity, versions.get(a)) if name is None
                 else f"{name}_{a}" + (f"_{versions[a]}" if a in versions else "")) for a in models}
    name = name or f"hpt_{spec.name}_{_digest(identity)}"
    rows = []
    dbs = {a: StudyStore(names[a], dirname=dirname, spec=spec, sync=sync) for a in models}
    sync_ledgers(dbs.values(), verbose=verbose)              # every arm's ledger: one listing, one parallel fetch
    for a in models:
        db = dbs[a]
        db.pull(verbose=False, fetch=False)
        study_name = f"{names[a]}_final"
        study = None
        if db.path.exists() and study_name in [s.study_name for s in
                                               optuna.get_all_study_summaries(db.url(), include_best_trial=False)]:
            study = optuna.load_study(study_name=study_name, storage=db.url())
        done = completed(study) if study is not None else []
        n_all = len(study.trials) if study is not None else 0
        row = {"arm": a, "study": names[a], "trials": n_all, "complete": len(done), "budget": int(trials),
               "best_value": np.nan, "best_params": ""}
        if done:
            best = best_trial(study)
            row["best_value"] = float(best.values[0])
            row["best_params"] = json.dumps(best.params, sort_keys=True, default=str)
        row["status"] = "absent" if n_all == 0 else "complete" if n_all >= trials else "partial"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(name=name, identity=identity, names=names)
    if verbose:
        print(f"{name}  ({sampler} sampler, budget {trials} per arm, partition seed {partition_seed}, "
              f"{store.n:,} rows)")
        for _, r in out.iterrows():
            found = f"best {r['best_value']:.4f}  " + paint(r["best_params"], "dim") if r["complete"] else ""
            print(f"    {r['arm']:28s} {state(f'{r['status']:9s}')} {r['trials']:3d}/{r['budget']:<3d} {found}")
    return out


def load_hpt_json(spec, name, dirname=None, sync=True):
    local = Path(dirname or ".") / f"{name}.json"
    if local.exists():
        return json.loads(local.read_text())
    if not sync:
        return None
    try:
        got = HitHub(spec=spec).pull_file(_tuning_json_path(spec, name), verbose=False)
    except Exception:                                          # noqa: BLE001
        return None
    return json.loads(Path(got).read_text()) if got else None
