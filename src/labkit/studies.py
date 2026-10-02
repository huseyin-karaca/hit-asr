"""Optuna studies that survive a Colab kernel and run in several sessions at once: SQLite on disk, one
trial-ledger file per writer on the Hub."""

__all__ = ['LEDGER_MIN_PUSH_S', 'WRITER_ID', 'completed', 'best_trial', 'ledger_writer', 'ledger_frame', 'StudyStore',
           'sync_ledgers']

import json
import os
import secrets
import socket
import time
import warnings
from pathlib import Path

import pandas as pd

try:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
except ImportError:                                            # pragma: no cover
    optuna = TrialState = None


def _sign(direction):
    return 1.0 if direction == "minimize" else -1.0


def completed(study):
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def best_trial(study):
    done = completed(study)
    s = _sign(study.directions[0].name.lower() if hasattr(study.directions[0], "name") else "minimize")
    return min(done, key=lambda t: s * t.values[0])


LEDGER_MIN_PUSH_S = 600                   # at most one ledger commit per study per 10 minutes per process
WRITER_ID = f"{socket.gethostname()[:12]}-{os.getpid()}-{secrets.token_hex(3)}"
_LEDGER_COLUMNS = ("ledger_key", "writer", "number", "state", "values", "params", "distributions",
                   "user_attrs", "system_attrs", "intermediate_values", "datetime_start", "datetime_complete")


def _frozen_to_row(t, writer):
    """One ledger row for a finished trial. Imported trials keep their original key."""
    from optuna.distributions import distribution_to_json
    key = t.system_attrs.get("ledger_key") or f"{writer}:{t.number}"
    return {"ledger_key": key, "writer": key.split(":")[0], "number": int(t.number), "state": t.state.name,
            "values": json.dumps(list(t.values) if t.values is not None else None),
            "params": json.dumps(t.params, default=str),
            "distributions": json.dumps({k: distribution_to_json(d) for k, d in t.distributions.items()}),
            "user_attrs": json.dumps(t.user_attrs, default=str),
            "system_attrs": json.dumps({k: v for k, v in t.system_attrs.items() if k != "ledger_key"}, default=str),
            "intermediate_values": json.dumps({int(k): v for k, v in t.intermediate_values.items()}),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None}


def _row_to_frozen(r):
    from optuna.distributions import json_to_distribution
    values = json.loads(r["values"])
    return optuna.trial.create_trial(
        state=TrialState[r["state"]],
        values=values if r["state"] == "COMPLETE" else None,
        params=json.loads(r["params"]),
        distributions={k: json_to_distribution(d) for k, d in json.loads(r["distributions"]).items()},
        user_attrs=json.loads(r["user_attrs"]),
        system_attrs={**json.loads(r["system_attrs"]), "ledger_key": r["ledger_key"]},
        intermediate_values={int(k): v for k, v in json.loads(r["intermediate_values"]).items()},
    )


def ledger_writer(study):
    """The writer id this study's own trials are keyed under — stored in the study the first time it is
    written out, so a later process on the same disk (a Colab kernel restart) derives the same keys
    instead of taking the trials it finds for new ones and adding them a second time."""
    w = study.user_attrs.get("ledger_writer")
    if w is None:
        w = WRITER_ID
        study.set_user_attr("ledger_writer", w)
    return w


def ledger_frame(study, writer=None):
    """Every finished trial of `study` as ledger rows (a DataFrame). Own trials are keyed as `writer` (the study's)."""
    writer = writer or ledger_writer(study)
    rows = [_frozen_to_row(t, writer) for t in study.trials
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
    return pd.DataFrame(rows, columns=list(_LEDGER_COLUMNS))


class StudyStore:
    """A resumable Optuna storage: SQLite on disk, a trial ledger on the Hub.

    Every session writes its own ledger file (`studies/ledger/<group>/<name>/<writer>.parquet`), so several
    sessions can run one study at once; `pull` rebuilds the local SQLite from the union of every writer's trials.

    Parameters
    ----------
    hub : a zero-argument callable returning the `labkit.hub.Hub` the ledger lives in.
    group : the ledger's folder under `studies/ledger/` (the corpus, for this project).
    sync : read the ledger from the Hub. `readonly` reads it but never pushes this session's trials.
    """

    def __init__(self, name, dirname=None, hub=None, group="shared", sync=True, sync_every=5, readonly=False):
        self.name = name
        self.path = Path(dirname or ".") / f"{name}.db"
        self.ledger_dir = Path(dirname or ".") / "ledger" / name
        self.path_in_repo = f"studies/{name}.db"               # the legacy mirror, read only
        self._hub, self.group, self.sync, self.readonly = hub, group, bool(sync), bool(readonly)
        self.sync_every = int(sync_every)
        self._pushed_at, self._pushed_time = 0, 0.0
        self._imported = None

    def __repr__(self):
        n = f"{self.path.stat().st_size / 1e6:.1f} MB" if self.path.exists() else "absent"
        return f"StudyStore({self.name!r}, {self.path} [{n}], {'sync' if self.sync else 'local only'})"

    def hub(self):
        if self._hub is None:
            raise ValueError("StudyStore: no hub (pass `hub=` or subclass `hub()`)")
        return self._hub()

    def url(self):
        return f"sqlite:///{self.path}"

    def ledger_prefix(self):
        return f"studies/ledger/{self.group}/{self.name}/"

    def trial_count(self, path=None):
        path = Path(path or self.path)
        if not path.exists():
            return 0
        try:
            url = f"sqlite:///{path}"
            return sum(len(optuna.load_study(study_name=s.study_name, storage=url).trials)
                       for s in optuna.get_all_study_summaries(url, include_best_trial=False))
        except Exception:                                      # noqa: BLE001
            return 0

    # ------------------------------------------------------------- ledger --

    def _local_keys(self):
        """`{study_name: {ledger_key}}` of every trial the local SQLite holds (own trials keyed as the study's writer)."""
        keys = {}
        if not self.path.exists():
            return keys
        for smry in optuna.get_all_study_summaries(self.url(), include_best_trial=False):
            st = optuna.load_study(study_name=smry.study_name, storage=self.url())
            w = ledger_writer(st)
            keys[smry.study_name] = {t.system_attrs.get("ledger_key") or f"{w}:{t.number}" for t in st.trials}
        return keys

    def ledger_targets(self, listing):
        """`{repo_path: local_path}` of this study's ledger files in a Hub `listing`."""
        prefix = self.ledger_prefix()
        return {f: self.ledger_dir / Path(f).name for f in listing if f.startswith(prefix) and f.endswith(".parquet")}

    def _read_ledger(self, verbose=True, fetch=True):
        """Fetch this study's ledger files that are new or changed on the Hub, then read every local one."""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        if self.sync and fetch:
            hub = self.hub()
            listing = hub.listing(self.ledger_prefix())
            hub.fetch(self.ledger_targets(listing), listing, what=f"ledger {self.name}", verbose=verbose)
        frames = []
        for f in sorted(self.ledger_dir.glob("*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception as e:                             # noqa: BLE001
                if verbose:
                    print(f"  ledger: unreadable {f.name} ({type(e).__name__}); skipped")
        if not frames:
            return pd.DataFrame(columns=("study", *_LEDGER_COLUMNS, "direction"))
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates("ledger_key", keep="last")

    def _rebuild(self, ledger, verbose=True):
        """Add to the local SQLite every ledger trial it lacks. Returns the number added."""
        if ledger.empty:
            return 0
        have = self._local_keys()
        added = 0
        for study_name, part in ledger.groupby("study"):
            missing = part[~part["ledger_key"].isin(have.get(study_name, set()))]
            if missing.empty:
                continue
            directions = json.loads(missing.iloc[0]["direction"])
            kw = dict(study_name=study_name, storage=self.url(), load_if_exists=True)
            st = (optuna.create_study(directions=directions, **kw) if len(directions) > 1
                  else optuna.create_study(direction=directions[0], **kw))
            st.add_trials([_row_to_frozen(r) for _, r in missing.iterrows()])
            added += len(missing)
        if verbose and added:
            print(f"  ledger: {added} trial(s) from other sessions added to {self.path.name}")
        return added

    def pull(self, verbose=True, fetch=True):
        """Reconcile with the Hub: the union of every session's trials. Returns self.

        `fetch=False` rebuilds from the ledger files already on disk — right
        after a `sync_ledgers` that fetched them for many studies at once.
        """
        if not self.sync:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ledger = self._read_ledger(verbose=verbose, fetch=fetch)
        if ledger.empty and not self.path.exists():
            # nothing on the ledger yet: a legacy SQLite mirror, if any, seeds the local copy
            got = self.hub().pull_file(self.path_in_repo, self.path, verbose=False)
            if verbose:
                print(f"  {'using the legacy Hub copy: ' + str(self.trial_count()) + ' trials' if got else 'starting a new database at ' + str(self.path)}")
            return self
        n = self._rebuild(ledger, verbose=verbose)
        if verbose:
            print(f"  {self.name}: {self.trial_count()} trial(s) locally" + (f" ({n} merged from the ledger)" if n else ""))
        return self

    def write_ledger(self):
        """The ledger files this disk owns: every finished trial the local SQLite holds that no other
        writer owns, one file per study writer (a restarted kernel keeps appending to the file its
        studies were born under). Returns the list of files, empty when there is nothing to write."""
        if not self.path.exists():
            return []
        rows = []
        for smry in optuna.get_all_study_summaries(self.url(), include_best_trial=False):
            st = optuna.load_study(study_name=smry.study_name, storage=self.url())
            df = ledger_frame(st)
            if df.empty:
                continue
            df = df[df["writer"] == ledger_writer(st)]
            if df.empty:
                continue
            df.insert(0, "study", smry.study_name)
            df["direction"] = json.dumps([d.name.lower() for d in st.directions])
            rows.append(df)
        if not rows:
            return []
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        for writer, part in pd.concat(rows, ignore_index=True).groupby("writer"):
            out = self.ledger_dir / f"{writer}.parquet"
            part.to_parquet(out, index=False)
            outs.append(out)
        return outs

    def push(self, message=None, verbose=True, force=True):
        """Write this writer's ledger and upload it (one commit). `force=False` respects the throttle."""
        if not self.sync or self.readonly or not self.path.exists():
            return None
        if not force and time.time() - self._pushed_time < LEDGER_MIN_PUSH_S:
            return None
        outs = self.write_ledger()
        if not outs:
            return None
        try:
            hub = self.hub()
            if len(outs) == 1:
                path = hub.push_file(outs[0], self.ledger_prefix() + outs[0].name,
                                     message=message or f"{self.name}: ledger", verbose=verbose)
            else:
                path = hub.push_files({self.ledger_prefix() + o.name: o for o in outs},
                                      message=message or f"{self.name}: ledger", verbose=verbose)
            self._pushed_time = time.time()
            return path
        except Exception as e:                                 # noqa: BLE001
            print(f"  (could not push the ledger of {self.name}: {str(e).splitlines()[0][:160]} — local copy kept)")
            return None

    def callback(self):
        def cb(study, trial):
            if not self.sync or self.readonly:
                return
            n = len(study.trials)
            if n - self._pushed_at >= self.sync_every:
                self._pushed_at = n
                self.push(message=f"{self.name}: {n} trials", verbose=False, force=False)
        return cb

    def stored_trials(self, study_name):
        """How many trials (any state but waiting) the local SQLite holds for `study_name`."""
        if not self.path.exists():
            return 0
        if study_name not in [s.study_name for s in optuna.get_all_study_summaries(self.url(), include_best_trial=False)]:
            return 0
        return sum(1 for t in optuna.load_study(study_name=study_name, storage=self.url()).trials
                   if t.state != TrialState.WAITING)

    def studies(self):
        if not self.path.exists():
            return pd.DataFrame(columns=["study", "trials", "complete", "pruned"])
        rows = []
        for s in optuna.get_all_study_summaries(self.url(), include_best_trial=False):
            trials = optuna.load_study(study_name=s.study_name, storage=self.url()).trials
            rows.append({"study": s.study_name, "trials": len(trials),
                         "complete": sum(t.state == TrialState.COMPLETE for t in trials),
                         "pruned": sum(t.state == TrialState.PRUNED for t in trials)})
        return pd.DataFrame(rows)


def sync_ledgers(stores, verbose=True):
    """Fetch the ledger files of many `StudyStore`s — one listing per corpus, one parallel fetch. Returns the count fetched.

    After this, `store.pull(fetch=False)` rebuilds each study from disk
    without another request.
    """
    stores = [s for s in stores if s.sync]
    fetched = 0
    by_corpus = {}
    for s in stores:
        by_corpus.setdefault(s.ledger_prefix().rsplit("/", 2)[0] + "/", []).append(s)
    for prefix, group in by_corpus.items():
        hub = group[0].hub()
        listing = hub.listing(prefix)
        targets = {}
        for s in group:
            s.ledger_dir.mkdir(parents=True, exist_ok=True)
            targets.update(s.ledger_targets(listing))
        got = hub.fetch(targets, listing, what=f"ledgers of {len(group)} stud(ies)", verbose=verbose)
        fetched += got.fetched
        if verbose:
            took = f", fetched {got.fetched} in {got.seconds:.0f}s" if got.fetched else ""
            print(f"  hub: trial ledgers of {len(group)} stud(ies) — {len(targets)} file(s) on the Hub, "
                  f"{got.current} already local{took}")
    return fetched
