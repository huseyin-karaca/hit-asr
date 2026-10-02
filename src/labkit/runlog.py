"""A content-addressed cache for expensive notebook steps, mirrored to a Hub records repo.

Every result is stored under a name hashed from what it depends on (`RunLog.digest`), so a re-run with the same
inputs reads instead of recomputing, and a changed input computes a new file beside the old one.
"""

__all__ = ['RunLog', 'CacheMiss']

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from labkit.hub import Fetched, HubRateLimited, HubUnavailable
from labkit.pretty import state


class CacheMiss(LookupError):
    """A strict run needed a result the cache does not hold."""


class RunLog:
    """A content-addressed cache and timing log for expensive notebook steps.

    Parameters
    ----------
    name : the run's stem. Files land in `<dirname>/<name>/` and, when synced,
        under `<root>/<name>/` on the Hub.
    hub : a zero-argument callable returning the `labkit.hub.Hub` to mirror to.
    root : the folder of the Hub the runs live under.
    sync : read from the Hub. `False` keeps everything local.
    readonly : read the Hub's results but keep new ones local (with `sync`).
    refresh : ignore every hit and recompute.
    strict : never compute — a result the cache lacks raises `CacheMiss`.
    """

    REFERENCES = "references"
    PUSH_HOLD_S = 600            # after a failed push, queue for this long before trying again (mirror() ignores it)

    def __init__(self, name, dirname=None, hub=None, root="results/runlog", sync=True, refresh=False, verbose=True,
                 readonly=False, strict=False):
        self.name = name
        self.dir = Path(dirname or ".") / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hub, self.root = hub, root.rstrip("/")
        self.sync, self.refresh, self.verbose = bool(sync), bool(refresh), bool(verbose)
        self.readonly, self.strict = bool(readonly), bool(strict)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self._pending = {}
        self._batching = 0
        self._hold_until = 0.0

    def __repr__(self):
        hits = sum(1 for e in self.manifest.values() if e.get("kind") == "frames")
        return (f"RunLog({self.name!r}, {self.dir}, {len(self.manifest)} entr(ies), "
                f"{hits} cached step(s), {'sync' if self.sync else 'local only'})")

    # ------------------------------------------------------------- plumbing --

    def hub(self):
        if self._hub is None:
            raise ValueError("RunLog: no hub (pass `hub=` or subclass `hub()`)")
        return self._hub()

    def _repo_path(self, filename):
        return f"{self.root}/{self.name}/{filename}"

    @staticmethod
    def digest(key):
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str))

    def pull(self, names, verbose=False):
        """Bring the local copies of `names` (files of this run) in line with the Hub. Returns a `Fetched`.

        One listing of the run's folder, one parallel fetch of the listed names
        that are missing or differ here. A name the Hub does not list comes
        back in `absent` (it will be computed); one it lists but will not serve
        raises `HubUnavailable` — nothing is computed over a read error. The
        Hub's manifest is folded into the local one on the way.
        """
        if not self.sync or not names:
            return Fetched(wanted=len(names), absent=[self._repo_path(n) for n in names])
        hub = self.hub()
        listing = hub.listing(self._repo_path(""))
        got = hub.fetch({self._repo_path(n): self.dir / n for n in names}, listing, what=f"runlog {self.name}",
                        verbose=self.verbose)
        self._merge_manifest(hub, listing)
        if verbose and got.fetched:
            print(got.line(f"runlog {self.name}"))
        return got

    def _merge_manifest(self, hub, listing):
        """Fold the Hub's manifest into the local one, so this session's next push does not drop another's entries."""
        repo = self._repo_path("manifest.json")
        if repo not in listing:
            return
        theirs = self.dir / "manifest.hub.json"
        hub.fetch({repo: theirs}, listing, verbose=False)
        try:
            merged = {**json.loads(theirs.read_text()), **self.manifest}
        except (OSError, ValueError):
            return
        if merged != self.manifest:
            self.manifest = merged
            self._write_manifest()

    def _push(self, path, message):
        if not self.sync or self.readonly or not path.exists():
            return
        self._pending[self._repo_path(path.name)] = (path, message)
        if not self._batching:
            self.flush()

    def unpushed(self):
        """How many artefacts are queued for the Hub and not yet on it."""
        return len(self._pending)

    def flush(self, message=None, force=False):
        """Send every queued artefact as ONE commit.

        A failed push (the Hub down past its retries, or the hourly commit
        quota) keeps everything queued and local, and further pushes wait
        `PUSH_HOLD_S` so a long run is not slowed by one retry cycle per seed;
        `force` (what `mirror()` does) tries regardless.
        """
        if not self._pending or (not force and time.monotonic() < self._hold_until):
            return
        pending, self._pending = self._pending, {}
        try:
            self.hub().push_files({repo: str(p) for repo, (p, _) in pending.items()},
                                  message or f"{self.name}: {len(pending)} artefact(s)", verbose=False)
            self._hold_until = 0.0
        except Exception as e:                                 # noqa: BLE001
            self._pending = {**pending, **self._pending}
            self._hold_until = time.monotonic() + self.PUSH_HOLD_S
            if self.verbose:
                why = "the repo's hourly commit quota, 429" if isinstance(e, HubRateLimited) else \
                    (str(e).splitlines() or [type(e).__name__])[0][:160]
                print(f"  runlog: Hub write failed ({why}); {len(self._pending)} file(s) kept locally and queued — "
                      "a later push or `mirror()` sends them")

    def mirror(self, message=None):
        """Push every local artefact of this run the Hub lacks, and anything still queued, in one commit.

        Returns the names sent; an empty list when there was nothing to send
        or the Hub would not take it (then they stay queued — `unpushed()`).
        """
        if not self.sync or self.readonly:
            return []
        try:
            listing = self.hub().listing(self._repo_path(""))
        except HubUnavailable as e:
            if self.verbose:
                print(f"  runlog: cannot list the Hub ({str(e).splitlines()[0][:120]}); "
                      f"{self.unpushed()} queued artefact(s) stay local")
            return []
        local = [p for p in sorted(self.dir.iterdir()) if p.is_file() and not p.name.startswith(".")
                 and p.name not in ("manifest.json", "manifest.hub.json")]
        missing = [p for p in local if self._repo_path(p.name) not in listing]
        for p in missing:
            self._pending[self._repo_path(p.name)] = (p, f"{self.name}: {p.name}")
        if not self._pending:
            return []
        if self.manifest_path.exists():
            self._pending[self._repo_path("manifest.json")] = (self.manifest_path, f"{self.name}: manifest")
        if self.verbose and missing:
            print(f"  runlog: mirroring {len(missing)} local artefact(s) the Hub lacks")
        names = [Path(r).name for r in self._pending]
        self.flush(message or f"{self.name}: mirror {len(names)} artefact(s)", force=True)
        return [] if self._pending else names

    class _Batch:
        def __init__(self, log):
            self.log = log

        def __enter__(self):
            self.log._batching += 1

        def __exit__(self, *exc):
            self.log._batching -= 1
            if not self.log._batching:
                self.log.flush()
            return False

    def batch(self):
        return RunLog._Batch(self)

    def step(self, tag, **key):
        """Time a block and record it. Use as a context manager."""
        return _Step(self, tag, key)

    def record(self, entry, seconds, kind="step", **extra):
        self.manifest[entry] = {"kind": kind, "seconds": float(seconds),
                                "at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        self._write_manifest()
        return self.manifest[entry]

    # ---------------------------------------------------------------- frames --

    @staticmethod
    def expand_summary(summary, scores=None):
        idx = np.repeat(summary.index.values, summary["n"].values.astype(int))
        s = summary.loc[idx]
        out = pd.DataFrame({"model": s["model"].values, "repeat": s["repeat"].values,
                            "fold": s["fold"].values, "row": -1, "errors": np.nan,
                            "choice": s["choice"].values})
        if scores is not None:
            stamp = [c for c in ("members", "repeat_seed") if c in scores.columns]
            if stamp:
                per_fold = scores.drop_duplicates(["model", "repeat", "fold"])
                out = out.merge(per_fold[["model", "repeat", "fold", *stamp]],
                                on=["model", "repeat", "fold"], how="left")
        return out

    def _read_picks(self, picks_p, summary_p, scores):
        if picks_p.exists():
            return pd.read_parquet(picks_p)
        if summary_p.exists():
            return self.expand_summary(pd.read_parquet(summary_p), scores)
        return None

    def _read(self, stem, h):
        """`(scores, picks)` from the local copies, or None."""
        scores_p = self.dir / f"{stem}_{h}_scores.parquet"
        if not scores_p.exists():
            return None
        scores = pd.read_parquet(scores_p)
        picks = self._read_picks(self.dir / f"{stem}_{h}_picks.parquet",
                                 self.dir / f"{stem}_{h}_picks_summary.parquet", scores)
        return None if picks is None else (scores, picks)

    @staticmethod
    def _files(stem, h):
        return [f"{stem}_{h}_{kind}.parquet" for kind in ("scores", "picks", "picks_summary")]

    @staticmethod
    def _arm_seconds(scores):
        cols = [c for c in ("fit_seconds", "predict_seconds") if c in scores.columns]
        if not cols:
            return float("nan")
        return float(scores.drop_duplicates(["model", "repeat", "fold"])[cols].sum().sum())

    def frames(self, tag, fn, key=None, picks_summary=True, refresh=None, push_picks=True):
        """Cached `(fold_scores, fold_picks)`. Runs `fn()` only on a miss.

        The local copy first; on a local miss the Hub is asked once (one
        listing, a fetch of this tag's files) right before computing.
        """
        refresh = self.refresh if refresh is None else refresh
        key = dict(key or {})
        h = self.digest(key)
        entry = f"{tag}[{h}]"
        got = None
        if not refresh:
            got = self._read(tag, h)
            if got is None and self.sync:
                self.pull(self._files(tag, h), verbose=self.verbose)
                got = self._read(tag, h)
        if got is not None:
            scores, picks = got
            if self.verbose:
                saved = self.manifest.get(entry, {}).get("seconds")
                print(f"  runlog {state('HIT')}  {tag} [{h}] — {len(scores):,} score rows, nothing refitted"
                      + (f" (saving ~{saved / 60:.1f} min)" if saved else ""))
            return scores, picks
        if self.strict:
            raise CacheMiss(f"{self.name}: {tag} [{h}] is not in the cache, and this run reads cached results only (strict)")
        if self.verbose:
            print(f"  runlog {state('MISS')} {tag} [{h}] — computing")
        t0 = time.perf_counter()
        scores, picks = fn()
        secs = time.perf_counter() - t0
        self._write(entry, tag, h, key, scores, picks, secs, picks_summary=picks_summary, push_picks=push_picks)
        if self.verbose:
            print(f"  runlog {state('WROTE')} {tag} [{h}] — {secs / 60:.1f} min, {len(scores):,} score rows")
        return scores, picks

    def _write(self, entry, tag, h, key, scores, picks, secs, picks_summary=True, push_picks=True, **extra):
        scores_p = self.dir / f"{tag}_{h}_scores.parquet"
        picks_p = self.dir / f"{tag}_{h}_picks.parquet"
        summary_p = self.dir / f"{tag}_{h}_picks_summary.parquet"
        scores.to_parquet(scores_p, index=False)
        picks.to_parquet(picks_p, index=False)
        self._push(scores_p, f"{self.name}: {tag} scores")
        if push_picks:
            self._push(picks_p, f"{self.name}: {tag} picks")
        files = [scores_p.name, picks_p.name]
        if picks_summary:
            summary = picks.groupby(["model", "repeat", "fold", "choice"]).size().rename("n").reset_index()
            summary.to_parquet(summary_p, index=False)
            self._push(summary_p, f"{self.name}: {tag} pick summary")
            files.append(summary_p.name)
        self.record(entry, secs, kind="frames", digest=h, key=key, tag=tag,
                    n_score_rows=int(len(scores)), n_pick_rows=int(len(picks)), files=files, **extra)
        self._push(self.manifest_path, f"{self.name}: manifest")

    def per_seed(self, tag, seeds, fn, key=None, refresh=None):
        """Cached one repetition at a time; `fn(seed)` runs ONE repetition. Returns concatenated frames."""
        refresh = self.refresh if refresh is None else refresh
        scores, picks = [], []
        if not refresh and self.sync:
            self.pull([f for seed in seeds for f in self._files(f"{tag}_seed{int(seed)}",
                                                                 self.digest({**(key or {}), "seed": int(seed)}))],
                      verbose=self.verbose)
        for i, seed in enumerate(seeds):
            with self.batch():
                s, p = self.frames(f"{tag}_seed{int(seed)}", lambda: fn(int(seed)),
                                   key={**(key or {}), "seed": int(seed)}, refresh=refresh)
            if s["repeat"].nunique() != 1:
                raise ValueError(f"fn({seed}) returned {s['repeat'].nunique()} repetitions")
            scores.append(s.assign(repeat=i))
            picks.append(p.assign(repeat=i))
        self.mirror()
        return pd.concat(scores, ignore_index=True), pd.concat(picks, ignore_index=True)

    def per_arm(self, tag, seeds, estimators, fn, key=None, arm_keys=None, refresh=None):
        """Cached per **arm** and seed. `fn(seed, estimators)` runs ONE repetition of the arms it is handed.

        `arm_keys[arm]` is what that arm's scores depend on given the folds
        (its selected configuration and version); `key` is everything shared.
        The parameter-free reference arms form one more group, `references`.
        """
        refresh = self.refresh if refresh is None else refresh
        key, arm_keys = dict(key or {}), dict(arm_keys or {})
        estimators = dict(estimators)
        missing = [a for a in estimators if a not in arm_keys]
        if missing:
            raise ValueError(f"no arm key for {missing}; pass `arm_keys=` for every estimator")
        seeds = [int(s) for s in seeds]
        groups = {self.REFERENCES: {"arm": self.REFERENCES}, **{a: arm_keys[a] for a in estimators}}

        def stem_digest(seed, g):
            return f"{tag}_seed{seed}_{g}", self.digest({**key, **groups[g], "seed": seed})

        if not refresh and self.sync:
            # every file the seeds and arm keys name, in ONE listing and ONE parallel fetch
            got = self.pull([f for seed in seeds for g in groups for f in self._files(*stem_digest(seed, g))])
            if self.verbose:
                cached = sum(1 for seed in seeds for g in groups
                             if (self.dir / self._files(*stem_digest(seed, g))[0]).exists())
                took = f"; fetched {got.fetched} file(s) in {got.seconds:.0f}s" if got.fetched else ""
                print(f"  hub: runlog {self.name} — {cached} of {len(seeds) * len(groups)} (seed, group) "
                      f"result(s) available{took}")
        out_scores, out_picks = [], []
        for i, seed in enumerate(seeds):
            have, need = {}, []
            with self.batch():
                for g in groups:
                    got = None if refresh else self._read(*stem_digest(seed, g))
                    if got is None:
                        need.append(g)
                    else:
                        have[g] = got
                if need and not refresh and self.sync:
                    # one fresh look before computing: a parallel session may have pushed some of them since
                    self.pull([f for g in need for f in self._files(*stem_digest(seed, g))], verbose=self.verbose)
                    for g in list(need):
                        got = self._read(*stem_digest(seed, g))
                        if got is not None:
                            have[g] = got
                            need.remove(g)
                if self.verbose:
                    n = len(groups)
                    word = state("HIT ") if not need else state("MISS")
                    print(f"  runlog {word} {tag} seed {seed} — {n - len(need)}/{n} group(s) cached"
                          + (f"; computing {', '.join(need)}" if need else ", nothing refitted"))
                if need and self.strict:
                    raise CacheMiss(f"{self.name}: {tag} seed {seed} — {', '.join(need)} not in the cache, and this run "
                                    "reads cached results only (strict)")
                if need:
                    fit = {a: estimators[a] for a in need if a in estimators}
                    t0 = time.perf_counter()
                    scores, picks = fn(seed, fit)
                    secs = time.perf_counter() - t0
                    if scores["repeat"].nunique() != 1:
                        raise ValueError(f"fn({seed}) returned {scores['repeat'].nunique()} repetitions")
                    for g in need:
                        s, p = self._split(scores, picks, g, estimators)
                        if s.empty:
                            raise ValueError(f"fn({seed}) returned no rows for {g!r}")
                        gkey = {**key, **groups[g], "seed": seed}
                        h = self.digest(gkey)
                        stem = f"{tag}_seed{seed}_{g}"
                        self._write(f"{stem}[{h}]", stem, h, gkey, s, p, self._arm_seconds(s),
                                    picks_summary=False, group=g, fitted_with=list(fit))
                        have[g] = (s, p)
                    if self.verbose:
                        print(f"  runlog {state('WROTE')} {tag} seed {seed} — {secs / 60:.1f} min for {', '.join(need)}")
            order = [self.REFERENCES] + list(estimators)
            out_scores += [have[g][0].assign(repeat=i) for g in order]
            out_picks += [have[g][1].assign(repeat=i) for g in order]
        self.mirror()
        return pd.concat(out_scores, ignore_index=True), pd.concat(out_picks, ignore_index=True)

    def _split(self, scores, picks, group, estimators):
        if group == self.REFERENCES:
            fitted = set(estimators)
            m_s, m_p = ~scores["model"].isin(fitted), ~picks["model"].isin(fitted)
        else:
            m_s, m_p = scores["model"] == group, picks["model"] == group
        return scores[m_s].reset_index(drop=True), picks[m_p].reset_index(drop=True)

    def timings(self):
        """Every recorded step, longest first."""
        rows = [{"step": v.get("tag", k), "entry": k, "kind": v.get("kind", "step"),
                 "seconds": v.get("seconds", np.nan), "minutes": v.get("seconds", np.nan) / 60.0,
                 "digest": v.get("digest", ""), "at": v.get("at", "")} for k, v in self.manifest.items()]
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values("seconds", ascending=False, ignore_index=True)
        total = out["seconds"].sum()
        out["share"] = out["seconds"] / total if total else np.nan
        return out


class _Step:
    def __init__(self, log, tag, key):
        self.log, self.tag, self.key = log, tag, key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        secs = time.perf_counter() - self.t0
        self.log.record(self.tag, secs, kind="step", failed=exc_type is not None, **self.key)
        if self.log.verbose:
            print(f"  runlog step {self.tag}: {secs:.1f}s" + (" (raised)" if exc_type is not None else ""))
        return False