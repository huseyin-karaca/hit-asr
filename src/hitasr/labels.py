__all__ = ['ERROR_PARTS', 'LABEL_KEEP', 'default_labels_dir', 'LabelStore']

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hitasr.core import active_dataset
from hitasr.hub import HitHub

ERROR_PARTS = ("sub", "dele", "ins")
LABEL_KEEP = ("id", "sub", "dele", "ins", "nref", "wer", "wer_raw", "n_frames",
              "transcription", "transcription_norm", "text_norm")


def default_labels_dir():
    root = Path(os.environ.get("HITASR_CACHE", Path.home() / ".cache" / "hitasr"))
    return root / "labels"


class LabelStore:
    """Every expert's per-utterance labels on one corpus, aligned, as one error matrix.

    Parameters
    ----------
    spec : the corpus. Defaults to the active dataset.
    experts : which experts to load. `None` loads every expert that has labels
        in either repo. Names are the extractors' `name`s.
    adopt : read fastt's `model_<expert>` configs for experts this project has
        no labels of its own for. On by default.
    cache_dir : local parquet mirror, keyed on the content ids of the config's
        own parquet files — so a commit elsewhere in the repo (another session's
        results, a trial ledger) does not invalidate it; re-pushed labels do.
    """

    def __init__(self, spec=None, experts=None, adopt=True, cache_dir=None):
        self.spec = active_dataset(spec)
        self.requested = None if experts is None else tuple(experts)
        self.adopt = bool(adopt)
        self.cache_dir = Path(cache_dir or default_labels_dir()) / self.spec.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._reset()

    def _reset(self):
        self.experts, self.source = [], {}
        self.ids, self.split, self.nref, self.ref_text = None, None, None, None
        self.err, self.err_raw, self.n_frames, self.texts, self.texts_norm = {}, {}, {}, {}, {}
        self.frame_rate_hz = {}
        self._pooled, self.n_nonfinite = {}, {}
        self._frames = {}

    def __repr__(self):
        if self.ids is None:
            return f"LabelStore({self.spec.name}, unloaded)"
        return (f"LabelStore({self.spec.name}, {len(self.ids):,} utterances, "
                f"{len(self.experts)} experts: {', '.join(self.experts)})")

    # ------------------------------------------------------------ sources --

    def available(self):
        """`{expert: ("hitasr" | "fastt")}` — every expert with labels on this corpus.

        hit-asr's own labels win; fastt's `model_<expert>` configs fill in
        behind them. `<prefix>model_` is an exact prefix match, so an
        unprefixed `model_x` never picks up `ami_sdm_model_x`, and `ami_sdm_`
        never picks up `ami_sdm_full_`. fastt also keeps non-transcript
        `model_<name>` configs (e.g. `model_mfcc_pooled`, pooled acoustic
        features with no reference word counts) under the same prefix, so a
        config is only adopted once it is confirmed to carry `nref`.

        That confirmation reads a parquet shard per config, so with `experts`
        named only the ones hit-asr lacks are looked up in fastt — none, for
        the pinned trios; unasked, it was 1.3 GB of shards on AMI.
        """
        hub = HitHub(spec=self.spec)
        out = {e: "hitasr" for e in hub.expert_names("labels")}
        need = None if self.requested is None else {e for e in self.requested if e not in out}
        if self.adopt and need != set():
            prefix = f"{self.spec.source_prefix}model_"
            src = hub.source()
            for c in src.configs():
                name = c[len(prefix):]
                if not c.startswith(prefix) or name in out or (need is not None and name not in need):
                    continue
                splits = src.split_names(c)
                if splits and "nref" in src.column_names(c, splits[0]):
                    out[name] = "fastt"
        return out

    def _read_one(self, expert, where):
        """One expert's frame: `LABEL_KEEP` + `pool_mean`, every split concatenated, cached."""
        hub = HitHub(spec=self.spec)
        if where == "hitasr":
            cfg, repo = self.spec.labels_config(expert), hub
        else:
            cfg, repo = self.spec.source_model_config(expert), hub.source()
        shards = sorted((p, oid) for p, (oid, _) in repo.listing(f"{cfg}/").items() if p.endswith(".parquet"))
        if not shards:
            raise KeyError(f"no parquet under {cfg}/ in {repo.repo_id}")
        rev = hashlib.sha1(json.dumps(shards).encode()).hexdigest()[:8]
        cached = self.cache_dir / f"{expert}__{where}__{rev}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)

        frames = []
        for split in self.spec.splits:
            cols = repo.column_names(cfg, split)
            pooled = "pool_mean" if "pool_mean" in cols else next(
                (c for c in cols if c.startswith("pool_l") and c.endswith("_mean")), None)
            want = [c for c in LABEL_KEEP if c in cols] + ([pooled] if pooled else [])
            t = repo.read_columns(cfg, split, want).to_pandas()
            if pooled and pooled != "pool_mean":
                t = t.rename(columns={pooled: "pool_mean"})
            t.insert(0, "split", split)
            frames.append(t)
        df = pd.concat(frames, ignore_index=True)
        if "pool_mean" in df.columns:
            df["pool_mean"] = [np.asarray(v, dtype=np.float32) for v in df["pool_mean"]]
        df.to_parquet(cached, index=False)
        return df

    def _frame_rate(self, expert, where):
        """The encoder frame rate, from the extractor's results JSON (either repo); None when there is none.

        A Hub that does not answer raises (`HubUnavailable`) rather than
        leaving the durations of Table 1 silently empty.
        """
        hub = HitHub(spec=self.spec)
        try:
            if where == "hitasr":
                return hub.load_results(f"extract_{expert}")["manifest"].get("frame_rate_hz")
            path = (f"results/{expert}.json" if self.spec.source_prefix == ""
                    else f"results/{self.spec.name}/{expert}.json")
            got = hub.source().pull_file(path, verbose=False)
            return json.loads(Path(got).read_text()).get("encoder", {}).get("frame_rate_hz") if got else None
        except (FileNotFoundError, KeyError, TypeError, ValueError, AttributeError):
            return None

    # -------------------------------------------------------------- load --

    def load(self, verbose=True):
        """Read every requested expert, align on `(split, id)`, build the matrices. Returns self."""
        t0 = time.perf_counter()
        self._reset()
        avail = self.available()
        names = list(avail) if self.requested is None else list(self.requested)
        missing = [e for e in names if e not in avail]
        if missing:
            raise KeyError(f"no labels for {missing} on {self.spec.name}; have {sorted(avail)}")

        key = None
        for e in names:
            df = self._read_one(e, avail[e])
            k = list(zip(df["split"], df["id"]))
            if key is None:
                key = k
                self.ids, self.split = key, df["split"].to_numpy()
                self.nref = df["nref"].to_numpy().astype(np.int64)
                self.ref_text = df["text_norm"].tolist() if "text_norm" in df else None
            elif k != key:
                pos = {kk: i for i, kk in enumerate(k)}
                order = [pos.get(kk) for kk in key]
                if any(o is None for o in order):
                    raise KeyError(f"{e}: {sum(o is None for o in order)} row(s) of the corpus "
                                   "have no label")
                df = df.iloc[order].reset_index(drop=True)
                if len(df) != len(key):
                    raise ValueError(f"{e}: {len(df)} rows, expected {len(key)}")
            if not np.array_equal(df["nref"].to_numpy(), self.nref):
                raise ValueError(f"{e}: reference word counts differ from {names[0]}'s — "
                                 "the labels were not scored on the same references")
            self.experts.append(e)
            self.source[e] = avail[e]
            self.err[e] = df[list(ERROR_PARTS)].to_numpy().sum(1).astype(np.float64)
            self.err_raw[e] = (df[[f"{c}_raw" for c in ERROR_PARTS]].to_numpy().sum(1).astype(np.float64)
                               if "sub_raw" in df else None)
            self.n_frames[e] = df["n_frames"].to_numpy().astype(np.int64) if "n_frames" in df else None
            self.texts[e] = df["transcription"].tolist() if "transcription" in df else None
            self.texts_norm[e] = df["transcription_norm"].tolist() if "transcription_norm" in df else None
            if "pool_mean" in df:
                P = np.stack(df["pool_mean"].to_list()).astype(np.float32)
                bad = ~np.isfinite(P).all(1)
                if bad.any():
                    # a NaN/inf pooled vector (an encoder overflowing on a near-silent or very short
                    # segment) would poison every consumer downstream: torch trains through NaN
                    # silently, XGBoost reads it as "missing", and LAPACK's SVD refuses to converge.
                    # The row is still a real utterance, so keep it and zero its features.
                    print(f"{e}: {int(bad.sum()):,} of {len(P):,} pooled vectors are non-finite -> zeroed")
                    P[bad] = 0.0
                self.n_nonfinite[e] = int(bad.sum())
                self._pooled[e] = P
            self.frame_rate_hz[e] = self._frame_rate(e, avail[e])
        if verbose:
            n_fastt = sum(1 for s in self.source.values() if s == "fastt")
            print(f"{self.spec.name}: {len(self.ids):,} utterances x {len(self.experts)} experts "
                  f"({n_fastt} adopted from fastt) in {time.perf_counter() - t0:.1f}s")
        return self

    # ------------------------------------------------------------- views --

    @property
    def n(self):
        return len(self.ids)

    def index(self):
        return pd.DataFrame({"split": self.split, "id": [i for _, i in self.ids]})

    def mask(self, splits):
        return np.isin(self.split, list(splits))

    def E(self, members, splits=None):
        """`(N, k)` error counts for `members`, optionally restricted to `splits`."""
        M = np.stack([self.err[m] for m in members], axis=1)
        return M if splits is None else M[self.mask(splits)]

    err_matrix = E

    def words(self, splits=None):
        return self.nref if splits is None else self.nref[self.mask(splits)]

    def pooled(self, expert):
        """`(N, D)` float32 mean-pooled frames for `expert`, from its labels."""
        if expert not in self._pooled:
            raise KeyError(f"{expert} has no pooled column in its labels")
        return self._pooled[expert]

    def corpus_wer(self, members=None, splits=None):
        members = list(members or self.experts)
        E, n = self.E(members, splits), self.words(splits)
        return pd.Series(E.sum(0) / n.sum(), index=members, name="corpus_wer")

    def seconds(self):
        """`(N,)` duration estimate from the first expert with a known frame rate, or None."""
        for e in self.experts:
            if self.frame_rate_hz.get(e) and self.n_frames.get(e) is not None:
                return self.n_frames[e] / float(self.frame_rate_hz[e])
        return None

    def summary(self, splits=None):
        """Corpus WER per expert, plus where its labels came from and its win rate."""
        members = list(self.experts)
        E, n = self.E(members, splits), self.words(splits)
        is_best = E == E.min(1, keepdims=True)
        share = (is_best / is_best.sum(1, keepdims=True)).mean(0)
        return pd.DataFrame({"expert": members, "source": [self.source[m] for m in members],
                             "corpus_wer": E.sum(0) / n.sum(),
                             "mean_utt_wer": (E / np.maximum(n, 1)[:, None]).mean(0),
                             "win_rate": share,
                             "frame_rate_hz": [self.frame_rate_hz.get(m) for m in members]}
                            ).sort_values("corpus_wer", ignore_index=True)
