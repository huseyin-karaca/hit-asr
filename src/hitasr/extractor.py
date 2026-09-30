__all__ = ['UNPUSHED', 'ExtractConfig', 'Encoded', 'Replay', 'RunResult', 'local_runs', 'push_run', 'is_rate_limited', 'pending_runs', 'push_leftovers', 'run_expert', 'sweep', 'is_oom', 'ASRFrameExtractor',
           'rebuild_plan', 'compare_with_published']

import json
import shutil
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from hitasr.core import SAMPLING_RATE, SOURCE_REPO, active_dataset, require_audio_datasets, use_dataset
from labkit.env import empty_cache
from hitasr.frames import (FrameWriter, default_frames_dir, labels_features,
                           write_manifest)
from hitasr.hub import HitHub, load_base
from hitasr.scoring import WerScorer, corpus_wer_table


@dataclass
class ExtractConfig:
    """Everything tunable about one extraction run.

    layer : which entry of the encoder's hidden-state stack to store. `-1` is
        the final encoder layer, resolved to an absolute index before use.
    batch_size : utterances per forward pass. `base` is duration-sorted, so
        padding is minimal — except for Whisper, which pads to 30 s regardless.
    shard_bytes : frame shard size; see `FrameWriter`. Large on purpose — a
        corpus-expert store is a dozen 2 GB files rather than hundreds of
        small ones, which is what keeps Colab <-> Hub traffic cheap.
    """
    layer: int = -1
    batch_size: int = 16
    shard_bytes: int = 2 << 30


@dataclass
class Encoded:
    """One batch after the encoder and before the decoder.

    `hidden` `(B, T, D)` at the requested layer, still on device; `lengths`
    the valid frame count per utterance; `state` whatever `_decode_batch`
    needs — logits for a CTC head, the encoder output for Whisper.
    """
    hidden: torch.Tensor
    lengths: torch.Tensor
    state: object = None

    def __len__(self):
        return int(self.lengths.shape[0])


class Replay:
    """A stand-in `forward` for a module whose outputs were recorded: returns them in the order they were produced.

    `module.forward = Replay(outputs)` (an instance attribute, removed with `del module.forward`) makes a model's
    own call path reuse an encoder pass instead of running it again — what `decode_encoded` does.
    """

    def __init__(self, outputs):
        self.outputs, self.i = list(outputs), 0

    def __call__(self, *args, **kwargs):
        if self.i >= len(self.outputs):
            raise RuntimeError(f"the encoder was called {self.i + 1} times, {len(self.outputs)} when recorded")
        self.i += 1
        return self.outputs[self.i - 1]


@dataclass
class RunResult:
    """What `ASRFrameExtractor.run()` hands back."""
    labels: dict                    # {split: DataFrame} in base order
    corpus: pd.DataFrame            # corpus WER per split
    frames_dir: Path                # the local frame store written
    manifest: dict
    payload: dict = None


UNPUSHED = ".unpushed"           # marker a finished run leaves in its store until `push_run` succeeds


def _labels_dataset(df, d):
    """A labels DataFrame as a `datasets.Dataset` with the fixed `labels_features(d)` schema."""
    from datasets import Dataset
    feats = labels_features(d)
    return Dataset.from_pandas(df[list(feats)], preserve_index=False, features=feats)


def local_runs(frames_root=None):
    """Every local frame store under `frames_root` that has a manifest: `{config_dir: manifest}`."""
    root = Path(frames_root or default_frames_dir())
    out = {}
    for man in sorted(root.glob("*/manifest.json")):
        try:
            out[man.parent] = json.loads(man.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return out


def push_run(frames_dir, frames=True, labels=True, verbose=True):
    """Push one local frame store — labels config, shards, manifest, results JSON — to the Hub.

    Needs nothing but the directory: the manifest names the corpus and the
    expert, `labels/<split>.parquet` holds the labels. That is what makes a
    failed upload recoverable after the model and the kernel are gone; the
    notebooks call it for anything left on disk. Labels are pushed first, so
    a failure never leaves an expert with frames but no labels.
    """
    from datasets import DatasetDict

    frames_dir = Path(frames_dir)
    man = json.loads((frames_dir / "manifest.json").read_text())
    spec = active_dataset(man["dataset"])
    name, model_id, d = man["expert"], man.get("model_id"), int(man["d"])
    hub = HitHub(spec=spec)
    label_files = {s: frames_dir / "labels" / f"{s}.parquet" for s in man["splits"]}
    have_labels = all(f.exists() for f in label_files.values())
    corpus = None
    if labels and have_labels:
        dfs = {s: pd.read_parquet(f) for s, f in label_files.items()}
        dd = DatasetDict({s: _labels_dataset(df, d) for s, df in dfs.items()})
        hub.push(dd, spec.labels_config(name), f"{model_id} on {spec.name}: transcripts, WER counters, pooled mean")
        corpus = corpus_wer_table(dfs)
    elif labels and verbose:
        print(f"  {frames_dir.name}: no labels/ on disk — pushing frames only")
    payload = {"expert": name, "model_id": model_id, "dataset": spec.name, "manifest": man,
               "corpus_wer": corpus.to_dict(orient="records") if corpus is not None else None}
    results_path = spec.results_path(f"extract_{name}")
    local_json = frames_dir / "extract.json"
    local_json.write_text(json.dumps(payload, indent=2, default=str))
    # One commit for the shards, the indexes, the manifest and the results
    # JSON: the Hub allows 128 commits per hour per repo, and one commit per
    # split blew through that with two kernels pushing. `create_commit`
    # pre-uploads every LFS file first, so a retry re-sends only what is
    # missing, and a store is either whole on the Hub or absent.
    files = {results_path: local_json}
    if frames:
        cfg = spec.frames_config(name)
        files[f"{cfg}/manifest.json"] = frames_dir / "manifest.json"
        for split in man["splits"]:
            for f in sorted((frames_dir / split).iterdir()):
                if f.is_file():
                    files[f"{cfg}/{split}/{f.name}"] = f
    hub.push_files(files, f"{model_id} on {spec.name}: frames, manifest, results", verbose=verbose)
    (frames_dir / UNPUSHED).unlink(missing_ok=True)
    return payload


def is_rate_limited(exc):
    """True for the Hub's 429 (128 commits per hour per repo); retrying sooner than an hour is pointless."""
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code == 429 or "429" in str(exc) or "Too Many Requests" in str(exc)


def pending_runs(frames_root=None):
    """Local frame stores that still need a push: marked unpushed, or absent from the Hub."""
    out = {}
    for d, man in local_runs(frames_root).items():
        try:
            spec = active_dataset(man["dataset"])
            on_hub = HitHub(spec=spec).frames_complete(man["expert"])
        except Exception:                                      # noqa: BLE001
            on_hub = False
        if (d / UNPUSHED).exists() or not on_hub:
            out[d] = man
    return out


def push_leftovers(frames_root=None, delete=True, verbose=True, wait_minutes=0, poll_minutes=10):
    """Push every pending local store (`pending_runs`) and delete it once it is up. Returns `[(name, status)]`.

    `wait_minutes > 0` keeps trying through a 429 for that long, polling every
    `poll_minutes` — the commit quota is per hour, so the end of a sweep is
    the right place to wait it out.
    """
    deadline = time.monotonic() + 60 * wait_minutes
    status = {}
    while True:
        limited = False
        for d, man in pending_runs(frames_root).items():
            if status.get(d.name) == "pushed":
                continue
            print(f"\n--- leftover {d.name} ({man['dataset']} / {man['expert']})")
            try:
                push_run(d, verbose=verbose)
                if delete:
                    shutil.rmtree(d, ignore_errors=True)
                status[d.name] = "pushed"
            except Exception as e:                             # noqa: BLE001
                limited = limited or is_rate_limited(e)
                print(f"  still failing: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
                status[d.name] = f"FAILED {type(e).__name__}"
        if not limited or time.monotonic() >= deadline:
            break
        print(f"  Hub commit quota exhausted — waiting {poll_minutes} min "
              f"({(deadline - time.monotonic()) / 60:.0f} min left)")
        time.sleep(60 * poll_minutes)
    if not status and verbose:
        print("no pending local stores")
    return list(status.items())


def run_expert(ex, ds, batch_size, min_batch=4, frames=True, push_retries=3, verify_min=0.85, decode="auto"):
    """smoke_test -> fit_batch_size -> run -> push, for one loaded-or-not extractor on one corpus.

    OOM halves the batch and restarts the expert; a verify-mode run whose
    adopted labels disagree with the decoder falls back to a full decode; a
    push that keeps failing leaves the store on disk for `push_leftovers`.
    `decode="full"` decodes every utterance even where labels could be adopted.
    Returns the log row. Never raises — the sweep must outlive one expert.
    """
    corpus, name = ex.spec.name, ex.name
    row = {"corpus": corpus, "expert": name, "status": "ok", "batch": None, "decode": None,
           "wer": None, "minutes": None, "error": ""}
    t0 = time.perf_counter()
    out_dir = ex.frames_root / ex.spec.frames_config(name)
    res = None
    try:
        ex.smoke_test(ds)
        bs = ex.fit_batch_size(ds, batch_size, floor=min_batch)
        while True:
            try:
                res = ex.run(ds, cfg=ExtractConfig(batch_size=bs), push=False, decode=decode,
                             verify_min=verify_min)
                break
            except Exception as e:                             # noqa: BLE001
                shutil.rmtree(out_dir, ignore_errors=True)
                if is_oom(e) and bs > min_batch:
                    empty_cache(report=True)
                    bs = max(min_batch, bs // 2)
                    print(f"  OOM mid-run -> retrying the whole expert at batch {bs}")
                elif "adopted labels disagree" in str(e) and decode != "full":
                    decode = "full"
                    print("  adopted labels disagree with this decoder -> re-running with decode='full'")
                else:
                    raise
        row.update(batch=bs, decode=res.manifest.get("decode"),
                   wer=round(float(res.corpus["errors_norm"].sum() / res.corpus["ref_words_norm"].sum()), 4))
        for attempt in range(1, push_retries + 1):
            try:
                ex.push(res, frames=frames, labels=True)
                shutil.rmtree(res.frames_dir, ignore_errors=True)
                break
            except Exception as e:                             # noqa: BLE001
                print(f"  push attempt {attempt}/{push_retries} failed: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:160]}")
                if attempt == push_retries or is_rate_limited(e):
                    row["status"] = "PUSH_PENDING"
                    print(f"  store kept at {res.frames_dir}; push_leftovers() will retry"
                          + (" once the commit quota resets" if is_rate_limited(e) else ""))
                    break
                time.sleep(30 * attempt)
    except Exception as e:                                     # noqa: BLE001
        row.update(status="FAILED", error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
        print(f"  FAILED {corpus}/{name}: {row['error']}")
        traceback.print_exc()
        shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        try:
            del ex.model
        except AttributeError:
            pass
        empty_cache(report=True)
        row["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    return row


def sweep(plan, experts, batch, min_batch=4, labels_only=(), push_retries=3, verify_min=0.85,
          final_wait_minutes=75, decode="auto"):
    """Labels + frames for every `(corpus, [expert, ...])` in `plan`; returns the log as a DataFrame.

    `experts` is `{name: class}` (from `load_expert_classes`), `batch` is
    `{name: int, "default": int}`. `base` is loaded once per corpus; after
    each corpus any store whose push failed is retried, and at the end the
    retry waits up to `final_wait_minutes` for the Hub's commit quota.
    """
    log = []
    for corpus, names in plan.items():
        if not names:
            continue
        use_dataset(corpus)
        ds = load_base()
        for name in names:
            print(f"\n===== {corpus} / {name}")
            log.append(run_expert(experts[name](), ds, batch.get(name, batch["default"]), min_batch=min_batch,
                                  frames=name not in labels_only, push_retries=push_retries,
                                  verify_min=verify_min, decode=decode))
            root = default_frames_dir()
            root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(root).free / 1e9
            print(f"  disk free: {free:.0f} GB")
        del ds
        push_leftovers()
    push_leftovers(wait_minutes=final_wait_minutes)
    return pd.DataFrame(log)


def is_oom(exc):
    """True when `exc` is a CUDA out-of-memory error, from torch or a library wrapping one."""
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc)
    return "out of memory" in msg.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in msg


def _free_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean_pool(hidden, lengths):
    """`(B, D)` float32 mean over valid frames, on the device."""
    h = hidden.float()
    mask = (torch.arange(h.shape[1], device=h.device)[None, :] < lengths.to(h.device)[:, None])
    s = (h * mask[..., None]).sum(1)
    return s / mask.sum(1).clamp(min=1)[:, None]


class ASRFrameExtractor(ABC):
    """Extract frame-level encoder states + WER labels for one expert; push both.

    Usage
    -----
        ex = WhisperLargeV3Extractor()
        ex.smoke_test(ds)                       # shapes, frame rate, one transcript
        res = ex.run(ds, push=False)            # frames -> labels -> corpus WER
        ex.push(res)                            # upload when the numbers look right
    """

    name: str = None                    # short id, used in config names
    model_id: str = None                # Hub checkpoint id
    sampling_rate: int = SAMPLING_RATE
    frame_rate_hz: float = None         # encoder frames per second; measured by smoke_test
    emits_transcription: bool = True
    grad_mode: str = "inference"        # "inference" | "no_grad"
    max_input_seconds: float = None
    min_input_samples: int = 0
    group: str = "A"                    # install group

    def __init__(self, model_id=None, device=None, dtype=torch.bfloat16, scorer=None,
                 spec=None, frames_root=None):
        self.model_id = model_id or self.model_id
        self.spec = active_dataset(spec)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.scorer = scorer or (WerScorer() if self.emits_transcription else None)
        self.frames_root = Path(frames_root or default_frames_dir())
        self._loaded = False
        self._clock = None
        self._warned_pad = False

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, model_id={self.model_id!r})"

    # ---------------------------------------------------------------- hooks --

    @abstractmethod
    def _load(self):
        """Populate `self.model` and anything else the subclass needs."""

    @property
    @abstractmethod
    def hidden_size(self):
        """Encoder width D."""

    @property
    @abstractmethod
    def num_encoder_layers(self):
        """Number of encoder blocks; a hidden-state stack has this many + 1 entries."""

    @abstractmethod
    def _encode_batch(self, arrays, layer):
        """One batch of waveforms -> `Encoded` at absolute layer index `layer`."""

    def _decode_batch(self, encoded):
        """`Encoded` -> transcriptions as emitted, or `None` when the expert has no decoder here."""
        return None

    # ------------------------------------------------------------ inference --

    def load(self):
        if not self._loaded:
            print(f"Loading {self.model_id} on {self.device} ({self.dtype})")
            t0 = time.perf_counter()
            self._load()
            self._loaded = True
            print(f"  ready in {time.perf_counter() - t0:.1f}s — "
                  f"{self.num_encoder_layers} encoder layers, d={self.hidden_size}")
        return self

    def _grad_ctx(self):
        return torch.inference_mode() if self.grad_mode == "inference" else torch.no_grad()

    def _mark_encoded(self):
        """Timing hook between the two halves of `_encode_and_decode` (armed by setting `self._clock = {}`)."""
        if self._clock is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._clock["encoded"] = time.perf_counter()

    def resolve_layer(self, layer):
        n = self.num_encoder_layers + 1
        a = layer if layer >= 0 else n + layer
        if not 0 <= a < n:
            raise IndexError(f"layer {layer} out of range for {n} hidden states")
        return a

    def encode(self, arrays, layer=-1):
        """Raw waveforms -> `Encoded`. Loads the model on first use."""
        self.load()
        arrays = self._pad_short([np.asarray(a, dtype=np.float32) for a in arrays])
        with self._grad_ctx():
            return self._encode_batch(arrays, self.resolve_layer(layer))

    def decode(self, encoded):
        if not self.emits_transcription:
            return None
        with self._grad_ctx():
            return self._decode_batch(encoded)

    def transcribe(self, arrays):
        return self.decode(self.encode(arrays))

    def _encode_and_decode(self, arrays, layer, decode=True):
        with self._grad_ctx():
            enc = self._encode_batch(arrays, layer)
            self._mark_encoded()
            hyps = self._decode_batch(enc) if (decode and self.emits_transcription) else None
        return enc, hyps

    # ----------------------------------------------------------- deployment --
    # What a router in front of the experts needs (`hitasr.deploy`): the frames it reads with the encoder alone run,
    # then the transcript of the ONE expert it picks, from the states already computed. The extraction path above is
    # untouched: an expert whose `_encode_batch` also decodes (an LLM transcriber) overrides both.

    reuses_encoder = True        # False: `decode_encoded` runs this expert's encoder again (reported by the timing)

    def encode_only(self, arrays, layer=-1):
        """Raw waveforms -> `Encoded`, running the encoder and nothing after it. The default is `encode`."""
        return self.encode(arrays, layer)

    def decode_encoded(self, encoded):
        """Transcripts for an `encode_only` batch, reusing its encoder states. The default is `decode`."""
        return self.decode(encoded)

    def _pad_short(self, arrays):
        """Zero-pad any clip below `min_input_samples`; AMI has 20 ms turns that some front ends reject."""
        if not self.min_input_samples:
            return arrays
        out, n_padded = [], 0
        for a in arrays:
            if len(a) < self.min_input_samples:
                a = np.pad(np.asarray(a), (0, self.min_input_samples - len(a)))
                n_padded += 1
            out.append(a)
        if n_padded and not self._warned_pad:
            self._warned_pad = True
            print(f"  note: zero-padding clips shorter than {self.min_input_samples} samples "
                  f"({1000 * self.min_input_samples / self.sampling_rate:.0f} ms); "
                  f"{n_padded} in this batch. Reported once per run.")
        return out

    # ------------------------------------------------------------ adoption --

    def adoption_source(self):
        """Where labels for this expert on this corpus already live.

        `("own", hub, config)` for hit-asr's own `<corpus>_labels_<expert>` — the
        same checkpoint and decoder, so preferred — else `("fastt", hub, config)`
        for fastt's `model_<expert>`, else `None`.
        """
        try:
            own = HitHub(spec=self.spec)
            if own.has(self.spec.labels_config(self.name)):
                return "own", own, self.spec.labels_config(self.name)
            src = HitHub(SOURCE_REPO, spec=self.spec)
            if src.has(self.spec.source_model_config(self.name)):
                return "fastt", src, self.spec.source_model_config(self.name)
        except Exception:                                      # noqa: BLE001
            pass
        return None

    def adoptable(self):
        """True if hit-asr or fastt holds labels for this expert on this corpus."""
        return self.adoption_source() is not None

    def adopt_labels(self, split):
        """The existing `(id, transcription, counters...)` for `split`, as a DataFrame in ITS order."""
        _, hub, config = self.adoption_source()
        cols = ["id", "transcription", "transcription_norm", "text_norm",
                "sub", "dele", "ins", "nref", "wer",
                "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw"]
        return hub.read_columns(config, split, cols).to_pandas()

    # ------------------------------------------------------------- pipeline --

    def run(self, ds, cfg=None, push=False, decode="auto", verify_n=64, verify_min=0.85, out_dir=None):
        """Frames -> labels -> corpus WER -> optionally push. One call.

        decode : `"full"` decodes every utterance; `"adopt"` takes existing
            labels and never runs the decoder; `"verify"` adopts and decodes
            `verify_n` utterances per split to check the transcripts agree;
            `"auto"` picks `"verify"` when labels exist and `"full"` otherwise.
        verify_min : the share of sampled transcripts that must match after
            normalisation. Not 1.0, because a different batch size pads
            differently and flips a few of AMI's 20 ms turns; a checkpoint
            that actually changed falls far below it.

        The labels are also written to `<out_dir>/labels/<split>.parquet`, so
        a run whose push failed can be pushed later from disk (`push_run`).
        """
        cfg = cfg or ExtractConfig()
        require_audio_datasets()
        self.load()
        layer = self.resolve_layer(cfg.layer)
        if decode == "auto":
            decode = "verify" if self.adoptable() else "full"
        if decode in ("adopt", "verify") and not self.adoptable():
            raise ValueError(f"{self.name} has no labels on {self.spec.name} to adopt")
        adopted_from = self.adoption_source()[0] if decode != "full" else None
        out_dir = Path(out_dir or self.frames_root / self.spec.frames_config(self.name))
        # a fresh store: the folder may hold frames fetched for level 2, hard-linked to the download cache, and
        # writing a shard over such a link would change the cached copy too
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"{self.name} on {self.spec.name}: layer {layer}, decode={decode}"
              f"{f' (labels from {adopted_from})' if adopted_from else ''}, -> {out_dir}")

        t0 = time.perf_counter()
        labels, verify_pairs = {}, []
        for split in self.spec.splits:
            if split not in ds:
                continue
            d = ds[split]
            adopted = self.adopt_labels(split) if decode != "full" else None
            if adopted is not None:
                pos = {i: k for k, i in enumerate(adopted["id"].tolist())}
                missing = [i for i in d["id"] if i not in pos]
                if missing:
                    raise KeyError(f"{split}: {len(missing)} base id(s) absent from fastt's "
                                   f"labels, e.g. {missing[:3]}")
            rng = np.random.default_rng(0)
            verify_rows = (set(rng.choice(d.num_rows, min(verify_n, d.num_rows), replace=False).tolist())
                           if decode == "verify" else set())
            writer = FrameWriter(out_dir / split, d=self.hidden_size, shard_bytes=cfg.shard_bytes)
            rows = []
            pbar = tqdm(total=d.num_rows, desc=f"  {split}", unit=" utts", leave=True)
            for start in range(0, d.num_rows, cfg.batch_size):
                batch = d[start:start + cfg.batch_size]
                ids = batch["id"]
                arrays = self._pad_short([a["array"] for a in batch["audio"]])
                need_decode = decode == "full" or any((start + j) in verify_rows for j in range(len(ids)))
                enc, hyps = self._encode_and_decode(arrays, layer, decode=need_decode)
                writer.add(ids, enc.hidden, enc.lengths)
                pooled = _mean_pool(enc.hidden, enc.lengths).cpu().float().numpy()
                lens = enc.lengths.cpu().tolist()
                if decode == "full":
                    scored = self.scorer.score(batch["text"], hyps)
                    for j, uid in enumerate(ids):
                        rows.append({"id": uid, "transcription": hyps[j],
                                     **{k: v[j] for k, v in scored.items()},
                                     "n_frames": int(lens[j]), "pool_mean": pooled[j]})
                else:
                    for j, uid in enumerate(ids):
                        src = adopted.iloc[pos[uid]]
                        rows.append({**{c: src[c] for c in adopted.columns},
                                     "n_frames": int(lens[j]), "pool_mean": pooled[j]})
                        if hyps is not None and (start + j) in verify_rows:
                            verify_pairs.append((uid, src["transcription"], hyps[j]))
                pbar.update(len(ids))
                pbar.set_postfix(frames=f"{writer.total_frames:,}",
                                 shard=writer.shard)
            pbar.close()
            index = writer.close()
            labels[split] = pd.DataFrame(rows)
            assert list(index["id"]) == list(labels[split]["id"]), "frame index and labels disagree"

        if verify_pairs:
            agree = sum(1 for _, a, b in verify_pairs
                        if self.scorer.normalizer(a) == self.scorer.normalizer(b))
            print(f"  verify: {agree}/{len(verify_pairs)} sampled transcripts identical to "
                  f"the {adopted_from} labels after normalisation")
            if agree < verify_min * len(verify_pairs):
                for uid, a, b in [p for p in verify_pairs
                                  if self.scorer.normalizer(p[1]) != self.scorer.normalizer(p[2])][:5]:
                    print(f"    {uid}\n      adopted: {a!r}\n      decoded: {b!r}")
                raise RuntimeError(f"adopted labels disagree with this checkpoint's decoder "
                                   f"({agree}/{len(verify_pairs)} < {verify_min:.2f}); run with decode='full'")
        for split, df in labels.items():
            (out_dir / "labels").mkdir(parents=True, exist_ok=True)
            _labels_dataset(df, self.hidden_size).to_parquet(out_dir / "labels" / f"{split}.parquet")
        (out_dir / UNPUSHED).touch()
        manifest = write_manifest(out_dir, self.spec, self.name, self.hidden_size,
                                  self.frame_rate_hz, layer, list(labels),
                                  extra={"model_id": self.model_id, "decode": decode,
                                         "adopted_from": adopted_from,
                                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "n_utts": {s: int(len(v)) for s, v in labels.items()},
                                         "n_frames": {s: int(v["n_frames"].sum()) for s, v in labels.items()}})
        corpus = corpus_wer_table(labels)
        print(f"\n--- Corpus WER ({self.model_id}) ---")
        print(corpus[["split", "n_utts", "wer_norm", "wer_raw"]].to_string(index=False))
        result = RunResult(labels=labels, corpus=corpus, frames_dir=out_dir, manifest=manifest)
        if push:
            self.push(result)
        else:
            print("\npush=False — nothing uploaded. Call ex.push(result) when ready.")
        print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")
        return result

    def push(self, result, frames=True, labels=True):
        """Push the labels config, the frame store and the results JSON — from the local store on disk."""
        result.payload = push_run(result.frames_dir, frames=frames, labels=labels)
        return result.payload

    # ---------------------------------------------------------- diagnostics --

    def smoke_test(self, ds, n=4):
        """Shapes, frame rate and one transcript. **Run this before a full extraction.**

        The sample is the `n-1` shortest rows plus the longest one; the frame
        rate is measured on the longest, because AMI's shortest turn is 20 ms
        and yields zero frames.
        """
        self.load()
        split = ds[list(ds)[0]]
        n = max(1, min(n, split.num_rows))
        rows = split.select(sorted({*range(n - 1), split.num_rows - 1}))
        arrays = self._pad_short([a["array"] for a in rows["audio"]])
        enc, hyps = self._encode_and_decode(arrays, self.resolve_layer(-1))
        i = int(np.argmax(enc.lengths.cpu().numpy()))
        seconds = len(arrays[i]) / self.sampling_rate
        if enc.lengths[i] > 0 and seconds > 0:
            self.frame_rate_hz = round(float(enc.lengths[i]) / seconds, 2)
        print(f"hidden  {tuple(enc.hidden.shape)}   (expect (B, T, {self.hidden_size}))")
        print(f"frames  {enc.lengths.tolist()}  -> {self.frame_rate_hz} frames/s")
        if enc.hidden.shape[-1] != self.hidden_size:
            print(f"  WARNING: last dim is {enc.hidden.shape[-1]}, not {self.hidden_size}")
        if self.emits_transcription and hyps is not None:
            print(f"ref : {rows[0]['text']}")
            print(f"hyp : {hyps[0]}")
        return enc, hyps

    def fit_batch_size(self, ds, batch_size, floor=2, decode=True):
        """The largest batch size <= `batch_size` whose worst batch fits on the device.

        `base` is duration-sorted, so the last `batch_size` rows of each split
        are the longest utterances the pass will see; one encode (and decode)
        of those is the stress test. Halves on OOM, down to `floor`.
        """
        self.load()
        layer = self.resolve_layer(-1)
        bs = int(batch_size)
        while True:
            try:
                for split in self.spec.splits:
                    if split not in ds:
                        continue
                    d = ds[split]
                    rows = d[max(0, d.num_rows - bs):]
                    arrays = self._pad_short([a["array"] for a in rows["audio"]])
                    self._encode_and_decode(arrays, layer, decode=decode)
                _free_cuda()
                print(f"  batch {bs}: the longest utterances fit")
                return bs
            except Exception as e:                             # noqa: BLE001
                if not is_oom(e) or bs <= floor:
                    raise
                _free_cuda()
                nxt = max(floor, bs // 2)
                print(f"  batch {bs}: OOM ({type(e).__name__}) -> trying {nxt}")
                bs = nxt

    def benchmark(self, ds, n=16):
        """Time one batch and print the real-time factor."""
        self.load()
        rows = ds[list(ds)[0]].select(range(n))
        arrays = [a["array"] for a in rows["audio"]]
        seconds = sum(len(a) for a in arrays) / self.sampling_rate
        arrays = self._pad_short(arrays)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._encode_and_decode(arrays, self.resolve_layer(-1))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{n} utterances, {seconds:.1f}s of audio: {dt:.2f}s  RTF {dt / seconds:.4f}")
        return dt / seconds


def rebuild_plan(corpora):
    """`{corpus: [expert, ...]}`: every trio expert whose labels and frames are not yet complete where the rebuild is
    written (`REPO_ID`: by default a local folder, see `hitasr.core.rebuild_dir`)."""
    from hitasr.core import use_dataset
    from hitasr.hub import HitHub
    plan = {}
    for corpus in corpora:
        spec = use_dataset(corpus, verbose=False)
        done = set(HitHub(spec=spec).complete_experts())
        plan[corpus] = [e for e in spec.members if e not in done]
        print(f"{corpus:16s} {plan[corpus] or '— complete'}")
    return plan


def compare_with_published(log):
    """Each rebuilt expert beside the published one: the share of utterances whose normalised transcript is identical,
    the share whose word-error count differs, and the corpus WER of both (the rebuild read from `REPO_ID`, the
    published labels from `PUBLIC_REPO`). The same checkpoint, decoder and normaliser agree up to the GPU's numerics:
    a few transcripts flip, the corpus WER moves in about the third decimal."""
    from hitasr.core import PUBLIC_REPO, use_dataset

    rows = []
    for corpus, expert in log.loc[log["status"] == "ok", ["corpus", "expert"]].itertuples(index=False):
        spec = use_dataset(corpus, verbose=False)
        cols = ["id", "transcription_norm", "sub", "dele", "ins", "nref"]
        cfg = spec.labels_config(expert)
        got = {}
        for name, hub in (("rebuilt", HitHub(spec=spec)), ("published", HitHub(PUBLIC_REPO, spec=spec))):
            got[name] = pd.concat([hub.read_columns(cfg, s, cols).to_pandas().assign(split=s) for s in spec.splits],
                                  ignore_index=True).set_index(["split", "id"])
        a, b = got["rebuilt"], got["published"].reindex(got["rebuilt"].index)
        if b["nref"].isna().any() or len(a) != len(got["published"]):
            raise ValueError(f"{corpus}/{expert}: the rebuilt utterances are not the published ones")
        err_a, err_b = a[["sub", "dele", "ins"]].sum(axis=1), b[["sub", "dele", "ins"]].sum(axis=1)
        rows.append({"corpus": corpus, "expert": expert, "utterances": len(a),
                     "identical_transcripts": float((a["transcription_norm"] == b["transcription_norm"]).mean()),
                     "error_count_differs": float((err_a != err_b).mean()),
                     "wer_rebuilt": float(err_a.sum() / a["nref"].sum()),
                     "wer_published": float(err_b.sum() / b["nref"].sum())})
    check = pd.DataFrame(rows)
    if len(check):
        check["difference"] = check["wer_rebuilt"] - check["wer_published"]
    return check.round(4)


def main(argv=None):
    """`python -m hitasr.extractor <corpus> [<corpus> ...] [--push-to REPO]`: `notebooks/extract` as a script — the
    trios' labels and frames into the local rebuild folder (`hitasr.core.rebuild_dir()`), then the comparison with the
    published labels printed; `--push-to` also uploads the folder to a dataset repository of yours."""
    import argparse

    from hitasr.configs import MAIN, REBUILD
    from hitasr.core import rebuild_dir, use_hub
    from hitasr.models.registry import load_expert_classes
    from labkit.env import describe_env, login_hf
    ap = argparse.ArgumentParser(description=main.__doc__.split(":")[0])
    ap.add_argument("corpora", nargs="+", choices=sorted(MAIN))
    ap.add_argument("--push-to", default="")
    a = ap.parse_args(argv)
    out = rebuild_dir()
    use_hub(f"local:{out}", records=f"local:{out}")
    login_hf(required=any(m in REBUILD.gated for c in a.corpora for m in MAIN[c].members))
    describe_env()
    print(f"writing to {out}")
    log = sweep(rebuild_plan(a.corpora), load_expert_classes(), REBUILD.batch, min_batch=REBUILD.min_batch,
                decode="full")
    print(log.to_string(index=False))
    if (log["status"] != "ok").any():
        raise SystemExit("some experts failed: see the log above")
    print(compare_with_published(log).to_string(index=False))
    if a.push_to:
        from huggingface_hub import HfApi
        login_hf()
        HfApi().create_repo(a.push_to, repo_type="dataset", private=True, exist_ok=True)
        HfApi().upload_large_folder(repo_id=a.push_to, repo_type="dataset", folder_path=str(out))


if __name__ == "__main__":
    main()
