"""HIT-ASR at run time, on raw audio: every expert's **encoder**, the **router**, and the **decoder of the one expert it
picks** — and the systems it is compared with, clocked on the same clips.

`DeployedRouter` is a fitted router with everything needed to apply it outside the store: the members, their widths,
the selected configuration, the frame cap and the weights of every seeded fit of the ensemble. `fit_deployed` fits
one from a store exactly as a fold of the main table does (`MODELS["hit_asr"]`, the same seed), so the deployed
router can be the very model a fold of Table 3 scored.

The systems (`transcribe(arrays) -> RouterOutput`), one clip at a time — the online setting of the cost claim:

* `HitASRSystem` — `encode_only` of every member, the router on those frames, `decode_encoded` of the chosen member
  (its encoder pass is reused, not repeated, wherever the expert allows it; `ex.reuses_encoder` says so);
* `SingleExpert` — one expert's own pipeline, the best single system of the tables;
* `DecodeAll` — every expert's full pipeline plus a ROVER vote: the cost every transcript-fusion method and the
  oracle must pay.

`time_experts` splits each expert's pipeline into its encoder and decoder halves on the same clips and checks that
the reused-encoder decode gives the pipeline's own transcript; `compare_systems` runs the systems end to end.
Every stage ends in a device synchronise, so the clocks are device time, not queue time.
"""

__all__ = ['DeployedRouter', 'fit_deployed', 'main_fold', 'rows_digest', 'router_frames', 'frame_agreement', 'RouterOutput', 'HitASRSystem',
           'SingleExpert', 'DecodeAll', 'time_experts', 'corpus_wer', 'compare_systems', 'weights_gb']

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hitasr.rover import build_network, vote


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------- the fitted router

class DeployedRouter:
    """A fitted HIT-ASR router, applicable to frames computed on the fly.

    Attributes
    ----------
    members, dims : the experts in the router's output order, and their frame widths.
    params : the selected configuration it was built from (`main_<corpus>.json`'s `params["hit_asr"]`).
    arch, max_frames, amp : the router's kwargs, the frame cap each expert's sequence is cut to, bf16 autocast.
    states : one `state_dict` per seeded fit; the prediction averages their softmax outputs.
    meta : where it came from (corpus, rows, seed, fold, fit seconds).
    """

    def __init__(self, members, dims, params, arch, max_frames, states, amp=True, meta=None):
        self.members, self.dims = tuple(members), {m: int(dims[m]) for m in members}
        self.params, self.arch = dict(params), dict(arch)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.states, self.amp, self.meta = list(states), bool(amp), dict(meta or {})
        if self.arch.get("pooled_skip"):
            raise NotImplementedError("a router with the pooled shortcut needs the pooled design at run time")
        self._models, self._device = None, None

    def __repr__(self):
        return (f"DeployedRouter({'+'.join(self.members)}, {len(self.states)} fit(s), {self.n_params() / 1e6:.2f}M "
                f"parameters, fitted on {self.meta.get('n_rows', '?')} rows)")

    @classmethod
    def from_arm(cls, arm, store, params, meta=None):
        """From a fitted `HitASRArm` (every model of its ensemble)."""
        models = arm.models or [arm.model]
        states = [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models]
        return cls(store.members, store.dims, params, arm.arch, arm.train.max_frames, states, amp=arm.train.amp,
                   meta=meta)

    def models(self, device=None):
        """The ensemble as `HitASRRouter`s on `device`, built once per device."""
        from hitasr.routers import HitASRRouter
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self._models is None or self._device != device:
            self._models = []
            for st in self.states:
                m = HitASRRouter(self.dims, self.members, **self.arch)
                m.load_state_dict(st)
                self._models.append(m.to(device).eval())
            self._device = device
        return self._models

    def n_params(self):
        """Trainable parameters of the whole ensemble (buffers such as the input statistics excluded)."""
        if getattr(self, "_n_params", None) is None:
            from hitasr.routers import HitASRRouter
            self._n_params = HitASRRouter(self.dims, self.members, **self.arch).n_params() * len(self.states)
        return self._n_params

    def probs(self, frames, lengths):
        """`(B, K)` routing distribution from `{member: (B, T, D)}` frames and `{member: (B,)}` lengths."""
        models = self.models(next(iter(frames.values())).device)
        use_amp = self.amp and self._device.type == "cuda"
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for m in models:
                out.append(torch.softmax(m(frames, lengths).float(), -1))
        return torch.stack(out).mean(0)

    def choose(self, frames, lengths):
        """Member index per clip (into `members`)."""
        return self.probs(frames, lengths).argmax(-1).cpu().numpy()

    def gflops(self, frames, lengths):
        """GFLOPs of one forward pass of the whole ensemble on this batch, per clip (`None` if torch cannot count)."""
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return None
        with FlopCounterMode(display=False) as fc:
            self.probs(frames, lengths)
        return fc.get_total_flops() / 1e9 / len(next(iter(lengths.values())))

    # ------------------------------------------------------------------ storage --

    def recipe(self):
        return {"members": list(self.members), "dims": self.dims, "params": self.params, "arch": self.arch,
                "max_frames": self.max_frames, "amp": self.amp, "n_fits": len(self.states),
                "n_params": self.n_params(), **self.meta}

    def save(self, path):
        """`<path>.pt` (the weights and the recipe) and `<path>.json` (the recipe alone). Returns both paths."""
        path = Path(path)
        pt, js = path.with_suffix(".pt"), path.with_suffix(".json")
        pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"recipe": self.recipe(), "states": self.states}, pt)
        js.write_text(json.dumps(self.recipe(), indent=2, default=str))
        return pt, js

    @classmethod
    def load(cls, path):
        blob = torch.load(Path(path).with_suffix(".pt"), map_location="cpu", weights_only=False)
        r = blob["recipe"]
        meta = {k: v for k, v in r.items() if k not in ("members", "dims", "params", "arch", "max_frames", "amp",
                                                        "n_fits", "n_params")}
        return cls(r["members"], r["dims"], r["params"], r["arch"], r["max_frames"], blob["states"], amp=r["amp"],
                   meta=meta)

    @staticmethod
    def hub_path(spec, name):
        return spec.results_path(f"deploy/{name}")[:-len(".json")]

    def push(self, hub, name, message=None):
        """Upload to `results/<corpus>/deploy/<name>.{pt,json}` of the records repository."""
        base = self.hub_path(hub.spec, name)
        pt, js = self.save(Path(base.replace("/", "_")))
        hub.push_files({f"{base}.pt": pt, f"{base}.json": js}, message or f"deploy: {name} on {hub.spec.name}")
        return base

    @classmethod
    def pull(cls, hub, name):
        """The router pushed under `name`, or `None`."""
        local = hub.pull_file(f"{cls.hub_path(hub.spec, name)}.pt", verbose=False)
        return None if local is None else cls.load(local)


def fit_deployed(store, params, rows, seed=42, ctx=None, meta=None):
    """Fit `MODELS["hit_asr"]` with `params` on `rows` of `store` — what a fold of the main table fits — and keep it."""
    from hitasr.arms import MODELS
    t0 = time.perf_counter()
    arm = MODELS["hit_asr"].build(dict(params), store, seed=seed, ctx=ctx)
    arm.fit(store, np.asarray(rows, dtype=int))
    meta = {"corpus": store.spec.name, "n_rows": int(len(rows)), "rows_digest": rows_digest(rows), "seed": int(seed),
            "fit_seconds": time.perf_counter() - t0, **(meta or {})}
    return DeployedRouter.from_arm(arm, store, params, meta=meta)


def main_fold(record, n_rows, fold=0, repetition=0):
    """One fold of a main run, rebuilt from its record (`main_<corpus>.json`) without its search.

    Returns `{seed, fold, fit_rows, eval_rows, cv_key, arm_key}`: the rows that fold's router was fitted on and
    scored on — the evaluation rows of the hold-out partition, halved by the repetition's seed exactly as `run_cv`
    halves them, **in the order `run_cv` hands them to the arm** (the trainer cuts its early-stopping rows off that
    order, so a sorted copy trains a different model) — and the keys its fold file is cached under in
    `results/<corpus>/runlog/main_<corpus>/`, so the main run's own picks for those rows can be read back
    (`RunLog.per_arm(..., strict=True)`).
    """
    from hitasr.crossval import ACC_TOLERANCES, METRICS
    from hitasr.tuning import cv_arm_key
    from labkit.cv import FiveByTwoSplit, random_partition
    cfg = record["config"]
    seed = int(record["seeds"]["seeds"][repetition]) if record.get("seeds") else int(cfg["cv_seeds"][repetition])
    part = random_partition(n_rows, cfg["holdout_frac"], cfg["inner_val_frac"], cfg["partition_seed"])
    rows = part["eval"]
    _, tr, ev = FiveByTwoSplit().repeat_folds(len(rows), seed)[fold]
    partition = {"dataset": cfg["dataset"], "members": list(record["members"]), "n_rows": int(n_rows),
                 "holdout_frac": cfg["holdout_frac"], "partition_seed": cfg["partition_seed"]}
    cv_key = {"partition": partition, "n_eval": int(len(rows)), "random_seed": cfg["cv_random_seed"],
              "member_arms": True, "tolerances": list(ACC_TOLERANCES), "metrics": sorted(METRICS),
              "epochs": cfg["epochs"]}
    arm_key = cv_arm_key(None, "hit_asr", record["params"]["hit_asr"], (cfg.get("arm_versions") or {}).get("hit_asr"))
    return {"seed": seed, "fold": int(fold), "fit_rows": rows[tr], "eval_rows": rows[ev], "cv_key": cv_key,
            "arm_key": arm_key}


def rows_digest(rows):
    """An order-sensitive fingerprint of the rows a router was fitted on (the order decides its early-stopping cut)."""
    import hashlib
    return hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:12]


def router_frames(encoded, members, max_frames=None, device=None):
    """`{member: Encoded}` -> the router's `(frames, lengths)`: fp16, cut to `max_frames`, zero past each length —
    what `FrameSet.batch` hands the router from the stored frames."""
    frames, lengths = {}, {}
    for m in members:
        e = encoded[m]
        lens = e.lengths.to(torch.long).cpu()
        if max_frames is not None:
            lens = lens.clamp(max=int(max_frames))
        T = max(int(lens.max()), 1)
        h = e.hidden
        dev = torch.device(device) if device is not None else h.device
        out = torch.zeros((h.shape[0], T, h.shape[-1]), dtype=torch.float16, device=dev)
        for j, n in enumerate(lens.tolist()):
            out[j, :n] = h[j, :n].to(dev, torch.float16)
        frames[m], lengths[m] = out, lens.to(dev)
    return frames, lengths


def frame_agreement(live, stored, n):
    """How close frames computed now are to the stored ones for one clip: cosine similarity of the pooled vectors,
    the median per-frame cosine, and the relative length difference. `live` `(T, D)`, `stored` `(T', D)`."""
    a, b = live.float(), stored.float()
    T = min(a.shape[0], b.shape[0], int(n))
    cos = torch.nn.functional.cosine_similarity(a[:T], b[:T], dim=-1)
    pooled = torch.nn.functional.cosine_similarity(a[:T].mean(0), b[:T].mean(0), dim=0)
    return {"pooled_cos": float(pooled), "frame_cos_median": float(cos.median()),
            "len_live": int(a.shape[0]), "len_stored": int(b.shape[0])}


# ---------------------------------------------------------------------------------------------- the systems

class RouterOutput:
    """What a system returns for a batch of clips: transcripts, choices and the clock per stage (seconds)."""

    def __init__(self, text, chosen, seconds, n):
        self.text, self.chosen, self.seconds, self.n = list(text), list(chosen), dict(seconds), int(n)

    def __repr__(self):
        ms = {k: f"{v * 1e3 / max(self.n, 1):.1f}" for k, v in self.seconds.items()}
        return f"RouterOutput({self.n} clip(s), ms/clip {ms})"


class _System:
    name = None

    def extractors_used(self):
        return {}

    def transcribe(self, arrays):
        """One clip at a time (online); the stage clocks are summed over the clips."""
        text, chosen, secs = [], [], {}
        for a in arrays:
            t, c, s = self._one(a)
            text.append(t); chosen.append(c)
            for k, v in s.items():
                secs[k] = secs.get(k, 0.0) + v
        return RouterOutput(text, chosen, secs, len(arrays))


class HitASRSystem(_System):
    """HIT-ASR: every member's encoder, the router, the chosen member's decoder (reusing its encoder pass)."""

    name = "hit_asr"

    def __init__(self, router, extractors, device=None):
        self.router, self.extractors = router, dict(extractors)
        missing = [m for m in router.members if m not in self.extractors]
        if missing:
            raise KeyError(f"no extractor for {missing}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def extractors_used(self):
        return {m: self.extractors[m] for m in self.router.members}

    def _one(self, audio):
        r = self.router
        _sync(); t0 = time.perf_counter()
        enc = {m: self.extractors[m].encode_only([audio]) for m in r.members}
        _sync(); t1 = time.perf_counter()
        frames, lengths = router_frames(enc, r.members, r.max_frames, self.device)
        k = int(r.choose(frames, lengths)[0])
        _sync(); t2 = time.perf_counter()
        m = r.members[k]
        text = self.extractors[m].decode_encoded(enc[m])[0]
        _sync(); t3 = time.perf_counter()
        return text, m, {"encode": t1 - t0, "route": t2 - t1, "decode": t3 - t2}


class SingleExpert(_System):
    """One expert's own pipeline, as extraction runs it."""

    def __init__(self, extractor):
        self.ex, self.name = extractor, extractor.name

    def extractors_used(self):
        return {self.ex.name: self.ex}

    def _one(self, audio):
        _sync(); t0 = time.perf_counter()
        text = self.ex.transcribe([audio])[0]
        _sync(); t1 = time.perf_counter()
        return text, self.ex.name, {"pipeline": t1 - t0}


class DecodeAll(_System):
    """Every expert's full pipeline, then a ROVER vote over the transcripts (`weights`: one per member, uniform by
    default): what every transcript-fusion method and the oracle pay before their own combination step."""

    name = "decode_all"

    def __init__(self, extractors, members, weights=None, null_penalty=1.0):
        self.extractors, self.members = dict(extractors), tuple(members)
        self.weights = np.ones(len(self.members)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.null_penalty = float(null_penalty)

    def extractors_used(self):
        return {m: self.extractors[m] for m in self.members}

    def _one(self, audio):
        _sync(); t0 = time.perf_counter()
        hyps = [str(self.extractors[m].transcribe([audio])[0] or "") for m in self.members]
        _sync(); t1 = time.perf_counter()
        text = vote(build_network(hyps), self.weights, self.null_penalty)
        t2 = time.perf_counter()
        return text, "rover", {"pipelines": t1 - t0, "fuse": t2 - t1}


# ---------------------------------------------------------------------------------------------- measurement

def weights_gb(extractor):
    """GB of an expert's weights (every parameter and buffer of its model)."""
    model = getattr(extractor, "model", None)
    if model is None:
        return float("nan")
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())) / 1e9


def _peak_gb(fn):
    """`(result, peak GB allocated above the level at the start)` on CUDA; NaN elsewhere."""
    if not torch.cuda.is_available():
        return fn(), float("nan")
    _sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    _sync()
    return out, (torch.cuda.max_memory_allocated() - base) / 1e9


def time_experts(extractors, arrays, warmup=2, sampling_rate=16000):
    """Each expert's pipeline on the same clips, one at a time: `encode_only`, `decode_encoded` and the full
    pipeline, in ms per clip, with the audio seconds, the pipeline's transcript and whether the reused-encoder decode
    gave that same transcript. One row per (expert, clip); the first `warmup` clips are run and dropped."""
    rows = []
    for name, ex in extractors.items():
        ex.load()
        for i, a in enumerate(arrays):
            _sync(); t0 = time.perf_counter()
            enc = ex.encode_only([a])
            _sync(); t1 = time.perf_counter()
            hyp_split = ex.decode_encoded(enc)[0]
            _sync(); t2 = time.perf_counter()
            (hyp_full,), peak = _peak_gb(lambda: ex.transcribe([a]))
            _sync(); t3 = time.perf_counter()
            if i < warmup:
                continue
            rows.append({"expert": name, "clip": i - warmup, "audio_seconds": len(a) / sampling_rate,
                         "encode_ms": 1e3 * (t1 - t0), "decode_ms": 1e3 * (t2 - t1), "pipeline_ms": 1e3 * (t3 - t2),
                         "frames": int(enc.lengths[0]), "reuses_encoder": bool(ex.reuses_encoder),
                         "same_text": (hyp_split or "").strip() == (hyp_full or "").strip(), "peak_gb": peak,
                         "text": hyp_full or ""})
    return pd.DataFrame(rows)


def corpus_wer(scorer, refs, hyps):
    """Corpus WER of `hyps` against `refs` with a `WerScorer` (errors over reference words)."""
    cols = scorer.score(list(refs), [h or "" for h in hyps])
    return (sum(cols["sub"]) + sum(cols["dele"]) + sum(cols["ins"])) / max(sum(cols["nref"]), 1)


def compare_systems(systems, arrays, refs=None, scorer=None, warmup=2, sampling_rate=16000):
    """Run every system end to end on the same clips. One row per system: ms per clip per stage and in total, the
    real-time factor, peak GPU memory above the resident weights, the choice distribution, and the WER on the clips
    when `refs` are given (`scorer`: a `WerScorer`)."""
    audio_s = sum(len(a) for a in arrays[warmup:]) / sampling_rate
    rows, outputs = [], {}
    for label, sys_ in systems.items():
        sys_.transcribe(arrays[:warmup])
        out, peak = _peak_gb(lambda s=sys_: s.transcribe(arrays[warmup:]))
        n = max(out.n, 1)
        row = {"system": label, **{f"{k}_ms": 1e3 * v / n for k, v in out.seconds.items()},
               "total_ms": 1e3 * sum(out.seconds.values()) / n,
               "rtf": sum(out.seconds.values()) / max(audio_s, 1e-9), "peak_gb": peak,
               "weights_gb": sum(weights_gb(e) for e in sys_.extractors_used().values())}
        if isinstance(sys_, HitASRSystem):
            for m in sys_.router.members:
                row[f"share_{m}"] = out.chosen.count(m) / n
        if refs is not None and scorer is not None:
            row["wer"] = corpus_wer(scorer, refs[warmup:], out.text)
        rows.append(row)
        outputs[label] = out
    df = pd.DataFrame(rows)
    df.attrs["audio_seconds"], df.attrs["n_clips"] = audio_s, len(arrays) - warmup
    return df, outputs
