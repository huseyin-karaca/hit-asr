__all__ = ['synthetic_spec', 'wer_table', 'synthetic_store', 'synthetic_experiment', 'run_synthetic', 'Synthetic']

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from hitasr.core import DatasetSpec, register_dataset, use_dataset
from hitasr.frames import FrameWriter, write_manifest
from hitasr.labels import LabelStore
from hitasr.store import RouterStore


def synthetic_spec(name="synthetic"):
    return DatasetSpec(name=name, source="synthetic", splits=("train", "test"), fit_splits=("train",),
                       eval_splits=("test",), base_config=f"{name}_base", source_prefix=f"{name}_",
                       config_prefix=f"{name}_", note="regime-switch benchmark, generated")


def wer_table(n_regimes, K, rng, best=0.05, worst=0.35, noise=0.02):
    """`(R, R, K)`: the WER of expert `k` on a clip whose halves are regimes `(r1, r2)`.

    No expert is uniformly best, the best expert depends on both halves, and
    the average regime does not determine it — the construction the manuscript
    describes.
    """
    table = np.full((n_regimes, n_regimes, K), worst, dtype=np.float64)
    for r1 in range(n_regimes):
        for r2 in range(n_regimes):
            table[r1, r2, (r1 + 2 * r2) % K] = best
    return np.clip(table + rng.normal(0, noise, table.shape), 0.01, 1.0)


def synthetic_store(n=10000, K=3, dims=None, T=128, n_regimes=4, noise_std=0.5, wer_noise=0.02,
                    seed=0, root=None, device=None, n_words=(8, 40), name="synthetic",
                    frame_rates=None, verbose=True):
    """Generate a regime-switch corpus, write it through `FrameWriter`, return an opened `RouterStore`.

    `T` is a fixed frame count (the manuscript's 128) or a `(lo, hi)` range.
    `dims` are the experts' widths (default 1024 / 512 / 1024 cycling);
    `frame_rates` subsample expert `k`'s frames by `frame_rates[k]` to
    simulate heterogeneous encoders.
    """
    rng = np.random.default_rng(seed)
    root = Path(root or tempfile.mkdtemp(prefix="hitasr_synth_"))
    spec = register_dataset(synthetic_spec(name))
    use_dataset(spec, verbose=False)
    members = tuple(f"expert{k}" for k in range(K))
    dims = tuple(dims or [(1024, 512, 1024)[k % 3] for k in range(K)])
    strides = tuple(frame_rates or [1] * K)
    d_regime = 32
    centers = rng.normal(0, 1, (n_regimes, d_regime))
    proj = [rng.normal(0, 1 / np.sqrt(d_regime), (d_regime, d)) for d in dims]
    table = wer_table(n_regimes, K, rng)
    n_train = int(n * 0.8)
    splits = {"train": n_train, "test": n - n_train}
    ids, split_col, nref_all, E_all, n_frames = [], [], [], [], {m: [] for m in members}
    pooled = {m: [] for m in members}
    meta = []
    for split, n_split in splits.items():
        writers = {m: FrameWriter(root / spec.frames_config(m) / split, d=d) for m, d in zip(members, dims)}
        for i in range(n_split):
            uid = f"{split}-{i:06d}"
            t = int(T) if np.isscalar(T) else int(rng.integers(T[0], T[1] + 1))
            rho = int(rng.integers(t // 4, 3 * t // 4))
            r1 = int(rng.integers(0, n_regimes)); r2 = (r1 + 1 + int(rng.integers(0, n_regimes - 1))) % n_regimes
            reg = np.concatenate([np.full(rho, r1), np.full(t - rho, r2)])
            z = centers[reg] + rng.normal(0, noise_std, (t, d_regime))
            nwords = int(rng.integers(n_words[0], n_words[1] + 1))
            wer = np.clip(table[r1, r2] + rng.normal(0, wer_noise, K), 0, 1)
            errs = np.round(wer * nwords).astype(np.int64)
            for k, m in enumerate(members):
                fr = (z @ proj[k]).astype(np.float16)[::strides[k]]
                writers[m].add([uid], fr[None], [len(fr)])
                n_frames[m].append(len(fr)); pooled[m].append(fr.astype(np.float32).mean(0))
            ids.append((split, uid)); split_col.append(split); nref_all.append(nwords); E_all.append(errs)
            meta.append({"split": split, "id": uid, "r1": r1, "r2": r2, "rho": rho, "T": t})
        for m in members:
            writers[m].close()
    for k, (m, d) in enumerate(zip(members, dims)):
        write_manifest(root / spec.frames_config(m), spec, m, d, 50.0 / strides[k], -1, list(splits),
                       extra={"synthetic": True, "seed": seed})
    E = np.stack(E_all)
    ls = LabelStore.__new__(LabelStore)
    ls.spec, ls.requested, ls.adopt, ls.cache_dir = spec, members, False, root / "labels"
    ls._reset()
    ls.experts, ls.source = list(members), {m: "synthetic" for m in members}
    ls.ids, ls.split, ls.nref = ids, np.array(split_col), np.array(nref_all, dtype=np.int64)
    ls.ref_text = [""] * len(ids)
    for k, m in enumerate(members):
        ls.err[m] = E[:, k].astype(np.float64); ls.err_raw[m] = ls.err[m]
        ls.n_frames[m] = np.asarray(n_frames[m]); ls.texts[m] = ls.texts_norm[m] = None
        ls._pooled[m] = np.stack(pooled[m]); ls.frame_rate_hz[m] = 50.0 / strides[k]
    store = RouterStore(ls, members, frames_root=root).open(device=device, verbose=verbose)
    store.meta = pd.DataFrame(meta)
    store.wer_table = table
    if verbose:
        print(f"synthetic: {n:,} clips, K={K}, dims={dims}, T={T}, R={n_regimes}, "
              f"oracle {E.min(1).sum() / ls.nref.sum():.4f}  bsm {min(E.sum(0) / ls.nref.sum()):.4f}")
    return store

def synthetic_experiment(store, arms, seed=42, verbose=True):
    """`{arm: factory}` -> a frame with WER, selection accuracy, the oracle gap and the share of the best single
    expert's gap to the oracle closed, on the test split. Rows: random, the best single expert (best on the training
    split), every arm (with its fit seconds and parameter count), the oracle."""
    import time
    tr = np.where(store.labels.split == "train")[0]
    te = np.where(store.labels.split == "test")[0]
    E, n = store.E, store.nref
    oracle = E[te].min(1).sum() / n[te].sum()
    best = E[te].min(1)
    k_bsm = int(np.argmin(E[tr].sum(0)))
    rows = [{"arm": "random", "wer": float(E[te].mean(1).sum() / n[te].sum()), "sel_acc": 1 / E.shape[1]},
            {"arm": "bsm", "wer": float(E[te, k_bsm].sum() / n[te].sum()), "sel_acc": float((E[te, k_bsm] <= best).mean())}]
    for name, factory in arms.items():
        arm = factory()
        t0 = time.perf_counter()
        arm.fit(store, tr)
        secs = time.perf_counter() - t0
        choice = np.asarray(arm.predict(store, te), dtype=int)
        picked = E[te, choice]
        n_params = arm.n_params() if hasattr(arm, "n_params") else None
        rows.append({"arm": name, "wer": float(picked.sum() / n[te].sum()), "sel_acc": float((picked <= best).mean()),
                     "fit_seconds": secs, "n_params": n_params})
    rows.append({"arm": "oracle", "wer": float(oracle), "sel_acc": 1.0})
    out = pd.DataFrame(rows)
    out["oracle_gap"] = out["wer"] - oracle
    bsm = float(out.loc[out["arm"] == "bsm", "wer"].iloc[0])
    out["gap_closed"] = (bsm - out["wer"]) / (bsm - oracle) if bsm > oracle else np.nan
    if verbose:
        print(out.round(4).to_string(index=False))
    return out


def run_synthetic(cfg, device="cuda"):
    """The experiment of `notebooks/synthetic`: for each K and seed a fresh regime-switch corpus, MLP-pool and HIT-ASR
    at their fixed configurations (one fit each), and the router's clock at each K. Returns the record."""
    import shutil

    import torch

    from hitasr.arms import MODELS
    from hitasr.latency import time_router
    from labkit.search import space_defaults

    hit = {**space_defaults(MODELS["hit_asr"].space()), **cfg.hit_asr}
    mlp = {**space_defaults(MODELS["mlp_pool"].space()), **cfg.mlp_pool}
    ctx = {"epochs": cfg.epochs, "device": device}
    rows, latency = [], []
    for K in cfg.ks:
        for seed in cfg.seeds:
            root = tempfile.mkdtemp(prefix="hitasr_synth_")          # K=10 writes ~22 GB of frames: one at a time
            store = synthetic_store(n=cfg.n, K=K, T=cfg.T, n_regimes=cfg.n_regimes, seed=seed, device=device,
                                    root=root, verbose=(seed == cfg.seeds[0]))
            arms = {"mlp_pool": lambda: MODELS["mlp_pool"].build(mlp, store, seed=seed, ctx=ctx),
                    "hit_asr": lambda: MODELS["hit_asr"].build(hit, store, seed=seed, ctx=ctx)}
            out = synthetic_experiment(store, arms, seed=seed, verbose=True)
            out.insert(0, "seed", seed)
            out.insert(0, "K", K)
            rows.append(out)
            if seed == cfg.seeds[0]:                # the router's clock at this K: a fit on a slice is enough
                arm = arms["hit_asr"]()
                arm.fit(store, np.where(store.labels.split == "train")[0][:2000])
                t = time_router(arm.model, arm._batcher(store), np.arange(120), n=100, batch_size=1)
                latency.append({"K": K, "router_ms": t["router_ms"], "router_ms_sd": t["router_ms_sd"],
                                "params_M": t["n_params"] / 1e6})
            del store, arms
            shutil.rmtree(root, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    from dataclasses import asdict
    import json
    return {"config": asdict(cfg), "hit_asr": hit, "mlp_pool": mlp, "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "results": json.loads(pd.concat(rows, ignore_index=True).to_json(orient="records")), "latency": latency}


class Synthetic:
    """The synthetic notebook. Level 1 reads `results/synthetic.json` from the Hub; level 2 runs the experiment
    (a GPU, ~30 min) and writes `synthetic.json` to the working directory."""

    ORDER = ("random", "bsm", "mlp_pool", "hit_asr", "oracle")
    NAMES = {"random": "Random", "bsm": "Best single expert", "mlp_pool": "MLP-pool", "hit_asr": "HIT-ASR",
             "oracle": "Oracle"}

    def __init__(self, cfg, level=1, device="cuda"):
        import json

        from hitasr.records import load_record
        from labkit.env import set_determinism
        if level not in (1, 2):
            raise ValueError("level is 1 or 2")
        set_determinism(42)
        self.cfg, self.level = cfg, level
        if level >= 2:
            import torch
            self.record = run_synthetic(cfg, device if torch.cuda.is_available() else "cpu")
            Path("synthetic.json").write_text(json.dumps(self.record, indent=1, default=str))
        else:
            self.record = load_record("synthetic")

    def results(self):
        """Per K: WER, selection accuracy, the gap to the oracle and the share of the best single expert's gap to the
        oracle closed — mean ± sd over the seeds."""
        from labkit.pretty import Table
        r, c = pd.DataFrame(self.record["results"]), self.record["config"]
        g = r.groupby(["K", "arm"], sort=False)
        s = pd.DataFrame({"wer": g["wer"].mean(), "wer_sd": g["wer"].std(), "sel_acc": g["sel_acc"].mean(),
                          "sel_acc_sd": g["sel_acc"].std(), "oracle_gap": g["oracle_gap"].mean(),
                          "gap_closed": g["gap_closed"].mean()}).reset_index()
        s["order"] = s["arm"].map(self.ORDER.index)
        s = s.sort_values(["K", "order"]).drop(columns="order")
        s["arm"] = s["arm"].map(self.NAMES)
        self.summary = s
        return Table(s.round(4), title="Synthetic regime switch: WER, selection accuracy, oracle gap, gap closed",
                     group="K", highlight=lambda row: row["arm"] == "HIT-ASR",
                     caption=f"mean ± sd over seeds {tuple(c['seeds'])}; N={c['n']}, T={c['T']}, {c['n_regimes']} "
                             "regimes. A feasibility check, not evidence under real acoustics.")

    def latency(self):
        """The router's latency and size against K (batch 1)."""
        from labkit.pretty import Table
        return Table(pd.DataFrame(self.record["latency"]).round(3),
                     title=f"The router against K (batch 1, {self.record['gpu']})")
