"""The cost measurement on raw audio (`notebooks/timing`): each corpus's trio, one clip at a time on one GPU.

Three systems run end to end on the same clips — the best single expert (its encoder and decoder), HIT-ASR (every
member's encoder, the router, the chosen member's decoder, whose encoder pass is reused) and decode-all + ROVER (every
encoder and decoder, then the vote) — and each expert's pipeline is split into its encoder and its decoder. The router
is a fold of Table 3, fitted on that fold's training half; the clips are drawn from the half it was scored on, and its
choices on raw audio are checked against the choices Table 3 counted for the same clips.

    t = Timing(TIMING, level=1)      # level 1: the records on the Hub; level 2: measure (a GPU)
    t.experts(); t.systems(); t.summary(); t.scaling()
"""

__all__ = ['Timing', 'measure']

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd


class _Experts:
    """The experts, loaded once and shared across corpora."""

    def __init__(self, device):
        from hitasr.models.registry import load_expert_classes
        self.classes, self.device, self.loaded = load_expert_classes(), device, {}

    def __call__(self, name):
        if name not in self.loaded:
            self.loaded[name] = self.classes[name](device=self.device).load()
        return self.loaded[name]

    def keep_only(self, members):
        """Unload the experts another corpus used and this one does not."""
        from labkit.env import empty_cache
        for m in [m for m in self.loaded if m not in members]:
            del self.loaded[m]
        empty_cache(report=False)


def _clip_audio(labels, rows):
    """`(arrays, refs)` of store rows, from the corpus's audio base."""
    from hitasr.hub import load_base
    splits = sorted({labels.ids[r][0] for r in rows})
    ds = load_base(splits=splits)
    pos = {s: {i: k for k, i in enumerate(ds[s]["id"])} for s in splits}
    arrays, refs = [], []
    for r in rows:
        s, i = labels.ids[r]
        item = ds[s][pos[s][i]]
        arrays.append(np.asarray(item["audio"]["array"], dtype=np.float32))
        refs.append(item["text"])
    return arrays, refs


def _throughput(experts, members, arrays, batch):
    """Each member's pipeline in batches of `batch` (clips sorted by length): seconds per batch."""
    import torch
    order = np.argsort([len(a) for a in arrays])
    rows = []
    for m in members:
        ex = experts(m)
        for s in range(0, len(order), batch):
            b = [arrays[i] for i in order[s:s + batch]]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ex.transcribe(b)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            if s:                                                # the first batch warms the kernels up
                rows.append({"expert": m, "n": len(b), "audio_seconds": sum(len(a) for a in b) / 16000,
                             "seconds": t1 - t0})
    return rows


def measure(corpus, cfg, experts, device="cuda"):
    """One corpus's timing record: the experts, the three systems and the router's agreement with Table 3."""
    import torch

    from hitasr.core import use_dataset
    from hitasr.deploy import (DecodeAll, DeployedRouter, HitASRSystem, compare_systems, corpus_wer, fit_deployed,
                               frame_agreement, main_fold, router_frames, rows_digest, time_experts, weights_gb)
    from hitasr.hub import HitHub
    from hitasr.labels import LabelStore
    from hitasr.runlog import RunLog
    from hitasr.scoring import WerScorer
    from hitasr.store import RouterStore
    from labkit.env import empty_cache, set_determinism

    spec = use_dataset(corpus, verbose=False)
    hub = HitHub(spec=spec)
    main = hub.load_results(f"main_{corpus}")
    members, params = tuple(main["members"]), main["params"]["hit_asr"]
    experts.keep_only(members)
    labels = LabelStore(experts=members).load()
    fold = main_fold(main, labels.n, fold=cfg.router_fold)
    log = RunLog(f"main_{corpus}", spec=spec, sync=True, readonly=True, strict=True, verbose=False)
    _, picks = log.per_arm("cv5x2", [fold["seed"]], {"hit_asr": None}, None, key=fold["cv_key"],
                           arm_keys={"hit_asr": fold["arm_key"]})
    picks = picks[(picks["model"] == "hit_asr") & (picks["fold"] == fold["fold"])].set_index("row")["choice"]
    assert np.array_equal(np.sort(picks.index.to_numpy()), np.sort(fold["eval_rows"])), "the fold does not match"
    # the timed clips: a random sample of the scored half; the warm-up clips: the longest others, so every buffer and
    # kernel is at its largest before the clock starts
    timed = np.random.default_rng(cfg.clip_seed).choice(fold["eval_rows"], cfg.n_clips, replace=False)
    rest = np.setdiff1d(fold["eval_rows"], timed)
    warm = rest[np.argsort(labels.seconds()[rest])[-cfg.warmup:]] if cfg.warmup else rest[:0]
    clips = np.concatenate([warm, timed])
    arrays, refs = _clip_audio(labels, clips)
    print(f"\n=== {corpus}: {', '.join(members)}; router = fold {fold['fold']} of repetition {fold['seed']} "
          f"({len(fold['fit_rows']):,} fit rows); {len(clips)} clips, {sum(map(len, arrays)) / 16000 / 60:.1f} min")

    name = f"hit_asr_seed{fold['seed']}_fold{fold['fold']}"
    router = DeployedRouter.pull(hub, name)
    if router is not None and (router.params != params or router.meta.get("rows_digest") != rows_digest(fold["fit_rows"])):
        print(f"  {name} on the Hub is not this fold's router (parameters or rows) — refitted")
        router = None
    store = None
    if (router is None and cfg.fit_router) or cfg.n_frame_checks:
        store = RouterStore(labels, members).open(device=device)
    if router is None and cfg.fit_router:
        mc = main["config"]
        set_determinism(mc["model_seed"])
        router = fit_deployed(store, params, fold["fit_rows"], seed=mc["model_seed"],
                              ctx={"epochs": mc["epochs"], "device": device, "nthread": mc.get("nthread", 4)},
                              meta={"fold": fold["fold"], "repeat_seed": fold["seed"]})
        print(f"  {router}: fitted in {router.meta['fit_seconds'] / 60:.1f} min")

    frames = []
    if store is not None:                        # the frames computed now against the stored ones
        for m in members:
            for r, a in list(zip(clips, arrays))[cfg.warmup:cfg.warmup + cfg.n_frame_checks]:
                enc = experts(m).encode_only([a])
                n = int(enc.lengths[0])
                st = store.frames[m].get(int(r))
                stored = st.cpu() if torch.is_tensor(st) else torch.as_tensor(np.asarray(st))
                frames.append({"expert": m, "row": int(r), **frame_agreement(enc.hidden[0, :n].cpu(), stored, n)})
        del store
        empty_cache(report=False)

    scorer = WerScorer()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    per = time_experts({m: experts(m) for m in members}, arrays, warmup=cfg.warmup)
    wer_single = {m: corpus_wer(scorer, refs[cfg.warmup:], per[per["expert"] == m].sort_values("clip")["text"])
                  for m in members}
    out = {"dataset": corpus, "members": list(members), "gpu": gpu, "config": asdict(cfg),
           "fold": {"seed": fold["seed"], "fold": fold["fold"], "n_fit": int(len(fold["fit_rows"])),
                    "n_scored": int(len(fold["eval_rows"]))},
           "clip_rows": [int(r) for r in clips], "audio_seconds": float(sum(map(len, arrays[cfg.warmup:])) / 16000),
           "corpus_hours": float(np.nansum(labels.seconds()) / 3600), "bsm": main["bsm"],
           "weights_gb": {m: weights_gb(experts(m)) for m in members},
           "per_expert": json.loads(per.drop(columns=["text"]).to_json(orient="records")), "wer_single": wer_single,
           "frames": frames, "throughput": _throughput(experts, members, arrays[cfg.warmup:], cfg.throughput_batch)}
    if router is not None:
        systems = {"HIT-ASR": HitASRSystem(router, {m: experts(m) for m in members}, device),
                   "Decode all + ROVER": DecodeAll({m: experts(m) for m in members}, members)}
        table, outs = compare_systems(systems, arrays, refs=refs, scorer=scorer, warmup=cfg.warmup)
        chosen = [members.index(c) for c in outs["HIT-ASR"].chosen]
        recorded = picks.loc[clips[cfg.warmup:]].to_numpy()
        enc = {m: experts(m).encode_only([arrays[cfg.warmup]]) for m in members}
        fr, ln = router_frames(enc, members, router.max_frames, device)
        out.update(router=router.recipe(), router_name=name, systems=json.loads(table.to_json(orient="records")),
                   agreement=float(np.mean(np.asarray(chosen) == recorded)), router_gflops=router.gflops(fr, ln),
                   choices=chosen, recorded_choices=[int(c) for c in recorded])
    return out


class Timing:
    """The timing notebook. Level 1 reads each corpus's `timing_<corpus>.json` from the Hub; level 2 measures (a GPU
    with room for the experts, ~15 min per corpus) and writes the records to the working directory."""

    def __init__(self, cfg, level=1, device="cuda"):
        from hitasr.models.registry import EXPERT_LABELS
        from hitasr.records import load_record
        from labkit.hub import set_read_only
        if level not in (1, 2):
            raise ValueError("level is 1 or 2")
        set_read_only(True)
        self.cfg, self.level, self.device, self.labels = cfg, level, device, dict(EXPERT_LABELS)
        self.records = {}
        if level >= 2:
            experts = _Experts(device)
            for c in cfg.corpora:
                self.records[c] = measure(c, cfg, experts, device)
                with open(f"timing_{c}.json", "w") as fh:
                    json.dump(self.records[c], fh, indent=1, default=str)
        else:
            self.records = {c: load_record(f"timing_{c}", c) for c in cfg.corpora}
        self.expert_tables, self.system_tables = {}, {}

    def experts(self):
        """Per expert: encoder, decoder and the decoder's share of the pipeline (ms per clip, batch 1); `same_text` is
        the share of clips on which decoding from the reused encoder pass gave the pipeline's own transcript,
        `label GPU-h` the one-off labelling pass over the corpus, `frame_cos` the median cosine between frames computed
        from the audio and the stored ones."""
        from hitasr.latency import expert_costs
        from labkit.pretty import Table
        out = []
        for c, rec in self.records.items():
            per = pd.DataFrame(rec["per_expert"])
            e = expert_costs(per, throughput=rec.get("throughput"), hours=rec.get("corpus_hours"))
            e["wer"] = pd.Series(rec["wer_single"])
            e["weights_gb"] = pd.Series(rec["weights_gb"])
            if rec.get("frames"):
                e["frame_cos"] = pd.DataFrame(rec["frames"]).groupby("expert")["frame_cos_median"].median()
            self.expert_tables[c] = e
            out.append(Table(e.rename(index=self.labels).reset_index().round(4),
                             title=f"{c} — per expert, ms per clip ({len(per) // max(len(rec['members']), 1)} clips, "
                                   f"{rec['gpu']})"))
        return out

    def systems(self):
        """The three systems end to end on the same clips; `agreement` is the share of clips on which the router,
        reading frames computed from the audio, chose what Table 3's fold chose from the stored frames."""
        from hitasr.latency import system_costs
        from labkit.pretty import Table
        if not self.expert_tables:
            self.experts()
        out = []
        for c, rec in self.records.items():
            if "systems" not in rec:
                continue
            t = system_costs(self.expert_tables[c], pd.DataFrame(rec["systems"]), rec["members"], rec["bsm"],
                             labels=self.labels)
            self.system_tables[c] = t
            out.append(Table(t.round(4), title=f"{c} — systems, ms per clip on raw audio",
                             caption=f"router: {rec['router']['n_params'] / 1e6:.2f}M parameters in "
                                     f"{rec['router']['n_fits']} fits, {rec['router_gflops']:.3f} GFLOPs per clip; "
                                     f"agreement with Table 3's choices {rec['agreement']:.1%}",
                             highlight=lambda r: r["system"] == "HIT-ASR"))
        return out

    def summary(self):
        """Across corpora: HIT-ASR against the best single expert and decode-all, ms per clip at batch 1."""
        from labkit.pretty import Table
        if not self.system_tables:
            self.systems()
        rows = []
        for c, t in self.system_tables.items():
            rec, e = self.records[c], self.expert_tables[c]
            hit = t[t["kind"] == "hit_asr"].iloc[0]
            alls = t[t["kind"] == "decode_all"].iloc[0]
            best = t[t["system"].str.endswith("(best single)")].iloc[0]
            rows.append({"corpus": c, "clips": rec["config"]["n_clips"], "encoders_ms": hit["encode_ms"],
                         "router_ms": hit["route_ms"], "decoder_ms": hit["decode_ms"], "hit_ms": hit["total_ms"],
                         "best_single_ms": best["total_ms"], "decode_all_ms": alls["total_ms"],
                         "hit_rtf": hit["rtf"], "best_rtf": best["rtf"], "all_rtf": alls["rtf"],
                         "x_best_single": hit["x_best_single"], "saved_vs_all": hit["saved_vs_decode_all"],
                         "decoder_share_min": e["decoder_share"].min(), "decoder_share_max": e["decoder_share"].max(),
                         "router_gflops": rec["router_gflops"], "router_M": rec["router"]["n_params"] / 1e6,
                         "agreement": rec["agreement"], "hit_wer": hit["wer"], "best_wer": best["wer"],
                         "weights_gb_hit": sum(rec["weights_gb"].values()),
                         "weights_gb_best": rec["weights_gb"][rec["bsm"]],
                         "label_gpu_h": e.get("label_gpu_h", pd.Series(dtype=float)).sum()})
        self.summary_table = s = pd.DataFrame(rows)
        return Table(s.round(3), title="Cost across corpora (ms per clip, batch 1)")

    def scaling(self):
        """The router alone against K experts: latency, parameters and GFLOPs of one fit of each corpus's selected
        router on synthetic frames of the members' mean width and length (measured at level 2)."""
        from labkit.pretty import Table
        rows = []
        if self.level >= 2:
            import torch

            from hitasr.latency import router_scaling
            for c, rec in self.records.items():
                if "router" not in rec or not torch.cuda.is_available():
                    continue
                arch = {k: v for k, v in rec["router"]["arch"].items() if k != "pooled_dim"}
                per = pd.DataFrame(rec["per_expert"])
                s = router_scaling(Ks=self.cfg.scaling_ks, d_in=int(np.mean(list(rec["router"]["dims"].values()))),
                                   T=int(min(per["frames"].mean(), rec["router"]["max_frames"] or 1e9)), arch=arch,
                                   device=self.device)
                s.insert(0, "corpus", c)
                rec["scaling"] = json.loads(s.to_json(orient="records"))
                with open(f"timing_{c}.json", "w") as fh:
                    json.dump(rec, fh, indent=1, default=str)
                rows.append(s)
        else:
            rows = [pd.DataFrame(r["scaling"]).assign(corpus=c) for c, r in self.records.items() if r.get("scaling")]
        if not rows:
            return None
        return Table(pd.concat(rows, ignore_index=True).round(4), title="The router alone against K", group="corpus")
