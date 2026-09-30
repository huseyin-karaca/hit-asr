"""The cost of routing, as the paper's cost section reports it.

The clocks themselves come from `hitasr.deploy` (`time_experts`, `compare_systems`), which runs the experts and the
systems on raw audio; this module turns those measurements into the tables — per expert (encoder, decoder, the
decoder's share, the real-time factor, the label pass) and per system (the best single expert, HIT-ASR, decoding all
experts) — and clocks the router alone: its latency, parameters and FLOPs, and how they grow with the number of
experts K.
"""

__all__ = ['time_router', 'count_flops', 'router_flops', 'router_scaling', 'expert_costs', 'system_costs']

import time

import numpy as np
import pandas as pd
import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_router(model, batcher, rows, n=200, batch_size=1, warmup=5):
    """Router forward wall-clock per clip, in ms, at `batch_size`."""
    device = next(model.parameters()).device
    model.eval()
    rows = np.asarray(rows, dtype=int)[:n + warmup * batch_size]
    warmup = min(warmup, max(0, len(rows) // (2 * batch_size)))
    times = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            b = rows[i:i + batch_size]
            args = batcher(b)
            _sync(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(*args)
            _sync(); t1 = time.perf_counter()
            if i >= warmup * batch_size:
                times.append(1000 * (t1 - t0) / len(b))
    return {"router_ms": float(np.mean(times)), "router_ms_sd": float(np.std(times)),
            "n_params": sum(p.numel() for p in model.parameters()), "batch_size": batch_size}


def count_flops(fn, *args, **kwargs):
    """Total FLOPs of one call, via torch's `FlopCounterMode`. `None` if unsupported."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    with FlopCounterMode(display=False) as fc:
        fn(*args, **kwargs)
    return int(fc.get_total_flops())


def router_flops(model, batcher, rows):
    """GFLOPs of the router's forward pass per clip, averaged over `rows` at batch size 1."""
    model.eval()
    vals = []
    with torch.no_grad():
        for r in np.asarray(rows, dtype=int):
            f = count_flops(lambda: model(*batcher([r])))
            if f is None:
                return None
            vals.append(f / 1e9)
    return float(np.mean(vals))

def expert_costs(per_expert, throughput=None, hours=None):
    """Per expert, from `deploy.time_experts` rows: the mean ms per clip of its encoder, its decoder and its whole
    pipeline (and their medians, which a rare slow clip does not move), the decoder's share of encoder + decoder, the
    pipeline's real-time factor, the share of clips on which the decode from the reused encoder pass gave the
    pipeline's own transcript, and the peak memory of a pipeline.

    `throughput` (rows `expert, audio_seconds, seconds` of the pipeline run in batches) and `hours` (the corpus's
    audio) add the batched real-time factor and the GPU-hours of labelling the corpus with that expert.
    """
    g = per_expert.groupby("expert", sort=False)
    out = pd.DataFrame({"encode_ms": g["encode_ms"].mean(), "decode_ms": g["decode_ms"].mean(),
                        "pipeline_ms": g["pipeline_ms"].mean(),
                        "rtf": g["pipeline_ms"].sum() / 1e3 / g["audio_seconds"].sum(),
                        "same_text": g["same_text"].mean(), "reuses_encoder": g["reuses_encoder"].first(),
                        "peak_gb": g["peak_gb"].max(), "audio_s_per_clip": g["audio_seconds"].mean(),
                        "encode_ms_median": g["encode_ms"].median(), "decode_ms_median": g["decode_ms"].median(),
                        "pipeline_ms_median": g["pipeline_ms"].median()})
    out["decoder_share"] = out["decode_ms"] / (out["encode_ms"] + out["decode_ms"])
    if throughput is not None and len(throughput):
        tg = pd.DataFrame(throughput).groupby("expert")
        out["rtf_batched"] = tg["seconds"].sum() / tg["audio_seconds"].sum()
        if hours is not None:
            out["label_gpu_h"] = out["rtf_batched"] * float(hours)
    return out


def system_costs(experts, systems, members, bsm, labels=None):
    """The cost table of one corpus: every member's own pipeline (from `expert_costs`, with a `wer` column), then
    HIT-ASR and decoding all members + ROVER end to end (from `deploy.compare_systems`, labelled `"HIT-ASR"` and
    `"Decode all + ROVER"`). `x_best_single` is a system's time over the best single expert's (`bsm`, the member
    with the lowest corpus WER); `saved_vs_decode_all` the share of decode-all's time it does not spend.

    `attrs` holds the same comparison composed from the per-expert clocks (every encoder + the router + the chosen
    decoder under HIT-ASR's own choice shares), a check on the end-to-end clock.
    """
    labels = labels or {}
    e = experts.loc[list(members)]
    s = systems.set_index("system")
    rows = [{"system": labels.get(m, m) + (" (best single)" if m == bsm else ""), "kind": "single",
             "encode_ms": e.loc[m, "encode_ms"], "decode_ms": e.loc[m, "decode_ms"], "total_ms": e.loc[m, "pipeline_ms"],
             "rtf": e.loc[m, "rtf"], "wer": e.loc[m].get("wer", np.nan)} for m in members]
    hit, alls = s.loc["HIT-ASR"], s.loc["Decode all + ROVER"]
    rows.append({"system": "HIT-ASR", "kind": "hit_asr", "encode_ms": hit["encode_ms"], "route_ms": hit["route_ms"],
                 "decode_ms": hit["decode_ms"], "total_ms": hit["total_ms"], "rtf": hit["rtf"], "wer": hit.get("wer")})
    rows.append({"system": "Decode all + ROVER", "kind": "decode_all", "decode_ms": alls["pipelines_ms"],
                 "fuse_ms": alls["fuse_ms"], "total_ms": alls["total_ms"], "rtf": alls["rtf"], "wer": alls.get("wer")})
    out = pd.DataFrame(rows)
    best, all_ms = e.loc[bsm, "pipeline_ms"], alls["total_ms"]
    out["x_best_single"] = out["total_ms"] / best
    out["saved_vs_decode_all"] = 1 - out["total_ms"] / all_ms
    shares = np.array([hit.get(f"share_{m}", np.nan) for m in members], dtype=float)
    composed = float(e["encode_ms"].sum() + hit["route_ms"] + (shares * e["decode_ms"].to_numpy()).sum())
    out.attrs.update(composed_hit_ms=composed, measured_hit_ms=float(hit["total_ms"]),
                     composed_decode_all_ms=float(e["pipeline_ms"].sum()), router_share=float(hit["route_ms"] / hit["total_ms"]),
                     decoder_share_of_hit=float(hit["decode_ms"] / hit["total_ms"]),
                     encoder_share_of_hit=float(hit["encode_ms"] / hit["total_ms"]))
    return out


def router_scaling(Ks=(3, 5, 10), d_in=1024, T=250, arch=None, device=None, batch_size=1, n=50):
    """Latency (ms/clip), parameters and GFLOPs of `HitASRRouter` at each K, on synthetic frames."""
    from hitasr.routers import HitASRRouter
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for K in Ks:
        members = [f"e{i}" for i in range(K)]
        model = HitASRRouter({m: d_in for m in members}, members, **(arch or {})).to(device).eval()
        frames = {m: torch.randn(batch_size, T, d_in, device=device, dtype=torch.float16) for m in members}
        lengths = {m: torch.full((batch_size,), T, device=device) for m in members}
        batcher = lambda b, f=frames, l=lengths: (f, l)
        t = time_router(model, batcher, np.arange(n), n=n, batch_size=batch_size)
        g = count_flops(lambda: model(frames, lengths))
        rows.append({"K": K, "router_ms": t["router_ms"], "params_M": t["n_params"] / 1e6,
                     "gflops": None if g is None else g / 1e9 / batch_size, "T": T, "d_in": d_in})
    return pd.DataFrame(rows)
