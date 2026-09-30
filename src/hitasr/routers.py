__all__ = ['FUSIONS', 'INPUT_NORMS', 'LOSS_PRESETS', 'SinusoidalPositionalEncoding', 'CrossAttentionBridge', 'HitASRRouter',
           'MLPPoolRouter', 'composite_loss']

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("bridge", "self_attn", "concat", "mean")
INPUT_NORMS = ("none", "scale", "standardize", "layernorm")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(f"sequence of {T} frames exceeds max_len={self.pe.size(1)}; "
                             "raise max_seq_len or lower max_frames")
        return x + self.pe[:, :T].to(x.dtype)


def _encoder(d_model, n_heads, ffn_dim, dropout, n_layers):
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                                       dropout=dropout, batch_first=True, activation="gelu",
                                       norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class CrossAttentionBridge(nn.Module):
    """Eq. 3: the summary attends to the other experts' frames; residual + LayerNorm."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, cls, others, key_padding_mask):
        out, _ = self.attn(query=cls, key=others, value=others, key_padding_mask=key_padding_mask)
        return self.norm(cls + self.drop(out))


class HitASRRouter(nn.Module):
    """The hierarchical transformer router. Input `{member: (B, T, D_m)}` + lengths; output `(B, K)` logits."""

    def __init__(self, dims, members, d_model=256, n_heads=4, stage1_layers=2, stage2_layers=1,
                 ffn_dim=512, dropout=0.15, fusion="bridge", share_stage1=True, max_seq_len=4096,
                 input_norm="none", pooled_skip=False, pooled_dim=None):
        super().__init__()
        if pooled_skip and not pooled_dim:
            raise ValueError("pooled_skip needs pooled_dim (the arm passes the store's pooled width + 1)")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")
        self.members, self.K, self.fusion, self.share_stage1 = tuple(members), len(members), fusion, share_stage1
        self.d_model, self.input_norm = d_model, input_norm
        self.proj = nn.ModuleDict({m: nn.Linear(int(dims[m]), d_model) for m in self.members})
        # per-expert input statistics: identity until `set_input_stats` — buffers, so `state_dict`
        # carries them and a restored best state is normalised the way it was trained
        for k, m in enumerate(self.members):
            self.register_buffer(f"in_mu_{k}", torch.zeros(int(dims[m])))
            self.register_buffer(f"in_sd_{k}", torch.ones(int(dims[m])))
        if input_norm == "layernorm":
            self.in_norm = nn.ModuleDict({m: nn.LayerNorm(int(dims[m])) for m in self.members})
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 1)
        self.identity = nn.Embedding(self.K, d_model)
        self.cls = nn.Parameter(torch.randn(self.K, 1, d_model) * 0.02)
        self.in_drop = nn.Dropout(dropout)
        if share_stage1:
            self.stage1 = _encoder(d_model, n_heads, ffn_dim, dropout, stage1_layers)
        else:
            self.stage1s = nn.ModuleDict({m: _encoder(d_model, n_heads, ffn_dim, dropout, stage1_layers)
                                          for m in self.members})
        if fusion == "bridge":
            self.bridges = nn.ModuleDict({m: CrossAttentionBridge(d_model, n_heads, dropout) for m in self.members})
        if fusion in ("bridge", "self_attn"):
            self.global_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.stage2 = _encoder(d_model, n_heads, ffn_dim, dropout, stage2_layers)
            head_in = d_model
        elif fusion == "concat":
            head_in = d_model * self.K
        else:
            head_in = d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, self.K))
        self.pooled_skip = bool(pooled_skip)
        if self.pooled_skip:
            self.skip = nn.Linear(int(pooled_dim), self.K)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def output_layers(self):
        """The layers that write the logits: the head's last `Linear`, then the pooled shortcut if any."""
        return [self.head[-1]] + ([self.skip] if self.pooled_skip else [])

    def set_input_stats(self, stats):
        """`{member: (mu, sd)}` per dimension, from the fit rows. `sd` is floored at 1e-6."""
        for k, m in enumerate(self.members):
            if m not in stats:
                continue
            mu, sd = stats[m]
            getattr(self, f"in_mu_{k}").copy_(torch.as_tensor(np.asarray(mu, dtype=np.float32)))
            getattr(self, f"in_sd_{k}").copy_(torch.as_tensor(np.asarray(sd, dtype=np.float32)).clamp(min=1e-6))

    def input_stats(self):
        return {m: (getattr(self, f"in_mu_{k}").detach().cpu().numpy(), getattr(self, f"in_sd_{k}").detach().cpu().numpy())
                for k, m in enumerate(self.members)}

    def _normalise(self, k, m, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.input_norm == "standardize":
            x = (x - getattr(self, f"in_mu_{k}")) / getattr(self, f"in_sd_{k}")
        elif self.input_norm == "scale":                     # `set_input_stats` stored the scalar rms in every entry of in_sd
            x = x / getattr(self, f"in_sd_{k}")
        elif self.input_norm == "layernorm":
            x = self.in_norm[m](x)
        return x

    def _stage1(self, k, m, x, lengths):
        B, T, _ = x.shape
        h = self.proj[m](self._normalise(k, m, x).to(self.proj[m].weight.dtype))
        h = self.pos(h) + self.identity.weight[k][None, None, :]
        h = self.in_drop(h)
        h = torch.cat([self.cls[k].unsqueeze(0).expand(B, -1, -1).to(h.dtype), h], dim=1)
        pad = torch.arange(T, device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        enc = self.stage1 if self.share_stage1 else self.stage1s[m]
        y = enc(h, src_key_padding_mask=pad)
        return y[:, :1], y, pad

    def forward(self, frames, lengths, pooled=None):
        summaries, seqs, pads = {}, {}, {}
        for k, m in enumerate(self.members):
            summaries[m], seqs[m], pads[m] = self._stage1(k, m, frames[m], lengths[m])
        if self.fusion == "bridge":
            updated = {}
            for m in self.members:
                others = [o for o in self.members if o != m]
                seq = torch.cat([seqs[o][:, 1:] for o in others], dim=1)
                pad = torch.cat([pads[o][:, 1:] for o in others], dim=1)
                updated[m] = self.bridges[m](summaries[m], seq, pad)
            summaries = updated
        S = torch.cat([summaries[m] for m in self.members], dim=1)          # (B, K, d)
        if self.fusion in ("bridge", "self_attn"):
            z = torch.cat([self.global_cls.expand(S.size(0), -1, -1).to(S.dtype), S], dim=1)
            z = self.stage2(z)[:, 0]
        elif self.fusion == "concat":
            z = S.reshape(S.size(0), -1)
        else:
            z = S.mean(1)
        logits = self.head(z)
        if self.pooled_skip:
            if pooled is None:
                raise ValueError("pooled_skip=True: the batcher must supply the pooled features")
            logits = logits + self.skip(pooled.to(self.skip.weight.dtype))
        return logits

class MLPPoolRouter(nn.Module):
    """`(B, sum D)` pooled features -> `(B, K)` logits."""

    def __init__(self, in_features, K, d_hidden=1024, n_layers=2, dropout=0.15):
        super().__init__()
        layers, prev = [], int(in_features)
        for _ in range(n_layers):
            layers += [nn.Linear(prev, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = d_hidden
        layers.append(nn.Linear(prev, K))
        self.net = nn.Sequential(*layers)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def output_layers(self):
        return [self.net[-1]]

    def forward(self, x):
        return self.net(x)

LOSS_PRESETS = {
    "default":      dict(lambda_wer=1.0, lambda_hard=0.0, lambda_soft=0.5, tau=0.1),
    "wer_only":     dict(lambda_wer=1.0, lambda_hard=0.0, lambda_soft=0.0, tau=0.1),
    "hard_ce_only": dict(lambda_wer=0.0, lambda_hard=1.0, lambda_soft=0.0, tau=0.1),
    "soft_ce_only": dict(lambda_wer=0.0, lambda_hard=0.0, lambda_soft=1.0, tau=0.1),
    "wer+hard":     dict(lambda_wer=1.0, lambda_hard=0.3, lambda_soft=0.0, tau=0.1),
    "all":          dict(lambda_wer=1.0, lambda_hard=0.3, lambda_soft=0.5, tau=0.1),
    "wer+soft_tau1": dict(lambda_wer=1.0, lambda_hard=0.0, lambda_soft=0.5, tau=1.0),
}


def composite_loss(logits, wer, lambda_wer=1.0, lambda_hard=0.0, lambda_soft=0.5, tau=0.1,
                   label_smoothing=0.1, wer_cap=1.0, class_weights=None, weights=None):
    """Eq. 6 on one batch. `wer` is `(B, K)` per-clip WER; returns `(loss, parts)`.

    `weights` `(B,)` weights the clips — the reference word counts make every
    term a (capped) **corpus** WER rather than a mean per-utterance WER, which
    is the number the tables report. `None` is the plain mean.
    """
    wer = wer.float().clamp(max=wer_cap) if wer_cap else wer.float()
    logp = F.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    if weights is None:
        w = torch.full((wer.shape[0],), 1.0 / max(wer.shape[0], 1), dtype=torch.float32, device=wer.device)
    else:
        w = weights.float()
        w = w / w.sum().clamp(min=1e-6)
    parts = {}
    loss = logits.new_zeros(())
    if lambda_wer:
        parts["wer"] = ((probs * wer).sum(-1) * w).sum()
        loss = loss + lambda_wer * parts["wer"]
    if lambda_hard:
        target = wer.argmin(-1)
        ce = F.cross_entropy(logits.float(), target, weight=class_weights, label_smoothing=label_smoothing,
                             reduction="none")
        parts["hard"] = (ce * w).sum()
        loss = loss + lambda_hard * parts["hard"]
    if lambda_soft:
        q = F.softmax(-wer / tau, dim=-1)
        parts["soft"] = (-(q * logp).sum(-1) * w).sum()
        loss = loss + lambda_soft * parts["soft"]
    return loss, parts
