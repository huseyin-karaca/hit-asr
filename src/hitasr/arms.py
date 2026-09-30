"""The arms of the paper: HIT-ASR, the pooled rivals (MLP-pool, ADASTT) and the hard-CE control, their search spaces,
and the registry every notebook looks them up in (`MODELS`). The fusion baselines register from `hitasr.rover`."""

__all__ = ['ARM_KEYS', 'FUSIONS_ALL', 'INPUT_NORMS_ALL', 'MODELS', 'OURS', 'RIVALS', 'CONTROLS',
           'Arm', 'HitASRArm', 'MLPPoolArm', 'AdaSTTArm', 'ModelSpec',
           'hit_asr_space', 'build_hit_asr', 'mlp_pool_space', 'build_mlp_pool', 'adastt_space', 'build_adastt',
           'register', 'known_params', 'check_spaces']

import os
from dataclasses import dataclass, replace

import numpy as np
import torch

from hitasr.routers import LOSS_PRESETS, HitASRRouter, MLPPoolRouter
from hitasr.training import TrainConfig, fit_with_restarts, predict_logits, wer_targets
from labkit.search import Param, cat


class Arm:
    """Base class: the contract above. Subclasses set `produces` and implement fit/predict_proba."""

    produces = "choice"
    wants_texts = False

    def fit(self, store, rows, callback=None):
        raise NotImplementedError

    def predict_proba(self, store, rows):
        raise NotImplementedError

    def predict(self, store, rows):
        return self.predict_proba(store, rows).argmax(1)

    def expected_wer(self, store, rows):
        rows = np.asarray(rows, dtype=int)
        Y = wer_targets(store.E, store.nref, rows)
        return float((self.predict_proba(store, rows) * Y).sum(1).mean())


def _torch_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class HitASRArm(Arm):
    """HIT-ASR, or any of its ablation variants, as a fittable arm.

    `arch` are `HitASRRouter` kwargs; `train` a `TrainConfig` (the objective
    lives there). The frames are gathered from the store per batch — on the
    device the store was opened on.
    """

    def __init__(self, arch=None, train=None, seed=42, device=None, n_seeds=1):
        self.arch, self.train = dict(arch or {}), (train or TrainConfig())
        self.train = replace(self.train, seed=seed)
        self.seed, self.device, self.n_seeds = seed, _torch_device(device), int(n_seeds)
        self.model, self.history, self.models = None, None, []
        self._pooled, self.pool_mu, self.pool_sd = None, None, None

    def _batcher(self, store):
        mf = self.train.max_frames
        pooled = self._pooled if self.arch.get("pooled_skip") else None

        def batcher(rows):
            frames, lengths = store.batch(rows, max_frames=mf)
            out = ({m: f.to(self.device, non_blocking=True) for m, f in frames.items()},
                   {m: l.to(self.device) for m, l in lengths.items()})
            if pooled is not None:
                out = out + (pooled[torch.as_tensor(np.asarray(rows, dtype=int), device=self.device)],)
            return out
        return batcher

    def _pooled_design(self, store, rows):
        """The pooled shortcut's input: mean-pooled frames + log duration, standardised on the fit rows."""
        X = np.concatenate([store.pooled(), np.log1p(store.n_frames())[:, None]], axis=1).astype(np.float32)
        rows = np.asarray(rows, dtype=int)
        self.pool_mu, self.pool_sd = X[rows].mean(0), X[rows].std(0).clip(1e-6)
        self._pooled = torch.as_tensor((X - self.pool_mu) / self.pool_sd, device=self.device)
        return self._pooled.shape[1]

    def build(self, store, rows=None):
        torch.manual_seed(self.seed)
        arch = dict(self.arch)
        if arch.get("pooled_skip"):
            arch["pooled_dim"] = self._pooled_design(store, rows if rows is not None else np.arange(store.n))
        self.model = HitASRRouter(store.dims, store.members, **arch).to(self.device)
        return self.model

    def input_stats(self, store, rows, n_sample=1024, scalar=False):
        """Per-expert, per-dimension frame mean and sd over (a sample of) the fit rows' frames —
        or, with `scalar`, a zero mean and one rms per expert (broadcast over the dimensions).

        Padded positions are excluded; non-finite frames are zeroed first, as
        the model does. `n_sample` rows bound the pass to a few seconds.
        """
        rows = np.asarray(rows, dtype=int)
        rng = np.random.default_rng(self.seed)
        sample = rows if len(rows) <= n_sample else rng.choice(rows, n_sample, replace=False)
        batcher = self._batcher(store)
        acc = {m: [torch.zeros(d, dtype=torch.float64, device=self.device),
                   torch.zeros(d, dtype=torch.float64, device=self.device), 0] for m, d in store.dims.items()}
        with torch.no_grad():
            for s in range(0, len(sample), self.train.batch_size):
                frames, lengths = batcher(sample[s:s + self.train.batch_size])
                for m, x in frames.items():
                    x = torch.nan_to_num(x.double(), nan=0.0, posinf=0.0, neginf=0.0)
                    valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[m][:, None]
                    xv = x[valid]
                    acc[m][0] += xv.sum(0); acc[m][1] += (xv * xv).sum(0); acc[m][2] += int(valid.sum())
        stats = {}
        for m, (s1, s2, n) in acc.items():
            n = max(n, 1)
            mu = s1 / n
            var = (s2 / n - mu * mu).clamp(min=0)
            if scalar:
                rms = float((s2.sum() / (n * len(s2))).sqrt())
                stats[m] = (np.zeros(len(s1), dtype=np.float32), np.full(len(s1), max(rms, 1e-6), dtype=np.float32))
            else:
                stats[m] = (mu.float().cpu().numpy(), var.sqrt().float().cpu().numpy())
        return stats

    def fit(self, store, rows, callback=None):
        Y = wer_targets(store.E, store.nref, np.arange(store.n))
        stats = None

        arch = dict(self.arch)
        if arch.get("pooled_skip"):
            arch["pooled_dim"] = self._pooled_design(store, rows)

        def build(seed):
            nonlocal stats
            torch.manual_seed(seed)
            model = HitASRRouter(store.dims, store.members, **arch).to(self.device)
            if model.input_norm in ("standardize", "scale"):
                if stats is None:
                    stats = self.input_stats(store, rows, scalar=model.input_norm == "scale")
                model.set_input_stats(stats)
            return model

        self.models, hists = [], []
        for i in range(self.n_seeds):                       # an ensemble averages independently seeded fits
            cfg = replace(self.train, seed=self.train.seed + 100 * i)
            model, hist = fit_with_restarts(build, self._batcher(store), Y, rows, cfg,
                                            callback=callback if i == 0 else None, nref=store.nref)
            self.models.append(model); hists.append(hist)
        self.model, self.history = self.models[0], dict(hists[0])
        self.history.update(n_seeds=self.n_seeds, seconds=float(sum(h["seconds"] for h in hists)),
                            members=[{k: h.get(k) for k in ("best_val_wer", "best_epoch", "epochs_run", "restarts", "no_gain")} for h in hists])
        return self

    def predict_proba(self, store, rows):
        batcher, rows = self._batcher(store), np.asarray(rows, dtype=int)
        probs = [torch.softmax(torch.as_tensor(predict_logits(m, batcher, rows, batch_size=self.train.batch_size,
                                                              amp=self.train.amp)), -1).numpy()
                 for m in (self.models or [self.model])]
        return np.mean(probs, axis=0)

    def n_params(self):
        return sum(m.n_params() for m in (self.models or [self.model])) if self.model is not None else None


class MLPPoolArm(Arm):
    """The pooled baseline on the labels' mean vectors (plus the log duration), standardised on the fit rows."""

    def __init__(self, arch=None, train=None, seed=42, device=None, use_duration=True, n_seeds=1):
        self.arch, self.train = dict(arch or {}), replace(train or TrainConfig(), seed=seed)
        self.seed, self.device, self.use_duration, self.n_seeds = seed, _torch_device(device), use_duration, int(n_seeds)
        self.model, self.history, self.mu, self.sd, self.models = None, None, None, None, []
        self._X = None

    def _design(self, store):
        if self._X is None:
            X = store.pooled()
            if self.use_duration:
                X = np.concatenate([X, np.log1p(store.n_frames())[:, None]], axis=1)
            self._X = X.astype(np.float32)
        return self._X

    def _fit_design(self, store, rows):
        """The design a fit on `rows` uses. The pooled descriptors do not depend on the rows; a subclass's may."""
        return self._design(store)

    def _batcher(self, X):
        Xt = torch.as_tensor(X, device=self.device)

        def batcher(rows):
            return (Xt[torch.as_tensor(np.asarray(rows, dtype=int), device=self.device)],)
        return batcher

    def fit(self, store, rows, callback=None):
        rows = np.asarray(rows, dtype=int)
        X = self._fit_design(store, rows)
        self.mu, self.sd = X[rows].mean(0), X[rows].std(0).clip(1e-6)
        Xs = (X - self.mu) / self.sd
        Y = wer_targets(store.E, store.nref, np.arange(store.n))

        def build(seed):
            torch.manual_seed(seed)
            return MLPPoolRouter(Xs.shape[1], len(store.members), **self.arch).to(self.device)

        self.models, hists = [], []
        for i in range(self.n_seeds):
            cfg = replace(self.train, seed=self.train.seed + 100 * i)
            model, hist = fit_with_restarts(build, self._batcher(Xs), Y, rows, cfg,
                                            callback=callback if i == 0 else None, nref=store.nref)
            self.models.append(model); hists.append(hist)
        self.model, self.history = self.models[0], dict(hists[0])
        self.history.update(n_seeds=self.n_seeds, seconds=float(sum(h["seconds"] for h in hists)))
        return self

    def predict_proba(self, store, rows):
        Xs = (self._design(store) - self.mu) / self.sd
        batcher, rows = self._batcher(Xs), np.asarray(rows, dtype=int)
        probs = [torch.softmax(torch.as_tensor(predict_logits(m, batcher, rows, batch_size=256, amp=self.train.amp)), -1).numpy()
                 for m in (self.models or [self.model])]
        return np.mean(probs, axis=0)

    def n_params(self):
        return sum(m.n_params() for m in (self.models or [self.model])) if self.model is not None else None


class AdaSTTArm(Arm):
    """ADASTT: gradient-boosted trees on the pooled descriptors.

    The pooled inter-model router of the prior work, refitted on this project's rows, features and folds.
    `objective="cross_entropy"` trains `multi:softprob` on the argmin label; `"expected_wer"` descends the expected
    WER instead.
    """

    produces = "choice"

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                 colsample_bytree=0.5, min_child_weight=4.0, reg_lambda=1.0,
                 early_stopping_rounds=50, nthread=4, seed=42, objective="expected_wer",
                 use_duration=True, wer_cap=1.0, gamma=0.0):
        self.p = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate),
                      max_depth=int(max_depth), subsample=float(subsample),
                      colsample_bytree=float(colsample_bytree), min_child_weight=float(min_child_weight),
                      reg_lambda=float(reg_lambda), gamma=float(gamma))
        self.early_stopping_rounds, self.nthread, self.seed = early_stopping_rounds, nthread, seed
        self.objective, self.use_duration, self.wer_cap = objective, use_duration, wer_cap
        self.bsm_ = None                                       # the fit rows' best single member
        self.model_, self.mu, self.sd, self._X = None, None, None, None

    def _margin(self, n, K):
        """The boosting's base margin for `n` rows (`None`: xgboost's default). A subclass may start elsewhere."""
        return None

    def _design(self, store):
        if self._X is None:
            X = store.pooled()
            if self.use_duration:
                X = np.concatenate([X, np.log1p(store.n_frames())[:, None]], axis=1)
            self._X = X.astype(np.float32)
        return self._X

    def _params(self, K):
        return {"num_class": K, "objective": "multi:softprob", "tree_method": "hist", "device": "cpu",
                "nthread": self.nthread if self.nthread > 0 else os.cpu_count(),
                "eta": self.p["learning_rate"], "max_depth": self.p["max_depth"],
                "min_child_weight": self.p["min_child_weight"], "subsample": self.p["subsample"],
                "colsample_bytree": self.p["colsample_bytree"], "lambda": self.p["reg_lambda"],
                "gamma": self.p["gamma"], "seed": self.seed}

    def fit(self, store, rows, callback=None):
        import xgboost as xgb
        X = self._design(store)
        rows = np.asarray(rows, dtype=int)
        self.mu, self.sd = X[rows].mean(0), X[rows].std(0).clip(1e-6)
        Xs = (X[rows] - self.mu) / self.sd
        Y = wer_targets(store.E, store.nref, rows, cap=self.wer_cap)
        K = Y.shape[1]
        self.bsm_ = int((store.E[rows].sum(0) / np.maximum(store.nref[rows].sum(), 1)).argmin())
        idx = np.random.default_rng(self.seed).permutation(len(rows))
        cut = int(round(len(rows) * 0.8))
        tr, va = idx[:cut], idx[cut:]
        if self.objective == "cross_entropy":
            dtr = xgb.DMatrix(Xs[tr], label=Y[tr].argmin(1), base_margin=self._margin(len(tr), K))
            dva = xgb.DMatrix(Xs[va], label=Y[va].argmin(1), base_margin=self._margin(len(va), K))
            self.model_ = xgb.train({**self._params(K), "eval_metric": "mlogloss"}, dtr,
                                    num_boost_round=self.p["n_estimators"], evals=[(dva, "val")],
                                    early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
            return self
        rng = np.random.default_rng(self.seed)
        dtr = xgb.DMatrix(Xs[tr], label=rng.integers(0, K, len(tr)), base_margin=self._margin(len(tr), K))
        dva = xgb.DMatrix(Xs[va], label=rng.integers(0, K, len(va)), base_margin=self._margin(len(va), K))
        wer_map = {id(dtr): Y[tr], id(dva): Y[va]}

        def obj(preds, dmat):
            z = preds.reshape(-1, K) if preds.ndim == 1 else preds
            e = np.exp(z - z.max(1, keepdims=True)); w = e / e.sum(1, keepdims=True)
            W = wer_map[id(dmat)]
            L = (w * W).sum(1, keepdims=True)
            grad = w * (W - L)
            hess = np.maximum(w * (1.0 - 2.0 * w) * (W - L), 0.02)
            return grad.astype(np.float32), hess.astype(np.float32)

        def metric(preds, dmat):
            z = preds.reshape(-1, K) if preds.ndim == 1 else preds
            e = np.exp(z - z.max(1, keepdims=True)); w = e / e.sum(1, keepdims=True)
            return "ExpectedWER", float((w * wer_map[id(dmat)]).sum(1).mean())

        self.model_ = xgb.train({**self._params(K), "disable_default_eval_metric": 1}, dtr,
                                num_boost_round=self.p["n_estimators"], obj=obj,
                                evals=[(dtr, "train"), (dva, "val")], custom_metric=metric,
                                early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        return self

    def predict_proba(self, store, rows):
        import xgboost as xgb
        Xs = (self._design(store)[np.asarray(rows, dtype=int)] - self.mu) / self.sd
        K = len(store.members)
        z = self.model_.predict(xgb.DMatrix(Xs, base_margin=self._margin(len(Xs), K)))
        if z.ndim == 1:
            z = z.reshape(len(Xs), -1)
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------- the registry --

@dataclass
class ModelSpec:
    """One tunable algorithm: its space, its sampler, its builder, its cost profile."""
    name: str
    space: object                  # () -> {name: Param}
    build: object                  # (params, store, seed, ctx) -> Arm
    family: str = "ours"
    prunable: bool = False         # `fit(callback=)` reports a per-epoch score
    frames: bool = True            # opens the frame store
    trials: float = 1.0            # share of a per-arm budget when not exact
    placeholder: bool = False
    label: str = ""
    torch: bool = False            # a torch arm: every `TrainConfig` field is a parameter it knows
    extra_params: tuple = ()       # further parameter names its builder accepts beyond the space

    def n_free(self, space=None):
        return sum(1 for p in (space or self.space()).values() if p.frozen is None)


ARM_KEYS = ("loss_preset", "n_seeds")      # parameters the arm consumes itself, not the network


def _split_params(params, config=TrainConfig):
    """`params` -> `(arch kwargs, config)` for the torch arms: the trainer's fields to `config`, the rest to the network."""
    tc = {f: params[f] for f in config.__dataclass_fields__ if f in params}
    arch = {k: v for k, v in params.items() if k not in tc and k not in ARM_KEYS}
    if "loss_preset" in params:
        tc.update(LOSS_PRESETS[params["loss_preset"]])
    return arch, config(**tc)


FUSIONS_ALL = ("bridge", "self_attn", "concat", "mean")
INPUT_NORMS_ALL = ("none", "scale", "standardize", "layernorm")


def hit_asr_space(fusion=None, share_stage1=None, loss_preset=None, input_norm="none", word_weighted=False,
                  pooled_skip=False, n_seeds=3):
    """The HIT-ASR space. Fix `fusion` / `share_stage1` / `loss_preset` / `input_norm` / `word_weighted` /
    `pooled_skip` / `n_seeds` for an ablation arm; `None` searches the switch. The defaults are the
    manuscript's design: raw frames, no shortcut, per-utterance objective, a 3-seed ensemble."""
    s = {p.name: p for p in [
        cat("d_model", 128, 256, 384),
        cat("n_heads", 4, 8),
        cat("stage1_layers", 1, 2, 3),
        cat("stage2_layers", 1, 2),
        cat("ffn_dim", 256, 512, 1024),
        cat("dropout", 0.05, 0.15, 0.3),
        Param("fusion", "cat", choices=FUSIONS_ALL, frozen=fusion),
        Param("share_stage1", "cat", choices=(True, False), frozen=share_stage1),
        Param("input_norm", "cat", choices=INPUT_NORMS_ALL, frozen=input_norm),
        Param("pooled_skip", "cat", choices=(False, True), frozen=pooled_skip),
        cat("lr", 3e-5, 1e-4, 3e-4, 1e-3),
        cat("weight_decay", 1e-3, 1e-2, 1e-1),
        cat("batch_size", 8, 16, 32, 64),
        cat("max_frames", 500, 1000, 2000),
        cat("patience", 5, 10),
        Param("loss_preset", "cat", choices=tuple(LOSS_PRESETS), frozen=loss_preset),
        cat("label_smoothing", 0.0, 0.1),
        Param("word_weighted", "cat", choices=(True, False), frozen=word_weighted),
        Param("n_seeds", "cat", choices=(1, 3), frozen=n_seeds),
    ]}
    return s



def build_hit_asr(p, store, seed=42, ctx=None, config=TrainConfig):
    """`epochs` comes from `ctx` (the protocol's ceiling) unless the space searches it."""
    arch, tc = _split_params({"epochs": (ctx or {}).get("epochs", 50), **p}, config)
    arch.setdefault("max_seq_len", max(4096, tc.max_frames + 1))
    return HitASRArm(arch=arch, train=tc, seed=seed, device=(ctx or {}).get("device"), n_seeds=p.get("n_seeds", 1))


def mlp_pool_space(loss_preset=None, word_weighted=False, n_seeds=3):
    """The pooled rival's space; `n_seeds` matches HIT-ASR's ensemble so the frame-vs-pooled
    comparison is not an ensemble-vs-single one."""
    return {p.name: p for p in [
        cat("d_hidden", 256, 512, 1024, 2048),
        cat("n_layers", 1, 2, 3),
        cat("dropout", 0.05, 0.15, 0.3),
        cat("lr", 3e-5, 1e-4, 3e-4, 1e-3),
        cat("weight_decay", 1e-3, 1e-2, 1e-1),
        cat("batch_size", 32, 64, 128, 256),
        cat("patience", 5, 10),
        Param("loss_preset", "cat", choices=tuple(LOSS_PRESETS), frozen=loss_preset),
        cat("label_smoothing", 0.0, 0.1),
        Param("word_weighted", "cat", choices=(True, False), frozen=word_weighted),
        Param("n_seeds", "cat", choices=(1, 3), frozen=n_seeds),
    ]}


def build_mlp_pool(p, store, seed=42, ctx=None, config=TrainConfig, arm=None):
    """`epochs` comes from `ctx` unless the space searches it."""
    arch, tc = _split_params({"epochs": (ctx or {}).get("epochs", 100), **p}, config)
    return (arm or MLPPoolArm)(arch=arch, train=tc, seed=seed, device=(ctx or {}).get("device"),
                               n_seeds=p.get("n_seeds", 1))


def adastt_space(objective=None):
    return {p.name: p for p in [
        cat("n_estimators", 100, 400, 1000),
        cat("learning_rate", 0.01, 0.03, 0.1, 0.3),
        cat("max_depth", 3, 5, 8),
        cat("subsample", 0.6, 0.8, 1.0),
        cat("colsample_bytree", 0.1, 0.3, 0.6, 1.0),
        cat("min_child_weight", 1.0, 4.0, 16.0),
        cat("reg_lambda", 0.3, 3.0, 30.0),
        Param("objective", "cat", choices=("expected_wer", "cross_entropy"), frozen=objective),
    ]}


def build_adastt(p, store, seed=42, ctx=None, arm=None):
    return (arm or AdaSTTArm)(seed=seed, nthread=(ctx or {}).get("nthread", 4), **p)


MODELS = {}


def register(spec):
    MODELS[spec.name] = spec
    return spec


# --- ours -----------------------------------------------------------------
register(ModelSpec("hit_asr", lambda: hit_asr_space(fusion="bridge", share_stage1=True, loss_preset="default"),
                   build_hit_asr, family="ours", prunable=True, torch=True, label="HIT-ASR"))
# --- rivals -------------------------------------------------------------------
register(ModelSpec("mlp_pool", lambda: mlp_pool_space(loss_preset="default"), build_mlp_pool,
                   family="rival", prunable=True, frames=False, torch=True, label="MLP-pool"))
register(ModelSpec("mlp_pool_hard_ce", lambda: mlp_pool_space(loss_preset="hard_ce_only"), build_mlp_pool,
                   family="control", prunable=True, frames=False, torch=True, label="MLP-pool (hard CE)"))
register(ModelSpec("adastt_ce", lambda: adastt_space(objective="cross_entropy"), build_adastt,
                   family="rival", frames=False, label="ADASTT"))
OURS = ("hit_asr",)
RIVALS = ("mlp_pool", "adastt_ce")
CONTROLS = ("mlp_pool_hard_ce",)


def known_params(spec):
    """The parameter names a space for `spec` may use: its default space, plus every `TrainConfig`
    field for the torch arms (`_split_params` routes those to the trainer), so a notebook can freeze
    a training knob the registry space does not search — `max_restarts`, say — without editing it —
    plus whatever its builder takes beyond the space (`ModelSpec.extra_params`)."""
    known = set(spec.space()) | set(spec.extra_params)
    if spec.torch:
        known |= set(TrainConfig.__dataclass_fields__)
    return known


def check_spaces(spaces, strict=True):
    """Every space names only parameters its arm knows (`known_params`). Raises on the first stranger."""
    for arm, space in spaces.items():
        if arm not in MODELS:
            raise KeyError(f"{arm!r} is not a registered arm")
        known = known_params(MODELS[arm])
        strangers = set(space) - known
        if strangers and strict:
            raise KeyError(f"{arm}: unknown parameter(s) {sorted(strangers)}; known: {sorted(known)}")
    return True


