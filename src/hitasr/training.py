"""The routers' training loop: the objective, early stopping on the soft validation WER, EMA weights, the restart
guard, and inference."""

__all__ = ['TrainConfig', 'wer_targets', 'train_router', 'fit_with_restarts', 'predict_logits']

import math
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from hitasr.routers import composite_loss


@dataclass
class TrainConfig:
    """Everything the loop needs that is not the architecture."""
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    epochs: int = 50
    batch_size: int = 32
    patience: int = 10
    max_frames: int = 2000
    grad_clip: float = 1.0
    val_frac: float = 0.1
    seed: int = 42
    amp: bool = True
    # the objective
    lambda_wer: float = 1.0
    lambda_hard: float = 0.0
    lambda_soft: float = 0.5
    tau: float = 0.1
    label_smoothing: float = 0.1
    wer_cap: float = 1.0
    class_balanced: bool = False
    word_weighted: bool = False    # every term and the validation WER weighted by reference words (noisier selection; off)
    max_restarts: int = 1          # refit from a new seed when the validation cut shows no gain over the BSM
    # model selection
    select_on: str = "soft"        # "soft": validation expected WER under the routing distribution; "hard": argmax WER
    ema_epochs: float = 2.0        # EMA horizon of the weights used for validation and the final model, in epochs; 0 = off
    min_epochs: int = 10           # early stopping cannot fire before this many epochs
    verbose: int = 0

    def base_lr(self, total_steps):
        """The optimiser's learning rate for a run of `total_steps` (before warm-up and cosine decay)."""
        return self.lr

    def start(self, model, bsm):
        """Called once before training, with the training cut's best single member. Does nothing here."""
        return None

    def loss_kwargs(self):
        return dict(lambda_wer=self.lambda_wer, lambda_hard=self.lambda_hard,
                    lambda_soft=self.lambda_soft, tau=self.tau,
                    label_smoothing=self.label_smoothing, wer_cap=self.wer_cap)


def wer_targets(E, nref, rows, cap=None):
    """`(n, K)` per-clip WER for `rows`, capped when asked. float32."""
    Y = (E[rows] / np.maximum(nref[rows], 1)[:, None]).astype(np.float32)
    return np.minimum(Y, cap) if cap else Y


def _device_of(model):
    return next(model.parameters()).device


def train_router(model, batcher, Y, rows, cfg, verbose=None, callback=None, nref=None, step_offset=0):
    """Fit `model` on `rows` (indices into `Y`) with `cfg`. Returns a history dict.

    `batcher(rows) -> model inputs` (a tuple the model is called with).
    `Y` is the `(N, K)` WER matrix over the store; `rows` index it. `nref`
    `(N,)` is the reference word count per clip — with `cfg.word_weighted`
    the loss and the validation WER are word-weighted (corpus WER).
    `callback(epoch, val_wer)` is the pruning hook the tuner arms.
    """
    verbose = cfg.verbose if verbose is None else verbose
    device = _device_of(model)
    rows = np.asarray(rows, dtype=int)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(rows))
    n_val = int(round(len(rows) * cfg.val_frac)) if cfg.val_frac else 0
    val_rows, tr_rows = rows[perm[:n_val]], rows[perm[n_val:]]
    Yt = torch.as_tensor(Y, device=device)
    if cfg.word_weighted and nref is not None:
        Wt = torch.as_tensor(np.maximum(np.asarray(nref, dtype=np.float32), 1.0), device=device)
    else:
        Wt = torch.ones(len(Y), dtype=torch.float32, device=device)
    # the reference point for the restart guard: the best single member on the training cut, scored on the validation cut
    w_tr = Wt[torch.as_tensor(tr_rows, device=device)]
    bsm = int(((Yt[torch.as_tensor(tr_rows, device=device)] * w_tr[:, None]).sum(0) / w_tr.sum()).argmin())
    cfg.start(model, bsm)

    class_weights = None
    if cfg.class_balanced and cfg.lambda_hard:
        counts = np.bincount(Y[tr_rows].argmin(1), minlength=Y.shape[1]).astype(np.float64)
        w = 1.0 / np.maximum(counts, 1)
        class_weights = torch.as_tensor(w / w.sum() * len(w), dtype=torch.float32, device=device)

    steps_per_epoch = max(1, math.ceil(len(tr_rows) / cfg.batch_size))
    total = steps_per_epoch * cfg.epochs
    lr = cfg.base_lr(total)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = bool(cfg.amp and device.type == "cuda")
    torch.manual_seed(cfg.seed)
    K = Y.shape[1]

    # the EMA of the weights: decay per step from the horizon in epochs, ramped up from the first step
    ema_decay = math.exp(-1.0 / (cfg.ema_epochs * steps_per_epoch)) if cfg.ema_epochs else 0.0
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if ema_decay else None

    def ema_update(step):
        d = min(ema_decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    ema[k].copy_(v)

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def load(state):
        cur = model.state_dict()
        model.load_state_dict({k: v.to(cur[k].dtype) for k, v in state.items()})

    def evaluate(idx):
        """`(argmax WER, soft expected WER, member shares)` on `idx` — with the EMA weights when configured."""
        if ema is not None:
            live = snapshot(); load(ema)
        model.eval()
        picked, soft, total_w = 0.0, 0.0, 0.0
        counts = torch.zeros(K, dtype=torch.long, device=device)
        with torch.no_grad():
            for s in range(0, len(idx), cfg.batch_size):
                b = idx[s:s + cfg.batch_size]
                bt = torch.as_tensor(b, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(*batcher(b))
                choice = logits.argmax(-1)
                probs = torch.softmax(logits.float(), -1)
                picked += float((Yt[bt, choice] * Wt[bt]).sum())
                soft += float(((probs * Yt[bt]).sum(-1) * Wt[bt]).sum())
                total_w += float(Wt[bt].sum())
                counts += torch.bincount(choice, minlength=K)
        model.train()
        if ema is not None:
            load(live)
        share = (counts.double() / max(int(counts.sum()), 1)).cpu().numpy().round(4).tolist()
        return picked / max(total_w, 1), soft / max(total_w, 1), share

    if n_val:
        vt = torch.as_tensor(val_rows, device=device)
        bsm_val = float((Yt[vt, bsm] * Wt[vt]).sum() / Wt[vt].sum())
    else:
        bsm_val = float("nan")
    history = {"epoch": [], "train_loss": [], "val_wer": [], "val_soft": [], "val_share": [], "bsm_val_wer": bsm_val,
               "lr": lr}
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0       # `best` is the selection criterion
    best_hard = float("inf")
    step, t0 = 0, time.perf_counter()
    model.train()
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(tr_rows))
        losses = []
        for s in range(0, len(tr_rows), cfg.batch_size):
            b = tr_rows[order[s:s + cfg.batch_size]]
            bt = torch.as_tensor(b, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(*batcher(b))
            loss, _ = composite_loss(logits, Yt[bt], class_weights=class_weights,
                                     weights=Wt[bt] if cfg.word_weighted else None, **cfg.loss_kwargs())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            if ema is not None:
                ema_update(step)
            step += 1
            losses.append(loss.item())
        if not np.isfinite(np.mean(losses)):
            # a NaN loss has already poisoned AdamW's moments; nothing after this epoch is meaningful
            print(f"    epoch {epoch + 1:3d}  loss is not finite — stopping; the best state so far is kept", flush=True)
            history["diverged"] = epoch
            break
        val, val_soft, share = evaluate(val_rows) if n_val else (float(np.mean(losses)), float(np.mean(losses)), None)
        crit = val_soft if cfg.select_on == "soft" else val
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_wer"].append(val)
        history["val_soft"].append(val_soft)
        history["val_share"].append(share)
        if verbose:
            shares = "" if share is None else "  share " + "/".join(f"{x:.2f}" for x in share)
            print(f"    epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val_wer {val:.4f}  val_soft {val_soft:.4f}{shares}"
                  f"  {time.perf_counter() - t0:.0f}s", flush=True)
        if callback is not None:
            callback(step_offset + epoch, val)
        if crit < best - 1e-6:
            best, best_epoch, bad, best_hard = crit, epoch, 0, val
            best_state = {k: v.clone() for k, v in ema.items()} if ema is not None else snapshot()
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience and epoch + 1 >= cfg.min_epochs:
                break
    if best_state is not None:
        load(best_state)
    model.eval()
    best_share = history["val_share"][best_epoch] if 0 <= best_epoch < len(history["val_share"]) else None
    history.update(best_val_wer=best_hard, best_val_crit=best, best_epoch=best_epoch, epochs_run=len(history["epoch"]),
                   best_val_share=best_share, select_on=cfg.select_on, ema_epochs=cfg.ema_epochs,
                   collapsed=bool(best_share is not None and max(best_share) > 0.98),
                   no_gain=bool(np.isfinite(bsm_val) and best_hard >= bsm_val - 1e-4),
                   seconds=time.perf_counter() - t0, n_train=int(len(tr_rows)), n_val=int(n_val))
    return history


def fit_with_restarts(build, batcher, Y, rows, cfg, callback=None, nref=None):
    """`train_router` with the restart guard: `build(seed) -> model`; a fit whose validation cut shows
    no gain over the best single member is retried from the next seed, up to `cfg.max_restarts`
    times, and the attempt with the lowest validation WER is kept. Returns `(model, history)`."""
    best_model, best_hist = None, None
    for attempt in range(1 + max(0, int(cfg.max_restarts))):
        model = build(cfg.seed + attempt)
        hist = train_router(model, batcher, Y, rows, replace(cfg, seed=cfg.seed + attempt), callback=callback,
                            nref=nref, step_offset=attempt * cfg.epochs)
        hist["attempt"] = attempt
        if best_hist is None or hist["best_val_wer"] < best_hist["best_val_wer"]:
            best_model, best_hist = model, hist
        if not hist["no_gain"] and not hist.get("diverged"):
            break
        if cfg.verbose or attempt < cfg.max_restarts:
            print(f"    attempt {attempt + 1}: validation WER {hist['best_val_wer']:.4f} vs BSM {hist['bsm_val_wer']:.4f}"
                  + (" — restarting from a new seed" if attempt < cfg.max_restarts else " — keeping the best attempt"), flush=True)
    best_hist["restarts"] = best_hist["attempt"]
    return best_model, best_hist


def predict_logits(model, batcher, rows, batch_size=64, amp=True):
    """`(n, K)` float32 logits for `rows`, in order."""
    device = _device_of(model)
    use_amp = bool(amp and device.type == "cuda")
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = np.asarray(rows[s:s + batch_size], dtype=int)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out.append(model(*batcher(b)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), dtype=np.float32)