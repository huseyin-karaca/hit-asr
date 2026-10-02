__all__ = ['WEIGHT_SCHEMES', 'FUSION_RIVALS', 'word_counters', 'build_network', 'vote', 'system_weights', 'edit_distance',
           'agreement', 'check_counters', 'RidgeProbe', 'FusionCache', 'fiscus_vote', 'ConfWeightedRover',
           'CNMBRFusion', 'conf_rover_space', 'cn_mbr_space']

import hashlib

import numpy as np
import pandas as pd

from hitasr.arms import Arm, ModelSpec, register
from labkit.search import Param

WEIGHT_SCHEMES = ("uniform", "inverse_wer", "log_odds")


def word_counters(ref, hyp):
    """`(sub, dele, ins, nref)` for one already-normalised pair, via `jiwer`."""
    import jiwer
    if not ref.strip():
        return 0, 0, len(hyp.split()), 0
    w = jiwer.process_words(ref, hyp)
    return w.substitutions, w.deletions, w.insertions, (w.hits + w.substitutions + w.deletions)


def _levenshtein_path(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    path, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            path.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            path.append((i - 1, None)); i -= 1
        else:
            path.append((None, j - 1)); j -= 1
    return path[::-1]


def _majority(words):
    if not words:
        return ""
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(words, key=lambda w: (counts[w], -words.index(w)))


def build_network(hyps):
    """Iterative ROVER alignment of `k` hypotheses into a word transition network (list of slots)."""
    k = len(hyps)
    tokens = [h.split() for h in hyps]
    slots = [[w] + [None] * (k - 1) for w in tokens[0]]
    for s in range(1, k):
        spine = [_majority([t for t in slot if t is not None]) for slot in slots]
        merged = []
        for slot_i, tok_j in _levenshtein_path(spine, tokens[s]):
            if slot_i is None:
                new = [None] * k
                new[s] = tokens[s][tok_j]
                merged.append(new)
            else:
                slot = slots[slot_i]
                slot[s] = None if tok_j is None else tokens[s][tok_j]
                merged.append(slot)
        slots = merged
    return slots


def vote(slots, weights, null_penalty=1.0):
    """Vote one network into a transcript. Ties break towards emitting a word."""
    words = []
    for slot in slots:
        scores = {}
        for w, weight in zip(slot, weights):
            scores[w] = scores.get(w, 0.0) + weight * (null_penalty if w is None else 1.0)
        best = max(scores, key=lambda key: (scores[key], key is not None))
        if best is not None:
            words.append(best)
    return " ".join(words)


def system_weights(scheme, wer, floor=1e-3):
    """Per-system voting weights from their WER on the training half."""
    wer = np.clip(np.asarray(wer, dtype=np.float64), floor, 1 - floor)
    if scheme == "uniform":
        w = np.ones_like(wer)
    elif scheme == "inverse_wer":
        w = 1.0 / wer
    elif scheme == "log_odds":
        w = np.log((1.0 - wer) / wer)
    else:
        raise ValueError(f"weight scheme must be one of {WEIGHT_SCHEMES}, got {scheme!r}")
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def agreement(hyps):
    """`(k,)` mean agreement of each hypothesis with the others, in `[0, 1]` — the `cn` confidence."""
    k = len(hyps)
    if k < 2:
        return np.ones(k)
    toks = [h.split() for h in hyps]
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            denom = max(len(toks[i]), len(toks[j]))
            D[i, j] = D[j, i] = 0.0 if denom == 0 else edit_distance(toks[i], toks[j]) / denom
    out = np.array([1.0 - D[i].sum() / (k - 1) for i in range(k)])
    return np.clip(out, 0.0, 1.0)


def check_counters(store, n=200, seed=0):
    """Re-score `n` random utterances from the stored strings; the counters must reproduce exactly."""
    labels = store.labels
    rng = np.random.default_rng(seed)
    picks = rng.choice(store.n, size=min(n, store.n), replace=False)
    rows = []
    for row in picks:
        ref = str(labels.ref_text[row])
        for j, m in enumerate(store.members):
            s, d, i, _ = word_counters(ref, str(labels.texts_norm[m][row]))
            rows.append({"row": int(row), "model": m, "recomputed": s + d + i, "stored": float(store.E[row, j])})
    out = pd.DataFrame(rows)
    out["match"] = out["recomputed"] == out["stored"]
    rate = out["match"].mean()
    print(f"{rate:.1%} of {len(out):,} (utterance, model) counters reproduce exactly")
    return out

class RidgeProbe:
    """Multi-output ridge with exact leave-one-out alpha selection, via one SVD."""

    def __init__(self, alphas=None):
        self.alphas = np.logspace(-2, 6, 12) if alphas is None else np.asarray(alphas, dtype=np.float64)
        self.coef_, self.intercept_, self.alpha_ = None, None, None

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        xm, ym = X.mean(0), Y.mean(0)
        Xc, Yc = X - xm, Y - ym
        if not (np.isfinite(Xc).all() and np.isfinite(Yc).all()):
            raise ValueError("RidgeProbe: non-finite design or targets — check the pooled features "
                             "(`Labels.n_nonfinite`) before fitting")
        try:
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd  # gesvd: slower, but converges where gesdd gives up
            U, s, Vt = svd(Xc, full_matrices=False, lapack_driver="gesvd")
        UtY = U.T @ Yc
        best, best_alpha, best_coef = np.inf, None, None
        for a in self.alphas:
            d = s / (s ** 2 + a)
            coef = (Vt.T * d) @ UtY
            h = ((U ** 2) * (s ** 2 / (s ** 2 + a))).sum(1)
            resid = Yc - Xc @ coef
            loo = ((resid / np.maximum(1 - h, 1e-8)[:, None]) ** 2).mean()
            if loo < best:
                best, best_alpha, best_coef = loo, a, coef
        self.coef_, self.alpha_ = best_coef, best_alpha
        self.intercept_ = ym - xm @ best_coef
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class FusionCache:
    """The memo tables every combination arm shares. One per run (`ctx["fusion_cache"]`)."""

    def __init__(self):
        self.networks, self.agreement, self.scores, self.probes = {}, {}, {}, {}

    @classmethod
    def get(cls, ctx):
        if ctx is None:
            return cls()
        got = ctx.get("fusion_cache")
        if got is None:
            got = ctx["fusion_cache"] = cls()
        return got

    def __repr__(self):
        return (f"FusionCache(networks={len(self.networks):,}, agreement={len(self.agreement):,}, "
                f"scores={len(self.scores):,}, probes={len(self.probes)})")

    @staticmethod
    def hypotheses(store, row):
        return [str(store.labels.texts_norm[m][int(row)]) for m in store.members]

    def network(self, store, row):
        key = int(row)
        if key not in self.networks:
            self.networks[key] = build_network(self.hypotheses(store, key))
        return self.networks[key]

    def agree(self, store, row):
        key = int(row)
        if key not in self.agreement:
            self.agreement[key] = agreement(self.hypotheses(store, key))
        return self.agreement[key]

    def errors(self, store, rows, hyps):
        out = np.empty(len(rows), dtype=np.float64)
        for i, (row, hyp) in enumerate(zip(np.asarray(rows, dtype=int), hyps)):
            key = (int(row), hyp)
            got = self.scores.get(key)
            if got is None:
                got = self.scores[key] = float(sum(word_counters(str(store.labels.ref_text[int(row)]), hyp)[:3]))
            out[i] = got
        return out

    def probe(self, X, Y, rows, alphas):
        key = (hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16],
               X.shape[1], round(float(X.sum()), 3), len(alphas))
        got = self.probes.get(key)
        if got is None:
            got = self.probes[key] = RidgeProbe(alphas=alphas).fit(X, Y)
        return got

def _blend(alpha, freq, conf, null_penalty, is_null):
    return (alpha * freq + (1.0 - alpha) * conf) * (null_penalty if is_null else 1.0)


def fiscus_vote(slots, mass, conf, alpha=1.0, null_penalty=1.0):
    """Vote one network with per-system mass and per-system confidence."""
    words = []
    for slot in slots:
        freq, conf_mass = {}, {}
        for w, m, c in zip(slot, mass, conf):
            freq[w] = freq.get(w, 0.0) + m
            conf_mass[w] = conf_mass.get(w, 0.0) + m * c
        best, best_score = None, -np.inf
        for w, f in freq.items():
            score = _blend(alpha, f, (conf_mass[w] / f) if f > 0 else 0.0, null_penalty, w is None)
            if (score, w is not None) > (best_score, best is not None):
                best, best_score = w, score
        if best is not None:
            words.append(best)
    return " ".join(words)


class _FusionArm(Arm):
    """Shared plumbing: the fold's WER-derived weights, the pooled design, the probe."""

    wants_texts = True

    def __init__(self, ctx=None, alphas=None):
        self.cache = FusionCache.get(ctx)
        self.alphas = np.logspace(-2, 6, 12) if alphas is None else alphas
        self.weights_, self.probe_, self.mu, self.sd = None, None, None, None

    def _member_wer(self, store, rows):
        return store.E[rows].sum(0) / store.nref[rows].sum()

    def _fit_probe(self, store, rows):
        X = store.pooled()
        self.mu, self.sd = X[rows].mean(0), X[rows].std(0).clip(1e-6)
        Y = store.E[rows] / np.maximum(store.nref[rows], 1)[:, None]
        self.probe_ = self.cache.probe((X[rows] - self.mu) / self.sd, Y, rows, self.alphas)

    def _probe_conf(self, store, rows):
        X = (store.pooled()[rows] - self.mu) / self.sd
        return np.clip(1.0 - self.probe_.predict(X), 0.0, 1.0)


class ConfWeightedRover(_FusionArm):
    """Confidence-weighted ROVER. `alpha=1`, `weight_by_conf=False`, `conf_source="cn"` is classical weighted ROVER."""

    produces = "errors"

    def __init__(self, alpha=1.0, conf_source="cn", weighting="inverse_wer", weight_by_conf=False,
                 conf_temp=1.0, null_penalty=1.0, ctx=None, **_):
        super().__init__(ctx)
        self.alpha, self.conf_source, self.weighting = float(alpha), conf_source, weighting
        self.weight_by_conf, self.conf_temp, self.null_penalty = bool(weight_by_conf), float(conf_temp), float(null_penalty)

    def fit(self, store, rows, callback=None):
        rows = np.asarray(rows, dtype=int)
        self.weights_ = system_weights(self.weighting, self._member_wer(store, rows))
        if self.conf_source in ("probe", "blend"):
            self._fit_probe(store, rows)
        return self

    def _confidence(self, store, rows):
        cn = np.vstack([self.cache.agree(store, r) for r in rows]) if self.conf_source in ("cn", "blend") else None
        pr = self._probe_conf(store, rows) if self.conf_source in ("probe", "blend") else None
        conf = cn if self.conf_source == "cn" else pr if self.conf_source == "probe" else \
            np.sqrt(np.clip(cn, 1e-6, None) * np.clip(pr, 1e-6, None))
        return np.clip(conf, 1e-6, 1.0) ** self.conf_temp

    def transcribe(self, store, rows):
        conf = self._confidence(store, rows)
        out = []
        for i, row in enumerate(rows):
            slots = self.cache.network(store, row)
            mass = (self.weights_ * conf[i]) if self.weight_by_conf else self.weights_
            out.append(fiscus_vote(slots, mass, conf[i], self.alpha, self.null_penalty))
        return out

    def predict(self, store, rows):
        rows = np.asarray(rows, dtype=int)
        return self.cache.errors(store, rows, self.transcribe(store, rows))

    def predict_proba(self, store, rows):
        raise NotImplementedError("a fusion arm selects no member")


class CNMBRFusion(_FusionArm):
    """Confusion-network MBR combination."""

    def __init__(self, decode="slot", candidates="members", prior="inverse_wer", temperature=1.0,
                 null_prior=1.0, use_conf=False, conf_source="cn", ctx=None, **_):
        super().__init__(ctx)
        self.decode, self.candidates, self.prior = decode, candidates, prior
        self.temperature, self.null_prior, self.use_conf, self.conf_source = float(temperature), float(null_prior), bool(use_conf), conf_source
        self.produces = "choice" if (decode == "hyp" and candidates == "members") else "errors"

    def fit(self, store, rows, callback=None):
        rows = np.asarray(rows, dtype=int)
        self.weights_ = system_weights(self.prior, self._member_wer(store, rows))
        if self.use_conf and self.conf_source in ("probe", "blend"):
            self._fit_probe(store, rows)
        return self

    def _confidence(self, store, rows):
        k = len(store.members)
        if not self.use_conf:
            return np.ones((len(rows), k))
        if self.conf_source == "cn":
            return np.clip(np.vstack([self.cache.agree(store, r) for r in rows]), 1e-6, None)
        pr = np.clip(self._probe_conf(store, rows), 1e-6, 1.0)
        if self.conf_source == "probe":
            return pr
        cn = np.vstack([self.cache.agree(store, r) for r in rows])
        return np.sqrt(np.clip(cn, 1e-6, None) * pr)

    def _posterior(self, slots, mass):
        out = []
        for slot in slots:
            raw = {}
            for w, m in zip(slot, mass):
                raw[w] = raw.get(w, 0.0) + m
            top = max(raw.values(), default=0.0)
            if top <= 0.0:
                raw, top = {w: 1.0 for w in raw}, 1.0
            post = {w: ((v / top) ** (1.0 / self.temperature)) for w, v in raw.items() if v > 0}
            if None in post:
                post[None] *= self.null_prior
            total = sum(post.values()) or 1.0
            out.append({w: v / total for w, v in post.items()})
        return out

    @staticmethod
    def _expected_error(slots, post, system):
        return sum(1.0 - p.get(slot[system], 0.0) for slot, p in zip(slots, post))

    @staticmethod
    def _consensus(post):
        return " ".join(w for w in (max(p, key=lambda k: (p[k], k is not None)) for p in post) if w is not None)

    def _decode_row(self, store, row, mass):
        slots = self.cache.network(store, row)
        post = self._posterior(slots, mass)
        if self.decode == "slot":
            return self._consensus(post), -1
        costs = [self._expected_error(slots, post, s) for s in range(len(store.members))]
        pick = int(np.argmin(costs))
        if self.candidates == "members":
            return None, pick
        cons_cost = sum(1.0 - max(p.values()) for p in post)
        if cons_cost < costs[pick]:
            return self._consensus(post), -1
        return str(store.labels.texts_norm[store.members[pick]][int(row)]), pick

    def predict(self, store, rows):
        rows = np.asarray(rows, dtype=int)
        conf = self._confidence(store, rows)
        hyps, picks = [], []
        for i, row in enumerate(rows):
            hyp, pick = self._decode_row(store, row, self.weights_ * conf[i])
            hyps.append(hyp); picks.append(pick)
        if self.produces == "choice":
            return np.asarray(picks, dtype=int)
        return self.cache.errors(store, rows, [str(h) for h in hyps])

    def predict_proba(self, store, rows):
        raise NotImplementedError("CN-MBR carries no routing distribution")


def conf_rover_space():
    return {p.name: p for p in [
        Param("alpha", "cat", choices=(1.0, 0.75, 0.5, 0.25, 0.0)),
        Param("conf_source", "cat", choices=("cn", "probe", "blend")),
        Param("weighting", "cat", choices=WEIGHT_SCHEMES),
        Param("weight_by_conf", "cat", choices=(False, True)),
        Param("conf_temp", "cat", choices=(0.25, 0.5, 1.0, 2.0, 4.0)),
        Param("null_penalty", "cat", choices=(0.25, 0.5, 1.0, 2.0, 4.0)),
    ]}


def cn_mbr_space():
    return {p.name: p for p in [
        Param("decode", "cat", choices=("slot", "hyp")),
        Param("candidates", "cat", choices=("members", "members+consensus")),
        Param("prior", "cat", choices=WEIGHT_SCHEMES),
        Param("temperature", "cat", choices=(0.1, 0.3, 1.0, 3.0, 10.0)),
        Param("null_prior", "cat", choices=(0.05, 0.2, 1.0, 2.0, 5.0)),
        Param("use_conf", "cat", choices=(False, True)),
        Param("conf_source", "cat", choices=("cn", "probe", "blend")),
    ]}


register(ModelSpec("rover_conf", conf_rover_space, lambda p, store, seed=42, ctx=None: ConfWeightedRover(ctx=ctx, **p),
                   family="external", frames=False, label="ROVER (confidence-weighted)"))
register(ModelSpec("cn_mbr", cn_mbr_space, lambda p, store, seed=42, ctx=None: CNMBRFusion(ctx=ctx, **p),
                   family="external", frames=False, label="CN-MBR"))

FUSION_RIVALS = ("rover_conf", "cn_mbr")
