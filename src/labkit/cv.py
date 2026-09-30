"""Resampling: Dietterich's 5x2 (and one repetition of it, addressed by its seed), shuffled k-fold, and the
pool-and-recut hold-out partition a search may see."""

__all__ = ['BaseSplitScheme', 'FiveByTwoSplit', 'RepetitionSplit', 'KFoldSplit', 'random_partition', 'fold_rng']

import hashlib

import numpy as np


class BaseSplitScheme:
    """A resampling scheme. Subclasses yield `(repeat, fold, train_idx, eval_idx)`."""

    name = "base"

    def split(self, n):
        raise NotImplementedError

    def repeat_folds(self, n, seed):
        raise NotImplementedError

    @property
    def n_folds(self):
        raise NotImplementedError


class FiveByTwoSplit(BaseSplitScheme):
    """Dietterich's 5x2: one shuffle per repetition, each cut into complementary halves.

    `seeds` gives one seed **per repetition**; repetition `i` is determined by
    `seeds[i]` and nothing else. Report the seeds beside any p-value —
    `seeds[0]` alone decides the *t*-test's numerator.
    """

    name = "5x2cv"

    def __init__(self, seeds=(11, 22, 33, 44, 55)):
        seeds = tuple(int(s) for s in seeds)
        if len(seeds) < 2:
            raise ValueError("5x2cv needs at least 2 repetitions")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"seeds must be distinct, got {seeds}")
        self.seeds = seeds
        self.n_repeats = len(seeds)

    @property
    def n_folds(self):
        return 2 * self.n_repeats

    def repeat_seed(self, repeat):
        return self.seeds[repeat]

    def repeat_folds(self, n, seed):
        order = np.random.default_rng(seed).permutation(n)
        a, b = order[: n // 2], order[n // 2:]
        return [(0, a, b), (1, b, a)]

    def split(self, n):
        for repeat in range(self.n_repeats):
            for fold, tr, ev in self.repeat_folds(n, self.repeat_seed(repeat)):
                yield repeat, fold, tr, ev


class RepetitionSplit(FiveByTwoSplit):
    """ONE repetition of a 5x2, addressed by its seed. Two folds, byte-identical to the 5x2's."""

    def __init__(self, seed):
        self.seeds, self.n_repeats = (int(seed),), 1


class KFoldSplit(BaseSplitScheme):
    """Plain shuffled k-fold — the inner loop a tuner may use, never the outer test."""

    name = "kfold"

    def __init__(self, k=3, seed=0):
        self.k, self.seed = int(k), seed

    @property
    def n_folds(self):
        return self.k

    def repeat_folds(self, n, seed):
        order = np.random.default_rng(seed).permutation(n)
        parts = np.array_split(order, self.k)
        return [(i, np.concatenate([q for j, q in enumerate(parts) if j != i]), parts[i])
                for i in range(self.k)]

    def split(self, n):
        for fold, tr, ev in self.repeat_folds(n, self.seed):
            yield 0, fold, tr, ev


def random_partition(n, holdout_frac=0.25, inner_val_frac=0.30, seed=20260901):
    """Pool-and-recut: `{tune, eval, fit, val}` row indices over `n` pooled rows.

    A function of `seed` alone. `eval` goes to `run_cv(rows=...)` and is never
    seen by a trial; `fit`/`val` cut `tune` again so a trial is scored on rows
    it did not fit on. Every array is sorted, so `X[rows]` keeps store order.
    """
    if not 0 < holdout_frac < 1 or not 0 < inner_val_frac < 1:
        raise ValueError("fractions must be in (0, 1)")
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(np.clip(round(n * holdout_frac), 1, n - 1))
    tune_rows, eval_rows = np.sort(perm[:cut]), np.sort(perm[cut:])
    inner = np.random.default_rng(seed + 1).permutation(len(tune_rows))
    icut = int(np.clip(round(len(tune_rows) * (1 - inner_val_frac)), 1, len(tune_rows) - 1))
    return {"tune": tune_rows, "eval": eval_rows,
            "fit": tune_rows[np.sort(inner[:icut])], "val": tune_rows[np.sort(inner[icut:])]}


def fold_rng(random_seed, repeat_key, fold):
    digest = hashlib.sha1(f"{random_seed}|{repeat_key}|{fold}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))
