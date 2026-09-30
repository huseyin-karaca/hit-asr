__all__ = ['RouterStore']

import numpy as np

from hitasr.frames import FrameSet
from hitasr.labels import LabelStore


class RouterStore:
    """The error matrix of `members` plus their frames, aligned in base order."""

    def __init__(self, labels, members, frames_root=None):
        if not isinstance(labels, LabelStore) or labels.ids is None:
            raise TypeError("labels must be a loaded LabelStore")
        self.labels, self.members = labels, tuple(members)
        missing = [m for m in self.members if m not in labels.experts]
        if missing:
            raise KeyError(f"labels for {missing} are not loaded")
        self.spec = labels.spec
        self.frames_root = frames_root
        self.frames = {}
        self.device = None
        self.E = labels.E(self.members)
        self.nref = labels.nref
        self.n = labels.n

    def __repr__(self):
        where = "closed" if not self.frames else ("cpu-mapped" if self.device is None else str(self.device))
        return f"RouterStore({self.spec.name}, {'+'.join(self.members)}, {self.n:,} rows, frames {where})"

    def open(self, device=None, verbose=True, frames=True):
        """Open every member's `FrameSet`, verify alignment, optionally move to `device`. Returns self.

        `frames=False` leaves the frames closed: the labels, the error matrix and the pooled vectors are all there,
        and only a fit of a frame-level arm (which asks for `batch`) would need more — what a run that reads its
        results from the cache uses, so it neither downloads the frames nor needs a GPU.
        """
        if not frames:
            self.device = device
            return self
        for m in self.members:
            fs = FrameSet(self.spec, m, root=self.frames_root).open(verbose=verbose)
            if fs.ids != self.labels.ids:
                if set(fs.ids) != set(self.labels.ids):
                    raise ValueError(f"{m}: frame store and labels cover different utterances")
                raise ValueError(f"{m}: frame store rows are not in the labels' order")
            if device is not None:
                fs.to(device, verbose=verbose)
            self.frames[m] = fs
        self.device = device
        return self

    @property
    def dims(self):
        """`{member: D}`."""
        return {m: fs.d for m, fs in self.frames.items()}

    def batch(self, rows, max_frames=None):
        """`({member: (B, T_m, D_m) fp16}, {member: (B,) int64})` for `rows`."""
        if not self.frames:
            raise RuntimeError("the frames are closed — open(frames=True) (a cached run never asks for them)")
        out, lens = {}, {}
        for m, fs in self.frames.items():
            out[m], lens[m] = fs.batch(rows, max_frames=max_frames)
        return out, lens

    def pooled(self, members=None):
        """`(N, sum D)` float32 — the labels' mean-pooled frames of `members`, concatenated."""
        return np.concatenate([self.labels.pooled(m) for m in (members or self.members)], axis=1)

    def n_frames(self, member=None):
        return self.labels.n_frames[member or self.members[0]]