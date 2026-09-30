__all__ = ['FRAME_DTYPE', 'SHARD_BYTES', 'LABEL_COLUMNS', 'default_frames_dir', 'FrameWriter', 'write_manifest', 'FrameSet', 'labels_features']

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hitasr.core import active_dataset
from hitasr.hub import HitHub

FRAME_DTYPE = np.float16
SHARD_BYTES = 2 << 30                  # ~2 GB per shard: few, large files move fastest between Colab and the Hub


def default_frames_dir():
    """Where frame shards live locally: `$HITASR_CACHE/frames` or `~/.cache/hitasr/frames`.

    On Colab, set `HITASR_CACHE=/content/hitasr_cache` — the local disk is
    what makes memory-mapped reads fast, and Drive is not local disk.
    """
    import os
    root = Path(os.environ.get("HITASR_CACHE", Path.home() / ".cache" / "hitasr"))
    return root / "frames"


class FrameWriter:
    """Append `(T, D)` frame arrays for one (corpus, expert, split); write ~1 GB shards.

    Usage
    -----
        w = FrameWriter(out_dir / "test", d=1024)
        for ids, hidden, lengths in batches:
            w.add(ids, hidden, lengths)          # (B, T, D) padded + valid lengths
        index = w.close()                        # one frame per utterance

    Frames are cast to fp16 on the way in; shards roll over when the buffered
    bytes exceed `shard_bytes`. Nothing is held beyond one shard's worth.
    """

    def __init__(self, out_dir, d, shard_bytes=SHARD_BYTES, dtype=FRAME_DTYPE):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.d, self.dtype, self.shard_bytes = int(d), dtype, int(shard_bytes)
        self._buf, self._buf_bytes, self._buf_rows = [], 0, []
        self._offset = 0                # frames written into the CURRENT shard
        self.shard = 0
        self.index_rows = []            # every utterance: id, shard, offset, n_frames
        self.total_frames = 0

    def add(self, ids, hidden, lengths):
        """One batch: `hidden` `(B, T, D)` (tensor or array), `lengths` valid frames per row."""
        arr = hidden.detach().to("cpu").float().numpy() if hasattr(hidden, "detach") else np.asarray(hidden)
        lens = [int(n) for n in (lengths.tolist() if hasattr(lengths, "tolist") else lengths)]
        for i, (uid, n) in enumerate(zip(ids, lens)):
            n = max(0, min(n, arr.shape[1]))
            x = np.ascontiguousarray(arr[i, :n].astype(self.dtype, copy=False))
            self._buf.append(x)
            self._buf_bytes += x.nbytes
            self.index_rows.append({"id": str(uid), "shard": self.shard,
                                    "offset": self._offset, "n_frames": n})
            self._offset += n
            self.total_frames += n
            if self._buf_bytes >= self.shard_bytes:
                self._flush()

    def _flush(self):
        if not self._buf:
            return
        block = np.concatenate(self._buf, axis=0) if len(self._buf) > 1 else self._buf[0]
        np.save(self.dir / f"shard-{self.shard:05d}.npy", block)
        self._buf, self._buf_bytes = [], 0
        self.shard += 1
        self._offset = 0

    def close(self):
        """Flush the last shard and write the index. Returns the index as a DataFrame."""
        self._flush()
        index = pd.DataFrame(self.index_rows, columns=["id", "shard", "offset", "n_frames"])
        pq.write_table(pa.Table.from_pandas(index, preserve_index=False),
                       self.dir / "index.parquet")
        return index


def write_manifest(out_dir, spec, expert, d, frame_rate_hz, layer, splits, extra=None):
    """`manifest.json` for one expert's frame store: what a reader must know before opening it."""
    man = {"dataset": spec.name, "expert": expert, "d": int(d),
           "frame_rate_hz": frame_rate_hz, "layer": layer, "dtype": np.dtype(FRAME_DTYPE).name,
           "layout": "per split: shard-NNNNN.npy (sum_T, d) fp16 + index.parquet (id, shard, offset, n_frames)",
           "splits": splits, **(extra or {})}
    Path(out_dir, "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    return man


@dataclass
class FrameSet:
    """Random access to one expert's frames on one corpus — every split, base order.

    Parameters
    ----------
    spec, expert : which store. `splits` defaults to the spec's.
    root : local directory holding `<frames_config>/<split>/shard-*.npy`; the
        Hub copy is fetched into it by `fetch()` when it is missing.

    After `open()`: `n` utterances, `ids` (the `(split, id)` keys in order),
    `n_frames` `(n,)`, and `get(i)` -> `(T_i, D)` fp16 view. `batch(rows,
    max_frames)` pads a set of rows into `(B, T, D)` + lengths, truncating to
    `max_frames` from the front — the manuscript's `T_max`.

    `to(device)` moves every shard onto the device as one contiguous tensor
    (fp16) and switches `get`/`batch` to slice it there. A corpus-expert pair
    is 5-15 GB, so three members fit on a 96 GB GPU with room for the router;
    on anything smaller, stay memory-mapped.
    """

    spec: object
    expert: str
    splits: tuple = None
    root: Path = None
    manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        self.spec = active_dataset(self.spec)
        self.splits = tuple(self.splits or self.spec.splits)
        self.root = Path(self.root or default_frames_dir())
        self.dir = self.root / self.spec.frames_config(self.expert)
        self._shards = {}               # (split, shard) -> memmap or tensor
        self._index = None
        self._device = None

    def __repr__(self):
        state = (f"{self.n:,} utts, {int(self.n_frames.sum()):,} frames, D={self.d}"
                 if self._index is not None else "closed")
        return f"FrameSet({self.spec.name}/{self.expert}, {state})"

    # ----------------------------------------------------------- fetching --

    def fetch(self, verbose=True):
        """Download this store from the Hub into `root` (skips what is present). Returns self.

        One parallel `snapshot_download`, retried as a whole: finished shards
        stay in the `huggingface_hub` cache and a partial one resumes, so a
        re-run after `HubUnavailable` picks up where the last one stopped. The
        indexes and the manifest are linked in last — a split counts as
        present by its index, so it must never have one without its shards.
        """
        hub = HitHub(spec=self.spec)
        cfg = self.spec.frames_config(self.expert)
        want = [f"{cfg}/manifest.json"] + [f"{cfg}/{s}/*" for s in self.splits]
        missing = [s for s in self.splits if not (self.dir / s / "index.parquet").exists()]
        if not missing and (self.dir / "manifest.json").exists():
            return self
        if verbose:
            print(f"  fetching {cfg} ({', '.join(missing) or 'manifest'}) from {hub.repo_id}")
        snap = hub.prefetch(want, verbose=verbose, strict=True)
        src = Path(snap) / cfg
        if not (src / "manifest.json").exists():
            raise FileNotFoundError(f"{cfg} is not on {hub.repo_id} and not in {self.root}")
        self.dir.mkdir(parents=True, exist_ok=True)
        last = ("index.parquet", "manifest.json")
        for p in sorted((p for p in src.rglob("*") if p.is_file()), key=lambda p: p.name in last):
            dst = self.dir / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                # A hard link where the cache and the root share a
                # filesystem, a copy otherwise — never a symlink into the
                # cache, which is replaced under you on the next download.
                try:
                    import os
                    os.link(p.resolve(), dst)
                except OSError:
                    import shutil
                    shutil.copyfile(p.resolve(), dst)
        return self

    def evict(self, verbose=True):
        """Free the disk this store takes: its local copy under `root` and the shards' blobs in the `huggingface_hub`
        cache (the local copy is hard-linked to them, so both names must go). Close it first. Returns GB freed."""
        import os
        import shutil
        from huggingface_hub import snapshot_download
        freed = 0
        cfg = self.spec.frames_config(self.expert)
        hub = HitHub(spec=self.spec)
        if hub.local_root(cfg) is not None:                    # a local repo: its copy stays, only the links go
            snap = None
            if self.dir.resolve().is_relative_to(hub.local_root(cfg)):
                self._shards = {}
                return 0.0
        else:
            try:
                snap = snapshot_download(hub.repo_for(cfg), repo_type="dataset",
                                         allow_patterns=[f"{cfg}/*", f"{cfg}/**"], local_files_only=True)
            except Exception:                                  # noqa: BLE001 — nothing cached
                snap = None
        if snap is not None and (Path(snap) / cfg).exists():
            for p in (Path(snap) / cfg).rglob("*"):
                if p.is_symlink() or p.is_file():
                    blob = p.resolve()
                    if blob.exists():
                        freed += blob.stat().st_size
                        os.remove(blob)
                    os.remove(p)
        if self.dir.exists():
            freed += sum(p.stat().st_size for p in self.dir.rglob("*") if p.is_file() and p.stat().st_nlink == 1)
            shutil.rmtree(self.dir, ignore_errors=True)
        self._shards = {}
        if verbose:
            print(f"  evicted {cfg}: {freed / 1e9:.1f} GB freed")
        return freed / 1e9

    # ------------------------------------------------------------ opening --

    def open(self, fetch=True, verbose=True):
        """Read the manifest and every split's index; memory-map nothing yet. Returns self."""
        if fetch:
            self.fetch(verbose=verbose)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        frames = []
        for s in self.splits:
            idx = pq.read_table(self.dir / s / "index.parquet").to_pandas()
            idx.insert(0, "split", s)
            frames.append(idx)
        self._index = pd.concat(frames, ignore_index=True)
        self.n_frames = self._index["n_frames"].to_numpy().astype(np.int64)
        self.ids = list(zip(self._index["split"], self._index["id"]))
        return self

    @property
    def n(self):
        return int(len(self._index))

    @property
    def d(self):
        return int(self.manifest["d"])

    @property
    def frame_rate_hz(self):
        return self.manifest.get("frame_rate_hz")

    def index(self):
        """The `(split, id, shard, offset, n_frames)` table, base order."""
        return self._index

    def _shard(self, split, shard):
        key = (split, int(shard))
        if key not in self._shards:
            arr = np.load(self.dir / split / f"shard-{int(shard):05d}.npy", mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != self.d:
                raise ValueError(f"{key}: shard is {arr.shape}, expected (*, {self.d})")
            self._shards[key] = arr
        return self._shards[key]

    def get(self, i):
        """Utterance `i`'s frames, `(T_i, D)` fp16 — a view, not a copy."""
        r = self._index.iloc[int(i)]
        block = self._shard(r["split"], r["shard"])
        o, n = int(r["offset"]), int(r["n_frames"])
        return block[o:o + n]

    def to(self, device, verbose=True):
        """Load every shard onto `device` as one fp16 tensor per shard. Returns self.

        After this `batch()` gathers on the device: no host copy, no transfer
        per step. `device="cpu"` loads the shards into RAM instead of mapping.
        """
        import torch
        t0 = time.perf_counter()
        total = 0
        for s in self.splits:
            for k in sorted(self._index.loc[self._index["split"] == s, "shard"].unique()):
                arr = np.load(self.dir / s / f"shard-{int(k):05d}.npy", mmap_mode="r")
                self._shards[(s, int(k))] = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
                total += arr.nbytes
        self._device = torch.device(device)
        if verbose:
            print(f"  {self.spec.name}/{self.expert}: {total / 1e9:.1f} GB on {device} "
                  f"in {time.perf_counter() - t0:.0f}s")
        return self

    def batch(self, rows, max_frames=None):
        """`(frames, lengths)` for `rows`: `(B, T, D)` fp16 padded with zeros, `(B,)` int64.

        Sequences longer than `max_frames` keep their first `max_frames` frames.
        On a device-resident set both come back on the device; otherwise as
        torch tensors on the CPU, built from the memory maps.
        """
        import torch
        rows = [int(r) for r in rows]
        lens = self.n_frames[rows]
        if max_frames is not None:
            lens = np.minimum(lens, int(max_frames))
        T = int(lens.max()) if len(rows) else 0
        if self._device is not None:
            out = torch.zeros((len(rows), max(T, 1), self.d), dtype=torch.float16,
                              device=self._device)
            for j, (i, n) in enumerate(zip(rows, lens)):
                r = self._index.iloc[i]
                block = self._shards[(r["split"], int(r["shard"]))]
                o = int(r["offset"])
                out[j, :n] = block[o:o + int(n)]
            return out, torch.as_tensor(lens, device=self._device)
        out = np.zeros((len(rows), max(T, 1), self.d), dtype=np.float16)
        for j, (i, n) in enumerate(zip(rows, lens)):
            out[j, :n] = self.get(i)[:int(n)]
        return torch.from_numpy(out), torch.as_tensor(lens)

    def pooled(self, kind="mean", max_frames=None):
        """`(n, D)` float32 pooled vectors, computed off the frames. Slow on a memmap; fine on a device."""
        import torch
        out = np.zeros((self.n, self.d), dtype=np.float32)
        for i in range(self.n):
            x = np.asarray(self.get(i)[:max_frames] if max_frames else self.get(i), dtype=np.float32)
            if len(x) == 0:
                continue
            out[i] = x.mean(0) if kind == "mean" else x.max(0)
        return out

LABEL_COLUMNS = ("id", "transcription", "transcription_norm", "text_norm",
                 "sub", "dele", "ins", "nref", "wer",
                 "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw",
                 "n_frames")


def labels_features(d):
    """The Arrow schema of a labels config: `LABEL_COLUMNS` plus `pool_mean` of width `d`."""
    from datasets import Features, Sequence, Value

    feats = {"id": Value("string"), "transcription": Value("string"),
             "transcription_norm": Value("string"), "text_norm": Value("string")}
    for suffix in ("", "_raw"):
        for c in ("sub", "dele", "ins", "nref"):
            feats[f"{c}{suffix}"] = Value("int32")
        feats[f"wer{suffix}"] = Value("float32")
    feats["n_frames"] = Value("int32")
    feats["pool_mean"] = Sequence(Value("float32"), length=int(d))
    return Features(feats)