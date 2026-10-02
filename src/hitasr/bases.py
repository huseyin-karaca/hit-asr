__all__ = ['AFRISPEECH_SPLITS', 'AFRISPEECH_EXTRA', 'RAW_LOADERS', 'to_16k_wav', 'scorable', 'finalize_base',
           'afrispeech_key', 'afrispeech_rows', 'raw_afrispeech', 'build_base']

import fnmatch
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .core import SAMPLING_RATE, active_dataset
from .hub import HitHub
from .scoring import load_english_normalizer


def to_16k_wav(data, target_sr=SAMPLING_RATE):
    """`(wav_bytes, nsamples)`: any clip soundfile can read, as 16-bit mono PCM at `target_sr`."""
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), target_sr, format="WAV", subtype="PCM_16")   # PCM write wraps, not clips
    return buf.getvalue(), int(len(y))


def scorable(spec, normalizer=None):
    """A `text -> bool` predicate: non-empty, not blocklisted, >= `spec.min_ref_words` normalised words."""
    norm = normalizer or load_english_normalizer()
    blocked = {b.strip().lower() for b in spec.text_blocklist}
    min_words = int(spec.min_ref_words or 0)

    def keep(text):
        t = (text or "").strip()
        if not t or t.lower() in blocked:
            return False
        return min_words <= 0 or len(norm(t).split()) >= min_words

    return keep


def finalize_base(ds, spec=None, target_sr=None, num_proc=None, normalizer=None):
    """`base` from a raw `DatasetDict`: scorable rows, 16 kHz audio, `nsamples`, sorted by duration.

    Every step is per split, so a split built alone equals the same split
    built with the others. Duplicate ids inside a split are an error (the
    `(split, id)` key must be unique). Returns the `DatasetDict` in the
    canonical column order `id, audio, text, <extra_columns>, nsamples`.
    """
    from datasets import Audio, DatasetDict, Value

    spec = active_dataset(spec)
    target_sr = target_sr or spec.sampling_rate
    keep = scorable(spec, normalizer)
    before = {s: d.num_rows for s, d in ds.items()}
    ds = DatasetDict({s: d.filter(keep, input_columns=["text"]) for s, d in ds.items()})
    print("  kept scorable rows: " + ", ".join(f"{s} {d.num_rows:,}/{before[s]:,}" for s, d in ds.items())
          + (f"  (>= {spec.min_ref_words} normalised words)" if spec.min_ref_words else ""))
    for s, d in ds.items():
        ids = d["id"]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{spec.name}/{s}: {len(ids) - len(set(ids))} duplicate id(s)")

    def resample(batch):
        audio, n = [], []
        for i, a in zip(batch["id"], batch["audio"]):
            raw = a["bytes"] if a.get("bytes") is not None else Path(a["path"]).read_bytes()
            b, k = to_16k_wav(raw, target_sr)
            audio.append({"bytes": b, "path": f"{i}.wav"})
            n.append(k)
        return {"audio": audio, "nsamples": n}

    out = {}
    for s, d in ds.items():
        feats = d.features.copy()
        feats["audio"] = Audio(decode=False)
        feats["nsamples"] = Value("int64")
        d = d.map(resample, batched=True, batch_size=64, features=feats,
                  num_proc=num_proc or min(os.cpu_count() or 1, 16), desc=f"{spec.name}/{s}: 16 kHz")
        d = d.sort("nsamples")
        nz = [i for i, k in enumerate(d["nsamples"]) if k > 0]           # sorted, so zero-length rows are a prefix
        if len(nz) < d.num_rows:
            print(f"  {s}: dropped {d.num_rows - len(nz)} zero-length clip(s)")
            d = d.select(nz)
        cols = ["id", "audio", "text", *spec.extra_columns, "nsamples"]
        missing = [c for c in cols if c not in d.column_names]
        if missing:
            raise KeyError(f"{spec.name}/{s}: missing column(s) {missing}; have {d.column_names}")
        out[s] = d.select_columns(cols).cast_column("audio", Audio(sampling_rate=target_sr))
    base = DatasetDict(out)
    hours = {s: sum(d["nsamples"]) / target_sr / 3600 for s, d in base.items()}
    print(f"  {spec.base_config}: " + ", ".join(f"{s} {d.num_rows:,} rows / {hours[s]:.1f} h" for s, d in base.items())
          + f"; {sum(hours.values()):.1f} h in all")
    return base

AFRISPEECH_SPLITS = {"validation": "dev", "test": "test"}
AFRISPEECH_EXTRA = {"speaker_id": "user_ids", "accent": "accent", "country": "country",
                    "domain": "domain", "gender": "gender", "age_group": "age_group"}


def afrispeech_key(path):
    """The join key between a CSV `audio_paths` entry and a tar member: `<split>/<file>.wav`."""
    return "/".join(str(path).replace("\\", "/").split("/")[-2:])


def afrispeech_rows(meta, tar_files, repo_id=None, local_dir=None):
    """Yield one raw row per clip of `meta` found in `tar_files`, reading one shard at a time.

    With `repo_id`, each name in `tar_files` is a file of that dataset repo:
    it is downloaded into `local_dir`, read, and deleted before the next one.
    Without it, `tar_files` are local paths and are left alone. (Plain strings
    only: `Dataset.from_generator` hashes its `gen_kwargs`, and a callable
    there sends the hasher into unbounded recursion.) A clip found twice keeps
    its first copy; a clip the CSV does not list is skipped.
    """
    want = {afrispeech_key(p): r for p, r in zip(meta["audio_paths"], meta.to_dict("records"))}
    seen = set()
    for name in tar_files:
        if repo_id:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id, name, repo_type="dataset", local_dir=local_dir)
        else:
            local = name
        try:
            with tarfile.open(local, "r:*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    key = afrispeech_key(m.name)
                    r = want.get(key)
                    if r is None or key in seen:
                        continue
                    seen.add(key)
                    yield {"id": str(r["audio_ids"]), "text": " ".join(str(r["transcript"]).split()),
                           "audio": {"bytes": tf.extractfile(m).read(), "path": key},
                           **{c: "" if pd.isna(r.get(src)) else str(r.get(src))
                              for c, src in AFRISPEECH_EXTRA.items()}}
        finally:
            if repo_id:
                Path(local).unlink(missing_ok=True)


def raw_afrispeech(spec, splits):
    """AfriSpeech-200's dev/test as a raw `DatasetDict`, straight from the CSVs and tar shards."""
    from datasets import Audio, Dataset, DatasetDict, Features, Value
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(spec.source, repo_type="dataset")
    feats = Features({"id": Value("string"), "text": Value("string"), "audio": Audio(decode=False),
                      **{c: Value("string") for c in AFRISPEECH_EXTRA}})
    out = {}
    for split in splits:
        src = AFRISPEECH_SPLITS[split]
        meta = pd.read_csv(hf_hub_download(spec.source, f"transcripts/{src}.csv", repo_type="dataset"))
        tars = sorted(f for f in files if fnmatch.fnmatch(f, f"audio/*/{src}/*.tar.gz"))
        tmp = tempfile.mkdtemp(prefix=f"afrispeech_{src}_")
        print(f"  {split} (source {src}): {len(meta):,} clips in the CSV, {len(tars)} tar shard(s)")
        out[split] = Dataset.from_generator(afrispeech_rows, features=feats, keep_in_memory=False,
                                            gen_kwargs={"meta": meta, "tar_files": tars,
                                                        "repo_id": spec.source, "local_dir": tmp})
        shutil.rmtree(tmp, ignore_errors=True)
        lost = len(meta) - out[split].num_rows
        if lost:
            print(f"  WARNING {split}: {lost} clip(s) of the CSV were in no tar shard")
    return DatasetDict(out)


RAW_LOADERS = {"afrispeech": raw_afrispeech}


def build_base(spec=None, push=False, num_proc=None):
    """Build (and optionally push) `spec`'s `base` into `spec.base_hub_repo()`. Returns it."""
    spec = active_dataset(spec)
    if spec.name not in RAW_LOADERS:
        raise KeyError(f"{spec.name}: its base comes from fastt ({spec.base_hub_repo()}); "
                       f"only {sorted(RAW_LOADERS)} are built here")
    print(f"building {spec.base_config} from {spec.source} into {spec.base_hub_repo()}")
    base = finalize_base(RAW_LOADERS[spec.name](spec, spec.splits), spec, num_proc=num_proc)
    if push:
        HitHub(spec.base_hub_repo(), spec=spec).push(
            base, spec.base_config,
            f"{spec.base_config}: {sum(d.num_rows for d in base.values()):,} utterances, 16 kHz, sorted by nsamples")
    return base
