"""The project's view of the Hub: `HitHub` names every artefact the active corpus's way — labels and frames
per expert, parquet configs, the results records — and reads the audio bases."""

__all__ = ['HitHub', 'ensure_audio_feature', 'load_base', 'base_index']

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

try:
    import datasets
    from datasets import Dataset, DatasetDict, load_dataset
    # Read feature types from the card, not from the viewer's `datasets`-4.x
    # re-export, which spells a fixed-size list as `List` and breaks 3.x.
    datasets.config.USE_PARQUET_EXPORT = False
except ImportError:                                            # pragma: no cover
    datasets = Dataset = DatasetDict = load_dataset = None

from labkit.hub import Hub
from hitasr.core import SOURCE_REPO, active_dataset


class HitHub(Hub):
    """The data repo and the records repo, named the active spec's way.

    Parameters
    ----------
    repo_id : the data repo. Defaults to the spec's (`REPO_ID`), with the records in `RECORDS_REPO`; an explicit
        repo — `SOURCE_REPO`, or `.source()`, for fastt's — holds its own records.
    spec : the `DatasetSpec` whose naming this instance uses.
    """

    def __init__(self, repo_id=None, spec=None):
        self.spec = active_dataset(spec)
        super().__init__(repo_id or self.spec.hub_repo(),
                         records_repo=None if repo_id else self.spec.records_hub_repo())
        self._configs = None

    def __repr__(self):
        return f"HitHub({self.repo_id!r}, dataset={self.spec.name!r})"

    def push_files(self, files, message=None, verbose=True, skip_identical=False):
        out = super().push_files(files, message, verbose=verbose, skip_identical=skip_identical)
        self._configs = None
        return out

    def push_folder(self, local_dir, path_in_repo, message=None, verbose=True):
        super().push_folder(local_dir, path_in_repo, message, verbose=verbose)
        self._configs = None

    def source(self):
        """The read-only twin bound to `SOURCE_REPO`, same spec."""
        return HitHub(SOURCE_REPO, spec=self.spec)

    def configs(self, refresh=False):
        """Every top-level directory that holds parquet or frame shards, sorted."""
        if refresh or self._configs is None:
            names = set()
            for f in self.files(refresh):
                head, _, tail = f.partition("/")
                if tail and (tail.endswith(".parquet") or tail.endswith(".npy")
                             or tail == "manifest.json"):
                    names.add("default" if head == "data" else head)
            self._configs = sorted(names)
        return self._configs

    def has(self, config_name):
        return config_name in self.configs()

    def expert_names(self, kind="labels"):
        """Short names of every expert with a `labels` (or `frames`) config for THIS corpus."""
        prefix = self.spec.labels_prefix if kind == "labels" else self.spec.frames_prefix
        return [c[len(prefix):] for c in self.configs() if c.startswith(prefix)]

    def frames_complete(self, expert):
        """True when `expert`'s frame store is whole: manifest, every split's index, and the results JSON.

        A store that an interrupted upload left half-pushed has shards but no
        manifest; `expert_names("frames")` would count it, this does not.
        """
        cfg, files = self.spec.frames_config(expert), set(self.files())
        return (f"{cfg}/manifest.json" in files
                and all(f"{cfg}/{s}/index.parquet" in files for s in self.spec.splits)
                and self.spec.results_path(f"extract_{expert}") in files)

    def complete_experts(self):
        """Experts of THIS corpus whose labels and frames are both fully on the Hub."""
        return [e for e in self.expert_names("labels") if self.frames_complete(e)]

    def split_names(self, config_name):
        """The splits `config_name` has parquet files for, from the layout alone."""
        pattern = re.compile(rf"^{re.escape(config_name)}/(.+)-\d+-of-\d+\.parquet$")
        return sorted({m.group(1) for m in map(pattern.match, self.files()) if m})

    def load(self, config_name, splits=None):
        """Load one parquet config as a `DatasetDict`, raw-parquet fallback included."""
        if config_name not in self.configs():
            raise KeyError(f"no config {config_name!r} in {self.repo_id}; "
                           f"have {self.configs()}")
        wanted = list(splits) if splits else None
        ds = None
        if wanted is not None and set(wanted) < set(self.split_names(config_name)):
            try:
                ds = load_dataset("parquet", data_files={
                    s: f"hf://datasets/{self.repo_id}/{config_name}/{s}-*.parquet"
                    for s in wanted})
            except Exception as e:                         # noqa: BLE001
                print(f"  note: reading {config_name}/{'+'.join(wanted)} directly "
                      f"failed ({type(e).__name__}: {e}); loading every split instead.")
        if ds is None:
            try:
                ds = load_dataset(self.repo_id, config_name)
            except Exception as e:                         # noqa: BLE001
                print(f"  note: load_dataset({config_name!r}) failed ({type(e).__name__}); "
                      "falling back to the raw parquet files.")
                ds = load_dataset("parquet", data_files={
                    s: f"hf://datasets/{self.repo_id}/{config_name}/{s}*.parquet"
                    for s in (wanted or self.split_names(config_name))})
        return DatasetDict({s: ds[s] for s in wanted}) if wanted else ds

    def parquet_paths(self, config_name, split):
        """The repo-relative parquet shards backing one config/split, sorted."""
        prefix = f"{config_name}/{split}-"
        paths = sorted(f for f in self.files()
                       if f.startswith(prefix) and f.endswith(".parquet"))
        if not paths:
            raise FileNotFoundError(f"no parquet for {config_name}/{split} in {self.repo_id}")
        return paths

    def column_names(self, config_name, split):
        """Column names of one config/split, read from the parquet footer."""
        path = self.parquet_paths(config_name, split)[0]
        return [f.name for f in pq.ParquetFile(self.local_path(path)).schema_arrow]

    def read_columns(self, config_name, split, columns):
        """Named columns of one config/split as a `pyarrow.Table` (local read after download)."""
        paths = [self.local_path(p) for p in self.parquet_paths(config_name, split)]
        return pq.read_table(paths, columns=list(columns))

    def push(self, ds, config_name, message=None, retries=3):
        """Push a `Dataset` / `DatasetDict` as `config_name`, overwriting it.

        `push_to_hub` is tried `retries` times (the Hub answers 500 in bursts
        that last minutes); if it keeps failing the parquet shards are written
        locally and committed as plain files under `config_name/`, which is
        the layout every reader here resolves from the file listing anyway —
        only the dataset card's `configs:` entry is not refreshed.
        """
        import time
        self._writable(f"push {config_name}")
        message = message or f"Update config {config_name}"
        for attempt in range(1, retries + 1):
            try:
                ds.push_to_hub(self.repo_id, config_name=config_name, commit_message=message)
                break
            except Exception as e:                             # noqa: BLE001
                print(f"  push_to_hub({config_name}) attempt {attempt}/{retries} failed: "
                      f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
                if attempt == retries:
                    print(f"  falling back to plain parquet files for {config_name}")
                    self._push_parquet_files(ds, config_name, message)
                else:
                    time.sleep(20 * attempt)
        self._configs = self._files = None
        print(f"pushed config  {config_name}")

    def _push_parquet_files(self, ds, config_name, message):
        """Write `config_name/<split>-00000-of-00001.parquet` locally and commit them, replacing old shards."""
        import tempfile
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        splits = dict(ds) if isinstance(ds, DatasetDict) else {"train": ds}
        adds = {s: f"{config_name}/{s}-00000-of-00001.parquet" for s in splits}
        ops = [CommitOperationDelete(path_in_repo=f) for f in self.files()
               if f.startswith(f"{config_name}/") and f.endswith(".parquet") and f not in adds.values()]
        with tempfile.TemporaryDirectory() as tmp:
            for split, d in splits.items():
                local = Path(tmp) / f"{split}.parquet"
                d.to_parquet(local)
                ops.append(CommitOperationAdd(path_in_repo=adds[split], path_or_fileobj=str(local)))
            self.api.create_commit(repo_id=self.repo_id, repo_type="dataset", operations=ops,
                                   commit_message=message)

    def push_dataframe(self, df, config_name, message=None):
        self.push(Dataset.from_pandas(df, preserve_index=False), config_name, message)

    def load_results(self, name):
        """`results/<corpus>/<name>.json`, parsed. `FileNotFoundError` when that notebook never wrote it."""
        path = self.pull_file(self.spec.results_path(name), verbose=False)
        if path is None:
            raise FileNotFoundError(f"no {self.spec.results_path(name)} in {self.repo_id}")
        return json.loads(Path(path).read_text())

def ensure_audio_feature(ds, fallback_sr=16000):
    """Guarantee a decoding `Audio` column. Returns `(ds, sampling_rate)`."""
    from datasets import Audio

    feats = ds.features if hasattr(ds, "features") else next(iter(ds.values())).features
    audio = feats.get("audio")
    if not isinstance(audio, Audio) or not audio.decode:
        print(f"  repairing 'audio' (found {type(audio).__name__}) "
              f"-> Audio(sampling_rate={fallback_sr})")
        return ds.cast_column("audio", Audio(sampling_rate=fallback_sr)), fallback_sr
    if audio.sampling_rate is not None:
        return ds, audio.sampling_rate
    probe = ds if hasattr(ds, "features") else next(iter(ds.values()))
    return ds, probe[0]["audio"]["sampling_rate"]


def load_base(spec=None, splits=None, sampling_rate=None):
    """The active corpus's `base` from `spec.base_hub_repo()`, ready for any extractor.

    The single entry point every model subsection uses, so the audio repair and
    the rate check happen once. `use_dataset(...)` then `load_base()` is the
    entire corpus switch.
    """
    spec = active_dataset(spec)
    sampling_rate = sampling_rate or spec.sampling_rate
    ds = HitHub(spec.base_hub_repo(), spec=spec).load(spec.base_config, splits)
    ds, sr = ensure_audio_feature(ds, fallback_sr=sampling_rate)
    if sr != sampling_rate:
        raise ValueError(f"{spec.base_config} decodes at {sr} Hz but {sampling_rate} Hz "
                         "is expected")
    missing = [s for s in (splits or spec.splits) if s not in ds]
    if missing:
        raise KeyError(f"{spec.base_config} is missing split(s) {missing}; has {list(ds)}")
    n = sum(d.num_rows for d in ds.values())
    print(f"{spec.name}/{spec.base_config}: {len(ds)} splits, {n:,} utterances, {sr} Hz")
    return ds


def base_index(spec=None, splits=None):
    """`{split: [id, ...]}` of `base` in stored order, WITHOUT downloading the audio.

    Reads the `id` column off the parquet footers. This is the canonical row
    order every artefact of the corpus aligns to, so it must be cheap.
    """
    spec = active_dataset(spec)
    src = HitHub(spec.base_hub_repo(), spec=spec)
    return {s: src.read_columns(spec.base_config, s, ["id"]).column("id").to_pylist()
            for s in (splits or spec.splits)}
