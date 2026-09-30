"""The corpus registry — every corpus the paper reports, its splits and its expert trio — and where the artefacts live
on the Hub."""

__all__ = ['PUBLIC_REPO', 'REPO_ID', 'RECORDS_REPO', 'SOURCE_REPO', 'SAMPLING_RATE', 'DATASETS', 'AMI_SDM', 'EARNINGS22',
           'PEOPLES_SPEECH', 'AFRISPEECH', 'DatasetSpec', 'register_dataset', 'use_dataset', 'active_dataset',
           'require_audio_datasets', 'rebuild_dir', 'use_hub']

import os
from dataclasses import dataclass, field
from pathlib import Path

# The Hub repos. PUBLIC_REPO is the published dataset: the audio bases, every expert's labels and frames. REPO_ID
# is where labels and frames are read and written — PUBLIC_REPO unless `HITASR_HUB` (or `use_hub`) names another: a
# local folder (`local:/path`, what `notebooks/extract` writes by default, see `rebuild_dir`) or a dataset repo of
# yours. RECORDS_REPO holds the records — `results/` (provenance JSON, fold caches) and `studies/` (search ledgers);
# `labkit.hub.Hub` sends every path under those two folders there.
PUBLIC_REPO = "huseyin-karaca/hit-asr"
REPO_ID = os.environ.get("HITASR_HUB", PUBLIC_REPO)
RECORDS_REPO = os.environ.get("HITASR_RECORDS", PUBLIC_REPO)
SOURCE_REPO = "huseyin-karaca/fastt"      # fastt's labels, adopted for experts hit-asr has none of
SAMPLING_RATE = 16000                     # every checkpoint here expects 16 kHz mono


@dataclass(frozen=True)
class DatasetSpec:
    """Everything corpus-specific about one evaluation set, in one object.

    Parameters
    ----------
    name : short id, e.g. `"ami_sdm"`. Also the `results/` subdirectory.
    source : the Hub repo the raw audio originally came from (provenance).
    splits : the split names as they appear in `base` and in every derived
        artefact. `fit_splits` and `eval_splits` must partition a subset of
        these and must not overlap.
    base_config : the config in `base_hub_repo()` holding the audio, the
        reference text and `nsamples` for this corpus, in the canonical `id` /
        `text` / `audio` schema. AMI, Earnings-22 and People's Speech come from
        fastt; AfriSpeech-200 is built by `hitasr.bases.build_base`.
    base_repo : where `base_config` lives. `None` means fastt's `SOURCE_REPO`;
        the corpora built here set `PUBLIC_REPO`.
    min_ref_words : `build_base` keeps a row only if its Whisper-normalised
        reference has at least this many words. `None` keeps every row with a
        non-empty raw reference (fastt's rule for the original five).
    source_prefix : what fastt's own per-model label configs are called for
        this corpus (`<source_prefix>model_<name>`), so its transcripts and WER
        counters can be adopted rather than recomputed.
    source_config, data_files, columns, id_columns, extra_columns, ... : the
        recipe `base` was built with. Carried as provenance so a spec can be
        audited without opening the fastt repo; nothing here reads them.
    config_prefix : namespaces every artefact this corpus writes in `REPO_ID`.
    repo_id : the Hub repo for artefacts. `None` means the global `REPO_ID`.
    members : the expert trio for this corpus — what `main_<corpus>` routes
        between.
    license : the licence of the corpus, and so of everything derived from it.
    """

    name: str
    source: str
    splits: tuple
    fit_splits: tuple
    eval_splits: tuple
    base_config: str
    source_prefix: str
    source_config: str = None
    data_files: dict = None
    columns: dict = field(default_factory=dict)
    id_columns: tuple = ()
    extra_columns: dict = field(default_factory=dict)
    text_blocklist: tuple = ()
    language: str = "en"
    config_prefix: str = ""
    repo_id: str = None
    sampling_rate: int = SAMPLING_RATE
    note: str = ""
    members: tuple = None
    base_repo: str = None
    min_ref_words: int = None
    license: str = ""

    def __post_init__(self):
        if self.members is not None:
            object.__setattr__(self, "members", tuple(sorted(self.members)))   # frozen dataclass
        overlap = set(self.fit_splits) & set(self.eval_splits)
        if overlap:
            raise ValueError(f"{self.name}: {sorted(overlap)} is both fit and eval")
        unknown = (set(self.fit_splits) | set(self.eval_splits)) - set(self.splits)
        if unknown:
            raise ValueError(f"{self.name}: {sorted(unknown)} not in splits={self.splits}")
        if not self.config_prefix:
            raise ValueError(f"{self.name}: every corpus is prefixed in {REPO_ID}")

    # ------------------------------------------------------ artefact names --

    def hub_repo(self, repo_id=None):
        """The repo to read and write artefacts. Explicit argument wins, then the spec."""
        return repo_id or self.repo_id or REPO_ID

    def records_hub_repo(self):
        """The repo `results/` and `studies/` live in."""
        return RECORDS_REPO

    def base_hub_repo(self):
        """The repo `base_config` is read from: fastt's unless the spec says otherwise."""
        return self.base_repo or SOURCE_REPO

    def source_model_config(self, name):
        """fastt's label config for `name` on this corpus (`model_<name>`, prefixed)."""
        return f"{self.source_prefix}model_{name}"

    @property
    def labels_prefix(self):
        """What every per-expert labels config starts with."""
        return f"{self.config_prefix}labels_"

    @property
    def frames_prefix(self):
        return f"{self.config_prefix}frames_"

    def labels_config(self, name):
        return f"{self.config_prefix}labels_{name}"

    def frames_config(self, name):
        return f"{self.config_prefix}frames_{name}"

    def results_path(self, name):
        """`results/<corpus>/<name>.json`."""
        return f"results/{self.name}/{name}.json"

    def base_column_dtypes(self):
        """`{column: dtype string}` for every non-audio column of `base`, `id` first."""
        return {"id": "string", "text": "string",
                **self.extra_columns, "nsamples": "int64"}


DATASETS = {}                              # name -> DatasetSpec
_ACTIVE = None                             # the spec every default resolves to

def register_dataset(spec, activate=False):
    """Add a spec to the registry (and optionally make it the active one)."""
    DATASETS[spec.name] = spec
    if activate or _ACTIVE is None:
        use_dataset(spec, verbose=False)
    return spec


def use_dataset(spec, verbose=True):
    """Make `spec` (a name or a `DatasetSpec`) the active dataset. Returns it.

    Every `spec=None` / `repo_id=None` default in the project resolves through
    `active_dataset()` at call time, so this one call switches the corpus for
    every section at once. It changes nothing on the Hub and nothing already
    loaded: rebuild any store after switching.
    """
    global _ACTIVE
    if isinstance(spec, str):
        if spec not in DATASETS:
            raise KeyError(f"no dataset {spec!r}; have {sorted(DATASETS)}")
        spec = DATASETS[spec]
    _ACTIVE = spec
    if verbose:
        print(f"active dataset : {spec.name}  (base {spec.base_hub_repo()}/{spec.base_config})")
        print(f"  splits       : {list(spec.splits)}  (fit {list(spec.fit_splits)} -> eval {list(spec.eval_splits)})")
        print(f"  artefacts    : {spec.labels_config('<expert>')}, {spec.frames_config('<expert>')}, "
              f"{spec.results_path('<run>')} in {spec.hub_repo()}")
    return spec


def active_dataset(spec=None):
    """The active `DatasetSpec`, or `spec` itself (a name or a spec) when one is passed."""
    if spec is not None:
        return DATASETS[spec] if isinstance(spec, str) else spec
    if _ACTIVE is None:
        raise RuntimeError("no active dataset; call use_dataset('ami_sdm')")
    return _ACTIVE


# --- three corpora as fastt built them ------------------------------------------
# The shard arithmetic is fastt's (`AMI_TRAIN_SHARDS`, `VOXPOPULI_TRAIN_SHARDS`,
# ...) and is reproduced in `data_files` for provenance only. The rows come
# from `SOURCE_REPO/<base_config>` as pushed; the recipe is the documentation
# of what those rows are.

AMI_SDM = register_dataset(DatasetSpec(
    name="ami_sdm", source="edinburghcstr/ami", source_config="sdm",
    base_config="ami_sdm_base", source_prefix="ami_sdm_",
    data_files={"train": [f"sdm/train-{i:05d}-of-00027.parquet" for i in range(4)],
                "validation": "sdm/validation-*.parquet", "test": "sdm/test-*.parquet"},
    splits=("train", "validation", "test"),
    fit_splits=("train", "validation"), eval_splits=("test",),
    columns={"id": "audio_id"},
    extra_columns={"meeting_id": "string", "speaker_id": "string", "microphone_id": "string",
                   "begin_time": "float32", "end_time": "float32"},
    text_blocklist=("ignore_time_segment_in_scoring",),
    config_prefix="ami_sdm_",
    members=("cohere_transcribe", "granite_speech_4_1_2b", "parakeet_tdt_0_6b_v3"),
    license="CC BY 4.0",
    note="Spontaneous meeting speech through a single distant microphone, cut into "
         "short conversational turns. The hardest corpus here and the one where WER "
         "spreads widest across experts — 28-36% for the strong ones — which is what "
         "a router needs. Train subsampled to 4/27 shards (~15.9k utterances); "
         "41,641 utterances in all.",
), activate=True)

EARNINGS22 = register_dataset(DatasetSpec(
    name="earnings22", source="distil-whisper/earnings22", source_config="chunked",
    base_config="earnings22_base", source_prefix="earnings22_",
    data_files={"validation": [f"chunked/test-{i:05d}-of-00038-*.parquet" for i in range(4)],
                "test": [f"chunked/test-{i:05d}-of-00038-*.parquet" for i in range(5, 9)]},
    splits=("validation", "test"),
    fit_splits=("validation",), eval_splits=("test",),
    columns={"text": "transcription"}, id_columns=("file_id", "segment_id"),
    extra_columns={"file_id": "string", "start_ts": "float32", "end_ts": "float32"},
    config_prefix="earnings22_",
    members=("cohere_transcribe", "kyutai_stt_2_6b", "parakeet_tdt_0_6b_v3"),
    license="CC BY-SA 4.0",
    note="Earnings calls from 125 companies in seven language regions: clean channel, "
         "professional references, hard ACCENTS. The source is one benchmark split, "
         "cut into 4 fit + 4 eval shards with a gap shard so no call straddles the "
         "line; 12,088 utterances.",
))

PEOPLES_SPEECH = register_dataset(DatasetSpec(
    name="peoples_speech", source="MLCommons/peoples_speech", source_config="dirty",
    base_config="peoples_speech_base", source_prefix="peoples_speech_",
    data_files={"train": ["dirty/train-00000-of-03140.parquet"],
                "validation": [f"dirty/validation-{i:05d}-of-00006.parquet" for i in range(2)],
                "test": [f"dirty/test-{i:05d}-of-00011.parquet" for i in range(2)]},
    splits=("train", "validation", "test"),
    fit_splits=("train", "validation"), eval_splits=("test",),
    extra_columns={"duration_ms": "int32"},
    config_prefix="peoples_speech_",
    members=("granite_speech_4_1_2b", "parakeet_ctc_1_1b", "parakeet_tdt_0_6b_v3"),
    license="CC BY 4.0 (the cc-by \"dirty\" subset)",
    note="Archive.org speech — council meetings, lectures, radio — on whatever hardware "
         "the room had. Noisy and, unlike every other corpus here, varied WITHIN the "
         "split. Transcripts are partly machine-sourced, so read the gap between "
         "experts rather than an absolute number; 14,299 utterances.",
))

# --- AfriSpeech-200, built here ---------------------------------------------------
# A corpus where the speaker's accent decides which expert wins. Its `base` is built by `hitasr.bases.build_base` into
# `REPO_ID`; `data_files` is again provenance.

AFRISPEECH = register_dataset(DatasetSpec(
    name="afrispeech", source="intronhealth/afrispeech-200",
    base_config="afrispeech_base", source_prefix="afrispeech_", base_repo=PUBLIC_REPO,
    data_files={"validation": ("transcripts/dev.csv", "audio/*/dev/*.tar.gz"),
                "test": ("transcripts/test.csv", "audio/*/test/*.tar.gz")},
    splits=("validation", "test"),
    fit_splits=("validation",), eval_splits=("test",),
    columns={"id": "audio_ids", "text": "transcript"},
    extra_columns={"speaker_id": "string", "accent": "string", "country": "string",
                   "domain": "string", "gender": "string", "age_group": "string"},
    min_ref_words=1,
    config_prefix="afrispeech_",
    members=("cohere_transcribe", "qwen3_asr_1_7b", "voxtral_mini_3b"),
    license="CC BY-NC-SA 4.0",
    note="AfriSpeech-200 dev + test: read English from 997 speakers of 108 African "
         "accents in 13 countries, clinical (57 %) and general domain (27.4 h, "
         "median 8.4 s). The train split (173 h) is not used. `id` is the source's "
         "`audio_ids` (an md5): the file names are hashes of the PROMPT, shared by every "
         "speaker who read it (62 collisions). `validation` is the source's `dev`. Audio "
         "comes as per-accent tar shards at 44.1 kHz plus CSV transcripts, so `build_base` "
         "reads it directly instead of through the dataset script; one test clip is in "
         "no shard and one has an empty reference, so 9,544 of 9,546 remain. "
         "CC BY-NC-SA 4.0.",
))

def require_audio_datasets():
    """Fail early, and legibly, if `datasets` is too new to decode audio here.

    `datasets` 4.0 replaced the audio decoder with `torchcodec`; every path that
    touches `example["audio"]["array"]` breaks on it fifteen frames deep inside
    `Dataset.map`. Called at the top of the audio paths, never at import — the
    label-reading sections are happy on 4.x.
    """
    import datasets

    major = int(datasets.__version__.split(".")[0])
    if major >= 4:
        raise RuntimeError(
            f"datasets {datasets.__version__} cannot decode audio for this project "
            "(4.x moved to torchcodec). Run the Group A/B install cell in this kernel "
            "— it pins `datasets<4.0` — then restart the runtime.")


def rebuild_dir():
    """The local folder `notebooks/extract` writes the rebuilt labels and frames to by default, and level 3 reads:
    `$HITASR_CACHE/rebuild` (`/content/hitasr_cache/rebuild` on Colab, `~/.cache/hitasr/rebuild` elsewhere)."""
    return Path(os.environ.get("HITASR_CACHE", Path.home() / ".cache" / "hitasr")) / "rebuild"


def use_hub(repo_id, records=None):
    """Read and write labels and frames in `repo_id` from here on — a local folder (`local:/path`) or a Hugging Face
    dataset repo; `records` moves `results/` and `studies/` as well. What `HITASR_HUB` / `HITASR_RECORDS` do at import,
    in a running session. Returns `repo_id`."""
    global REPO_ID, RECORDS_REPO
    REPO_ID = str(repo_id)
    if records is not None:
        RECORDS_REPO = str(records)
    return REPO_ID
