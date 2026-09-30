# The package

`pip install "hit-asr[paper] @ git+https://github.com/huseyin-karaca/hit-asr"` installs two packages.

## `hitasr` — the paper

| Module | What |
|---|---|
| `configs` | every configuration the paper reports, fixed and named: `MAIN[<corpus>]`, the spaces of Table 2 (`SPACES`), `ABLATION`, `TIMING`, `SYNTHETIC`, `REBUILD` |
| `experiment` | the main experiment on one corpus (`MainExperiment`): the search, the 5×2 cross-validation, Tables 1, 2, 3 and 6, the record |
| `paper` | the manuscript's tables from the records, shown and written as LaTeX (`ManuscriptTables`) |
| `ablation`, `timing`, `synthetic` | the ablation, the cost measurement on raw audio, the synthetic regime-switch check |
| `records` | the records the notebooks read from the Hub |
| `core` | the corpus registry (`DatasetSpec`): splits, trio, licence; the Hub repositories and the local rebuild folder (`rebuild_dir`, `use_hub`) |
| `hub` | `HitHub`: the dataset repository named the active corpus's way; the audio bases |
| `scoring` | the WER scorer (Whisper's English normaliser) |
| `extractor`, `models.*` | one extractor per expert: decode, word-error counters, frame-level encoder states; the rebuild (`python -m hitasr.extractor <corpus>`) and its check against the published labels |
| `labels`, `frames`, `store` | the labels as an error matrix, the frame shards memory-mapped or on the GPU, the two aligned (`RouterStore`) |
| `routers`, `training` | the HIT-ASR router and the MLP-pool router; the objective and the training loop |
| `arms` | every model as `fit` / `predict`, its search space, the registry `MODELS` |
| `rover` | the transcript-fusion baselines: confidence-weighted ROVER, CN-MBR |
| `tuning` | the hold-out search (`hpt`) and the names its studies and folds are stored under |
| `crossval` | the routing metrics, one fold of every model, Table 3 and the routing behaviour |
| `eda`, `ensemble` | the dataset table, the oracle arithmetic of a group of experts |
| `deploy`, `latency`, `bases` | the router on raw audio, the cost measurements, the builder of the AfriSpeech-200 audio base |

## `labkit` — the experiment machinery

Independent of speech; HIT-ASR is built on it.

| Module | What |
|---|---|
| `hub` | Hugging Face dataset repositories as a cache: one listing and one parallel fetch per step, every Hub error retried and never taken for a missing file, a read-only mode; a local folder (`local:/path`) in the same layout stands in for a repository |
| `runlog` | a content-addressed result cache: a step's result stored under a hash of what it depends on, per model and seed; a strict mode that never computes |
| `studies` | Optuna studies stored as trial ledgers on the Hub |
| `search` | search spaces as data (`Param`) and as source |
| `cv` | the 5×2 splits (and one repetition of them by seed), k-fold, the hold-out partition |
| `significance` | the paired test of Table 6 on the 5×2 folds (see the paper) and its table |
| `pretty`, `env` | tables in a notebook and in a terminal; seeds and the GPU |
