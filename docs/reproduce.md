# Reproduce the paper

Every notebook runs at one of three levels; set `LEVEL` in its first code cell.

| Level | What runs | What it reads | Needs |
|---|---|---|---|
| **1** | every table | the searches, folds and records stored on the Hub, the experts' labels | a CPU, minutes |
| **2** | every model retrained; in the main notebooks `RETUNE = True` runs the hyperparameter search again too | the experts' labels and frames | a GPU, hours |
| **3** | level 2, on labels and frames you rebuilt from the audio | your own Hub repository (`extract`) | a GPU, a Hub account |

Levels 1 and 2 read only public repositories and need no account and no token.

## Level 1 — the tables from the stored results

Open a notebook in Colab (the badges on the [notebooks page](notebooks.md)) and run it top to bottom; any runtime will
do. A main notebook downloads the three experts' labels of its corpus and the stored searches and folds, and shows
Tables 1, 2, 3 and 6 and the routing behaviour.

Level 1 **trains nothing**. A search or fold missing from the Hub stops it with `CacheMiss` rather than training it
quietly; it never opens the frames and never writes to the Hub.

## Level 2 — retrain the models

Set `LEVEL = 2` on a GPU runtime (a 96 GB GPU fits every corpus). The frames of the three experts are downloaded
(3–26 GB per corpus) and every fold is retrained locally — every model, on the same folds. With `RETUNE = True` the
hyperparameter search runs again too (10 random draws per model on the hold-out). The notebook keeps what it computes
on the local disk and writes its record (`main_<corpus>.json`) next to itself.

GPU training is not bit-exact across hardware and library versions, so expect agreement with the published tables to
the noise of the fold means, not to the last digit.

To read your own records in the `ablation` and `manuscript_tables` notebooks, set `HITASR_RECORDS_DIR` to the folder
that holds them before the imports.

## Level 3 — rebuild the experts' outputs

`notebooks/extract` decodes every corpus with each expert of its trio and stores their frame-level encoder states, from
the audio in the published dataset, into a Hugging Face dataset repository of yours:

1. create an empty dataset repository on the Hub and add a write token as the Colab secret `HF_TOKEN`;
2. run `extract` with `REBUILD_REPO` set to it — it ends by comparing your WERs with the published ones;
3. run a main notebook at `LEVEL = 3` with `HITASR_HUB` set to your repository before its imports.

Cohere Transcribe is gated on the Hub: accept its terms there before step 2.

## Locally

```bash
pip install "hit-asr[paper] @ git+https://github.com/huseyin-karaca/hit-asr"
git clone https://github.com/huseyin-karaca/hit-asr && cd hit-asr
jupyter lab notebooks/main_ami_sdm.ipynb
```

Level 1 runs on a laptop. A main notebook also runs as a script, with the tables printed as text:

```bash
python -m hitasr.experiment ami_sdm --device cpu              # level 1
python -m hitasr.experiment ami_sdm --level 2                 # on a GPU
```

## Docker

`docker/Dockerfile` builds a CPU image with the package and the notebooks; it runs level 1 of a main notebook and
leaves the record in the mounted folder:

```bash
docker build -f docker/Dockerfile -t hit-asr .
docker run --rm -v "$PWD/out:/out" hit-asr                                  # AMI
docker run --rm -v "$PWD/out:/out" hit-asr python -m hitasr.experiment earnings22 --device cpu
```
