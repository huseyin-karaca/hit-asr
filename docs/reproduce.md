# Reproduce the paper

Every notebook runs at one of three levels; set `LEVEL` in its first code cell.

| Level | What runs | What it reads | Needs |
|---|---|---|---|
| **1** | every table | the searches, folds and records stored on the Hub, the experts' labels | a CPU, minutes |
| **2** | every model retrained; in the main notebooks `RETUNE = True` runs the hyperparameter search again too | the experts' labels and frames | a GPU, hours |
| **3** | level 2, on labels and frames you rebuilt from the audio | the local folder `extract` wrote, on the same machine | a GPU, a read token |

Levels 1 and 2 read only public repositories and need no account and no token. Level 3 writes nothing anywhere but
the machine it runs on; its one token is a free read token, because Cohere Transcribe is gated on the Hub.

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

`notebooks/extract` decodes every corpus with each expert of its trio, from the audio in the published dataset, and
stores their transcripts, word-error counters and frame-level encoder states in a local folder,
`$HITASR_CACHE/rebuild` (on Colab `/content/hitasr_cache/rebuild`):

1. accept the terms of [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) on the Hub
   and add a **read** token as the Colab secret `HF_TOKEN` (the trios of AMI, Earnings-22 and AfriSpeech use it);
2. run `extract` on a GPU runtime, with `CORPORA` cut to the corpora you want — it ends by comparing your transcripts
   and WERs with the published ones;
3. on the same runtime, run a main notebook at `LEVEL = 3`: it reads the labels and frames from that folder and the
   published searches from the Hub, and retrains every fold.

The rebuilt frames and transcripts match the published ones for most experts; they differ for Parakeet-TDT's transcripts on AMI, Earnings-22 and People's Speech (taken from an earlier decoding pass) and for Granite-Speech on AMI and People's Speech (batch-dependent) — see [the data page](data.md#how-the-labels-were-made). Level 3 therefore moves the tables of those three corpora more than level 2 does.

A corpus's trio takes 7-18 GB of disk and one to a few GPU hours to extract. To run level 3 on another machine, set
`PUSH_TO` in `extract` to a dataset repository of yours (this needs a write token) and `HITASR_HUB` to that repository
before the main notebook's imports; `HITASR_HUB=/some/folder` reads a folder instead.

As scripts:

```bash
python -m hitasr.extractor earnings22                         # the trio's labels and frames, into $HITASR_CACHE/rebuild
python -m hitasr.experiment earnings22 --level 3              # level 3 on them
```

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

Two images on the GitHub container registry, built from `docker/` by the repository's `docker` workflow; each holds
the package and the notebooks, keeps what it downloads in `/cache` and writes the records to `/out`:

| Image | For | Torch |
|---|---|---|
| `ghcr.io/huseyin-karaca/hit-asr:cpu` | level 1 | CPU |
| `ghcr.io/huseyin-karaca/hit-asr:cuda` | levels 2 and 3 (and 1) | CUDA 12.8 — an NVIDIA GPU from Ampere on (bf16), driver R570 or newer |

```bash
docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/hit-asr:cpu                  # AMI, level 1
docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/hit-asr:cpu python -m hitasr.experiment earnings22 --device cpu

docker run --rm --gpus all -v "$PWD/out:/out" -v "$PWD/cache:/cache" ghcr.io/huseyin-karaca/hit-asr:cuda \
    python -m hitasr.experiment earnings22 --level 2
docker run --rm --gpus all -e HF_TOKEN=<a read token> -v "$PWD/out:/out" -v "$PWD/cache:/cache" \
    ghcr.io/huseyin-karaca/hit-asr:cuda \
    sh -c "python -m hitasr.extractor earnings22 && python -m hitasr.experiment earnings22 --level 3"
```

The CUDA image pins the library versions the reproduction was checked with (torch 2.11.0, transformers 5.16.1,
datasets 3.6.0, XGBoost 3.4.1, Optuna 5.0.0). `docker build -f docker/Dockerfile.cuda -t hit-asr:cuda .` builds it
locally.
