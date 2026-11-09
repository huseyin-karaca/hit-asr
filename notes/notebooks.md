# The notebooks

| Notebook | What it reproduces | Open |
|---|---|---|
| `main_ami_sdm` | Tables 1, 2, 3 and 6 on AMI (single distant microphone) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/main_ami_sdm.ipynb) |
| `main_earnings22` | the same on Earnings-22 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/main_earnings22.ipynb) |
| `main_peoples_speech` | the same on People's Speech | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/main_peoples_speech.ipynb) |
| `main_afrispeech` | the same on AfriSpeech-200 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/main_afrispeech.ipynb) |
| `ablation` | the design switches of HIT-ASR, and the number of experts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/ablation.ipynb) |
| `timing` | the cost on raw audio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/timing.ipynb) |
| `synthetic` | the synthetic regime-switch check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/synthetic.ipynb) |
| `manuscript_tables` | every table of the paper, as LaTeX in `tables/` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/manuscript_tables.ipynb) |
| `extract` | the experts' labels and frames, rebuilt from the audio (level 3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyin-karaca/hit-asr/blob/main/notebooks/extract.ipynb) |

Every configuration is fixed in `hitasr.configs` and imported by its notebook; the notebooks hold the calls and the
results, the package holds the code.

## A main notebook, section by section

| § | What it does | Shows |
|---|---|---|
| — | `MainExperiment(MAIN["<corpus>"], level=LEVEL)`: the corpus, its expert trio, the protocol and the seeds | |
| 1 | the hyperparameter spaces of every model (`hitasr.configs.SPACES`) | **Table 2** |
| 2 | the three experts' labels (and, from level 2, their frames); a check that the stored word-error counters reproduce from the stored transcripts | |
| 3 | the search: a random 25 % of the clips is held out, and every model's hyperparameters are chosen there by 10 random draws, each trained on 70 % of the hold-out and scored on the rest | **Table 1**, the selected configurations |
| 4 | the 5×2 cross-validation on the other 75 %: each of five repetitions halves the clips with its seed, every model is trained on one half and tested on the other, then the other way round | |
| 5 | every model's metrics as mean ± sd over the ten folds; the routers' selection distributions | **Table 3**, routing behaviour |
| 6 | HIT-ASR against every other system on the same folds, with a custom statistical test — for full details, please see the paper | **Table 6** |
| 7 | the record: everything above in one JSON file | |

## The models

| Model | What it is |
|---|---|
| HIT-ASR | the hierarchical transformer router on the experts' frame-level encoder states |
| MLP-pool | a router on the experts' mean-pooled encoder states |
| ADASTT | gradient-boosted trees on the pooled states, trained with cross-entropy |
| ROVER (confidence-weighted), CN-MBR | transcript-level fusion of the three experts' outputs |
| Best single expert, Random, Oracle | the references: the best expert on the training half, a uniform pick, the per-clip best |

MLP-pool (hard CE), MLP-pool trained with hard cross-entropy only, is searched and trained as a control; it is kept in
each record and in the ablation notebook, and is not a row of Tables 3 and 6.
