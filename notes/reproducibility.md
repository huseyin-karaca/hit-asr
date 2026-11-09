# Reproducibility report

We did not stop at publishing the code and the data. We took the public package as it stands — the GitHub
repository, the Hugging Face dataset, a read-only token — and ran it again from scratch on machines we had never
used for the paper: Google Colab notebooks and rented cloud GPUs on [vast.ai](https://vast.ai) running our Docker
images. This page reports what came back, what it took, and what you need to do the same.

## At a glance

| What we ran | Where | Result |
|---|---|---|
| **Level 1** — every table from the stored searches and folds | a fresh download on a laptop; the `:cpu` container on a vast.ai machine | **identical to the last digit**: Tables 1, 2, 3 and 6, the routing behaviour, the ablation, the timing, the synthetic check and every LaTeX table of the paper |
| **Level 2** — every model retrained from the published features | Colab (NVIDIA RTX PRO 6000 Blackwell) | **every baseline identical to the last digit** — the experts, the best single expert, Random, ROVER, CN-MBR, ADASTT, MLP-pool and its hard-CE control; HIT-ASR varies run to run as GPU-trained transformers do, and matches the published corpus WER to the fifth decimal on Earnings-22 and People's Speech |
| **Level 3** — the experts' outputs rebuilt from the raw audio, then level 2 on them | Colab | **the experts re-decode the corpora to the published numbers**: identical transcripts for five of the seven experts, frame-level encoder states matching with cosine similarity 1.000000 for six, every expert's corpus WER within 0.0015 of the published one; AfriSpeech-200 rebuilt bit for bit |
| **Levels 2 and 3 in the CUDA container** | vast.ai, NVIDIA RTX 3090 and RTX A5000 (24 GB) | the whole pipeline — extraction from audio, retraining, tests, tables — runs end to end on a commodity 24 GB card; every expert's corpus WER within 0.0003 of the published one |

All four corpora went through level 1; three of them (Earnings-22, People's Speech, AfriSpeech-200) also through
levels 2 and 3 on Colab, and Earnings-22 through all three levels in the containers on vast.ai.

## Level 1: exact

Level 1 recomputes every table from the stored hyperparameter searches and the stored 5×2 cross-validation folds. It
trains nothing, so it reproduces **exactly**: we compare every numeric cell with the published record and every
LaTeX table byte for byte, and they match on all four corpora, from a fresh download of the public dataset as well
as from the `:cpu` container on a rented machine. This is the reference against which the other two levels are read.

## Level 2: retraining reproduces the results

Level 2 throws the stored folds away and trains every model again — the hyperparameter search is read, everything
else is recomputed: 5 repetitions × 2 folds × every system, then the paired tests.

- Every deterministic system and the pooled neural router (MLP-pool) come back **identical to the published
  numbers** on the Colab runtime, on every corpus we retrained.
- HIT-ASR's transformer trains through GPU kernels that are not bit-deterministic (for example the backward pass of
  fused attention), so a retrained HIT-ASR varies from run to run as any GPU-trained transformer does; on
  Earnings-22 and People's Speech its corpus WER came back to the fifth decimal. This is why every number in the paper
  is a mean over ten folds reported with its standard deviation: the conclusions rest on the fold means, not on the
  last digit of one run.

## Level 3: the experts' outputs rebuilt from the audio

Level 3 starts from the audio alone: every expert of each corpus's trio decodes the corpus again, its frame-level
encoder states are stored, and the main experiment is run on these rebuilt features.

- **Frames.** The rebuilt encoder states match the published ones with a mean frame-level cosine similarity of
  1.000000 and the same frame count for every utterance (500 utterances sampled per expert, plus the pooled vector of
  every utterance) — for every expert but Granite-Speech, whose encoder output depends on how a batch is padded
  (mean cosine 0.998).
- **Transcripts.** Cohere-Transcribe, Kyutai-STT, Parakeet-CTC, Qwen3-ASR and Voxtral reproduce their published
  transcripts **for 100 % of the utterances**, with identical word-error counts. AfriSpeech-200's trio is rebuilt
  bit for bit. Parakeet-TDT and Granite-Speech agree on 95–97 % of the utterances (Parakeet-TDT's published transcripts
  come from an earlier decoding pass, Granite-Speech's depend on the batching; see
  [how the labels were made](data.md#how-the-labels-were-made)); every expert's corpus WER is within 0.0015 of the
  published value.
- **Tables.** Where the rebuilt features are bit-identical (AfriSpeech-200), every baseline row of Table 3 at level 3
  is identical to the published one; elsewhere the systems move only with the few transcripts that differ, by at
  most a few thousandths of WER.

## On other hardware

The CUDA image ran the complete pipeline on two rented 24 GB Ampere cards (RTX 3090, RTX A5000) — a different GPU
generation from the one the paper used. The experts decode the corpora to the published word error rates (every
expert's corpus WER within 0.0003) and the baselines retrain to the published numbers (within 0.0001). A small share
of transcripts differs at the token level, as expected between GPU generations whose bf16 arithmetic differs; for
bit-identical expert outputs use the GPU generation of the published pass (NVIDIA Blackwell, Colab's G4).

## What it takes

### Time and cost per corpus

Measured on Colab's **G4** runtime (NVIDIA RTX PRO 6000 Blackwell, 96 GB) — the runtime the paper's results were
produced on. Colab bills it at about 8.7 compute units an hour (about US$0.10 per unit pay-as-you-go).

| Corpus | Level 1 (CPU) | Level 2 | Extraction (level 3) | Level 3 run | Level 3 in all |
|---|---|---|---|---|---|
| Earnings-22 | minutes | 36 min (~5 units) | 29 min | 36 min | ~65 min (~10 units) |
| People's Speech | minutes | 39 min (~6 units) | 9 min | 37 min | ~47 min (~7 units) |
| AfriSpeech-200 | minutes | 78 min (~11 units) | 17 min | 74 min | ~91 min (~13 units) |
| AMI (SDM) | minutes | ~2.5 h (~22 units, estimated) | not measured | ~2.5 h (estimated) | — |

The extraction times are the three experts of the trio together (on Earnings-22: Cohere-Transcribe 2.3 min,
Kyutai-STT 23.7 min, Parakeet-TDT 1.9 min). A level-2 run costs about one compute unit per repetition on Earnings-22
and People's Speech.

On **vast.ai** with the containers (prices of September 2026):

| Run | Machine | Time | Cost |
|---|---|---|---|
| Level 1, Earnings-22 (`:cpu`) | any small instance | ~5 min, image pull included | ~US$0.02 |
| Level 2, Earnings-22 (`:cuda`) | RTX 3090 24 GB, US$0.23/h | ~2 h (~22 min per repetition) | ~US$0.45 |
| Level 3, Earnings-22 (`:cuda`) | RTX A5000 24 GB, US$0.24/h | ~4 h (extraction 75 min, then the folds) | ~US$1.00 |

The whole campaign behind this page — three corpora through every level on Colab, plus the container runs — cost
about 50 Colab compute units and US$1.61 on vast.ai.

### Minimum specifications

| | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| **GPU** | none | NVIDIA, Ampere or newer (bf16), driver for CUDA 12.8 (R570+) | same as level 2 |
| **GPU memory** | — | 24 GB for Earnings-22 (tested); 40 GB or more for AMI, People's Speech and AfriSpeech-200, whose trio's frames (16–18 GB) are kept on the GPU | same as level 2 |
| **RAM** | 8 GB | 32 GB | 32 GB |
| **Disk** | 5 GB | 60 GB | 150 GB |
| **Download** | labels and records: 0.2–0.7 GB per corpus | + the trio's frames: 6.6 GB (Earnings-22) to 18.4 GB (AfriSpeech-200) | + the audio (1.9–5.8 GB) and the three experts' checkpoints (10–20 GB) |
| **Account / token** | none | none | a free Hugging Face **read** token (Cohere-Transcribe is gated) |
| **Container** | `ghcr.io/huseyin-karaca/hit-asr:cpu` (0.9 GB) | `ghcr.io/huseyin-karaca/hit-asr:cuda` (5 GB) | `ghcr.io/huseyin-karaca/hit-asr:cuda` |

On a rented machine, pick a host with a fast downlink (1 Gb/s or more): the CUDA image and the frames then arrive in
a few minutes. The commands are on the [Reproduce](reproduce.md#docker) page.

## How we compared

- **Tables.** Every numeric cell of Tables 1, 3 and 6 and of the routing behaviour, side by side with the published
  record of the corpus (`results/<corpus>/main_<corpus>.json` on the dataset); the LaTeX tables byte for byte.
- **Experts' outputs.** Per expert and utterance: the normalised transcript, the word-error counters, the number of
  frames, the cosine similarity of the frame-level encoder states (500 utterances per expert, frame by frame) and of
  the pooled vector (every utterance).
- **Environment.** Python 3.12 (container) and 3.13 (Colab), torch 2.11.0 (CUDA 12.8), transformers 5.16.1,
  datasets 3.6.0, XGBoost 3.4.1, Optuna 5.0.0 — the versions the CUDA image pins.
