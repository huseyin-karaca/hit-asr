# Data and licences

Everything the notebooks read is the Hugging Face dataset
[`huseyin-karaca/hit-asr`](https://huggingface.co/datasets/huseyin-karaca/hit-asr).

## What it holds

| Folder | What |
|---|---|
| `<corpus>_base/` | the corpus's audio at 16 kHz mono with its reference transcripts — the rows the paper uses |
| `<corpus>_labels_<expert>/` | one expert on one corpus, per utterance: the transcript (raw and normalised), the WER counters against the reference, the mean of its encoder states |
| `<corpus>_frames_<expert>/` | that expert's frame-level encoder states: fp16 `.npy` shards of about 2 GB per split, an `index.parquet` of `(id, shard, offset, n_frames)`, a `manifest.json` |
| `results/<corpus>/` | the records: the main record (`main_<corpus>.json`), the extraction records, the stored 5×2 folds (`runlog/`), the timing record and the routers it deployed (`deploy/`), the hold-out exploration and pool-size records of the ablation |
| `results/synthetic.json` | the synthetic check |
| `studies/ledger/<corpus>/` | the hyperparameter searches, one trial ledger per study |

The audio of AMI, Earnings-22 and People's Speech is read from the companion dataset
[`huseyin-karaca/fastt`](https://huggingface.co/datasets/huseyin-karaca/fastt) (`<corpus>_base`). For People's Speech
the labels and frames of the ten experts of the pool-size study are here as well.

## The corpora

| Corpus | Source | Used splits | Licence of everything derived from it |
|---|---|---|---|
| AMI, single distant microphone | [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) (`sdm`) | train (4 of 27 shards), validation, test | CC BY 4.0 |
| Earnings-22 | [Rev](https://github.com/revdotcom/speech-datasets/tree/main/earnings22), via [distil-whisper/earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) (`chunked`) | 8 of 38 shards, split by call | CC BY-SA 4.0 |
| People's Speech | [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech) (`dirty`, the CC-BY subset) | 1 train shard, 2 validation and 2 test shards | CC BY 4.0 |
| AfriSpeech-200 | [intronhealth/afrispeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) | dev, test | CC BY-NC-SA 4.0 |

Each corpus's audio, labels and frames are distributed under that corpus's licence — the folders are kept apart
per corpus, so the dataset is a collection of separately licensed parts, not one combined work. In short:
**attribution** is required for all of them, **ShareAlike** for Earnings-22 (derivatives under the same
licence), and AfriSpeech-200 is **non-commercial** as well. Please cite the corpora you use.

## The experts

The labels and frames are outputs of these pretrained models; their licences allow redistributing them with
attribution.

| Expert | Model | Licence |
|---|---|---|
| Cohere-Transcribe | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (gated) | Apache-2.0 |
| Granite-Speech-4.1-2b | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | Apache-2.0 |
| Qwen3-ASR-1.7B | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache-2.0 |
| Voxtral-Mini-3B | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Apache-2.0 |
| Parakeet-TDT-0.6b-v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC BY 4.0 |
| Parakeet-CTC-1.1b | [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CC BY 4.0 |
| Kyutai-STT-2.6b | [kyutai/stt-2.6b-en-trfs](https://huggingface.co/kyutai/stt-2.6b-en-trfs) | CC BY 4.0 |
| Whisper-large-v3, Whisper-large-v3-turbo, Distil-Whisper-large-v3.5 (pool-size study only) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) | Apache-2.0, MIT, MIT |

## The trios

| Corpus | Experts the router chooses among |
|---|---|
| AMI (SDM) | Cohere-Transcribe, Granite-Speech-4.1-2b, Parakeet-TDT-0.6b-v3 |
| Earnings-22 | Cohere-Transcribe, Kyutai-STT-2.6b, Parakeet-TDT-0.6b-v3 |
| People's Speech | Granite-Speech-4.1-2b, Parakeet-CTC-1.1b, Parakeet-TDT-0.6b-v3 |
| AfriSpeech-200 | Cohere-Transcribe, Qwen3-ASR-1.7B, Voxtral-Mini-3B |

## How the labels were made

Every frame store, and the transcripts of every trio expert but one, come from one full decoding pass of the corpus
(`decode = full` in `results/<corpus>/extract_<expert>.json`). The exception is Parakeet-TDT-0.6b-v3 on AMI (SDM),
Earnings-22 and People's Speech: its frames were extracted in that pass, but its transcripts and word-error counters
were taken from an earlier decoding pass of the same checkpoint (the companion dataset `huseyin-karaca/fastt`, batch size 8), checked by re-decoding 64 utterances per split
(`decode = verify`, at least 85 % identical after normalisation). `notebooks/extract` decodes every expert in full, so a
level-3 rebuild reproduces those frames but not every one of those transcripts: on Earnings-22 a full re-decode gives
the same normalised transcript for 96.2 % of the utterances (corpus WER 0.1390 against the published 0.1392), and the
tables of a level-3 run move accordingly.

Granite-Speech-4.1-2b (AMI (SDM), People's Speech) was decoded in full, but its tower does not mask the padding
of a batch and its frame count is estimated from the batch's longest clip, so its frames and transcripts depend on
how the corpus was batched and on the library versions of the pass, which were not recorded for the published one.
A rebuild of People's Speech gives the same frame count for 86 % of the utterances (mean frame cosine 0.998) and
the same normalised transcript for 94.8 % (corpus WER 0.1791 against the published 0.1776). The rebuild's
extraction records now state their batch size.
