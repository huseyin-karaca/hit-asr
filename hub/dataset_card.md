---
pretty_name: HIT-ASR
license: other
license_name: per-corpus
license_link: https://huseyin-karaca.github.io/hit-asr/data/
language:
- en
task_categories:
- automatic-speech-recognition
tags:
- speech-recognition
- model-selection
- routing
- mixture-of-experts
- encoder-representations
- reproducibility
---

# HIT-ASR — data and results

The data behind **HIT-ASR: Hierarchical Transformer Routing for Adaptive ASR Expert Selection** (A. Samil Namli,
Huseyin Karaca, Suleyman S. Kozat — Bilkent University): what the pretrained ASR experts of the paper produce on its
four English corpora, and the stored results every notebook of the code repository reads.

- **Code and notebooks:** [github.com/huseyin-karaca/hit-asr](https://github.com/huseyin-karaca/hit-asr)
- **Documentation:** [huseyin-karaca.github.io/hit-asr](https://huseyin-karaca.github.io/hit-asr)

## What is here

| Config / folder | What |
|---|---|
| `<corpus>_base` | the corpus's audio at 16 kHz mono and its reference transcripts — the rows the paper uses |
| `<corpus>_labels_<expert>` | one expert on one corpus, per utterance: its transcript (raw and Whisper-normalised), the WER counters (substitutions, deletions, insertions, reference words) against the reference, and the mean of its final-layer encoder states |
| `<corpus>_frames_<expert>/` | that expert's final-layer encoder states per utterance, at the encoder's own frame rate: fp16 `.npy` shards of ~2 GB per split, `index.parquet` (`id, shard, offset, n_frames`), `manifest.json` |
| `results/<corpus>/` | the main record of each corpus (`main_<corpus>.json`: configuration, selected hyperparameters, every table), the extraction records, the stored 5×2 folds (`runlog/`), the timing record and its routers (`deploy/`), and the ablation's hold-out exploration and pool-size records |
| `results/synthetic.json` | the synthetic regime-switch check |
| `studies/ledger/<corpus>/` | the hyperparameter searches of the main experiment, one trial ledger per study |

The labels load with `datasets` (`load_dataset("huseyin-karaca/hit-asr", "ami_sdm_labels_cohere_transcribe")`); the
frames are read by `hitasr.frames.FrameSet`, memory-mapped or moved to the GPU. Every corpus holds its trio (below);
People's Speech holds the ten experts of the pool-size study. The audio of AMI, Earnings-22 and People's Speech is in
the companion dataset [`huseyin-karaca/fastt`](https://huggingface.co/datasets/huseyin-karaca/fastt).

## The corpora and their licences

Every part keeps the licence of the corpus it is derived from — the folders are kept apart per corpus, so this
dataset is a collection of separately licensed parts. **Attribution** is required for all of them, **ShareAlike**
for Earnings-22, and AfriSpeech-200 is **non-commercial** as well.

| Corpus | Source | Licence |
|---|---|---|
| AMI, single distant microphone (`ami_sdm_*`) | [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) | CC BY 4.0 |
| Earnings-22 (`earnings22_*`) | [Rev](https://github.com/revdotcom/speech-datasets/tree/main/earnings22), via [distil-whisper/earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) | CC BY-SA 4.0 |
| People's Speech, `dirty` (CC-BY) subset (`peoples_speech_*`) | [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | CC BY 4.0 |
| AfriSpeech-200 (`afrispeech_*`) | [intronhealth/afrispeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) | CC BY-NC-SA 4.0 |

The `results/` and `studies/` records are released under CC BY 4.0, except those of AfriSpeech-200 (CC BY-NC-SA
4.0) and of Earnings-22 (CC BY-SA 4.0).

## The experts

The labels and frames are outputs of pretrained models, redistributed under their licences with attribution:

| Expert | Model | Licence |
|---|---|---|
| Cohere-Transcribe | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Apache-2.0 |
| Granite-Speech-4.1-2b | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | Apache-2.0 |
| Qwen3-ASR-1.7B | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache-2.0 |
| Voxtral-Mini-3B | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Apache-2.0 |
| Whisper-large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Apache-2.0 |
| Whisper-large-v3-turbo | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT |
| Distil-Whisper-large-v3.5 | [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) | MIT |
| Parakeet-TDT-0.6b-v3, Parakeet-CTC-1.1b | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CC BY 4.0 |
| Kyutai-STT-2.6b | [kyutai/stt-2.6b-en-trfs](https://huggingface.co/kyutai/stt-2.6b-en-trfs) | CC BY 4.0 |

## The paper's trios

| Corpus | Experts the router chooses among |
|---|---|
| AMI (SDM) | Cohere-Transcribe, Granite-Speech-4.1-2b, Parakeet-TDT-0.6b-v3 |
| Earnings-22 | Cohere-Transcribe, Kyutai-STT-2.6b, Parakeet-TDT-0.6b-v3 |
| People's Speech | Granite-Speech-4.1-2b, Parakeet-CTC-1.1b, Parakeet-TDT-0.6b-v3 |
| AfriSpeech-200 | Cohere-Transcribe, Qwen3-ASR-1.7B, Voxtral-Mini-3B |

## Citation

```bibtex
@article{namli2026hitasr,
  title  = {{HIT-ASR}: Hierarchical Transformer Routing for Adaptive {ASR} Expert Selection},
  author = {Namli, A. Samil and Karaca, Huseyin and Kozat, Suleyman S.},
  year   = {2026}
}
```

Please cite the corpora and the models you use as well.
