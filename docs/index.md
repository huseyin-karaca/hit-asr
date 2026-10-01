# HIT-ASR

**Hierarchical Transformer Routing for Adaptive ASR Expert Selection**

A. Samil Namli\*, Huseyin Karaca\*, Suleyman S. Kozat — Bilkent University (\* equal contribution)

[Code](https://github.com/huseyin-karaca/hit-asr){ .md-button } [Data](https://huggingface.co/datasets/huseyin-karaca/hit-asr){ .md-button } [Reproduce](reproduce.md){ .md-button .md-button--primary } [Reproducibility report](reproducibility.md){ .md-button }

Pretrained ASR models have complementary strengths: on any one clip, one of them is usually clearly better than the
others. HIT-ASR picks that expert per clip. It reads the **frame-level encoder states** of every expert — not a
pooled summary — with a two-stage transformer: a first stage that models the temporal evidence within each expert,
and a cross-attention bridge that lets the experts' streams inform one another before one of them is chosen. All
expert encoders run once per clip; only the chosen expert's decoder runs.

![The HIT-ASR router](assets/router_figure.svg)

## What this site covers

- **[Reproduce](reproduce.md)** — every notebook at three levels, from the stored results (a CPU, minutes) to
  rebuilding the experts' outputs from the audio (a GPU), in Colab or in our Docker images on any GPU machine.
- **[Reproducibility report](reproducibility.md)** — the package re-run from scratch on Colab and on rented cloud GPUs:
  every table reproduced exactly at level 1, every baseline retrained to the last digit, the experts' outputs rebuilt
  from the audio — with the time, the cost and the hardware each level takes.
- **[The notebooks](notebooks.md)** — what each notebook reproduces, and what each section of a main notebook does.
- **[Data and licences](data.md)** — the Hugging Face dataset: what it holds, how it is laid out, and under which
  licence each part may be used.
- **[The package](package.md)** — `hitasr` (the paper) and `labkit` (the experiment machinery underneath).

## Citation

```bibtex
@article{namli2026hitasr,
  title  = {{HIT-ASR}: Hierarchical Transformer Routing for Adaptive {ASR} Expert Selection},
  author = {Namli, A. Samil and Karaca, Huseyin and Kozat, Suleyman S.},
  year   = {2026}
}
```
