---
template: home.html
title: Home
hide:
  - navigation
  - toc
---

## No single expert wins

Each corpus in the study comes with three pretrained recognizers, the experts. None of them is best on every clip.
Choosing the best of the three for each clip in hindsight, the per-clip oracle, gives a much lower word error rate
than the best single expert (WER in %):

--8<-- "snippets/home_stats.html"

Reaching that headroom without running every recognizer is the routing problem. It is hard for three reasons.
The cues that decide which expert will do well, such as a burst of overlapping speech, a change of microphone or a
stretch of accented speech, sit somewhere inside the clip. The router has to compare the experts with one another,
not score each one in isolation. And transcribing with every expert to pick the best output means paying for every
decoder.

## Why averaging over time is not enough

A router that averages each expert's encoder states over the clip sees one vector per clip. Two clips that hold the
same conditions in opposite order give it the same vector, even when they need different experts. A router that
reads the frames sees which condition comes first.

--8<-- "snippets/fig_pooling.html"

## The idea

<ul class="hit-points">
<li><b>Read the frames</b><span>HIT-ASR keeps every expert's encoder states frame by frame, at each expert's own frame rate and width, instead of a pooled summary.</span></li>
<li><b>Summarise, then compare</b><span>A first transformer stage summarises each expert's stream over time; a second stage compares the experts' summaries and scores them jointly.</span></li>
<li><b>All encoders, one decoder</b><span>Every encoder runs once per clip; the router picks one expert and only that expert's decoder runs.</span></li>
</ul>

The [method page](method.md) walks through the router step by step, with the training objective and the cost.

## Results at a glance

--8<-- "snippets/home_result.md"

--8<-- "snippets/fig_gap.html"

<div class="hit-cards">
<a class="hit-card" href="method/"><b>Method</b><span>The problem, the router step by step, the training objective, what runs at inference.</span></a>
<a class="hit-card" href="results/"><b>Results</b><span>Four corpora, the synthetic regime switch, routing behaviour, ablation, cost.</span></a>
<a class="hit-card" href="reproduce/"><b>Reproduce</b><span>Every table from the stored results on a CPU, every model retrained on a GPU, the experts rebuilt from audio.</span></a>
<a class="hit-card" href="data/"><b>Data</b><span>The audio, the experts' transcripts and encoder states, and the stored results, on the Hugging Face Hub.</span></a>
</div>

## Citation

```bibtex
@article{karaca2026hitasr,
  title  = {{HIT-ASR}: Hierarchical Transformer Routing for Adaptive {ASR} Expert Selection},
  author = {Karaca, Huseyin and Namli, A. Samil and Kozat, Suleyman S.},
  year   = {2026},
  note   = {Under review at IEEE Transactions on Audio, Speech and Language Processing}
}
```
