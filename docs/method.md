---
hide:
  - navigation
---

# Method

<p class="hit-lede">HIT-ASR routes each clip to one of several pretrained speech recognizers. It reads the
frame-level encoder states of all of them, summarises each expert's frames with a transformer, compares the experts in
a second stage, and decodes the clip with the chosen expert only.</p>

## The problem

Take \(K\) pretrained recognizers, the experts \(A_1, \dots, A_K\). Each is a complete system, an encoder followed by
a decoder. On a clip \(x_n\) with reference transcript \(R_n\), expert \(k\) makes \(e_{n,k}\) word errors, so its
word error rate on that clip is \(e_{n,k} / |R_n|\). A router looks at the clip and picks one expert
\(\hat{\jmath}_n\); the goal is a low corpus word error rate,

\[
\mathrm{WER} \;=\; \frac{\sum_n e_{n,\hat{\jmath}_n}}{\sum_n |R_n|}.
\]

Two references bound what a router can do. The **best single expert** sends every clip to the expert with the lowest
WER on the training data. The **per-clip oracle** picks, for each clip, an expert with the fewest errors,
\(\jmath^\star_n \in \arg\min_k e_{n,k}\); no router that picks one expert per clip can do better. The results report
how much of the distance between the two a router covers, the *gap closed*,

\[
\mathrm{GC} \;=\; \frac{\mathrm{WER}_{\text{single}} - \mathrm{WER}_{\text{router}}}{\mathrm{WER}_{\text{single}} - \mathrm{WER}_{\text{oracle}}},
\]

and its *selection accuracy*, the share of clips on which the chosen expert has the fewest errors (ties count).

## Why it is hard

- **The evidence is local.** What makes one expert better on a clip can be a short stretch of it: overlapping speech,
  a change of microphone, a few seconds of a hard accent. A summary vector averaged over the whole clip blurs where,
  and in what order, these happen.
- **The choice is relative.** Whether expert 2 should win depends on how experts 1 and 3 handle the same clip, so the
  router has to compare the experts rather than score each one on its own.
- **Decoding is the expensive part.** Running every expert in full and keeping the best transcript costs every
  decoder on every clip; the router has to decide before any decoder runs.

## The router, step by step

The experts differ in architecture: their encoders emit states of different widths at different frame rates (from
12.5 to 50.0 frames per second in this study). HIT-ASR takes each expert's sequence as it is and builds the decision in
two stages.

/// tab | Frames

--8<-- "snippets/fig_arch-1.html"

Every expert's encoder runs once on the clip. Its frame-level states \(\mathbf{H}^{(k)} \in \mathbb{R}^{T_k \times
D_k}\) are kept as a sequence, \(T_k\) frames of width \(D_k\), with no pooling. Each sequence is projected to a
common width, given sinusoidal positions and a learned embedding that says which expert it came from. Long clips keep
their first frames up to a cap that the hyperparameter search sets.

///

/// tab | Stage 1

--8<-- "snippets/fig_arch-2.html"

**Stage 1** is a transformer encoder over each expert's own sequence, with a learned summary token in front. After a
few layers of self-attention over time, the summary token holds that expert's view of the whole clip, a summary
learned by attention over the ordered frames rather than an average. The stage's weights are either shared by all experts or separate for
each; the search decides.

///

/// tab | Compare

--8<-- "snippets/fig_arch-3.html"

**Stage 2** puts the experts side by side. It fuses the \(K\) summaries into one representation, from which a small
MLP head computes one score per expert; a softmax turns the scores into routing weights \(\mathbf{w}\). How the
summaries are fused is chosen per corpus on the hold-out data, from the four options below.

///

/// tab | Decode

--8<-- "snippets/fig_arch-4.html"

At inference the expert with the largest weight is chosen, \(\hat{\jmath} = \arg\max_k w_k\), and only its decoder
runs. Its encoder states already exist from the first step, so nothing is encoded twice. The router itself is small
next to the experts; the [cost results](results.md#cost) give the time of every part.

///

### Stage 2 options

--8<-- "snippets/fig_fusion.html"

The **cross-attention bridge** lets each expert's summary attend to the frame-level outputs of the other experts
before a transformer mixes the summaries through a global summary token. The **transformer** option keeps the
global token and the mixing but skips the bridge. **Concatenation** and **mean** join the summaries directly and
hand them to the head. The [synthetic regime switch](results.md#synthetic-regime-switch) runs the full design, bridge
included.

## Training

Before training, every expert transcribes every training clip once, which gives each clip its vector of per-expert
errors; the encoder states are stored in the same pass. The router is then trained on these fixed inputs and targets,
with three terms that can be combined:

\[
\mathcal{L} \;=\; \lambda_{\text{wer}} \sum_{k} w_{k}\, \tilde e_{k}
\;+\; \lambda_{\text{hard}}\, \mathrm{CE}\big(\mathbf{w},\, \jmath^\star\big)
\;+\; \lambda_{\text{soft}}\, \mathrm{CE}\big(\mathbf{w},\, \mathbf{q}\big),
\]

\[
q_{k} \;=\; \frac{\exp(-\tilde e_{k}/\tau)}{\sum_{m} \exp(-\tilde e_{m}/\tau)},
\]

where \(\tilde e_k\) is the clip's word error rate under expert \(k\), capped at 100 %.

- The **expected-WER** term charges the router the error rate it would get under its own weights.
- The **hard cross-entropy** term pushes towards an expert with the fewest errors, the oracle's choice.
- The **soft cross-entropy** term pushes towards a target that gives an expert more weight the closer its error rate
  is to the best, with the temperature \(\tau\) setting how sharply. When two experts are nearly tied on a clip, the
  target splits its weight between them; when one is far ahead, it gets nearly all of it.

Which terms are on is a hyperparameter: hard CE alone, soft CE alone, WER with hard CE, or all three. Each corpus's
search picks one on held-out data. Training uses AdamW with warm-up and cosine decay. The kept model is the one with
the lowest expected WER on a validation cut, averaged over the run's recent weights. Every reported router averages
the routing weights of three independently seeded fits; the pooled baseline, MLP-pool, is built the same way.

## What runs at inference

| | Encoders | Router | Decoders |
|---|---|---|---|
| Best single expert | one | — | one |
| Decode every expert, then fuse | all \(K\) | — | all \(K\) |
| HIT-ASR | all \(K\) | one pass | one |

HIT-ASR adds the other experts' encoders and the router to the cost of one expert, and saves every decoder but one.
How that balances depends on how much of each expert's time is spent in its decoder; the
[cost results](results.md#cost) measure it on every corpus.
