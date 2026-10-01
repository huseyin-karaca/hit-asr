---
hide:
  - navigation
---

# Results

<p class="hit-lede">HIT-ASR is compared with the best single expert, a random choice, two transcript-fusion methods
that run every expert, and two pooled routers, on four English corpora, each with its own trio of experts. Every number
on this page is generated from the paper's tables.</p>

## Setup

Each corpus is split once: a random quarter is held out for the hyperparameter search, where every model gets the
same budget of ten random draws. On the remaining clips, every model is trained and tested on the same five
repetitions of two-fold cross-validation (5×2), and the tables give the mean and standard deviation over the ten test
folds.

The systems compared:

- **Best single expert**: the expert with the lowest WER on the training half, used for every clip.
- **Random**: an expert drawn uniformly for every clip.
- **ROVER (confidence-weighted)** and **CN-MBR**: transcript-level fusion of all three experts' outputs, which needs
  every expert to decode every clip.
- **MLP-pool**: a neural router on the experts' encoder states averaged over time, trained with all three terms of
  HIT-ASR's objective and the same search budget, three seeds averaged as well.
- **ADASTT**: gradient-boosted trees on the same pooled states, trained with cross-entropy.
- **Oracle**: an expert with the fewest errors on each clip, the upper bound for picking one expert.

--8<-- "snippets/table_datasets.html"

--8<-- "snippets/table_experts.html"

## Main results

--8<-- "snippets/results_main.md"

--8<-- "snippets/fig_gap.html"

--8<-- "snippets/table_main.html"

The share of the oracle gap each system closes, and how often it picks a best expert:

--8<-- "snippets/table_main_gc.html"

### Significance

HIT-ASR against every other system on the same folds:

--8<-- "snippets/table_significance.html"

## Synthetic regime switch

A controlled check of the central design choice. Each synthetic clip switches from one acoustic regime to another at
a random point, and which expert is best depends on both regimes and their order, so averaging the frames over time
removes the information the choice needs. HIT-ASR runs here with its full design: shared Stage 1, cross-attention
bridge and a Stage 2 transformer, trained on expected WER plus soft cross-entropy.

--8<-- "snippets/results_synthetic.md"

--8<-- "snippets/fig_synthetic.html"

--8<-- "snippets/table_synthetic.html"

## Routing behaviour

Which expert each router picks, and how often the pick is a best one.

--8<-- "snippets/results_routing.md"

--8<-- "snippets/fig_routing.html"

--8<-- "snippets/table_routing.html"

## Ablation

An exploration on the hold-out data drew HIT-ASR configurations at random, with each design choice as a switch.
Grouping its configurations by one switch at a time shows what that switch does across all the others.

--8<-- "snippets/results_ablation.md"

--8<-- "snippets/fig_objective.html"

--8<-- "snippets/table_ablation.html"

## Cost

Measured on raw audio, one clip at a time on one GPU, with a router of the main experiment deployed in front of the
real experts.

--8<-- "snippets/results_cost.md"

--8<-- "snippets/fig_cost.html"

--8<-- "snippets/table_cost.html"

## Pool size

The People's Speech trio grown to five and to ten experts, adding at each step the expert that lowers the oracle's
WER most, with HIT-ASR and MLP-pool refitted on the hold-out at each size:

--8<-- "snippets/table_pool.html"
