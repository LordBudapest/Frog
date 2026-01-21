Title: Permutation-Regularized Expander Graph Propagation

Abstract:
Expander-based message passing has been proposed as an effective mechanism to mitigate over-squashing in graph neural networks (GNNs) by enabling rapid global information flow. However, fixed expander structures can overfit and generalize poorly on realistic long-range graph benchmarks. In this work, we propose Permuted Expander Graph Propagation (P-EGP), a simple modification to expander-based GNNs that introduces controlled permutation as a form of regularization. Empirically, we show that permutation significantly improves performance at shallow depths, yielding large gains on TreeNeighborMatch as well as consistent improvements over EGP and CGP on Peptides-func and Peptides-struct. At larger depths, we observe that permutation can become detrimental, indicating a trade-off between rapid mixing and layer-to-layer coherence. Our results suggest that permutation acts as an effective regularizer when applied judiciously, improving generalization with negligible additional cost (up to constant factors)

Rough Draft:
Introduction (1–1.5 pages)

Problem: expanders help global mixing but over-regularize

Observation: EGP/CGP overfit training, generalize poorly

Insight: permutation acts as implicit regularization

Contributions (3 bullets, concrete)

Method

Define p-EGP

Full permutation vs first-layer permutation

Complexity note (constant-factor)

Experimental Setup

Datasets

Baselines

Metrics

Seeds