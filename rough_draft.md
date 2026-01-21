Title: Permutation-Regularized Expander Graph Propagation

Abstract:
Expander-based message passing has been proposed as an effective mechanism to mitigate over-squashing in graph neural networks (GNNs) by enabling rapid global information flow. However, fixed expander structures can overfit and generalize poorly on realistic long-range graph benchmarks. In this work, we propose Permuted Expander Graph Propagation (P-EGP), a simple modification to expander-based GNNs that introduces controlled permutation as a form of regularization. Empirically, we show that permutation significantly improves performance at shallow depths, yielding large gains on TreeNeighborMatch as well as consistent improvements over EGP and CGP on Peptides-func and Peptides-struct. At larger depths, we observe that permutation can become detrimental, indicating a trade-off between rapid mixing and layer-to-layer coherence. Our results suggest that permutation acts as an effective regularizer when applied judiciously, improving generalization with negligible additional cost (up to constant factors)

Rough Draft:
Day 1–2: Skeleton + Core Sections (Deepened Content)
1. Introduction (1–1.5 pages)
Problem framing (expanders help, but…)

You want to clearly articulate a tension, not a flaw.

Key idea to convey:

Expander-based propagation improves global information flow, but aggressive or repeated mixing can harm learning by overwhelming local structure and inducing over-regularization.

Explain it like this (conceptually):

Many graph learning tasks require both:

Local pattern recognition (chemistry motifs, functional groups)

Long-range aggregation (global context, matching constraints)

Expander Graph Propagation (EGP) and Cayley Graph Propagation (CGP) were introduced to combat oversquashing by:

Replacing sparse graph neighborhoods with highly connected expander structures

Ensuring rapid information mixing across the graph

However, this introduces a new failure mode:

Excessively fast mixing can collapse distinguishability

Node representations become overly smooth or biased toward global averages

Training accuracy can become high while validation/test stagnate or degrade

This positions your work as corrective, not adversarial.

Empirical observation (EGP/CGP behavior)

Here you state what you actually observed, without interpretation yet.

Core observations to articulate:

On medium-sized, noisy benchmarks (e.g. Peptides-func, Peptides-struct):

EGP and CGP often reach very high training performance

Validation and test performance lag behind

On synthetic diagnostic tasks (TreeNeighborMatch):

EGP/CGP excel at depth-specific reasoning

But performance can degrade sharply with deeper propagation or misalignment

Important:
You are not claiming EGP/CGP are bad — only that their behavior suggests over-regularization or over-mixing.

Insight: permutation as implicit regularization

This is the intellectual heart of the paper.

You should frame permutation not as increased expressivity, but as controlled noise.

Key intuition to articulate:

Applying a fixed expander at every layer enforces a rigid global communication pattern

Introducing a permutation:

Breaks exact alignment of message-passing paths

Prevents the model from overfitting to specific long-range shortcuts

Acts analogously to regularization techniques like dropout or data augmentation

Crucially:

The permutation does not remove connectivity

It only perturbs which long-range interactions are emphasized

This leads to your core thesis:

Permutation does not help by increasing mixing, but by tempering it.

This reframes the entire expander narrative.

Contributions (keep concrete)

You already have these, but here’s how to phrase them cleanly:

Methodological
We propose Permuted Expander Graph Propagation (p-EGP), a simple modification of EGP that introduces controlled permutation of expander edges to regularize message passing.

Empirical
Across Peptides-func, Peptides-struct, and TreeNeighborMatch, p-EGP consistently improves generalization over EGP and CGP, often by substantial margins.

Conceptual
We provide empirical evidence that excessive deterministic mixing can harm performance, and that shallow, controlled randomness offers a better tradeoff between global communication and generalization.

No theory claim. No guarantees.

2. Method
Define p-EGP clearly

Explain it operationally, not abstractly.

Start from standard EGP:

At each layer, node features are propagated along a fixed expander graph

p-EGP modification:

Before propagation, apply a permutation to node indices

Perform expander-based message passing

Optionally invert the permutation (or not, depending on variant)

Stress:

The permutation is fixed per run, not resampled every batch

No additional learnable parameters

Full permutation vs first-layer permutation

This is very important — it shows you understand failure modes.

Explain the distinction:

Full permutation (every layer):

Introduces high stochasticity

Can destroy multi-layer path coherence

Often detrimental on structured or low-noise tasks

First-layer-only permutation:

Injects diversity early

Preserves stable propagation paths in deeper layers

Works particularly well on TreeNeighborMatch

This gives you a clean ablation narrative:

“Where randomness helps depends on depth.”

Complexity note (constant-factor)

You don’t need math — just clarity.

Explain:

p-EGP uses the same number of message-passing operations as EGP

Permutation is an 
𝑂
(
𝑛
)
O(n) index reordering

Asymptotic time and memory complexity are unchanged

Empirically, runtime overhead is negligible

This defuses reviewer concerns early.

3. Experimental Setup
Datasets (why each one matters)

Explain why you chose them:

Peptides-func
Medium-sized molecular graphs requiring long-range interactions under noise.

Peptides-struct
Same graphs, regression task — tests robustness of representation learning.

TreeNeighborMatch
Synthetic diagnostic task explicitly designed to stress oversquashing and depth.

This shows intentionality, not cherry-picking.

Baselines

Explicitly state:

Same backbone architecture (MPNN)

Same depth, hidden size, optimizer, and training protocol

Only difference: propagation graph

This is critical for fairness.

Metrics, seeds

Report mean ± standard deviation

Use multiple seeds (≥5)

Select checkpoints based on validation performance

You want reviewers to trust your numbers.

Day 3–4: Results + Figures (Interpretive Guidance)
Main tables

When writing the results section, do not just report numbers.

For each table, answer:

Does p-EGP outperform EGP and CGP?

Is the improvement consistent?

Is variance reduced?

For Peptides-func:

Emphasize generalization gap reduction

Note that base GNN < EGP/CGP < p-EGP

For Peptides-struct:

Emphasize robustness across seeds

Note that improvements hold for regression, not just classification

TreeNeighborMatch (diagnostic framing)

Explain why this matters:

TNM isolates depth-dependent reasoning

Many methods fail abruptly as depth increases

p-EGP (first-layer permutation) improves:

Validation and test accuracy

While slightly reducing training accuracy

This strongly supports your regularization hypothesis.

Optional figure: train vs val/test gap

This is very powerful if you include it.

Explain what it shows:

EGP/CGP: rapid training convergence, stagnant validation

p-EGP: slower training, higher validation peak

You don’t need many figures — one good one is enough.

Ablations

Explain ablations as tests of hypotheses:

Full vs first-layer permutation → tests coherence

Depth sensitivity → tests when randomness helps vs hurts

Even 1–2 ablation tables are enough.

Day 5: Discussion + Related Work
Discussion: explain why it works

Structure it around tradeoffs:

Why permutation helps at shallow depth

Prevents brittle reliance on specific long-range shortcuts

Encourages distributed representations

Acts as inductive noise

Why deeper permutation hurts

Breaks alignment across layers

Prevents accumulation of relational evidence

Similar to excessive dropout

Big-picture lesson

The problem is not lack of mixing, but lack of control over mixing.

This is a strong conceptual takeaway.

Related work (keep factual)

Don’t summarize papers — categorize them:

Oversquashing: motivation

Graph rewiring: alternative solutions

Expander-based methods: closest prior work

Then state:

Our work complements these by highlighting a regularization perspective.