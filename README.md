# Neural Tissue Fields

This project explores continuous neural representations for spatial omics.

Goal: learn a function

f(x, y, z) → molecular state / gene expression

from measured spatial transcriptomics sections, then test whether the model can predict a held-out tissue section.

First milestone:
- Load public spatial transcriptomics data
- Convert it into x, y, z → expression format
- Train a simple coordinate MLP
- Predict a held-out section
- Plot real vs predicted gene expression
