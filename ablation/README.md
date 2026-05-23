# Ablation Experiments

This directory is separated into two isolated experiment groups:

- `ablation/backbone_ablation`: module/backbone ablations of the main network
- `ablation/env_factor_ablation`: one-by-one environmental-factor ablations

The output directories are also isolated:

- `model_outcomes/ablation/backbone_ablation/`
- `model_outcomes/ablation/env_factor_ablation/`

Use each group's own training entry script to run experiments.

Both groups now generate statistical significance reports (paired-seed CI + p-values) in their respective output roots.
