# Backbone Ablation

This folder contains ablation experiments for the **main model backbone/modules** only.

## Variants

- `baseline`: `STLSTM` only
- `plus_arp`: `STLSTM + ARP`
- `plus_mssp`: `STLSTM + MSSP`
- `plus_arp_mssp`: `STLSTM + ARP + MSSP` with fixed two-branch fusion
- `full`: `STLSTM + ARP + MSSP + CAR(ContextAwareRouter) + ReLU`

The `full` variant is **not retrained** here. The script reuses the existing saved outputs from:

`model_outcomes/checkpoints_gpr_fishnet_final/`

and recomputes hotspot-aware metrics at the requested threshold.

Default training/primary evaluation threshold follows the current main training script: `0.2755`.
By default, the script also exports an additional threshold-recomputed set at `0.3175` for paper/ranking use.

## Run

```bash
python ablation/backbone_ablation/train_backbone_ablation.py
```

Run selected variants only:

```bash
python ablation/backbone_ablation/train_backbone_ablation.py --variants baseline,plus_arp,full
```

Use a different existing main-model result folder for the imported `full` variant:

```bash
python ablation/backbone_ablation/train_backbone_ablation.py --full-source-dir path/to/checkpoints_gpr_fishnet_final
```

## Output Isolation

All outputs are written under:

`model_outcomes/ablation/backbone_ablation/`

Each variant has its own subfolder and checkpoints, metrics, rollout outputs are isolated.

Additional statistical report files are written in the group root:

- `backbone_significance_vs_full.csv`
- `backbone_significance_vs_full.json`
- `backbone_ablation_summary_threshold_0p3175.csv`
- `backbone_ablation_summary_threshold_0p3175.json`
- `backbone_significance_vs_full_threshold_0p3175.csv`
- `backbone_significance_vs_full_threshold_0p3175.json`

These include paired-seed mean differences, 95% CI, and two-sided paired sign-flip p-values against `full`.
