# Environmental Factor Ablation (One-by-One)

This folder contains **one-by-one (leave-one-out)** ablation for environmental factors.

## Protocol

- Backbone kept fixed to GPR-FishNet: `STLSTM + ARP + MSSP + CAR(ContextAwareRouter) + ReLU`
- Base factors: `thetao, chl, uo, vo, so, zos, o2`
- Reference run: full factors (`full_factors`)
- One-by-one runs: `drop_thetao`, `drop_chl`, ..., `drop_o2`
- Fairness control: keep model input channels fixed at `8` (`7 env + 1 AIS`) for all runs; the dropped factor channel is zero-filled instead of changing `in_chans`.

Each run removes exactly one factor while keeping all others unchanged.

Default training/primary evaluation threshold follows the main training setup: `0.2755`.
By default, the script also exports an additional threshold-recomputed set at `0.3175` for paper/ranking use.

## Run

```bash
python ablation/env_factor_ablation/train_env_factor_ablation.py
```

Run selected drop factors only:

```bash
python ablation/env_factor_ablation/train_env_factor_ablation.py --drop-factors thetao,chl
```

Skip the full-factor reference run:

```bash
python ablation/env_factor_ablation/train_env_factor_ablation.py --skip-full
```

## Output Isolation

All outputs are written under:

`model_outcomes/ablation/env_factor_ablation/`

Each experiment has its own isolated folder (`full_factors`, `drop_*`).

Additional statistical report files are written in the group root:

- `env_factor_significance_vs_full_factors.csv`
- `env_factor_significance_vs_full_factors.json`
- `env_factor_leave_one_out_summary_threshold_0p3175.csv`
- `env_factor_leave_one_out_summary_threshold_0p3175.json`
- `env_factor_significance_vs_full_factors_threshold_0p3175.csv`
- `env_factor_significance_vs_full_factors_threshold_0p3175.json`

These include paired-seed mean differences, 95% CI, and two-sided paired sign-flip p-values against `full_factors`.
