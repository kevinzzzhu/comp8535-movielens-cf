# 2026-04-27_ablations_v2

| Fusion | Head | RMSE | MAE | Accuracy | NLL |
|---|---|---|---|---|---|
| additive | ordinal | 0.9228 ± 0.0012 | 0.7221 ± 0.0012 | 0.4311 ± 0.0018 | 1.2637 ± 0.0046 |
| additive | sigmoid | 0.9259 ± 0.0004 | 0.7321 ± 0.0004 | 0.4166 ± 0.0002 | — |
| gated | ordinal | **0.9179 ± 0.0029** | **0.7182 ± 0.0032** | **0.4335 ± 0.0014** | **1.2560 ± 0.0007** |
| gated | sigmoid | 0.9182 ± 0.0007 | 0.7230 ± 0.0015 | 0.4256 ± 0.0012 | — |
| none | ordinal | 0.9289 ± 0.0006 | 0.7309 ± 0.0008 | 0.4237 ± 0.0015 | 1.2682 ± 0.0031 |
| none | sigmoid | 0.9258 ± 0.0004 | 0.7328 ± 0.0002 | 0.4194 ± 0.0002 | — |

Bold = best per column (lower-is-better for RMSE/MAE/NLL, higher for Acc).
Variance: 3 seeds {42, 43, 44}.
Protocol: ablation (patience=10, max 30 epochs).

Raw: `ablation_summary.csv` · per-run JSONs: `<fusion>_<head>_seed<N>/results.json`
