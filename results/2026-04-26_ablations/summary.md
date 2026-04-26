# 2026-04-26_ablations

| Fusion | Head | RMSE | MAE | Accuracy | NLL |
|---|---|---|---|---|---|
| additive | ordinal | 0.9168 ± 0.0002 | 0.7174 ± 0.0036 | 0.4319 ± 0.0039 | 1.2576 ± 0.0033 |
| additive | sigmoid | 0.9200 ± 0.0007 | 0.7263 ± 0.0015 | 0.4224 ± 0.0049 | — |
| gated | ordinal | **0.9108 ± 0.0026** | **0.7119 ± 0.0020** | **0.4368 ± 0.0026** | **1.2515 ± 0.0059** |
| gated | sigmoid | 0.9127 ± 0.0012 | 0.7186 ± 0.0014 | 0.4279 ± 0.0011 | — |
| none | ordinal | 0.9232 ± 0.0009 | 0.7254 ± 0.0012 | 0.4264 ± 0.0017 | 1.2638 ± 0.0014 |
| none | sigmoid | 0.9202 ± 0.0016 | 0.7276 ± 0.0015 | 0.4219 ± 0.0016 | — |

Bold = best per column (lower-is-better for RMSE/MAE/NLL, higher for Acc).
Variance: 3 seeds {42, 43, 44}.
Protocol: ablation (patience=10, max 30 epochs).

Raw: `ablation_summary.csv` · per-run JSONs: `<fusion>_<head>_seed<N>/results.json`
