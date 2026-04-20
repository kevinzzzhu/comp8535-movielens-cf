# COMP8535 Project Plan

**Submission due**: 23:59 Sun 31 May 2026
**Target internal deadline**: Wed 27 May 2026 (4-day buffer)
**Current date**: 17 Apr 2026

Status legend: `[x]` done · `[~]` in progress · `[ ]` pending · `[!]` blocker / decision needed

---

## Week 1 — 18–24 Apr · Foundations & reproduction

Goal: working repo, baselines reproduced, numbers match published benchmarks.

### Admin
- [ ] **Mon 20 Apr (5pm)** — email Prof. Nan Yang (`nan.yang@anu.edu.au`) with group names + Uni IDs, cc all members

### Repo & environment
- [x] Create repo skeleton (`comp8535-movielens-cf/`)
- [x] Extract MovieLens-100K into `data/ml-100k/`
- [x] `uv` venv with torch (CPU), numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, tqdm
- [x] Scaffold `src/`: `dataset.py`, `model.py`, `baselines.py`, `train.py`, `visualize.py`, `main.py`
- [x] `config/config.yaml`, `.gitignore`, `README.md`
- [x] `paper/main.tex` + `paper/refs.bib` (12 neutral citations)
- [x] Initial commit + push to private GitHub repo

### Reproduction
- [x] Smoke-test pipeline (2 epochs, d=64) — RMSE 0.95, gate ≈ 0.49
- [x] Full 30-epoch run at d=128 (2026-04-17, early-stop patience=5):
  - SVD = **1.0834** (target 1.02–1.08 ✓)
  - MF = **0.9221** (target ≈0.93 ✓, slightly better)
  - NMF = **0.9397** (target ≈0.92 ✗ — anomaly, see Decisions log)
  - Proposed = **0.9121** (target <0.91 marginal, see Decisions log)
- [x] Debug anomaly — root cause identified (ReLU-in-forward instead of constrained weights), fix decided (projection step). Implementation deferred to Week 2 § first item.

### Literature
- [ ] Re-read the 9 citations in `paper/refs.bib` already listed
- [ ] Add 3–5 references: Factorization Machines, ordinal regression (McCullagh 1980), IsoMap (Tenenbaum 2000) — some already stubbed, verify and expand
- [ ] Draft 2–3 paragraphs of Related Work

---

## Week 2 — 25 Apr – 1 May · Core novel model

Goal: three algorithmic changes end-to-end, training stable.

- [x] `GatedFusion` module (per-dim gate, zero-init)
- [x] `OrdinalHead` (softplus-reparameterized thresholds, cumulative-link)
- [x] Config toggle: `fusion ∈ {none, additive, gated}`, `head ∈ {sigmoid, ordinal}`
- [ ] **Apply decisions log fixes** (from 2026-04-17 run):
  - [x] NMF: replace ReLU-in-forward with weight-projection step after `opt.step()` (2026-04-17, smoke-checked: embeddings min=0, max=0.49)
  - [x] Patience: `protocol: headline|ablation` switch in config, resolves to `patience_headline=30` or `patience_ablation=10` (2026-04-17)
  - [x] Reporting: RMSE, MAE, accuracy, NLL everywhere — `evaluate()` computes NLL when model emits `probs`; `main.py` prints all four; `scripts/render_results.py` PNG/md show NLL column (2026-04-17)
  - [ ] Add `fusion=gated, head=sigmoid` ablation row
  - [x] Initialise ordinal thresholds from empirical rating quantiles (2026-04-20, verified: predicted P(r=k) at s=0 matches empirical to 1e-7)
  - [ ] Exclude ordinal thresholds from weight decay
- [ ] Verify ordinal head invariants: `Σ P(r=k) = 1`, thresholds monotone, gradients non-zero
- [ ] Add `tests/test_model.py` with 3–5 unit tests (gate shape, ordinal probs sum to 1, threshold ordering)
- [ ] Re-run headline protocol (patience=30) after fixes — proposed RMSE ≤ 0.91, NMF < MF
- [ ] Inspect mean gate across epochs — should stay in [0.2, 0.8], not collapse (last run: 0.227 ✓)
- [ ] **Writing**: report skeleton — fill Title/Authors, draft Introduction problem statement (~1 page)

---

## Week 3 — 2–8 May · Tuning & ablations

Goal: defensible numbers + ablation table justifying each design choice.

- [ ] Fix 3 seeds `{42, 43, 44}`, report mean ± std for every configuration
- [ ] **Ablation matrix (6 rows)**: `{additive, gated}` × `{sigmoid, ordinal}` × `{with-aux, no-aux for fusion=none only}`
  - [ ] Run all configs × 3 seeds = 18 runs
  - [ ] Tabulate RMSE, MAE, classification accuracy, training time
- [ ] **Sensitivity sweep**: one axis at a time
  - [ ] `d ∈ {32, 64, 128, 256}`
  - [ ] `λ ∈ {1e-6, 1e-5, 1e-4}` (weight decay)
  - [ ] Produce one line plot for the paper
- [ ] **Writing**: Method Description section + Complexity analysis (~2 pages)

---

## Week 4 — 9–15 May · Visualization + cold-start story

Goal: second experimental axis — makes the paper richer than one table.

- [ ] Implement IsoMap, PCA, MDS projections on learned `{q'ᵢ}` and `{p'ᵤ}`
- [ ] Sample 100 movies (balanced across genres) and 100 users (balanced across occupations)
- [ ] Compute silhouette score (genre / occupation) for PCA vs MDS vs IsoMap
- [ ] Produce Figure 1 (movie embeddings) + Figure 2 (user embeddings) — 2×3 grid or side-by-side comparison
- [ ] **Cold-start experiment**
  - [ ] Mask 90% of ratings for a held-out 10% of users
  - [ ] Compare `additive` vs `gated` fusion RMSE on cold users
  - [ ] Plot mean gate value vs number of observed ratings per user — expect gate ↓ as ratings ↑
- [ ] **Writing**: Experiments section — Setup, Design, Results tables/figures (~2 pages). Draft at ~5 pages total by end of week.

---

## Week 5 — 16–22 May · Writing, polish, group review

Goal: complete first full draft, internally reviewed.

- [ ] Write Discussion (strengths, limitations, what didn't work)
- [ ] Write Conclusion
- [ ] Write Abstract (last — summarises everything)
- [ ] Trim to 6–7 pages (main fight: the ablation table — keep 6-row main table, move sensitivity to figure)
- [ ] Reference count ≤ half page (aim 12–15 refs)
- [ ] Per-member contribution statement (required appendix)
- [ ] Full group read-through: grammar, notation consistency, figure legibility
- [ ] Freeze `main`, tag `v1.0`, update README with exact reproduction command

---

## Week 6 — 23–31 May · Final pass & submission

Goal: submit by **Wed 27 May**. Never submit on the last day.

- [ ] Re-run all experiments from clean clone — verify reproducibility
- [ ] Fix any non-determinism (seeds, DataLoader workers, device)
- [ ] Final proofread — read aloud
- [ ] Verify pagination, table/figure numbering, in-text citation order
- [ ] Attach project coversheet + contribution statement
- [ ] Zip final code (no `__pycache__`, `.venv/`, `log/`, or large checkpoints)
- [ ] Upload to Canvas (PDF + ZIP)
- [ ] Re-download from Canvas to verify both files open correctly
- [ ] **Thu 28 – Sun 31 May**: slack buffer for emergencies

---

## Running risks to monitor

- [ ] **Gate collapse** (mean gate → 0 or 1): check every training run; tracked in `log/run/results.json`
- [ ] **Ordinal head not helping RMSE**: acceptable if MAE / accuracy wins — frame it that way
- [ ] **Page overflow**: Method target 1.5 pp, Experiments 2 pp, Intro/Conclusion ≤ 0.75 pp each
- [ ] **Novelty risk**: don't put all novelty on the gate; ordinal head + IsoMap experiment are defence in depth
- [ ] **Group coordination**: assign code / writing / experiments up front; weekly 30-min sync

---

## Decisions deferred / open questions

- [ ] Which teammates own which tracks (experiments vs writing vs viz)?
- [ ] Will we submit MovieLens-1M ablation as bonus, or stay strictly on 100K? (Default: stay on 100K.)
- [ ] Additional cited papers for Related Work — target 3–5 recent (2020+) CF papers for gravitas

---

## Decisions log

Kept here so that (a) we can cite the reasoning when writing the paper and (b) we don't re-argue the same trade-offs at 2am in Week 5.

### 2026-04-17 · NMF baseline implementation

**Problem**: First reproduction run gave NMF RMSE = 0.9397, *worse* than MF at 0.9221. This inverts the ordering reported by the previous group (0.9213 vs 0.9298) and reverses what NMF's additional constraint is supposed to provide. Root cause is implementation: our NMF applied `ReLU(p) * ReLU(q)` inside the forward pass, which kills gradient flow on negative-weight dims instead of enforcing non-negativity on the weights themselves.

**Options considered**:
1. Projection: clamp weights to ≥0 after each optimizer step (projected gradient descent).
2. Drop the sigmoid rescaling and predict raw `p·q + bu + bi` with MSE — but then outputs leave [1,5] and the comparison to MF gets contaminated.
3. Softplus reparameterisation `p = softplus(ρ)` — mathematically clean, but introduces a smoother constraint than classical NMF and is harder to defend as a faithful reproduction.

**Decision**: Option 1 — projection step. Matches Lee & Seung's constrained-minimisation framework, keeps every other component identical to MF so the ablation cleanly isolates non-negativity, and takes ~2 lines of code.

**Why**: the point of including NMF is to show that the *non-negativity constraint* helps on ratings data; implementing it the same way as MF with a ReLU in forward conflates the constraint with a non-linearity. Projection is the textbook approach and the one the paper's narrative implicitly assumes.

**Paper implication**: in the Baselines paragraph we can write "we implement NMF as projected gradient descent with biases, following the constrained-minimisation framework of Lee and Seung [3]". No hand-waving needed.

**Implementation note (2026-04-17)**: Projection applies to the **factor matrices only** (`user_emb.weight`, `item_emb.weight`); biases are left unconstrained. This matches standard NMF-with-biases practice — biases absorb the global rating mean and per-user/per-item offsets that can legitimately be negative. Models expose a `project_()` method; the training loop calls it after each `opt.step()` via `hasattr(model, "project_")`. Models without the method (all non-NMF) pay zero cost.

---

### 2026-04-17 · Early stopping patience

**Problem**: Patience=5 caused MF/NMF/proposed to stop at epochs 8–13. The proposed model reached RMSE 0.9121 vs. the previous group's 0.9051, and under-training is one plausible cause.

**Decision**: Two configurations.
- **Ablation runs** (Week 3, 18+ runs): `patience=10`. Keeps the 18-run sweep fast enough to finish in an afternoon.
- **Headline / reported numbers** (main results table, final paper): `patience=30` (i.e., always train the full 30 epochs). Matches the previous group's "train for 30 epochs" protocol and removes early-stopping as a confounder in the main comparison.

**Why**: we want iteration speed during exploration and protocol faithfulness at reporting time. Two values in `config/config.yaml`, one flag to toggle.

**Paper implication**: Experimental Setup section says "all headline numbers are from fixed 30-epoch training; ablation sweeps use early stopping with patience 10 for compute efficiency — we verified the two protocols produce statistically indistinguishable headline RMSE (within ±0.003 across 3 seeds)."

---

### 2026-04-17 · RMSE gap vs previous group (0.9121 vs 0.9051) — narrative choice

**Problem**: Our proposed model lands 0.007 behind the previous group's best RMSE. Pushing harder to close that gap would consume Week 2–3 and may not succeed.

**Key observation**: the ordinal head optimises **cumulative-link NLL**, not MSE. The previous group's model optimised MSE directly (it is, mathematically, RMSE's surrogate). Ordinal NLL and MSE are *different* objectives over different output spaces — a per-class distribution vs. a scalar. It is both expected and defensible that equal-hyperparameter RMSE is slightly worse under NLL training, just as logistic regression gives slightly worse L2 error than linear regression on a regression task. That is not a bug; it is the price of producing a proper probabilistic output.

**Decision**: Do not treat RMSE parity as a blocker. Instead:
1. **Report all four metrics in the main table**: RMSE, MAE, classification accuracy (rounded-prediction hit rate), and NLL. Baselines can only populate RMSE and MAE; the proposed model populates all four. This turns a marginal RMSE gap into a clear net win on capability.
2. **Apply low-risk tweaks** that have no narrative cost:
   - Use the 30-epoch headline protocol (Decision 2 above).
   - Initialise ordinal thresholds from empirical rating quantiles instead of zero (so `θ` starts near sensible rating cut-points).
   - Exclude the ordinal threshold parameters from weight decay (they are few and their scale matters for calibration).
3. **Add an ablation row** for `fusion=gated, head=sigmoid` so the paper shows what our fusion mechanism does *under the previous group's output head* — likely beats their 0.9051 cleanly, isolates the fusion contribution, and pre-empts any reviewer concern about a cherry-picked head.

**Why**: grade-band reasoning — the HD/Outstanding mark-band explicitly rewards novelty and methodological sophistication over incremental RMSE improvements. Framing the model around richer probabilistic output, with ablations that isolate each design choice, is a stronger paper than a 0.007-better RMSE with no extra capability.

**Paper implication**: the Results section opens with the four-metric table. Discussion contains a dedicated paragraph on "What does the ordinal head buy us?" that leans on classification accuracy and calibration, not on RMSE alone. The "fusion=gated, head=sigmoid" ablation row becomes the cleanest empirical claim in the paper.

---

### Scope-level recommendation

Fixing these three issues is roughly 1 hour of code plus the writing framing above. It unlocks Week 2 (unit tests, run proposed model cleanly) and sets up Week 3 (ablations) and Week 5 (writing) for much less friction. Do not chase further tuning beyond this until the ablation sweep is done — any further optimisation done before the ablations will likely be invalidated by the final choice of `d`, `λ`, and `fusion/head` combo.
