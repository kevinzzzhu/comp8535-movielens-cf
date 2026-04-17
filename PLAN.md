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
- [ ] Full 30-epoch run at d=128 — confirm:
  - SVD ≈ 1.02–1.08
  - MF ≈ 0.93
  - NMF ≈ 0.92
  - Proposed < 0.91
- [ ] If MF/NMF > 0.94, debug before proceeding

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
- [ ] Verify ordinal head invariants: `Σ P(r=k) = 1`, thresholds monotone, gradients non-zero
- [ ] Add `tests/test_model.py` with 3–5 unit tests (gate shape, ordinal probs sum to 1, threshold ordering)
- [ ] Full training run, proposed model, RMSE ≤ 0.91 on `u1.test`
- [ ] Inspect mean gate across epochs — should stay in [0.2, 0.8], not collapse
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
