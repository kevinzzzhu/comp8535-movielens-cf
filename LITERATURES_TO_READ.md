---
created: 2026-04-22
last_edited: 2026-04-26
tags: [literature, citations, paper-writing]
---

# Literature & Citations — Reading Guide and Status

Companion to `RESULTS.md`. Empirical results live there; this file focuses on
**what to read** and **what is already cited where** in `paper/main.tex`.

---

## 1. Citation status (auto-grep this before each commit)

```sh
grep -oE "\\\\cite\{[^}]+\}" paper/main.tex \
  | sed 's/\\cite{//; s/}//' | tr ',' '\n' | sort -u
```

Status as of 2026-04-26 — coverage = 25 of 27 entries cited.

| Key | Paper | Used in main.tex | Why Relevant |
|-----|-------|:--:|--------------|
| `koren2011ordrec` | OrdRec (Koren & Sill 2011) | ✅ §1 | **Primary precedent for our ordinal head** |
| `zaman2025oprfm` | OPRFM (Zaman & Jana 2025) | ✅ §1 | **Closest scoop risk**; ordinal+FM on ML-100K |
| `gouvert2020ordnmf` | OrdNMF (Gouvert et al. 2020) | ✅ §1 | Ordinal + NMF precedent |
| `ma2019gate` | GATE (Ma et al. 2019) | ✅ §1 | **Gated fusion precedent** (item-side only) |
| `vargas2020clm` | CLM (Vargas et al. 2020) | ✅ §1, §3 | Justifies softplus-monotone trick |
| `cao2020coral` | CORAL (Cao et al. 2020) | ✅ §1, §3 | Cumulative-link in deep ordinal regression |
| `liu2014orf` | ORF (Liu et al. 2014) | ✅ §1 | Ordinal CF (methodological breadth) |
| `koren2013ordinalfeedback` | Koren & Sill 2013 | ✅ §1 | Earlier Koren ordinal CF work |
| `mccullagh1980ordinal` | McCullagh 1980 | ✅ §1, §3 | Foundational cumulative-link model |
| `schafer2007cf` | CF survey (2007) | ✅ §1 | Foundational CF background |
| `koren2009mf` | MF techniques (2009) | ✅ §1 | MF baseline theory |
| `harper2015movielens` | MovieLens (2015) | ✅ §1, §4 | Dataset paper |
| `he2017ncf` | NCF (He et al. 2017) | ✅ §1 | Neural CF lineage |
| `wang2015cdl` | CDL (Wang et al. 2015) | ✅ §1 | Deep CF lineage |
| `rendle2010fm` | FM (Rendle 2010) | ✅ §1 | Factorisation machines |
| `he2017nfm` | NFM (He & Chua 2017) | ✅ §1 | Deep CF lineage |
| `xiao2017afm` | AFM (Xiao et al. 2017) | ✅ §1 (×2) | Attention-based feature fusion |
| `guo2017deepfm` | DeepFM (Guo et al. 2017) | ✅ §1 | Deep FM architecture |
| `cheng2016widedeep` | Wide&Deep (Cheng et al. 2016) | ✅ §1 | Wide+deep philosophy |
| `lian2018xdeepfm` | xDeepFM (Lian et al. 2018) | ✅ §1 | Extended DeepFM |
| `tenenbaum2000isomap` | IsoMap (Tenenbaum et al. 2000) | ✅ §1 | Manifold visualisation primary |
| `maaten2008tsne` | t-SNE (van der Maaten & Hinton 2008) | ✅ §1 | Manifold visualisation alt |
| `mcinnes2018umap` | UMAP (McInnes et al. 2018) | ✅ §1 | Modern manifold visualisation |
| `lee2000nmf` | NMF (Lee & Seung 2000) | ✅ §4 | NMF baseline justification |
| `kingma2014adam` | Adam (Kingma & Ba 2014) | ✅ §4 | Optimiser citation |
| `jolliffe2002pca` | PCA (Jolliffe 2002) | ⏳ Week 4 | Reserve for §4.5 PCA-vs-IsoMap baseline |
| `gower2014netflix` | Netflix prize/SVD (Gower 2014) | ⏸ unused | Drop in Week 5 cleanup if still unused |

All **four scoop-risk citations** from the 2026-04-21 audit (OrdRec, OPRFM, OrdNMF, GATE) are in the §1 Related Work paragraph.

---

## 2. The four must-read precedents (read these before next writing pass)

These four papers cover the entire conceptual perimeter of the proposed model. Read them, in order, to internalise where our novelty lives.

### 1. **Koren & Sill (2011) — OrdRec**
*RecSys, pp. 117–124* · key `koren2011ordrec`

Canonical ordinal recommender. Predicts P(r=k) over MovieLens ratings using cumulative-link models. **Differences from us**: OrdRec uses logistic-form thresholds; we use softplus-monotone with empirical-quantile init. OrdRec has no side-info fusion; we do. OrdRec has no manifold viz.

### 2. **Zaman & Jana (2025) — OPRFM**
*Int. J. Data Science and Analytics 20(3): 2615–2629* · key `zaman2025oprfm`

Combines ordinal probit regression with factorisation machines. Tested on ML-100K/1M/10M. **Closest scoop risk** — same dataset, same ordinal-regression family. **Differences from us**: FM backbone (quadratic feature interactions) vs our MF + per-dim gated fusion; probit link vs our logit; no manifold diagnostic.

### 3. **Gouvert, Oberlin, Févotte (2020) — OrdNMF**
*ICML, pp. 3680–3689* · key `gouvert2020ordnmf`

Ordinal head on top of NMF via quantisation thresholds. Demonstrates ordinal beats sigmoid for calibration. **Differences from us**: hard quantisation vs our softplus-reparameterised continuous thresholds; NMF backbone vs MF; no fusion mechanism.

### 4. **Ma et al. (2019) — GATE**
*WSDM, pp. 519–527* · key `ma2019gate`

Sigmoid-gated fusion of item content with item embedding. **Differences from us**: GATE gates only the item side; we gate both sides symmetrically. GATE uses per-item global gates; we use per-dimension. GATE inits randomly; we zero-init so g=0.5 at start.

---

## 3. Reading strategy

### If you have 1 hour
Read the four precedents above. That's enough to defend the novelty positioning in §1.

### If you have 3 hours
Add:
- **Tenenbaum et al. (2000)** — IsoMap — needed before Week 4 visualisation
- **Vargas et al. (2020)** — CLM — justifies the softplus-monotone trick we already use

### If you have 6+ hours
Add:
- Schafer et al. (2007) — CF survey, sets context
- Koren, Bell & Volinsky (2009) — MF techniques, our baseline theory
- McCullagh (1980) — original cumulative-link model

---

## 4. Reserves and unused refs

- **`jolliffe2002pca`** — Reserve for §4.5 (Week 4): PCA as a linear baseline against IsoMap for embedding visualisation. If we don't run PCA, drop.
- **`gower2014netflix`** — Currently unused; covers Netflix prize / SVD context. Drop in Week 5 cleanup unless §1 needs a historical-context sentence.

---

**Last updated**: 2026-04-26 — citation status auto-grep'd before this commit.
