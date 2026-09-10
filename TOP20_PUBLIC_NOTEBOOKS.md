# Top-20 public notebooks (S6E9)

Research notes for [playground-series-s6e9](https://www.kaggle.com/competitions/playground-series-s6e9) (*Predicting Electric Vehicle Purchases*).

## Review log

Append a line each time we re-check what the top 20 are doing (public LB + public notebooks).

| Date | Notes |
|------|--------|
| 2026-09-09 | First pass: public LB top 20 snapshot (≈0.9467 ceiling); public kernels found for Deotte, Ravi, cstdy; rest mostly private. |

Pulled from the public leaderboard (~2026-09-09) and any **public** competition notebooks tied to those teams. Top public LB then ≈ **0.9467**.

Most of the top 20 have **no** public S6E9 notebook. The ones that do are summarized below. Near-top community kernels that mirror the same ideas are noted at the end.

## Leaderboard snapshot (top 20)

| Rank | Team | Public score |
|------|------|--------------|
| 1 | Chris Deotte | 0.94672 |
| 2 | Leo | 0.94664 |
| 3 | haya | 0.94657 |
| 4 | Laura Liepa | 0.94656 |
| 5 | yalpha | 0.94655 |
| 6 | M & M | 0.94654 |
| 7 | Rapha | 0.94654 |
| 8 | Jasper Dekoninck | 0.94653 |
| 9 | Kaggler Sergio | 0.94652 |
| 10 | Maher el Ouahabi | 0.94652 |
| 11 | Don Mani | 0.94650 |
| 12 | Optimistix | 0.94649 |
| 13 | CG | 0.94648 |
| 14 | Randy | 0.94647 |
| 15 | Ravi Ramakrishnan | 0.94647 |
| 16 | Christoffer Thimsen | 0.94647 |
| 17 | cstdy | 0.94647 |
| 18 | tomasa2 | 0.94647 |
| 19 | Tilii | 0.94646 |
| 20 | scikit-learn | 0.94645 |

## Top-20 with public competition notebooks

### 1. Chris Deotte (0.94672)

Public:

- [Fable 5.1 — XGB Starter](https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter)
- [Fable 5.1 — EDA / original-data insights](https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights)

Approach:

- Reverse-engineers a simple “EV recipe” from the **original** dataset (income, subsidy, environmental concern, range anxiety, related factors).
- Adds a few plain interaction / helper features.
- Trains **three XGBoost** models with the same 5-fold CV:
  1. plain baseline
  2. recipe as XGB **base_margin** (trees learn corrections only)
  3. recipe score as an **extra feature**
- Equal-weight blends the three OOF/test preds.
- Emphasis: use the generative recipe + small diversity, not a huge stack.

**In this repo:** implemented as `--strategy deotte` (see `STRATEGIES.md`, `src/ev_s6e9/deotte.py`).

### 15. Ravi Ramakrishnan (0.94647)

Public:

- [PlaygroundS6E9\|Public\|Ensemble\|V1](https://www.kaggle.com/code/ravi20076/playgrounds6e9-public-ensemble-v1)
- Related: [DataCollation V1](https://www.kaggle.com/code/ravi20076/playgrounds6e9-public-datacollation-v1), [EDA V1](https://www.kaggle.com/code/ravi20076/playgrounds6e9-eda-v1)

Approach:

- Ensemble notebook does **not** train base models itself.
- Loads **OOF + test predictions** from a collation of public models.
- Fits a **Logistic Regression** meta-model on those OOFs (C sweep, 5-fold stratified).
- **Rank-blends** roughly 20% meta vs 80% a strong public submission (daily rank-average ensemble style).

### 17. cstdy (`kirill0212`) (0.94647)

Public:

- [S6E9 LightGBM](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) (title undersells the content)

Approach:

- Multi-model CV: **XGBoost + CatBoost + LightGBM + TabM** (neural tabular).
- Nested / fold-safe **target encoding** (multiple smoothings / seeds).
- Community notebooks often cite this as a baseline for TE + digit-style feature ideas.

## Rest of top 20

No public S6E9 notebooks found (via kernel/user search) for:

Leo, haya, Laura Liepa, yalpha, M & M, Rapha, Jasper Dekoninck, Kaggler Sergio, Maher el Ouahabi, Don Mani, Optimistix, CG, Randy, Christoffer Thimsen, tomasa2, Tilii, scikit-learn.

Assume private stacks or unpublished code.

## Themes from near-top public code

Same ideas show up in high-LB public kernels even when the author is not in the top 20 (e.g. Naji pure LGBM / OOF blends, Transformer+GBDT ensembles):

- Exploit **synthetic artifacts** (income cliffs/spikes, digit / trailing-zero features, original-data target means).
- **Fold-safe target (+ frequency) encoding**, often with multiple smoothings.
- **GBDT ensembles** (LGBM / XGB / CatBoost), sometimes plus a neural tabular model.
- **OOF rank blends**; single tree families are highly correlated (~0.98+), so gains past ~0.946 are tiny.

## Our baseline (context)

This repo’s first LightGBM baseline: **CV 0.94171 ± 0.00073**, **LB 0.94150** — still below the ~0.946 plateau described above.
