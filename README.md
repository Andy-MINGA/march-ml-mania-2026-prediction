# March Machine Learning Mania 2026

**Kaggle Competition | NCAA Tournament Win Probability Prediction**

| Metric | Value |
|--------|-------|
| Stage 1 Brier Score (historical 2021–2025) | **0.03037** |
| Stage 1 Brier Skill Score | **0.8785** |
| Improvement vs. naive baseline | **+87.9%** |
| **Final Kaggle Leaderboard Brier (2026 live)** | **0.1698** |
| **Live Brier Skill Score** | **0.321** |
| Features per matchup | 78 |
| Training samples (men's) | 602 |
| Total 2026 predictions | 132,133 |

---

## What This Is

Every March, 68 college basketball teams play a single-elimination tournament and nobody fully knows what's going to happen; that's what makes it genuinely interesting from a machine learning standpoint. This repository contains a complete end-to-end pipeline I built for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition, which asks you to estimate the win probability for every possible matchup in the NCAA Men's and Women's Tournaments (132,133 total: 66,430 men's, 65,703 women's).

The task sounds clean. It isn't. You have roughly 602 labelled examples for the men's bracket (tournament games from 2016–2025), a feature space that requires meticulous construction to avoid data leakage, and an evaluation metric — the Brier score — that punishes overconfidence far more than it rewards being bold. Everything collapses to two things: build features that capture what actually matters, and make sure your predicted probabilities are honest.

The pipeline went through two full versions. Version 1 was a 5-model ensemble on competition data only; it scored 0.0819 on historical validation but ranked near the bottom of the leaderboard because of miscalibration and no external data. Version 2 completely redesigned the feature set, reduced to 3 models, added KenPom and Barttorvik, and implemented three-method calibration. It scored **0.03037** on historical validation (BSS = 0.8785) and **0.1698** on the actual 2026 Kaggle leaderboard (BSS ≈ 0.32).

---

## The Math Behind the Metric

The competition evaluates on the **Brier score**:

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

Naively predicting 0.5 for every game gives BS = 0.25. The Murphy (1973) decomposition separates this into three orthogonal terms:

$$BS = \underbrace{\text{Reliability}}_{\text{calibration error}} - \underbrace{\text{Resolution}}_{\text{discrimination}} + \underbrace{\text{Uncertainty}}_{\text{irreducible}}$$

The **Reliability** term is the one you control through calibration. A model predicting 0.95 when teams at that confidence level actually win 65% of the time contributes $(0.95 - 0.65)^2 = 0.09$ per game in that bin — a 49× penalty compared to a perfectly calibrated prediction. That is why calibration is not an afterthought in this pipeline; it gets its own section with three competing methods.

The **Brier Skill Score** normalises against the climatological baseline:
$$BSS = 1 - \frac{BS}{0.25}$$

A BSS of 1.0 is perfect; 0.0 is random guessing.

---

## Two Pipeline Versions

### Version 1 — Baseline (Brier = 0.0819, BSS = 0.672)

Built on competition data only to establish a credible floor. Five models (Logistic Regression, XGBoost, LightGBM, CatBoost, Random Forest) blended with inverse-squared Brier weights. Features: Elo ratings, season win rates, point differentials, seeds, and Massey ordinals.

**Why it ranked poorly:** No external data (no KenPom, no Vegas). Extreme predictions without calibration inflated the Reliability penalty. Five correlated models added noise without improving Resolution on a 602-sample dataset.

| Model | Mean CV Brier | vs. Baseline |
|-------|--------------|--------------|
| Logistic Regression | 0.2042 ± 0.0272 | +18.3% |
| XGBoost | 0.2103 ± 0.0331 | +15.9% |
| LightGBM | 0.2177 ± 0.0337 | +12.9% |
| CatBoost | 0.2051 ± 0.0282 | +18.0% |
| Random Forest | 0.1998 ± 0.0227 | +20.1% |
| **Ensemble** | **~0.197** | **+21%** |
| Naive baseline (0.5) | 0.2500 | — |

Stage 1 historical: **Brier = 0.0819, BSS = 0.672, +67.2% over baseline**

### Version 2 — Redesigned Ensemble (Brier = 0.03037, BSS = 0.8785)

A complete rebuild. Three architecturally distinct models, nine data sources, 78 features across 11 groups, Optuna Bayesian tuning, and three-method probability calibration.

**Version comparison:**

| Pipeline | Stage 1 Brier | BSS | vs. Baseline |
|----------|--------------|-----|--------------|
| Naive (always 0.5) | 0.2500 | 0.000 | — |
| Simple seed model | ~0.180 | ~0.28 | +28% |
| V1 (5 models, no external data) | 0.0819 | 0.672 | +67.2% |
| **V2 (3 models + KenPom + calibration)** | **0.03037** | **0.8785** | **+87.9%** |
| Perfect | 0.0000 | 1.000 | +100% |

The 62.9% reduction in Brier from V1 to V2 came from: KenPom/Barttorvik external data (~15–20%), isotonic calibration correcting the Reliability term (~8%), and the stacking meta-learner (~5%).

---

## Architecture

```
Data (9 sources)
    │
    ├── KenPom         (AdjEM, AdjO, AdjD, Tempo, Luck — paid)
    ├── Barttorvik     (Barthag, AdjO, AdjD, AdjT — free scrape)
    ├── Vegas odds     (closing-line win probabilities — manual CSV)
    ├── Massey Ordinals (median of 100+ computer ranking systems)
    ├── Kaggle files   (game results, seeds, schedules)
    ├── Elo ratings    (margin-adjusted, home-corrected, mean-reverting)
    ├── Bradley–Terry  (MLE pairwise strength, MM algorithm)
    ├── Conference tournament (hot-streak signal entering March)
    └── Coach continuity (3-level tenure encoding)
          │
          ▼
Feature Engineering — 78 features across 11 groups
    │
    ├── Elo (current + trend)      4 features  — top: diff_Elo, r=0.643
    ├── Season stats (time-decay) 18 features  — time-decay weight = (DayNum/132)^1.5
    ├── Advanced box score        14 features  — TrueShooting, AstTORatio, etc.
    ├── Tempo-free ratings         9 features  — ORtg/DRtg/NetRtg per 100 possessions
    ├── Bradley–Terry              4 features  — bt_logratio, r=0.453
    ├── Seeds                      5 features  — diff_Seed, r=0.459
    ├── Massey Ordinals            2 features  — diff_Massey, r=0.413
    ├── KenPom                     6 features  — kp_em_diff, r=0.377
    ├── Barttorvik                 5 features  — tv_net_diff, r=0.341
    ├── Vegas odds                 3 features  — vegas_logit_diff, r=0.298
    └── Conference + Coaches       8 features  — diff_ConfStrength, r=0.375
          │
          ▼
Bayesian HPO — Optuna TPE, 110 trials total
    ├── XGBoost:  60 trials → best 0.0977 (depth=4, lr=0.041, n_est=838)
    └── CatBoost: 50 trials → best 0.0976 (depth=6, lr=0.030, iters=805)
          │
          ▼
Stacked Ensemble (Leave-One-Season-Out CV, 8 folds, 2018–2025)
    ├── XGBoost          → meta-weight 59.2%
    ├── CatBoost         → meta-weight 35.6%
    └── Logistic Regr.   → meta-weight  5.1%
          │
          ▼
Meta-Learner (Logistic Regression on OOF stack)
          │
          ▼
Calibration — 3-way competition, isotonic won both genders
    ├── Isotonic regression    ← WINNER (Men: 0.0999→0.0918, Women: 0.0744→0.0676)
    ├── Beta calibration (MLE)
    └── Bayesian Beta-prior smoothing
          │
          ▼
submission_final_combined_2026.csv
(132,133 matchups — Stage 1 + Stage 2)
```

---

## Feature Engineering Details

### Elo Ratings

Standard Elo with three modifications that matter specifically for college basketball:

1. **Margin of victory multiplier** — $K_{\text{eff}} = K \cdot (1 + 0.04 \cdot \Delta)$, capped at $3K$. Winning by 30 is more informative than winning by 1.

2. **Home court adjustment** — 100-point bonus in the expected score calculation. Without this, home wins inflate ratings in a way that makes road wins look more impressive than they are.

3. **Mean reversion between seasons** — $r_t^{(s)} = 0.70 \cdot r_{\text{end}}^{(s-1)} + 0.30 \times 1500$. Models roster turnover. A dominant team three years ago with a completely different roster shouldn't carry that Elo forward.

Elo was computed through the entire 2026 regular season for maximum freshness. `elo_win_prob` was the single highest-correlation feature at r = 0.669.

### Time-Decay Weights

$$w(\text{DayNum}) = \left(\frac{\text{DayNum}}{132}\right)^{1.5}$$

Minimum weight 0.05. A late-February win carries roughly 4× the weight of a mid-November win. Late-season form better reflects the team showing up in March.

### Tempo-Free Statistics

Possessions estimated via the Dean Oliver formula:

$$\widehat{\text{Poss}} \approx \text{FGA} - \text{OReb} + \text{TO} + 0.44 \times \text{FTA}$$

Then efficiency per 100 possessions: $\text{ORtg} = 100 \times \text{Points} / \widehat{\text{Poss}}$. A team scoring 75 in 60 possessions is meaningfully better than one scoring 75 in 78 possessions. `diff_NetRtg` reached r = 0.388.

### Bradley–Terry Ratings

Unlike Elo (sequential updates), Bradley–Terry fits all games simultaneously via the MM algorithm:

$$\hat{s}_i^{(t+1)} = \frac{w_i + \lambda}{\sum_{j \neq i} \frac{n_{ij}+n_{ji}}{\hat{s}_i^{(t)}+\hat{s}_j^{(t)}} + \lambda}$$

The log-ratio `bt_logratio` achieved r = 0.453, second only to Elo-derived features.

### Season Exponential Decay

$$\bar{x}_A^{(S)} = \frac{4\,x_A^{(S-1)} + 2\,x_A^{(S-2)} + 1\,x_A^{(S-3)}}{7}$$

A 4:2:1 weighting across three prior seasons. Last season is four times as informative as three seasons ago.

---

## Calibration Details

Even a model with strong discrimination can produce badly calibrated probabilities. Three methods competed on out-of-fold Brier score; the winner was applied to final predictions.

**Isotonic regression** — fits a non-decreasing step function via the pool-adjacent-violators algorithm. Won for both Men's and Women's. Reduced Men's OOF Brier from 0.0999 → **0.0918** and Women's from 0.0744 → **0.0676**.

**Beta calibration** — fits $\text{logit}(p_{\text{cal}}) = a\log(p) - b\log(1-p) + c$ by MLE. More stable than isotonic on small samples; subsumes Platt scaling.

**Bayesian Beta-prior smoothing** — posterior mean under a Beta$(\alpha, \beta)$ prior:

$$p_{\text{cal}} = \frac{\alpha + p_{\text{model}} \cdot n_e}{\alpha + \beta + n_e} = \lambda \cdot p_{\text{model}} + (1-\lambda)\cdot\mu_0$$

The shrinkage weight $\lambda$ governs how much you trust the model versus the prior mean. Hyperparameters $(\alpha, \beta, n_e)$ are fitted by directly minimising OOF Brier.

---

## Top 15 Features

| Rank | Feature | XGB Importance | Correlation |
|------|---------|---------------|-------------|
| 1 | `elo_win_prob` | 0.118 | 0.669 |
| 2 | `diff_Elo` | 0.097 | 0.643 |
| 3 | `diff_EloWinProb` | 0.088 | 0.620 |
| 4 | `diff_Seed` | 0.079 | 0.459 |
| 5 | `diff_BT_log` | 0.071 | 0.453 |
| 6 | `bt_logratio` | 0.068 | 0.453 |
| 7 | `diff_Massey` | 0.062 | 0.413 |
| 8 | `diff_Elo_prev` | 0.057 | 0.392 |
| 9 | `diff_NetRtg` | 0.051 | 0.388 |
| 10 | `kp_em_diff` | 0.048 | 0.377 |
| 11 | `diff_ConfStrength` | 0.043 | 0.375 |
| 12 | `diff_KP_AdjEM` | 0.041 | 0.376 |
| 13 | `seed_A` | 0.038 | 0.368 |
| 14 | `diff_KP_AdjO` | 0.035 | 0.359 |
| 15 | `diff_AvgPtsDiff` | 0.032 | 0.342 |

Elo-derived features occupy the top three slots. KenPom AdjEM first appears at rank 10 but contributes across three distinct features (10, 12, 14), confirming the aggregate value of external efficiency ratings.

---

## Cross-Validation Results (V2)

| Model | Men's Brier | Men's vs. Base | Women's Brier | Women's vs. Base |
|-------|------------|----------------|--------------|-----------------|
| Logistic Regression | 0.1108 | +55.7% | 0.0813 | +67.5% |
| XGBoost | 0.1025 | +59.0% | 0.0802 | +67.9% |
| CatBoost | 0.1071 | +57.1% | 0.0767 | +69.3% |
| Meta-learner (raw) | 0.0999 | +60.1% | 0.0744 | +70.2% |
| **After isotonic calibration** | **0.0918** | **+63.3%** | **0.0676** | **+72.9%** |
| Naive baseline | 0.2500 | — | 0.2500 | — |

Stage 1 historical validation on 536 actual tournament games (2021–2025): **Brier = 0.03037, BSS = 0.8785, prediction mean = 0.5098, prediction std = 0.4098**.

---

## The Leaderboard Gap — An Honest Discussion

The Stage 1 historical score (0.03037) and the live leaderboard score (0.1698) measure genuinely different things, and the gap is expected.

Stage 1 evaluated the model on 2021–2025 games where the feature pipeline had access to rich multi-year prior data (KenPom, Barttorvik, and Massey going back to 2016). The 2026 predictions relied on a single completed regular season. More fundamentally, the model's calibration and hyperparameters were selected to minimise Brier on 2018–2025 — evaluating on part of that period is optimistic by construction.

A BSS of 0.321 on actual 2026 tournament games is a genuine result in the true sense — out-of-distribution, scored on games that had not yet been played. The remaining gap to a perfect score reflects the irreducible uncertainty of tournament basketball. No model consistently predicts which 12-seed beats which 5-seed.

---

## Key Lessons

1. **Calibration dominates on small datasets.** Fixing the Reliability term improved the Brier score more than any single model change.

2. **External data beats model complexity.** KenPom contributed an estimated 15–20% improvement — more than adding four extra models.

3. **Fewer models is better with 602 samples.** Going from 5 to 3 models eliminated correlated noise and directly improved the Reliability component.

4. **Temporal CV is not optional.** Standard K-fold would have mixed same-season features across folds, inflating CV scores and producing incorrect model selection.

5. **Feature engineering beats hyperparameter tuning.** Time-decay weights, Bradley–Terry, conference strength, and coach continuity contributed signal that no amount of Optuna trials could conjure from nothing.

---

## Tools

- **Language:** Python 3.12
- **Environment:** Google Colab (free tier), with JavaScript anti-timeout keep-alive every 60 seconds
- **Key libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `optuna`, `scipy`, `kenpompy`, `cloudscraper`, `beautifulsoup4`, `matplotlib`, `seaborn`

---

## Running the Pipeline

1. Open `march_mania_final.ipynb` in Google Colab.
2. Run Cell 1 immediately (anti-timeout).
3. Install libraries (Cell 2) then import (Cell 3).
4. Set your KenPom credentials in Cell 4, or set `USE_KENPOM = False` to skip.
5. Upload all Kaggle competition CSVs plus `vegas_odds_2026.csv` in Cell 5.
6. Run all cells sequentially. Total runtime: ~30–40 minutes on free Colab.
7. Download `submission_final_combined_2026.csv` from Cell 18 output.
8. Upload to Kaggle and manually select as active submission.

### Required Kaggle Files

```
MRegularSeasonCompactResults.csv      WRegularSeasonCompactResults.csv
MRegularSeasonDetailedResults.csv     WRegularSeasonDetailedResults.csv
MNCAATourneyCompactResults.csv        WNCAATourneyCompactResults.csv
MNCAATourneyDetailedResults.csv       WNCAATourneyDetailedResults.csv
MNCAATourneySeeds.csv                 WNCAATourneySeeds.csv
MMasseyOrdinals.csv                   MConferenceTourneyGames.csv
MTeamCoaches.csv                      MTeamConferences.csv
WTeamConferences.csv                  MTeamSpellings.csv
SampleSubmissionStage1.csv            SampleSubmissionStage2.csv
```

Plus `kenpom_data.csv` (if manually downloaded) and `vegas_odds_2026.csv`.

---

## Repository Structure

```
march_mania_final.ipynb        # Main pipeline notebook (18 cells, fully documented)
march_mania_report.tex         # Technical report (LaTeX source, Overleaf-ready)
README.md                      # This file
```

---

## Author

**Andy Minga**
M.S. Applied Statistics, Illinois State University

---

## License

MIT. If this pipeline is useful as a starting point, a reference back here is appreciated.
