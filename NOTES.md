# YMS Financial Assignment — Solution Notes

## 1. Problem Formulation

Predict **before market open (09:15 IST)** on trading day $D$ whether each Nifty-50 stock will trade more than 2% away from its previous close (`pclose`) at any point during day $D$:

$$\text{moved} = \frac{\max(\text{high} - \text{pclose},\; \text{pclose} - \text{low})}{\text{pclose}} > 0.02$$

The official evaluation optimizes a precision-weighted harmonic mean (75% weight on precision, 25% on recall) with a hard pass condition of $\text{Precision} > 0.60$:

$$\text{Score} = \sqrt{\frac{1}{\frac{0.75}{\text{Precision}} + \frac{0.25}{\text{Recall}}}}$$

Because the test period is substantially calmer than the training set (34.4% positive base rate vs. 41.2% overall training base rate), a threshold tuned naively for 60–63% precision on training dates drops below 60% on test dates (e.g., the template `predict.py` achieves 62.7% on train but fails at 53.3% on test). Therefore, our design focuses on:
1. Strong out-of-sample ranking (`ROC-AUC` / `PR-AUC`) across regimes without overfitting to early-period high volatility.
2. Continuous volatility modeling (`log(max_move_pct)`) combined with calibrated classification probabilities.
3. Conservative, regime-stress-tested threshold selection preserving a comfortable safety margin above 60% precision under calmer 32–37% base-rate regimes.

---

## 2. Dataset & EDA Findings

### Dataset Structure (`data/`)
- **Training set**: 97 trading days (`2026-03-04` to `2026-10-30`), 50 stocks per day = 4,850 rows, 1,996 positives (**41.15%**).
- **Masked vs. unmasked training files**: 29 of the 97 training dates in `data/historical_info/` have `pclose`, `phi`, `plo`, and all `vix*` columns blanked out because the preceding trading day was a test date; only `pvolume` and `mddv_20` remain populated on those 29 dates. The remaining **68 unmasked training dates** (3,400 rows, **42.94%** positive rate) have complete pre-open columns, except `phi`/`plo` are blank on 8 dates (400 rows, 11.76%) due to incomplete previous-day OHLC capture, and `vix` is missing on 18 individual `(date, name)` rows (0.53%). During official grading (`evaluator/data/info_all.csv`), all dates $\le D$ are unmasked.
- **Temporal regime shift**: Monthly positive rates vary dramatically:
  - Mar 2026: **67.1%**, Apr 2026: **58.0%**, May 2026: **46.8%**
  - Jun 2026: **31.6%**, Jul 2026: **29.1%**, Aug 2026: **18.3%**
  - Sep 2026: **31.5%**, Oct 2026: **46.0%**
- **Cross-stock heterogeneity**: `SHRIRAMFIN` (69.1%), `ETERNAL` (64.9%), and `INDIGO` (58.8%) move >2% more than 3× as often as `ITC` (17.5%), `SUNPHARMA` (20.6%), and `CIPLA` (23.7%). Crucially, stocks also differ systematically in their realized intraday excursion relative to option-implied daily SD ($\text{max\_move\_pct} / (\text{vix}/\sqrt{252})$ ranges from ~1.03 for `WIPRO` to ~1.49 for `TITAN` and ~1.44 for `SHRIRAMFIN` / `BAJFINANCE`).
- **Univariate correlations with `moved` (unmasked rows)**:
  - `daily_vol` (`vix / (100 * sqrt(252))`): `0.350`
  - `vix_zscore_raw` / `vix_zscore_smooth`: `0.327`
  - `vix_1m`: `0.288`, `vix_3m`: `0.251`, `vix_3m_ratio`: `0.241`
  - `prev_range` (`(phi - plo) / pclose`): `0.241`
  - `log_pvolume`: `0.121`, `vol_ratio` (`pvolume / mddv_20`): `0.055`

---

## 3. Feature Engineering & No-Look-Ahead Guarantees

Every feature is constructed strictly from information public before **09:15 IST on day $D$**:

### A. Provided Info & Cross-Sectional Features (Day $D$ pre-open)
1. `log_daily_vol`: $\log(\max(\text{vix} / (100\sqrt{252}), 10^{-5}))$ — log-scale option-implied daily SD, matching the log scale of `max_move_pct`.
2. `vix_zscore_smooth`: Stock's implied volatility relative to the ~224-stock F&O universe.
3. `vix_vs_1m`, `vix_vs_3m`: Short- and medium-term VIX term-structure / spike ratios ($\text{vix}/\text{vix\_1m}$, $\text{vix}/\text{vix\_3m}$).
4. `vix_rel_mkt`: Stock's `vix` divided by the same-day pre-open cross-sectional mean `vix` across the 50 Nifty-50 stocks.
5. `log_prev_range`: $\log(\max(\text{prev\_range\_filled}, 10^{-5}))$, where missing `prev_range` (`phi`/`plo` blank) is first imputed with the stock's causal 14-day realized range `hist_range_14d` before falling back to the training median.
6. `log_vol_ratio`: $\log(\max(\text{pvolume} / \text{mddv\_20}, 10^{-6}))$ — volume surge relative to 20-day median traded value.

### B. Strictly Causal Historical Stock Features (Dates $< D$ only)
Computed dynamically inside `predict.py` using only `<date>_n50.info.csv` files with `file_date < D`:
7. `hist_range_14d`: Mean of `(phi - plo) / pclose` over available unmasked days within the past 14 calendar days ($< D$; falls back to expanding mean $< D$ if empty).
8. `shrunk_ratio`: Bayesian-shrunk stock-specific intraday range to implied-vol multiplier:
   $$\text{shrunk\_ratio}_{i,D} = \frac{\sum_{t < D} \frac{\text{prev\_range}_{i,t}}{\text{daily\_vol}_{i,t}} + 8 \times 1.28}{N_{i,<D} + 8}$$

### C. Pre-Market External Features (`prepare_user_historical_info.py` → `user_historical_info.csv`)
Fetched once from Yahoo Finance daily bars and aligned strictly using **`date_ist < D`** (Indian market closes at 15:30 IST on a trading day strictly before $D$, ~17.75 hours prior to day $D$'s 09:15 IST open):
9. `indiavix`, `log_indiavix`: Previous trading day's closing India VIX (`^INDIAVIX`).
10. `nsei_rv5d`: Previous trading day's 5-day annualized close-to-close realized volatility (%) of Nifty 50 (`^NSEI`).

For the 29 masked training dates (or rare rows missing `vix`), a 7-feature **fallback model** uses only always-available signals: `log(hist_iv_14d)`, `hist_range_14d`, `shrunk_ratio`, `log_vol_ratio`, `log(mddv_20)`, `log_indiavix`, and `nsei_rv5d`.

---

## 4. Temporal Validation Methodology

To prevent temporal leakage and verify robustness across regimes without touching `evaluator/data/` test labels:
1. **Expanding-Window Walk-Forward (`wf_28`)**: Minimum initial training window of 28 unmasked dates (`2026-03-04` to `2026-05-29`), stepping forward in 5-date blocks across the remaining 40 unmasked dates (`2,000` OOS rows, calm base rate `30.8%`). (We also inspected `wf_18`, 50 OOS dates, base rate `33.1%`, where early March-only training folds suffer cold-start base-rate drift.)
2. **Contiguous Blocked 5-Fold CV (`blocked5`)**: Splits all 68 unmasked dates (`3,400` rows, base rate `42.9%`) into 5 contiguous chronological blocks so both high-vol (Mar–Apr) and low-vol (Jun–Aug) regimes are evaluated strictly out-of-sample without interleaved date proximity.
3. **Interleaved Date-Grouped 5-Fold CV (`interleaved5`)**: Assigns every 5th chronological date to each fold (`3,400` OOS rows), matching the assignment's random-date train/test split structure while keeping all historical stock features strictly causal ($< D$).
4. **Calm-Regime Bootstrap Stress Test**: Resamples 1,000 subsets of 41 dates from `blocked5` OOF predictions conditioned on calm base rates $\in [0.32, 0.37]$ (matching the ~34.4% test base rate reported in `README.md`) to measure median Precision, 10th-percentile Precision, and empirical pass rate ($\text{Precision} > 0.60$).

---

## 5. Model Experimentation & Comparison

On the curated 11-feature set (`F_R11`), all preprocessing (`SimpleImputer(median)` + `StandardScaler`) was fit strictly inside each training fold. Continuous targets (`log(max_move_pct)`) consistently outperformed binary-only classifiers because continuous move magnitude preserves information near the 2% boundary, and regularized linear/Huber models outperformed tree ensembles on this 68-date panel:

| Model (on `F_R11`) | `wf_28` AUC / AP | `blocked5` AUC / AP | `interleaved5` AUC / AP |
|---|---:|---:|---:|
| `HuberRegressor` (`log_move`, $\alpha=80$) | 0.6903 / 0.4853 | **0.7313** / **0.6685** | **0.7384** / **0.6753** |
| `Ridge` (`log_move`, $\alpha=150$) | 0.7001 / 0.4941 | 0.7291 / 0.6624 | 0.7382 / 0.6734 |
| `LogisticRegression` ($C=0.02$) | 0.7013 / 0.5030 | 0.7295 / 0.6656 | 0.7366 / 0.6727 |
| `ExtraTreesClassifier` (depth=5) | 0.6685 / 0.4596 | 0.7106 / 0.6239 | 0.7284 / 0.6495 |
| `RandomForestClassifier` (depth=4) | 0.6800 / 0.4735 | 0.7054 / 0.6226 | 0.7211 / 0.6437 |
| `HistGradientBoostingClassifier` (depth=2) | 0.6716 / 0.4786 | 0.7016 / 0.5994 | 0.7193 / 0.6262 |
| **`ens_lin_dv` (Final Ensemble)** | **0.7031** / **0.4986** | 0.7299 / 0.6671 | 0.7377 / 0.6738 |

### Final Model Architecture (`ens_lin_dv`)
Continuous `log(max_move_pct)` predictions from `HuberRegressor` and `Ridge` are mapped to probability space via $\Phi\left(\frac{\hat{y} - \log(2.0)}{0.48}\right)$ and combined with `LogisticRegression` and a direct `daily_vol` Gaussian anchor $\Phi\left(\frac{\text{daily\_vol} - 0.017}{0.004}\right)$:

$$p = 0.45\,p_{\text{Huber}} + 0.20\,p_{\text{Ridge}} + 0.20\,p_{\text{LogReg}} + 0.15\,p_{\text{daily\_vol}}$$

Blending the `daily_vol` anchor raised `wf_28` OOS AUC from `0.6903` to `0.7031` (and AP from `0.4853` to `0.4986`) by stabilizing forward regime extrapolation. All learned parameters are exported to `model_params.json`, so `predict.py` executes with zero external library dependencies at inference time (~20 ms per date).

---

## 6. Threshold Selection & Validation Results

Threshold sweep on `ens_lin_dv` across `blocked5` (42.9% base rate), `interleaved5`, `wf_28` (30.8% base rate), and 1,000 calm bootstrap draws (41 dates, base rate $\in [0.32, 0.37]$):

| Threshold | `blocked5` P / R / Score ($N$) | `interleaved5` P / R / Score | `wf_28` P / R ($N$) | Calm Med P | Calm p10 P | Calm Pass % | Calm Med Score |
|---:|---|---|---|---:|---:|---:|---:|
| 0.560 | 0.663 / 0.445 / 0.7684 (981) | — | — | 0.602 | 0.549 | 52.6% | 0.7240 |
| 0.600 | 0.694 / 0.382 / 0.7592 (804) | 0.706 / 0.383 / 0.7635 | 0.617 / 0.149 (149) | 0.633 | 0.574 | 75.6% | 0.7093 |
| 0.620 | 0.712 / 0.344 / 0.7495 (705) | 0.715 / 0.349 / 0.7525 | 0.623 / 0.107 (106) | 0.649 | 0.585 | 83.9% | 0.6944 |
| **0.640** | **0.724 / 0.318 / 0.7407 (641)** | **0.726 / 0.322 / 0.7435** | **0.616 / 0.086 (86)** | **0.665** | **0.600** | **89.9%** | **0.6865** |
| 0.650 | 0.734 / 0.308 / 0.7382 (612) | 0.731 / 0.304 / 0.7357 | 0.595 / 0.071 (74) | 0.676 | 0.609 | 92.5% | 0.6828 |

We selected **`threshold = 0.640`**, where even the 10th percentile of calm-regime bootstrap draws hits `0.600` precision (~90% pass probability) while retaining strong recall (`0.318` on `blocked5`, `Score = 0.7407`).

---

## 7. Official Grader Results

Executed after freezing all features, weights, and threshold (`0.640`) without using test labels for any tuning:

- **Practice evaluation (`python3 evaluator/run_evaluation.py .`, 97 training dates, 4,850 rows, 0 missing, 0 failed)**:
  - TP = 583, FP = 205, TN = 2649, FN = 1413
  - **Precision**: `0.7398` | **Recall**: `0.2921` | **Score**: `0.7313` | **Result**: **PASS**
- **Official test evaluation (`python3 evaluator/run_evaluation.py . --data evaluator/data`, 41 test dates, 2,050 rows, 0 missing, 0 failed)**:
  - TP = 173, FP = 99, TN = 1245, FN = 533 (272 predicted positives)
  - **Precision**: `0.6360` | **Recall**: `0.2450` | **Score**: `0.6743` | **Result**: **PASS** (beats all baselines in `evaluator/baselines.py`)

---

## 8. Failed Experiments & Key Observations

1. **46-feature kitchen sink**: Training on all 46 engineered features dropped `LogReg(C=0.1)` walk-forward AUC to `0.5946` due to multicollinearity and date-level overfitting with only 68 distinct unmasked dates.
2. **US VIX (`^VIX`) and overnight US returns (`^GSPC`, `^IXIC`)**: `us_vix` improved `interleaved5` AUC to `0.7400` because of slow trend interpolation across adjacent dates, but **degraded** `wf_28` (`0.6982 -> 0.6776`) and `blocked5` (`0.7242 -> 0.7128`). Dropped as spurious macro overfitting.
3. **Tree ensembles (`RF`, `ET`, `HistGB`, `GB`)**: Consistently lagged linear/Huber models by ~0.025–0.035 AUC on `blocked5` (`0.702–0.711` vs `0.731`) and `wf_28` (`0.672–0.680` vs `0.703`) due to greedy axis-aligned splits on noisy volatility tails in a small 68-date sample.
4. **Historical `moved` frequency (`hist_moved_shrunk`)**: Adding binary historical movement rate slightly reduced `blocked5` AUC (`0.7333 -> 0.7329`) because `shrunk_ratio` and `hist_range_14d` already capture the continuous signal without 0/1 quantization loss.
5. **Stock-specific thresholds & Platt recalibration**: Adjusting scores by stock residual `shrunk_ratio` produced zero AUC gain (`0.7286 == 0.7286`), and fitting an extra Platt logistic layer on `Huber` outputs slightly hurt `blocked5` Brier (`0.2058 -> 0.2063`).
6. **Naive threshold selection (`0.50–0.56`)**: Achieved `0.768–0.781` score on `blocked5` (42.9% base rate), but failed 47–83% of calm-regime bootstrap draws (`base_rate in [0.32, 0.37]`).

---

## 9. How to Reproduce

```bash
pip install -r requirements.txt
python prepare_user_historical_info.py --dates data/all_dates.txt --names data/n50_names.txt --out user_historical_info.csv
python train_model.py
python evaluator/run_evaluation.py .
python evaluator/run_evaluation.py . --data evaluator/data
```
