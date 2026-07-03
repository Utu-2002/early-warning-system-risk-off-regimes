# Early Warning System for Risk-Off Market Regimes

ML-based Early Warning System predicting risk-off market regimes one week ahead using cross-asset signals.

**Authors:** Axel Benedetti, Michele Schettino, Utku Apaydın, Valfredo De Lorenzo, Giorgio Piccinini, Luca Piermarini

## Overview

This project builds an Early Warning System (EWS) that predicts whether financial markets will enter a "risk-off" regime one week in advance. Using 1,109 weekly cross-asset observations (Jan 2000 – Apr 2021) spanning 42 indicators across 7 asset classes (equities, bonds, rates/yields, FX, commodities, volatility, macro), four machine learning models were trained and benchmarked on a strict chronological train/validation/test split to avoid look-ahead bias.

## Key Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Autoencoder (best)** | **0.754** | **0.611** | 0.379 | **0.468** |
| Isolation Forest | 0.743 | 0.667 | 0.276 | 0.390 |
| MVG | 0.743 | 0.152 | 0.897 | 0.260 |
| SVM | 0.680 | 0.210 | 0.586 | 0.309 |

- All four models outperform the random AUC baseline, confirming cross-asset signals carry forward-looking information about regime transitions.
- **Bond indices** show the strongest pre-crisis signal elevation (1.77×), ahead of equities (1.60×) and even VIX (ranked 5th).
- The Autoencoder correctly flags **March 2020 (COVID-19 shock)** as the largest anomaly in the test period.

## Methodology

- **Data split:** chronological, no shuffling — Train (2000–2012, 665 obs), Validation (2012–2017, 222 obs), Test (2017–2021, 222 obs)
- **Models:** Multivariate Gaussian (anomaly score via log-likelihood), SVM (RBF kernel, Platt scaling — the only supervised model), Autoencoder (42→24→8→24→42, ReLU, Adam), Isolation Forest (300 trees)
- **Feature engineering:** log-returns for price indices, first differences for rates/yields, stationarity confirmed via ADF tests

## Tech Stack

Python · NumPy · pandas · scikit-learn · PyTorch · statsmodels · matplotlib / seaborn

## Files

- `EWS_-_Code.ipynb` — full analysis notebook (data prep, feature engineering, model training, evaluation, visualizations)

## Next Steps

Gradient boosting ensemble, multi-horizon targets (t+2, t+4 weeks), SHAP explainability, SMOTE / cost-sensitive training for class imbalance.
