# Forecasting Oil Prices Using Time Series Models

This repo contains my end-to-end project on forecasting daily oil prices using a mix of classical time series models, a neural model, and a hybrid mixture model called **Kenshi**.

Oil prices are a good stress test for forecasting because the series has sharp swings, disruptions, and regime changes. A model that looks fine during a calm period can fall behind quickly when volatility shifts. For that reason, the project is built like an engineering workflow: establish strong baselines, evaluate them properly with chronological splits, then build a hybrid that can adapt when the relative quality of the baselines changes.

## What’s in this project

### Models
- **ARMA/ARIMA baseline (from scratch)**  
  Implemented with explicit ARMA recursion and conditional sum-of-squares (CSS) fitting. Includes grid search over (p,d,q), AIC-based selection, and rolling one-step evaluation.

- **Delta-LSTM**  
  A neural model trained to predict next-day **price differences** instead of raw levels to reduce drift and keep rollouts stable.

- **Kenshi (Hybrid Mixture)**  
  A dynamic mixture of a **returns-based ARIMA expert** and the **Delta-LSTM expert**. The mixture weights update over time based on recent squared errors (EWMA memory + softmax gate), with a small bias correction term.  
  Named after Kenshi Yonezu.

## Repository structure

- `CS1_Notebook_1_EDA.ipynb`  
  Exploratory analysis + diagnostics (stationarity checks, ACF/PACF, volatility behaviour).

- `CS1_Notebook_2_ARMA.ipynb`  
  ARMA/ARIMA from scratch:
  - differencing + inversion helpers
  - ARMA recursion filter
  - CSS fitting (L-BFGS-B)
  - (p,d,q) grid search using AIC
  - rolling 1-step test evaluation
  - Monte Carlo simulation for forecast intervals

- `CS1_Notebook_3_Delta_LSTM.ipynb`  
  Delta-LSTM pipeline:
  - delta construction + scaling
  - sequence building (lookback = 30)
  - training loop + gradient clipping
  - rolling 1-step evaluation
  - 24-month rollout with residual bootstrap

- `CS_1_Notebook_4_.ipynb`  
  Kenshi hybrid:
  - ARIMA expert on log returns (AIC selection)
  - pretrained Delta-LSTM expert
  - Kenshi gate + bias correction
  - validation grid search for Kenshi hyperparameters
  - rolling 1-step test comparison
  - 24-month Monte Carlo hybrid forecast

- `Final_Comparison.ipynb`  
  Final comparison and visuals:
  - leaderboard table
  - winner highlighting
  - plots (forecast overlay, absolute error, cumulative error)

## Evaluation setup

All models use rolling one-step-ahead evaluation on a held-out test window: each day’s forecast uses only the history available up to the previous day. This is closer to how the model would work in practice compared to random splits.

## Outputs

The notebooks save:
- forecast CSVs (including 24-month quantiles for the hybrid)
- metrics/config pickles (best params, RMSE results, final expert weights)

Paths may need adjusting depending on whether you run locally or in Colab.

## How to run

1. Open the notebooks in order:
   1) EDA
   2) ARMA/ARIMA baseline
   3) Delta-LSTM
   4) Kenshi hybrid
   5) Final comparison

2. Update paths for:
   - the dataset CSV
   - the pretrained `lstm_delta.pt` (if needed)

3. Run all cells.

## Notes on the 24-month forecast

The long-horizon forecast is generated as a scenario distribution using Monte Carlo simulation and bootstrapped noise, then summarized using median and 95% intervals. It’s best read as a range of plausible outcomes under the model assumptions, not a confident long-term call.

## Author

Keerthivasan Kannan
