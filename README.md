# GHI Forecasting Project

This project builds and compares a set of solar irradiance forecasting models for an NSRDB weather site. It started as a baseline forecasting pipeline, then expanded into an attention model, and finally into a residual hybrid system that learns to correct the baseline model's errors.

The main goal is to forecast hourly Global Horizontal Irradiance (GHI) and produce practical evaluation plots for 30-day horizons, 1-day case studies, and weather-regime breakdowns.

## What This Project Does

The repository ingests hourly NSRDB CSV files, builds time-series features, trains several models, evaluates them on a chronological holdout split, and saves reports and plots.

In plain terms, the project answers these questions:

1. Can we forecast GHI from past weather and solar features?
2. Does an attention model improve the baseline LSTM?
3. Can a residual model correct the baseline's systematic errors?
4. Which model is best on sunny, cloudy, and mixed-irradiance periods?

## Dataset

The current dataset folder contains hourly NSRDB CSV files for a single site:

- Location ID: 820809
- Latitude: 34.12
- Longitude: -118.39
- Time zone: UTC-8
- Years included: 2018 to 2021

The hourly rows include GHI plus meteorological features such as temperature, pressure, wind speed, and relative humidity. The project also derives solar geometry and time-based features from the timestamps.

## Technologies Used

This project is built with:

- Python
- TensorFlow / Keras for the neural forecasting models
- LightGBM for the residual correction model and regime classifier
- scikit-learn for scaling, metrics, and time-series validation utilities
- pandas and NumPy for data loading and feature engineering
- Matplotlib for all plots and comparison charts
- JSON and CSV files for reports and saved evaluation outputs

## Project Structure

Key files and folders:

```text
code/
  dataset/                        NSRDB CSV files
  models/
    baseline_lstm/                Baseline forecasting pipeline
    attention_lstm/               Attention model pipeline
    residual_hybrid/              Residual hybrid pipeline
    analysis/                     Correlation heatmap workflow
    testing/                      Basic test runner
    agent/                        Sprint automation and training orchestration
  outputs/
    artifacts/                    Saved .h5, .txt, and .joblib artifacts
    plots/                        Saved PNG charts
    reports/                      CSV and JSON summaries
  compare_model_forecasts.py      Multi-model comparison script
  generate_sprint_plots.py        Reporting and sprint plot generation
  lstm_model.py                   Root compatibility entry point for baseline
  attention_lstm_model.py         Root compatibility entry point for attention
  residual_hybrid_model.py        Root compatibility entry point for hybrid
```

## End-to-End Workflow

The project is organized as a full time-series forecasting pipeline:

1. Load and merge the NSRDB CSV files.
2. Sort the data chronologically.
3. Build time and solar features.
4. Create lag and rolling-window features from previous hours.
5. Split the data chronologically into train, validation, and test sets.
6. Train the baseline LSTM.
7. Train the attention LSTM.
8. Train the residual hybrid model on the baseline's residuals.
9. Evaluate all models on the same held-out test window.
10. Save plots, metrics tables, and comparison reports.

## Feature Engineering

The feature table includes:

- Raw GHI
- Temperature, relative humidity, wind speed, pressure
- Solar zenith angle and cosine zenith
- Daylight flag
- Hour and day-of-year cyclical encodings
- Lag features such as `ghi_lag_1`, `ghi_lag_2`, `ghi_lag_3`, `ghi_lag_24`
- Change features such as `ghi_diff_1` and `ghi_diff_3`
- Rolling statistics such as mean and standard deviation over 3, 6, 12, and 24 hours

These features let the models learn both short-term persistence and longer seasonal or regime patterns.

## Models

### 1. Baseline LSTM

The baseline model is the main direct forecasting network. It uses a sequence of past hours to predict the next GHI value.

Why it matters:

- It is the simplest strong reference model in the project.
- It gives a stable benchmark for later experiments.
- It already uses train-only scaling and chronological splits.

### 2. Attention LSTM

The attention model was added to see whether the network could focus on the most useful timesteps inside the historical window.

Why it matters:

- It tests whether the model can learn which recent hours are most informative.
- It is useful as an experimental comparison point.
- It did not beat the baseline on the saved holdout comparison, which is still an important result.

### 3. Residual Hybrid

The residual hybrid model is the strongest model in the current project.

It works like this:

1. A baseline LSTM produces an initial forecast.
2. A second model learns the residual error: `actual_ghi - baseline_pred`.
3. The residual prediction is added back to the baseline forecast.
4. The final value is clipped to remain physically valid.

Why it matters:

- It corrects systematic bias instead of trying to relearn GHI from scratch.
- It uses baseline prediction context, residual lags, clear-sky features, and weather-regime probabilities.
- It produced the best current holdout performance by a large margin.

## Model Comparison

The following metrics come from the saved comparison run on the current held-out test window.

| Model           |  RMSE |   MAE |     R2 | Day MAE | Peak MAE |
| --------------- | ----: | ----: | -----: | ------: | -------: |
| Baseline LSTM   | 53.92 | 22.68 | 0.9723 |   40.56 |    29.05 |
| Attention LSTM  | 83.31 | 33.53 | 0.9338 |   62.61 |    48.75 |
| Residual Hybrid |  8.34 |  3.36 | 0.9993 |    6.53 |     5.58 |

### What Improved

The residual hybrid is the biggest improvement in the project.

Compared with the baseline LSTM, the residual hybrid reduced:

- RMSE by 84.52%
- MAE by 85.18%
- Day MAE by 83.89%
- Peak MAE by 80.78%

Compared with the baseline, the attention model did not improve this final holdout. In the saved comparison it was worse on every core metric, which makes the residual hybrid even more important as the final winning approach.

### Regime View

The comparison script also breaks errors down by weather regime.

In the current reporting flow, regimes are derived from clear-sky index and daylight conditions:

- Sunny
- Partly cloudy
- Cloudy
- Night

Rain is not directly labeled in the hourly NSRDB rows used here, so cloudy and low-irradiance conditions are the closest practical proxy.

## Reporting and Plots

The repo saves several report files and visualizations:

- `outputs/reports/FINAL_SPRINT_REPORT.json`
- `outputs/reports/residual_hybrid_report.json`
- `outputs/reports/model_forecast_comparison_metrics.csv`
- `outputs/reports/model_forecast_comparison_regime_metrics.csv`
- `outputs/reports/model_forecast_comparison_predictions.csv`
- `outputs/plots/model_forecast_comparison_30_days.png`
- `outputs/plots/model_forecast_day_sunny.png`
- `outputs/plots/model_forecast_day_cloudy.png`
- `outputs/plots/model_forecast_day_mixed.png`
- `outputs/plots/model_forecast_regime_mae.png`

There are also sprint-era plots and reports in `outputs/reports/` and `outputs/plots/` that summarize the earlier baseline, attention, and loss-tuning experiments.

## How to Run

### Baseline LSTM

```powershell
python models\baseline_lstm\run.py --data-path .\dataset --no-plot
```

### Attention LSTM

```powershell
python models\attention_lstm\run.py --data-path .\dataset --no-plot
```

### Residual Hybrid

```powershell
python residual_hybrid_model.py --data-path .\dataset --no-plot
```

### Multi-Model Comparison

```powershell
python compare_model_forecasts.py --data-path .\dataset
```

Useful options:

- `--focus-month 5` to choose representative sunny/cloudy/mixed days from a month
- `--focus-date 2021-06-15` to compare one exact day and start the 30-day plot from that date
- `--baseline-model-path`, `--attention-model-path`, and `--hybrid-predictions-csv` to override artifact locations

### Sprint Plots

```powershell
python generate_sprint_plots.py
```

### Analysis and Tests

```powershell
python models\analysis\run.py --data-path .\dataset --output outputs\plots\pearson_correlation_heatmap.png
python models\testing\run.py
```

## Why the Improvement Matters

The project improvement is not just about lower error numbers. It comes from better modeling of the problem:

- Peak-aware training helps the models care more about the hours that matter most.
- Chronological splitting keeps evaluation realistic for time-series forecasting.
- Residual learning lets the final model focus on the baseline's mistakes instead of the whole signal.
- Regime-aware features help the model react differently to sunny, cloudy, and mixed conditions.
- Automated comparison plots make it easier to inspect where each model succeeds or fails.

In short, the project improved from a single direct forecaster to a more structured forecasting system that combines baseline prediction, error correction, and richer diagnostics.

## Important Notes

The repository is useful for experimentation and reporting, but there are two important caveats to keep in mind:

1. Some feature builders still use backfilling after lag and rolling feature creation. That can leak future values into early rows if the raw data has gaps.
2. The sprint-era baseline selection was based on comparing multiple variants on the same holdout split. That makes the final baseline comparison useful, but not a perfect untouched final benchmark.

If you want a publication-grade evaluation, the next step would be to remove future-looking fill logic and reserve a separate final test split that is never used for model selection.

## Main Entry Points

- `lstm_model.py` and `models/baseline_lstm/run.py` for the baseline pipeline
- `attention_lstm_model.py` and `models/attention_lstm/run.py` for the attention pipeline
- `residual_hybrid_model.py` and `models/residual_hybrid/run.py` for the hybrid pipeline
- `compare_model_forecasts.py` for the all-model comparison plots and tables
- `generate_sprint_plots.py` for the sprint summary visuals

## Summary

This project is a complete hourly GHI forecasting workflow for an NSRDB site. It begins with a baseline LSTM, explores attention-based forecasting, and ends with a residual hybrid model that currently delivers the best results. The repo also includes a full reporting stack so you can inspect the models on 30-day windows, single-day case studies, and weather-regime slices.
