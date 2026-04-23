# Solar Irradiance Forecasting — Benchmark Report

**Generated**: 2026-04-23 20:01:45

## Executive Summary

**Best Model**: GBT (LightGBM) (Composite Score: 5.12)
- RMSE: 8.35 W/m²
- MAE: 3.34 W/m²
- R²: 0.9993
- Peak MAE: 6.78 W/m²

## Leakage Audit Report

**Overall**: ✅ ALL CHECKS PASSED

- ✅ **temporal_ordering**: Timestamps are monotonically non-decreasing.
- ✅ **split_no_overlap**: No temporal overlap between splits. train_max=2020-10-19T15:00:00.000000000  val_min=2020-10-19T16:00:00.000000000  val_max=2021-05-26T19:00:00.000000000  test_min=2021-05-26T20:00:00.000000000
- ✅ **no_future_features**: ghi_lag_1 matches ghi.shift(1) — 0 mismatches in 35063 rows.
- ✅ **scaler_train_only**: Scaler fit on 24544 samples (train size = 24544).
- ✅ **target_not_in_features**: Target column 'ghi' is not in the tabular feature list.
- ✅ **no_backfill_after_split**: No NaN in any target partition.

## Model Comparison

| model           |     rmse |      mae |       r2 |   day_mae |   peak_mae |   cloud_mae |   transition_mae |   composite_score |   rank |
|:----------------|---------:|---------:|---------:|----------:|-----------:|------------:|-----------------:|------------------:|-------:|
| GBT (LightGBM)  |  8.35128 |  3.34069 | 0.999334 |   1.51588 |    6.78465 |    0.846243 |          3.07468 |           5.11859 |      1 |
| DNN             | 22.0196  | 11.9607  | 0.995372 |   9.03834 |   17.9575  |    5.12111  |         11.1579  |          15.0314  |      2 |
| ANN             | 19.5236  | 14.8436  | 0.996362 |  11.4842  |   26.4616  |    9.86148  |         15.5783  |          17.9341  |      3 |
| Hybrid Residual | 50.2711  | 16.5721  | 0.97588  |  12.1537  |   23.9007  |    5.42673  |         20.3753  |          27.0462  |      4 |
| CNN-LSTM        | 52.0018  | 19.8837  | 0.974191 |  16.5476  |   31.3896  |    6.75268  |         24.7695  |          30.5835  |      5 |
| LSTM            | 53.693   | 21.9059  | 0.972485 |  16.9591  |   30.5173  |    9.65653  |         24.5094  |          31.7174  |      6 |
| CNN-DNN         | 53.2862  | 22.9479  | 0.9729   |  19.2691  |   37.5559  |    8.18805  |         26.6082  |          33.3061  |      7 |
| CNN-A-LSTM      | 58.0023  | 24.4058  | 0.967891 |  17.6533  |   34.5803  |    9.89781  |         27.8531  |          34.8605  |      8 |
| SVM (SVR-RBF)   | 53.3663  | 43.0863  | 0.972819 |  41.6105  |   34.6785  |   42.1624   |         39.6703  |          43.8378  |      9 |

## Regime-Wise Performance

| regime        |   count |      rmse |      mae |       bias |   peak_mae | model           |
|:--------------|--------:|----------:|---------:|-----------:|-----------:|:----------------|
| clear         |     486 |   7.17977 |  4.38244 |  -0.193934 |    5.19945 | GBT (LightGBM)  |
| partly_cloudy |      55 |   2.8321  |  2.20429 |  -0.289964 |  nan       | GBT (LightGBM)  |
| cloudy        |    4719 |   8.50512 |  3.24664 |   0.03081  |    7.34593 | GBT (LightGBM)  |
| clear         |     486 |  47.568   | 37.4468  |  15.315    |   29.5271  | SVM (SVR-RBF)   |
| partly_cloudy |      55 |  67.1449  | 56.9557  | -14.2223   |  nan       | SVM (SVR-RBF)   |
| cloudy        |    4719 |  53.7483  | 43.5055  |  25.3104   |   36.5026  | SVM (SVR-RBF)   |
| clear         |     486 |  21.2272  | 16.6828  | -12.8179   |   18.9364  | ANN             |
| partly_cloudy |      55 |  30.716   | 28.5962  | -21.2161   |  nan       | ANN             |
| cloudy        |    4719 |  19.1695  | 14.4939  |   0.366824 |   29.1261  | ANN             |
| clear         |     486 |  33.3837  | 24.5244  | -11.5349   |   25.879   | DNN             |
| partly_cloudy |      55 |  33.0783  | 24.3998  |  -1.39725  |  nan       | DNN             |
| cloudy        |    4719 |  20.3204  | 10.5218  |  -3.47232  |   15.1526  | DNN             |
| clear         |     486 |  75.591   | 43.2194  |   4.32881  |   42.0733  | LSTM            |
| partly_cloudy |      55 | 113.765   | 68.6503  |  12.9545   |  nan       | LSTM            |
| cloudy        |    4719 |  49.7407  | 19.166   |   6.04748  |   26.408   | LSTM            |
| clear         |     486 |  88.5684  | 59.4944  | -41.4664   |   58.8421  | CNN-DNN         |
| partly_cloudy |      55 | 123.027   | 94.3714  | -23.57     |  nan       | CNN-DNN         |
| cloudy        |    4719 |  46.6976  | 18.3516  |   1.56309  |   29.9865  | CNN-DNN         |
| clear         |     486 |  86.2222  | 51.8542  | -29.3958   |   48.1231  | CNN-LSTM        |
| partly_cloudy |      55 | 110.934   | 85.141   | -38.6471   |  nan       | CNN-LSTM        |
| cloudy        |    4719 |  45.8818  | 15.8305  |   1.56099  |   25.4391  | CNN-LSTM        |
| clear         |     486 |  78.7852  | 44.8959  |  12.5133   |   42.8901  | CNN-A-LSTM      |
| partly_cloudy |      55 | 139.033   | 78.2635  |  18.4448   |  nan       | CNN-A-LSTM      |
| cloudy        |    4719 |  53.716   | 21.6678  |  11.4264   |   31.6254  | CNN-A-LSTM      |
| clear         |     486 |  72.6331  | 36.275   |  -7.17419  |   31.7806  | Hybrid Residual |
| partly_cloudy |      55 | 111.219   | 60.3834  |  17.7133   |  nan       | Hybrid Residual |
| cloudy        |    4719 |  46.1456  | 14.0323  |   2.53396  |   21.0986  | Hybrid Residual |

## Robustness Analysis

| model           |   n_nan |   n_negative |   n_exploding |   n_total_corrected |   robustness_rank |
|:----------------|--------:|-------------:|--------------:|--------------------:|------------------:|
| GBT (LightGBM)  |       0 |            0 |             0 |                   0 |                 1 |
| CNN-LSTM        |       0 |            0 |             0 |                   0 |                 2 |
| CNN-DNN         |       0 |            0 |             0 |                   0 |                 3 |
| LSTM            |       0 |            0 |             0 |                   0 |                 4 |
| CNN-A-LSTM      |       0 |            0 |             0 |                   0 |                 5 |
| Hybrid Residual |       0 |            0 |             0 |                   0 |                 6 |
| ANN             |       0 |           80 |             0 |                  80 |                 7 |
| SVM (SVR-RBF)   |       0 |          233 |             0 |                 233 |                 8 |
| DNN             |       0 |         1882 |             0 |                1882 |                 9 |

## Model Architectures

### GBT (LightGBM)
```json
{
  "name": "GBT (LightGBM)",
  "model_type": "tabular",
  "n_estimators": 200,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "best_iteration": 200
}
```

### SVM (SVR-RBF)
```json
{
  "name": "SVM (SVR-RBF)",
  "model_type": "tabular",
  "kernel": "rbf",
  "C": 100.0,
  "epsilon": 0.1,
  "train_samples_used": 3958,
  "n_support_vectors": 79
}
```

### ANN
```json
{
  "name": "ANN",
  "model_type": "tabular",
  "hidden_units": [
    128,
    64
  ],
  "dropout_rate": 0.2,
  "epochs": 10
}
```

### DNN
```json
{
  "name": "DNN",
  "model_type": "tabular",
  "hidden_units": [
    256,
    128,
    64,
    32
  ],
  "dropout_rates": [
    0.3,
    0.3,
    0.2,
    0.1
  ],
  "use_batch_norm": true,
  "epochs": 10
}
```

### LSTM
```json
{
  "name": "LSTM",
  "model_type": "sequence",
  "units": [
    64,
    32
  ],
  "dropout_rate": 0.2,
  "sequence_length": "from bundle",
  "epochs": 10
}
```

### CNN-DNN
```json
{
  "name": "CNN-DNN",
  "model_type": "sequence",
  "conv_filters": [
    64,
    32
  ],
  "dense_units": [
    64,
    32
  ],
  "epochs": 10
}
```

### CNN-LSTM
```json
{
  "name": "CNN-LSTM",
  "model_type": "sequence",
  "conv_filters": [
    64,
    32
  ],
  "lstm_units": [
    64,
    32
  ],
  "epochs": 10
}
```

### CNN-A-LSTM
```json
{
  "name": "CNN-A-LSTM",
  "model_type": "sequence",
  "conv_filters": [
    64,
    32
  ],
  "lstm_units": [
    64,
    32
  ],
  "attention_units": 32,
  "epochs": 10
}
```

### Hybrid Residual
```json
{
  "name": "Hybrid Residual",
  "model_type": "tabular",
  "ensemble_size": 3,
  "n_residual_features": 38
}
```

## Validation Statement

This benchmark employs the following safeguards to ensure scientific validity:

1. **Chronological splitting only** — no random shuffling at any stage.
2. **Scalers fit on training data only** — preventing information leakage from validation/test.
3. **Causal features only** — all lag and rolling features use `shift(≥1)`.
4. **Fixed future holdout** — the test set (final 15%) is never used during model selection.
5. **Automated leakage audit** — 6 checks run before training with hard failure on violation.
6. **Prediction sanitization** — NaN, negative, and exploding predictions are logged and corrected.
7. **Deterministic seeds** — all random sources seeded for full reproducibility.

## Generated Plots

- `actual_vs_predicted_all.png`
- `actual_vs_predicted_subplots_30days.png`
- `actual_vs_predicted_subplots_1day.png`
- `training_curves.png`
- `residual_distributions.png`
- `feature_importance.png`
- `attention_weights.png`
- `model_ranking.png`
- `regime_comparison.png`
- `scatter_actual_vs_predicted.png`
- `hourly_error_profile.png`
- `peak_prediction_comparison.png`
- `cloud_event_errors.png`
