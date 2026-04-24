# 🚀 Quick Start Guide — Colab Notebook

## What You Got

**Main File**: `Solar_Irradiance_Forecasting_Colab.ipynb` (197 KB, 2063 lines)

A complete, production-ready Google Colab notebook for solar irradiance forecasting with:

✅ **9 fully implemented models**

- GBT (LightGBM)
- SVM (Support Vector Machine)
- ANN (Artificial Neural Network)
- DNN (Deep Neural Network)
- LSTM (Long Short-Term Memory)
- CNN-DNN (Convolutional + Dense)
- CNN-LSTM (Convolutional + LSTM)
- CNN-A-LSTM (Convolutional + Attention-LSTM)
- Hybrid Residual (Baseline + Residual Correction)

✅ **Complete pipeline**

1. Data loading from NSRDB CSVs
2. Feature engineering (42 features including lags, rolling stats, solar geometry)
3. Leakage-safe chronological splits (70/15/15)
4. Scaling (train-only fit)
5. Model training & evaluation
6. Comprehensive metrics & ranking

✅ **11 publication-quality figures**

- Correlation heatmap
- Actual vs predicted (all 9 models)
- Performance comparison
- Clear sky predictions
- Cloudy conditions predictions
- Feature importance
- Training history
- Residual distributions
- Regime-wise performance
- Time series comparison
- Error vs solar zenith angle

---

## 30-Second Setup

### In Google Colab:

```
1. Go to https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Select "Solar_Irradiance_Forecasting_Colab.ipynb"
4. Click the dataset path cell (2.1) and update to your data location
5. Runtime → Change runtime type → Select GPU (optional but recommended)
6. Click "Run all" (Ctrl+F9)
```

### Dataset Format:

Your CSVs need these columns:

- `ghi` or `GHI` — Global Horizontal Irradiance (W/m²)
- `temperature` or `Temperature` — °C
- `relative_humidity` or `RelativeHumidity` — %
- `wind_speed` or `WindSpeed` — m/s
- `pressure` or `Pressure` — mb
- `solar_zenith_angle` or `SolarZenithAngle` — °

Plus a timestamp column OR Year/Month/Day/Hour/Minute columns.

---

## Notebook Sections

| Section         | Purpose                                       | Cells |
| --------------- | --------------------------------------------- | ----- |
| 1. Setup        | Config, imports, GPU check                    | 5     |
| 2. Data Loading | NSRDB CSV loading, parsing                    | 4     |
| 3. Features     | Lags, rolling stats, solar geometry, regime   | 6     |
| 4. Split        | Train/Val/Test (70/15/15), scaling, sequences | 4     |
| 5. Models       | 9 models with training & prediction           | 10    |
| 6. Metrics      | Comparison table & ranking                    | 1     |
| 7. Figures      | 11 publication-quality plots                  | 11    |
| 8. Results      | Summary & checklist                           | 3     |

**Total: 44 cells**

---

## What's Included

✅ **No external dependencies** — everything in the notebook
✅ **Colab auto-detection** — installs packages automatically
✅ **Leakage-safe** — chronological splits, no future data in training
✅ **GPU-ready** — deep learning models optimized for GPU
✅ **Production quality** — clean code, clear structure, publication-ready figures
✅ **Self-contained** — all 42 features engineered in notebook
✅ **Fully configurable** — adjust epochs, batch size, split ratios

---

## Customization

### For faster testing:

Edit cell 1.5:

```python
QUICK_MODE = True  # Uses 20 epochs instead of 100, 50 estimators for GBT instead of 200
```

### For different sequence length:

Edit cell 1.5:

```python
SEQUENCE_LENGTH = 48  # Change to 24, 36, or 72 hours
```

### For different train/val/test split:

Edit cell 1.5:

```python
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

---

## Expected Output

**Model Rankings Example:**

```
Rank  Model               RMSE    MAE     R²      Day MAE Peak MAE
1     CNN-LSTM            45.23   22.15   0.8932  12.50   31.45
2     Hybrid Residual     46.81   23.02   0.8901  13.02   33.22
3     CNN-DNN             47.15   23.45   0.8876  13.15   34.10
...
9     SVM                 68.45   35.20   0.7654  20.33   48.92
```

**11 Figures:**

- figure_01_correlation_heatmap.png
- figure_02_actual_vs_predicted.png
- figure_03_model_performance.png
- figure_04_clear_sky_comparison.png
- figure_05_cloudy_comparison.png
- figure_06_feature_importance.png
- figure_07_training_history.png
- figure_08_residuals.png
- figure_09_regime_performance.png
- figure_10_time_series_sample.png
- figure_11_error_vs_zenith.png

---

## Runtime Estimate (on GPU)

- Data loading & feature engineering: ~2 min
- GBT: ~3 min
- SVM: ~5 min
- ANN: ~8 min
- DNN: ~12 min
- LSTM: ~15 min
- CNN-DNN: ~15 min
- CNN-LSTM: ~18 min
- CNN-A-LSTM: ~15 min
- Hybrid Residual: ~12 min
- Figure generation: ~5 min

**Total: 60–90 minutes** (or 20 min with QUICK_MODE=True)

---

## Key Features (Detailed)

### Data Pipeline

- ✅ Automatic CSV concatenation
- ✅ Timestamp parsing (flexible format)
- ✅ Chronological sorting (never shuffled)
- ✅ Forward-fill for missing values (safe)
- ✅ Clipping to valid ranges (0–1500 W/m²)

### Features (42 total)

- **Weather**: temperature, humidity, wind speed, pressure, dew point
- **Solar Geometry**: zenith angle, elevation, air mass, clear sky GHI, clear sky index
- **Time**: hour (sin/cos), day-of-year (sin/cos), is_daylight
- **Lags**: 1h, 2h, 3h, 24h past GHI
- **Rolling**: 3/6/24h mean, std, min, max, range, variance
- **Volatility**: gradients, accelerations, absolute differences
- **Regime**: clear sky, partly cloudy, cloudy (one-hot)

### Model Architecture

- **Tabular**: GBT, SVM, ANN, DNN (fit on 42 features)
- **Sequence**: LSTM, CNN variants (48-step sequences with 20 features each)
- **Hybrid**: Combines baseline predictions + residual correction

### Evaluation Metrics

- **RMSE** — overall prediction accuracy
- **MAE** — mean absolute error
- **R²** — coefficient of determination
- **Day MAE** — performance during daylight only
- **Peak MAE** — performance above 75th percentile (high irradiance events)

---

## Support & Debugging

### If dataset isn't found:

```python
# Cell 2.1: Update this line
dataset_path = 'dataset'  # Change to your actual path
```

### If packages fail to install:

```python
# Manually run cell 1.2 (package installation) again
```

### If running out of GPU memory:

```python
# Cell 1.5: Reduce batch size
BATCH_SIZE = 16  # Try 8 or 4
```

### If figures don't display:

Check the Colab working directory — they're saved as PNG files.

---

## What Was NOT Added (By Design)

❌ No logging frameworks
❌ No experiment trackers
❌ No report generators
❌ No dashboards
❌ No unnecessary abstractions
❌ No complex deployment systems

**Focus**: Clean, readable, executable code for analysis & presentation.

---

## Next Steps

1. **Upload to Colab** → Open the notebook
2. **Set your dataset path** → Point to your NSRDB data
3. **Run all cells** → Complete pipeline execution
4. **Download figures** → Use in presentations/papers
5. **Share results** → Model rankings & metrics
6. **Iterate** → Adjust hyperparameters and re-run

---

## For Questions

Everything is self-documented:

- Each section has clear markdown headings
- Each cell prints status messages
- Variables have descriptive names
- All outputs show metrics & results

The notebook is ready for:
✅ Presentations
✅ Thesis work
✅ Publication
✅ Educational purposes
✅ Operational forecasting systems

---

**Happy forecasting!** 🌞
