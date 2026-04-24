# Solar Irradiance Forecasting — Google Colab Notebook

## Overview

`Solar_Irradiance_Forecasting_Colab.ipynb` is a complete, clean, professional-grade notebook that consolidates your entire forecasting project into a single, well-organized execution pipeline.

**No external dependencies on local modules** — everything is self-contained in the notebook.

---

## Notebook Structure

### ✅ SECTION 1: SETUP (5 cells)
- Library imports (numpy, pandas, sklearn, tensorflow, lightgbm)
- Colab package installation (auto-detected)
- Google Drive mount (optional)
- Random seed configuration
- Configuration constants

### ✅ SECTION 2: DATASET LOADING (4 cells)
- Load NSRDB CSV files
- Parse timestamps
- Chronological sorting
- Missing value handling
- Basic GHI statistics

### ✅ SECTION 3: FEATURE ENGINEERING (5 cells)
- **3.1** Solar geometry (clear sky GHI, air mass, dew point)
- **3.2** Weather & time features (cyclical encoding)
- **3.3** Lag features (1, 2, 3, 24 hours)
- **3.4** Rolling statistics (mean, std, min, max, range)
- **3.5** Regime classification (clear, partly cloudy, cloudy)
- **3.6** Final data cleaning

### ✅ SECTION 4: TRAIN/VAL/TEST SPLIT (4 cells)
- Feature list definition
- Chronological 70/15/15 split (NO shuffling)
- StandardScaler + MinMaxScaler (fit on train only)
- Sequence generation for deep learning (48-hour lookback)

### ✅ SECTION 5: MODEL IMPLEMENTATIONS (10 cells)
Each model in its own cell:

1. **Model 1: GBT** — LightGBM gradient boosted trees
2. **Model 2: SVM** — Support Vector Machine (RBF kernel)
3. **Model 3: ANN** — Artificial Neural Network (4 layers)
4. **Model 4: DNN** — Deep Neural Network (6 layers, batch norm)
5. **Model 5: LSTM** — Long Short-Term Memory (2 layers)
6. **Model 6: CNN-DNN** — Convolutional + Dense hybrid
7. **Model 7: CNN-LSTM** — Convolutional + LSTM hybrid
8. **Model 8: CNN-A-LSTM** — Convolutional + Attention-LSTM
9. **Model 9: Hybrid Residual** — Baseline + residual correction
10. Metrics function (compute_metrics)

### ✅ SECTION 6: METRIC COMPARISON (1 cell)
- Combined metrics table (RMSE, MAE, R², Day MAE, Peak MAE)
- Composite scoring and ranking
- Final comparison dataframe

### ✅ SECTION 7: FIGURE GENERATION (11 cells)
Each figure in its own cell:

1. **Figure 1** — Correlation heatmap
2. **Figure 2** — Actual vs predicted (9 model scatter plots)
3. **Figure 3** — Model performance comparison (bar charts)
4. **Figure 4** — Clear sky prediction comparison
5. **Figure 5** — Cloudy/partly cloudy prediction comparison
6. **Figure 6** — Feature importance (GBT top 20)
7. **Figure 7** — Training history (ANN, DNN, LSTM, CNN-LSTM)
8. **Figure 8** — Residual error distribution
9. **Figure 9** — Regime-wise performance comparison
10. **Figure 10** — Time series comparison (sample week)
11. **Figure 11** — Error vs solar zenith angle

### ✅ SECTION 8: FINAL RESULTS (3 cells)
- Ranked model summary
- All figures listed
- Pipeline validation checklist

---

## How to Use in Google Colab

### 1. Upload the Notebook
- Go to [Google Colab](https://colab.research.google.com)
- Upload `Solar_Irradiance_Forecasting_Colab.ipynb`

### 2. Prepare Your Dataset
- Option A: Upload dataset directly to Colab
- Option B: Mount Google Drive and provide path
  - Uncomment the Drive mount cell (1.3)
  - Update `dataset_path` to your Drive path

### 3. Run the Notebook
- Execute cells sequentially, or **Run All** (Ctrl+F9)
- GPU acceleration recommended for deep learning models
  - Runtime → Change runtime type → GPU

### 4. Adjust Configuration
- Edit `SECTION 1.5` constants if needed:
  - `QUICK_MODE = True` for rapid iteration (20 epochs instead of 100)
  - `BATCH_SIZE`, `EPOCHS_FULL`, `SEQUENCE_LENGTH`
  - Split ratios: `TRAIN_RATIO`, `VAL_RATIO`

---

## Dataset Requirements

Your dataset should contain:
- **GHI**: Global Horizontal Irradiance (W/m²)
- **Temperature**: Ambient temperature (°C)
- **RelativeHumidity**: Relative humidity (%)
- **WindSpeed**: Wind speed (m/s)
- **Pressure**: Atmospheric pressure (mb)
- **SolarZenithAngle**: Solar zenith angle (°)
- **Timestamps**: Year, Month, Day, Hour, Minute (or pre-built timestamp column)

Multiple CSV files are supported and will be concatenated automatically.

---

## Key Features

✅ **Leakage-Safe**
- Chronological splits (no random shuffling)
- Scalers fit on training data only
- Sequences created without future leakage

✅ **Clean Organization**
- Separate cells for each model
- Separate cells for each figure
- Clear markdown headings
- Readable variable names

✅ **Professional Output**
- 11 publication-quality figures
- Comprehensive metrics table
- Model rankings and scores
- Ready for presentation

✅ **Colab Compatible**
- Auto-detects and installs packages
- GPU-compatible
- Works with Google Drive
- No local dependencies

---

## Output Files

The notebook generates:

**Figures** (11 PNG files):
- `figure_01_correlation_heatmap.png`
- `figure_02_actual_vs_predicted.png`
- `figure_03_model_performance.png`
- `figure_04_clear_sky_comparison.png`
- `figure_05_cloudy_comparison.png`
- `figure_06_feature_importance.png`
- `figure_07_training_history.png`
- `figure_08_residuals.png`
- `figure_09_regime_performance.png`
- `figure_10_time_series_sample.png`
- `figure_11_error_vs_zenith.png`

**Tables** (in notebook):
- Model comparison table
- Ranked model summary

---

## Model Details

### Tabular Models (GBT, SVM, ANN, DNN)
- Input: 42 hand-engineered features
- No sequence length requirement
- Faster training

### Sequence Models (LSTM, CNN variants)
- Input: 48-hour lookback (sequence length)
- 20 temporal features per timestep
- Better capture of temporal dynamics

### Hybrid Model
- Combines baseline LSTM predictions with residual correction
- Adaptive error learning

---

## Approximate Runtime (on GPU)

- GBT, SVM: ~2–5 minutes (CPU-friendly)
- ANN, DNN: ~5–10 minutes
- LSTM: ~10–15 minutes
- CNN variants: ~15–20 minutes
- Hybrid Residual: ~10–15 minutes
- Figure generation: ~5 minutes

**Total: ~60–90 minutes (or ~20 minutes with QUICK_MODE=True)**

---

## Tips for Success

1. **Start with QUICK_MODE = True** to test the pipeline quickly
2. **Check your dataset** before running — missing columns will cause errors
3. **Use GPU** in Colab for significantly faster training
4. **Scroll through outputs** — figures will display inline
5. **Save figures** — they're automatically saved as PNG files
6. **Export results** — download figures and metrics for your report

---

## Troubleshooting

**ImportError: No module named X**
→ The notebook auto-installs packages. If it fails, manually run cell 1.2

**Out of memory**
→ Reduce `BATCH_SIZE` in cell 1.5, or reduce dataset size

**Missing dataset**
→ Update `dataset_path` in cell 2.1, ensure CSV files exist

**Figures not displaying**
→ Figures are saved as PNG files. Check the notebook working directory or Google Drive

---

## Next Steps

After running the notebook:

1. **Review metrics** — Check which model performs best
2. **Analyze figures** — Look for patterns in predictions vs actuals
3. **Export results** — Download PNG figures for presentations
4. **Iterate** — Adjust hyperparameters and re-run
5. **Share** — The notebook is publication-ready for presentations/reports

---

## Citation

This notebook consolidates a complete solar irradiance forecasting pipeline with:
- 9 models (from classical ML to advanced deep learning)
- 42 engineered features
- Leakage-safe chronological validation
- Publication-quality visualizations

Suitable for research, thesis work, or operational forecasting systems.

---

**Questions?** The notebook is self-documenting with inline comments and output summaries. Each cell prints status messages and produces clear outputs.
