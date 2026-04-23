# Plot Library

This folder is organized by purpose so the charts are easier to find.

## Folders

- [baseline](baseline/) - baseline LSTM forecast plots
- [attention](attention/) - standalone attention model plots
- [comparison](comparison/) - multi-model comparison plots
- [residual_hybrid](residual_hybrid/) - residual hybrid diagnostics and plots
- [sprint](sprint/) - sprint / experiment summary plots
- [analysis](analysis/) - correlation and analysis charts

## Common Files

### Baseline

- [prediction_30_days.png](baseline/prediction_30_days.png)
- [prediction_1_day.png](baseline/prediction_1_day.png)
- [training_improvement.png](baseline/training_improvement.png)

### Attention

- [attention_actual_vs_pred.png](attention/attention_actual_vs_pred.png)
- [attention_weights.png](attention/attention_weights.png)

### Comparison

- [model_forecast_comparison_30_days.png](comparison/model_forecast_comparison_30_days.png)
- [model_forecast_month_day_comparison.png](comparison/model_forecast_month_day_comparison.png)
- [model_forecast_jan_mar_2021_comparison.png](comparison/model_forecast_jan_mar_2021_comparison.png)
- [model_forecast_regime_mae.png](comparison/model_forecast_regime_mae.png)

### Residual Hybrid

- [residual_hybrid_metrics.png](residual_hybrid/residual_hybrid_metrics.png)
- [residual_hybrid_scatter.png](residual_hybrid/residual_hybrid_scatter.png)
- [residual_hybrid_walk_forward.png](residual_hybrid/residual_hybrid_walk_forward.png)

### Sprint

- [sprint_summary_report.png](sprint/sprint_summary_report.png)
- [sprint_target_achievement.png](sprint/sprint_target_achievement.png)
- [sprint_task_e_final_metrics.png](sprint/sprint_task_e_final_metrics.png)

### Analysis

- [pearson_correlation_heatmap.png](analysis/pearson_correlation_heatmap.png)

## How To Use

Open this folder and start from the category you need. For example:

- baseline performance: [baseline](baseline/)
- model comparison: [comparison](comparison/)
- residual debugging: [residual_hybrid](residual_hybrid/)
- sprint history: [sprint](sprint/)

The plotting scripts can also be updated to save directly into these folders.
