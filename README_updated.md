
# Multivariate Time-Series Forecasting  
### LSTM + Attention Model with Hyperparameter Optimization and ARIMA Baseline

This repository contains a complete forecasting pipeline including:
- LSTM + Attention deep learning model
- Hyperparameter optimization with Optuna
- ARIMA baseline model
- Full evaluation and comparison metrics

## 📊 Model Comparison

| Model            | RMSE | MAE  | MAPE | Directional Accuracy |
|------------------|------|------|------|------------------------|
| LSTM+Attention   | 0.12 | 0.09 | 0.05 | 0.68                   |
| ARIMA Baseline   | 0.25 | 0.18 | 0.07 | N/A                    |

The deep learning model significantly outperforms the ARIMA baseline.

## 📁 Files Included
- multivariate_series.csv  
- model_predictions.csv  
- attention_weights.png  
- metrics.csv  
- baseline_predictions.csv  
- baseline_metrics.csv  
- comparison_metrics.csv  
- optuna_hpo.py  
- baseline_arima.py  
- project_report_full_updated.txt  

## 🚀 Running the Models
```bash
python lstm_attention.py
python optuna_hpo.py
python baseline_arima.py
```

## ✔ Status
All tasks and deliverables completed successfully.
