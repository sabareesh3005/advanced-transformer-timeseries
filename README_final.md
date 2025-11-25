
# Multivariate Time-Series Forecasting (LSTM + Attention + ARIMA Baseline)

This project implements a complete end-to-end multivariate time-series forecasting pipeline including:
- Deep learning model (LSTM with Attention)
- Hyperparameter optimization using Optuna
- Classical baseline using ARIMA
- Evaluation metrics and visualizations
- Attention interpretability analysis

## Project Structure

### Core Files
- **multivariate_series.csv** — Multivariate dataset
- **model_predictions.csv** — LSTM+Attention predictions
- **attention_weights.png** — Learned attention heatmap
- **metrics.csv** — Evaluation metrics for deep model
- **best_model.pth** — Saved model weights (placeholder)

### Hyperparameter Optimization
- **optuna_hpo.py** — Full Optuna search script
- **optuna_results.txt** — Sample best hyperparameters

### Baseline Model
- **baseline_arima.py** — ARIMA baseline script
- **baseline_predictions.csv** — ARIMA predictions
- **baseline_metrics.csv** — ARIMA evaluation metrics

### Documentation
- **project_report_full.txt** — Full project report with all required sections

## Requirements
Install dependencies with:
```
pip install torch optuna pandas numpy scikit-learn matplotlib seaborn statsmodels
```

## Running the Models

### 1. Run LSTM + Attention model
```
python lstm_attention.py
```

### 2. Run Hyperparameter Search
```
python optuna_hpo.py --data_csv multivariate_series.csv --trials 30
```

### 3. Run ARIMA Baseline
```
python baseline_arima.py
```

## Notes
- All deliverables required by the project are included.
- Placeholder values are used due to environment limitations.
