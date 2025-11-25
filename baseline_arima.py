# baseline_arima.py
# Placeholder baseline ARIMA model script.
# Replace with real ARIMA code when running locally.

import pandas as pd

df = pd.read_csv('multivariate_series.csv')
y = df['feat_1']

# Fake ARIMA predictions (placeholder)
preds = y.tail(20).reset_index(drop=True)

preds.to_csv('baseline_predictions.csv', index=False)
