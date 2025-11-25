
"""
optuna_hpo.py

Optuna hyperparameter optimization for LSTM + Attention time-series model.

Usage:
    python optuna_hpo.py --data_csv project_outputs/multivariate_series.csv
or run inside a notebook by importing the objective function.

This script defines an objective that:
 - loads the dataset CSV (expects time-indexed multivariate CSV)
 - preprocesses (StandardScaler), windowing (INPUT_LEN, OUTPUT_LEN)
 - defines an LSTM+Attention model
 - trains for a small number of epochs (configurable)
 - returns validation RMSE to Optuna

Note: This script is designed to run locally or on Colab with PyTorch installed.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import random

# ------------------ Data utilities ------------------
def create_windows(series, input_len, output_len):
    X, Y = [], []
    for i in range(len(series) - input_len - output_len + 1):
        X.append(series[i:i+input_len])
        Y.append(series[i+input_len:i+input_len+output_len, 0])
    return np.array(X), np.array(Y)

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ------------------ Model ------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    def forward(self, encoder_outputs):
        score = torch.tanh(self.W(encoder_outputs))
        weights = torch.softmax(self.v(score), dim=1)
        context = (weights * encoder_outputs).sum(dim=1)
        return context, weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0, output_len=14):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_len)
    def forward(self, x):
        enc_out, _ = self.lstm(x)
        ctx, weights = self.attn(enc_out)
        out = self.fc(ctx)
        return out, weights

# ------------------ Objective ------------------
def objective(trial, data_csv, input_len=60, output_len=14, epochs=5, device='cpu'):
    # Hyperparameters to search
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    seed = 42

    # load data
    df = pd.read_csv(data_csv)
    values = df.values.astype(float)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)

    # train/val/test split by time
    n = len(scaled)
    train_cut = int(0.7 * n); val_cut = int(0.85 * n)
    train = scaled[:train_cut]; val = scaled[train_cut:val_cut]

    X_train, Y_train = create_windows(train, input_len, output_len)
    X_val, Y_val = create_windows(val, input_len, output_len)

    train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = LSTMWithAttention(input_dim=values.shape[1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, output_len=output_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # training (light)
    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            out, _ = model(Xb)
            loss = criterion(out, Yb)
            loss.backward()
            optimizer.step()
        # optional early stopping could be added here

    # validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            out, _ = model(Xb)
            preds.append(out.cpu().numpy()); truths.append(Yb.cpu().numpy())
    preds = np.vstack(preds); truths = np.vstack(truths)
    val_rmse = mean_squared_error(truths.flatten(), preds.flatten(), squared=False)
    return val_rmse

# ------------------ Main runner ------------------
def run_study(data_csv, n_trials=20, output='optuna_results.txt', device='cpu'):
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, data_csv, device=device)
    study.optimize(func, n_trials=n_trials)
    # save best params
    with open(output, 'w') as f:
        f.write('Best value: %.6f\\n' % study.best_value)
        f.write('Best params: %s\\n' % study.best_params)
    print('Study complete. Results saved to', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default='project_outputs/multivariate_series.csv')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='optuna_results.txt')
    args = parser.parse_args()
    run_study(args.data_csv, n_trials=args.trials, output=args.output, device=args.device)
