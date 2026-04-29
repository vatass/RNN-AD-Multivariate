#!/usr/bin/env python
"""5-fold cross-validation training of an MLP for 145-ROI regression.

Data format (subjectsamples_longclean_dl_muse_allstudies.csv):
    PTID  : subject identifier (multiple rows per subject)
    X     : 151-dimensional embedding stored as a Python list literal
    Y     : 145-dimensional ROI target stored as a Python list literal

Folds are assigned at the SUBJECT level so that all observations
from the same subject end up in the same fold, preventing leakage.
"""
from __future__ import print_function, division

import argparse
import ast
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cbig.Nguyen2020.mlp_model import MLPRegressor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Return arrays X [N, 151], Y [N, 145] and subject id per row."""
    xs, ys, ptids = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            xs.append(ast.literal_eval(row['X']))
            ys.append(ast.literal_eval(row['Y']))
            ptids.append(row['PTID'])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(ptids)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_stats(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0          # avoid division by zero for constant features
    return mean, std


def normalise(X, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------------
# 5-fold subject-level split
# ---------------------------------------------------------------------------

def make_subject_folds(ptids, n_folds=5, seed=42):
    """
    Assign each subject to one of n_folds folds.

    Returns a list of length n_folds; each element is a boolean mask
    over *rows* (not subjects) indicating the test set for that fold.
    """
    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(ptids)
    rng.shuffle(unique_subjects)

    subject_fold = {s: i % n_folds for i, s in enumerate(unique_subjects)}
    row_fold = np.array([subject_fold[p] for p in ptids])

    test_masks = []
    for f in range(n_folds):
        test_masks.append(row_fold == f)
    return test_masks


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_loader(X, Y, batch_size, shuffle):
    tx = torch.from_numpy(X)
    ty = torch.from_numpy(Y)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * xb.size(0)
            total_mae  += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def train_fold(fold_idx, X_train, Y_train, X_test, Y_test, args, device):
    """Train one fold, return best test MAE and per-epoch history."""
    loader_tr = build_loader(X_train, Y_train, args.batch_size, shuffle=True)
    loader_te = build_loader(X_test,  Y_test,  args.batch_size, shuffle=False)

    hidden = [int(h) for h in args.hidden_sizes.split(',')]
    model = MLPRegressor(
        input_size=X_train.shape[1],
        output_size=Y_train.shape[1],
        hidden_sizes=hidden,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss() if args.loss == 'mse' else nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=False)

    best_mae  = float('inf')
    best_path = os.path.join(args.out_dir, 'fold%d_best.pt' % fold_idx)
    history   = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss          = train_epoch(model, loader_tr, optimizer, criterion, device)
        te_loss, te_mae  = eval_epoch(model, loader_te, criterion, device)
        scheduler.step(te_loss)

        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'test_loss': te_loss, 'test_mae': te_mae})

        if te_mae < best_mae:
            best_mae = te_mae
            torch.save(model, best_path)

        if args.verbose and epoch % 10 == 0:
            elapsed = time.time() - t0
            print('  fold %d  epoch %3d/%d  [%.0fs]  '
                  'train_loss=%.4f  test_loss=%.4f  test_mae=%.4f'
                  % (fold_idx, epoch, args.epochs, elapsed,
                     tr_loss, te_loss, te_mae))

    return best_mae, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- load raw data ----
    print('Loading data from', args.data)
    X, Y, ptids = load_csv(args.data)
    print('  X shape:', X.shape, '  Y shape:', Y.shape,
          '  unique subjects:', len(np.unique(ptids)))

    # ---- 5-fold split masks (subject level) ----
    test_masks = make_subject_folds(ptids, n_folds=args.n_folds, seed=args.seed)

    all_results = []
    for fold_idx, test_mask in enumerate(test_masks):
        train_mask = ~test_mask

        X_tr_raw, Y_tr = X[train_mask], Y[train_mask]
        X_te_raw, Y_te = X[test_mask],  Y[test_mask]

        # normalise inputs using training-set statistics only
        mean, std   = compute_stats(X_tr_raw)
        X_tr        = normalise(X_tr_raw, mean, std)
        X_te        = normalise(X_te_raw, mean, std)

        print('\n=== Fold %d/%d  train=%d  test=%d ==='
              % (fold_idx, args.n_folds - 1, X_tr.shape[0], X_te.shape[0]))

        best_mae, history = train_fold(
            fold_idx, X_tr, Y_tr, X_te, Y_te, args, device)

        # save per-fold stats
        stats_path = os.path.join(args.out_dir, 'fold%d_stats.json' % fold_idx)
        with open(stats_path, 'w') as fh:
            json.dump({'mean': mean.tolist(), 'std': std.tolist()}, fh)

        history_path = os.path.join(
            args.out_dir, 'fold%d_history.json' % fold_idx)
        with open(history_path, 'w') as fh:
            json.dump(history, fh)

        all_results.append({'fold': fold_idx, 'best_test_mae': best_mae})
        print('  Fold %d best test MAE: %.4f' % (fold_idx, best_mae))

    # ---- summary ----
    maes = [r['best_test_mae'] for r in all_results]
    summary = {
        'folds': all_results,
        'mean_mae': float(np.mean(maes)),
        'std_mae':  float(np.std(maes)),
    }
    summary_path = os.path.join(args.out_dir, 'cv_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n=== 5-fold CV summary ===')
    print('  Mean MAE: %.4f  Std: %.4f' % (summary['mean_mae'], summary['std_mae']))
    print('  Results saved to', args.out_dir)


def get_args():
    parser = argparse.ArgumentParser(
        description='5-fold CV MLP training for 145-ROI regression')

    parser.add_argument('--data', required=True,
                        help='Path to subjectsamples_longclean_dl_muse_allstudies.csv')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='Directory for checkpoints and logs')

    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',  type=float, default=0.2)
    parser.add_argument('--hidden_sizes', type=str, default='512,256',
                        help='Comma-separated list of hidden layer widths')
    parser.add_argument('--loss', choices=['mse', 'mae'], default='mse',
                        help='Regression loss: mse (MSE) or mae (L1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
