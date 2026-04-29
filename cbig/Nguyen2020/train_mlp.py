#!/usr/bin/env python
"""5-fold cross-validation training of an MLP for 145-ROI regression.

Data format (subjectsamples_longclean_dl_muse_allstudies.csv):
    PTID  : subject identifier
    X     : 151-dimensional embedding stored as a Python list literal
    Y     : 145-dimensional ROI target stored as a Python list literal

Train/test splits are taken from the pre-defined pkl files:
    train_subject_allstudies_ids_dl_hmuse{fold}.pkl
    test_subject_allstudies_ids_dl_hmuse{fold}.pkl

Each pkl file is a plain Python list of subject IDs.  All visits (rows)
belonging to a subject are placed in the same split to prevent leakage.

Usage
-----
python -m cbig.Nguyen2020.train_mlp \\
    --data  subjectsamples_longclean_dl_muse_allstudies.csv \\
    --fold_dir . \\
    --out_dir output/mlp_cv \\
    --epochs 100 --verbose
"""
from __future__ import print_function, division

import argparse
import ast
import csv
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cbig.Nguyen2020.mlp_model import MLPRegressor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Return per-subject dicts: {ptid -> list of X vectors}, {ptid -> list of Y vectors}."""
    xs, ys = defaultdict(list), defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            pid = str(row['PTID'])
            xs[pid].append(ast.literal_eval(row['X']))
            ys[pid].append(ast.literal_eval(row['Y']))
    return xs, ys


def subjects_to_arrays(subject_ids, xs, ys):
    """Flatten all visits for the given subject IDs into (X, Y) numpy arrays."""
    X_rows, Y_rows = [], []
    for sid in subject_ids:
        key = str(sid)
        if key not in xs:
            continue
        X_rows.extend(xs[key])
        Y_rows.extend(ys[key])
    return (np.array(X_rows, dtype=np.float32),
            np.array(Y_rows, dtype=np.float32))


def load_fold_ids(fold_dir, fold):
    """Load train and test subject ID lists for the given fold index."""
    train_pkl = os.path.join(
        fold_dir, 'train_subject_allstudies_ids_dl_hmuse%d.pkl' % fold)
    test_pkl = os.path.join(
        fold_dir, 'test_subject_allstudies_ids_dl_hmuse%d.pkl' % fold)

    for p in (train_pkl, test_pkl):
        if not os.path.exists(p):
            raise FileNotFoundError(
                'Fold pkl not found: %s\n'
                'Expected files: train/test_subject_allstudies_ids_dl_hmuse{fold}.pkl' % p)

    with open(train_pkl, 'rb') as fh:
        train_ids = pickle.load(fh)
    with open(test_pkl, 'rb') as fh:
        test_ids = pickle.load(fh)

    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_stats(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def normalise(X, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_loader(X, Y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_mae = 0.0
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
        tr_loss         = train_epoch(model, loader_tr, optimizer, criterion, device)
        te_loss, te_mae = eval_epoch(model, loader_te, criterion, device)
        scheduler.step(te_loss)

        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'test_loss': te_loss, 'test_mae': te_mae})

        if te_mae < best_mae:
            best_mae = te_mae
            torch.save(model, best_path)

        if args.verbose and epoch % 10 == 0:
            print('  fold %d  epoch %3d/%d  [%.0fs]  '
                  'train_loss=%.4f  test_loss=%.4f  test_mae=%.4f'
                  % (fold_idx, epoch, args.epochs,
                     time.time() - t0, tr_loss, te_loss, te_mae))

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

    print('Loading CSV data from', args.data)
    xs, ys = load_csv(args.data)
    print('  Unique subjects in CSV:', len(xs))

    all_results = []
    for fold_idx in range(args.n_folds):
        train_ids, test_ids = load_fold_ids(args.fold_dir, fold_idx)

        X_tr_raw, Y_tr = subjects_to_arrays(train_ids, xs, ys)
        X_te_raw, Y_te = subjects_to_arrays(test_ids,  xs, ys)

        if X_tr_raw.shape[0] == 0:
            raise RuntimeError(
                'Fold %d: no training rows matched. '
                'Ensure PTID in the CSV matches the subject IDs in the pkl.' % fold_idx)
        if X_te_raw.shape[0] == 0:
            raise RuntimeError(
                'Fold %d: no test rows matched. '
                'Ensure PTID in the CSV matches the subject IDs in the pkl.' % fold_idx)

        # Normalise inputs using training-set statistics only
        mean, std = compute_stats(X_tr_raw)
        X_tr      = normalise(X_tr_raw, mean, std)
        X_te      = normalise(X_te_raw, mean, std)

        print('\n=== Fold %d/%d  train=%d visits (%d subjects)  '
              'test=%d visits (%d subjects) ==='
              % (fold_idx, args.n_folds - 1,
                 X_tr.shape[0], len(train_ids),
                 X_te.shape[0], len(test_ids)))

        best_mae, history = train_fold(
            fold_idx, X_tr, Y_tr, X_te, Y_te, args, device)

        # Save normalisation stats for inference
        stats_path = os.path.join(args.out_dir, 'fold%d_stats.json' % fold_idx)
        with open(stats_path, 'w') as fh:
            json.dump({'mean': mean.tolist(), 'std': std.tolist()}, fh)

        history_path = os.path.join(
            args.out_dir, 'fold%d_history.json' % fold_idx)
        with open(history_path, 'w') as fh:
            json.dump(history, fh)

        all_results.append({'fold': fold_idx, 'best_test_mae': best_mae})
        print('  Fold %d best test MAE: %.4f' % (fold_idx, best_mae))

    maes = [r['best_test_mae'] for r in all_results]
    summary = {
        'folds':    all_results,
        'mean_mae': float(np.mean(maes)),
        'std_mae':  float(np.std(maes)),
    }
    with open(os.path.join(args.out_dir, 'cv_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n=== %d-fold CV summary ===' % args.n_folds)
    print('  Mean MAE: %.4f  Std: %.4f' % (summary['mean_mae'], summary['std_mae']))
    print('  Results saved to', args.out_dir)


def get_args():
    parser = argparse.ArgumentParser(
        description='5-fold CV MLP training for 145-ROI regression '
                    'using pre-defined train/test subject splits')

    parser.add_argument('--data', required=True,
                        help='Path to subjectsamples_longclean_dl_muse_allstudies.csv')
    parser.add_argument('--fold_dir', default='.',
                        help='Directory containing '
                             'train/test_subject_allstudies_ids_dl_hmuse{fold}.pkl '
                             '(default: current directory)')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='Directory for checkpoints and logs')

    parser.add_argument('--n_folds',    type=int,   default=5)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--hidden_sizes', type=str, default='512,256',
                        help='Comma-separated hidden layer widths (default: 512,256)')
    parser.add_argument('--loss', choices=['mse', 'mae'], default='mse',
                        help='Regression loss function (default: mse)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
