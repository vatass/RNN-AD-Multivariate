#!/usr/bin/env python
"""5-fold cross-validation training of an MLP for 145-ROI regression.

Data format (subjectsamples_longclean_dl_muse_allstudies.csv):
    PTID  : subject identifier matching the RID used in the TADPOLE pkl files
    X     : 151-dimensional embedding stored as a Python list literal
    Y     : 145-dimensional ROI target stored as a Python list literal

Train/test splits are taken from the pre-generated pkl files produced by
gen_cv_pickle.py.  Each pkl contains a 'train' and 'test' dataloader whose
.subjects attribute lists the RIDs belonging to that split.  The MLP uses
those same RIDs to select rows from the CSV so results are directly
comparable to the RNN-AD experiments.

Usage example (5 folds, pkl files named fold0.pkl … fold4.pkl):

    python -m cbig.Nguyen2020.train_mlp \
        --data  subjectsamples_longclean_dl_muse_allstudies.csv \
        --pkl_dir output/ \
        --pkl_pattern "fold{fold}.pkl" \
        --n_folds 5 \
        --out_dir output/mlp_cv \
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cbig.Nguyen2020.mlp_model import MLPRegressor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load CSV and return dicts keyed by PTID.

    Returns:
        xs  : dict {ptid -> list[float]}  (151-dim)
        ys  : dict {ptid -> list[list[float]]}  one entry per timepoint
              (multiple rows per subject are collected as a list)
    """
    xs, ys = {}, {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            pid = str(row['PTID'])
            x   = ast.literal_eval(row['X'])
            y   = ast.literal_eval(row['Y'])
            if pid not in xs:
                xs[pid] = []
                ys[pid] = []
            xs[pid].append(x)
            ys[pid].append(y)
    return xs, ys


def subjects_to_arrays(subject_ids, xs, ys):
    """Flatten all timepoints for the given subjects into (X, Y) arrays."""
    X_rows, Y_rows = [], []
    for sid in subject_ids:
        key = str(sid)
        if key not in xs:
            continue
        X_rows.extend(xs[key])
        Y_rows.extend(ys[key])
    return (np.array(X_rows, dtype=np.float32),
            np.array(Y_rows, dtype=np.float32))


# ---------------------------------------------------------------------------
# Load subject lists from a pkl file
# ---------------------------------------------------------------------------

def load_subjects_from_pkl(pkl_path):
    """Return (train_subjects, test_subjects) from a gen_cv_pickle pkl file."""
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    train_subj = list(data['train'].subjects)
    test_subj  = list(data['test'].subjects)
    return train_subj, test_subj


# ---------------------------------------------------------------------------
# Normalisation helpers
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
        pkl_name = args.pkl_pattern.format(fold=fold_idx)
        pkl_path = os.path.join(args.pkl_dir, pkl_name)

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                'pkl file not found: %s\n'
                'Generate it first with gen_cv_pickle.py' % pkl_path)

        train_subj, test_subj = load_subjects_from_pkl(pkl_path)

        X_tr_raw, Y_tr = subjects_to_arrays(train_subj, xs, ys)
        X_te_raw, Y_te = subjects_to_arrays(test_subj,  xs, ys)

        if X_tr_raw.shape[0] == 0:
            raise RuntimeError(
                'Fold %d: no training rows found. '
                'Check that PTID in the CSV matches RID in the pkl.' % fold_idx)
        if X_te_raw.shape[0] == 0:
            raise RuntimeError(
                'Fold %d: no test rows found. '
                'Check that PTID in the CSV matches RID in the pkl.' % fold_idx)

        # normalise inputs using training-set statistics only
        mean, std = compute_stats(X_tr_raw)
        X_tr      = normalise(X_tr_raw, mean, std)
        X_te      = normalise(X_te_raw, mean, std)

        print('\n=== Fold %d/%d  (pkl: %s)  train=%d  test=%d ==='
              % (fold_idx, args.n_folds - 1, pkl_name,
                 X_tr.shape[0], X_te.shape[0]))

        best_mae, history = train_fold(
            fold_idx, X_tr, Y_tr, X_te, Y_te, args, device)

        # save normalisation stats for inference
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
    summary_path = os.path.join(args.out_dir, 'cv_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n=== %d-fold CV summary ===' % args.n_folds)
    print('  Mean MAE: %.4f  Std: %.4f' % (summary['mean_mae'], summary['std_mae']))
    print('  Results saved to', args.out_dir)


def get_args():
    parser = argparse.ArgumentParser(
        description='5-fold CV MLP training for 145-ROI regression '
                    'using pre-defined pkl train/test splits')

    parser.add_argument('--data', required=True,
                        help='Path to subjectsamples_longclean_dl_muse_allstudies.csv')
    parser.add_argument('--pkl_dir', required=True,
                        help='Directory containing the fold pkl files '
                             '(produced by gen_cv_pickle.py)')
    parser.add_argument('--pkl_pattern', default='fold{fold}.pkl',
                        help='Filename pattern for pkl files, '
                             '{fold} is replaced by the fold index '
                             '(default: fold{fold}.pkl)')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='Directory for checkpoints and logs')

    parser.add_argument('--n_folds',    type=int,   default=5)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--hidden_sizes', type=str, default='512,256',
                        help='Comma-separated hidden layer widths')
    parser.add_argument('--loss', choices=['mse', 'mae'], default='mse',
                        help='Regression loss: mse or mae (L1)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
