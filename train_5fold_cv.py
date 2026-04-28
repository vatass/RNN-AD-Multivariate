"""
5-fold cross-validation for the VolumetricRNN model.

Each fold uses pre-defined train/test subject splits stored as pkl files:
  train_subject_allstudies_ids_dl_hmuse{fold}.pkl
  test_subject_allstudies_ids_dl_hmuse{fold}.pkl

Results for each fold are saved to results/fold_{fold}/:
  best_model.pt          — best model weights
  history.json           — per-epoch training history
  test_predictions.npy   — predicted volumes (N_visits, 145)
  test_targets.npy       — ground-truth volumes (N_visits, 145)
  per_timepoint_mae.npy  — MAE per visit index (max_T,)
  per_timepoint_mse.npy  — MSE per visit index (max_T,)
  per_region_mae.npy     — MAE averaged across all timepoints per region (145,)
  per_region_mse.npy     — MSE averaged across all timepoints per region (145,)
  metrics.json           — summary metrics for this fold
  subject_ids.json       — train/test subject IDs used

Usage
-----
python train_5fold_cv.py \
    --data subjectsamples_longclean_dl_muse_allstudies.csv \
    --fold_dir .  \
    --output_dir results

# Specific fold only
python train_5fold_cv.py --data ... --fold 2
"""

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
from torch.utils.data import DataLoader

from cbig.VolumetricRNN.dataset import VolumetricDataset, collate_fn
from cbig.VolumetricRNN.model import VolumetricRNN
from cbig.VolumetricRNN.train import run_epoch


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_subjects(csv_path):
    """
    Load every subject from the CSV and return a dict:
        {ptid_str: (X_seq, Y_seq)}
    where X_seq is (T, 151) and Y_seq is (T, 145), sorted by time (X[:, -1]).
    """
    rows_by_subj = defaultdict(list)
    with open(csv_path) as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            ptid = str(row[0])
            X = np.array(ast.literal_eval(row[1]), dtype=np.float32)
            Y = np.array(ast.literal_eval(row[2]), dtype=np.float32)
            rows_by_subj[ptid].append((X, Y))

    subjects = {}
    for ptid, visits in rows_by_subj.items():
        # sort by time (last element of X)
        visits.sort(key=lambda v: v[0][-1])
        X_seq = np.stack([v[0] for v in visits])   # (T, 151)
        Y_seq = np.stack([v[1] for v in visits])   # (T, 145)
        subjects[ptid] = (X_seq, Y_seq)
    return subjects


def filter_subjects(all_subjects, ids):
    """Return list of (X_seq, Y_seq) tuples for the given subject ID strings."""
    return [all_subjects[str(sid)] for sid in ids if str(sid) in all_subjects]


def normalize_time(subjects, max_time=None):
    """Normalize last feature of X (months) by max_time."""
    if max_time is None:
        max_time = max(float(xs[-1, -1]) for xs, _ in subjects)
    normalized = []
    for xs, ys in subjects:
        xs = xs.copy()
        xs[:, -1] = xs[:, -1] / max_time
        normalized.append((xs, ys))
    return normalized, float(max_time)


# ---------------------------------------------------------------------------
# Per-timepoint metric computation
# ---------------------------------------------------------------------------

def compute_per_timepoint_metrics(preds_by_subj, trues_by_subj):
    """
    Compute per-visit-index MAE and MSE.

    preds_by_subj : list of (T_i, 145) arrays
    trues_by_subj : list of (T_i, 145) arrays

    Returns:
        mae_per_tp  : (max_T,) — mean MAE across regions and subjects at each visit
        mse_per_tp  : (max_T,) — mean MSE across regions and subjects at each visit
        counts      : (max_T,) — number of subjects contributing to each visit index
    """
    max_T = max(p.shape[0] for p in preds_by_subj)
    mae_sum = np.zeros(max_T)
    mse_sum = np.zeros(max_T)
    counts  = np.zeros(max_T, dtype=int)

    for pred, true in zip(preds_by_subj, trues_by_subj):
        T = pred.shape[0]
        for t in range(T):
            mae_sum[t] += np.abs(pred[t] - true[t]).mean()
            mse_sum[t] += ((pred[t] - true[t]) ** 2).mean()
            counts[t] += 1

    valid = counts > 0
    mae_per_tp = np.where(valid, mae_sum / counts, np.nan)
    mse_per_tp = np.where(valid, mse_sum / counts, np.nan)
    return mae_per_tp, mse_per_tp, counts


# ---------------------------------------------------------------------------
# Single-fold training + evaluation
# ---------------------------------------------------------------------------

def run_fold(fold, train_subjects, test_subjects, output_dir, args, device):
    os.makedirs(output_dir, exist_ok=True)

    # Normalize time using training set max
    train_subjects, max_time = normalize_time(train_subjects)
    test_subjects, _ = normalize_time(test_subjects, max_time=max_time)

    # Build model
    model = VolumetricRNN(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)

    train_loader = DataLoader(
        VolumetricDataset(train_subjects),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0)

    # Use a small portion of training subjects as validation for early stopping
    n_val = max(1, int(0.1 * len(train_subjects)))
    val_subjects = train_subjects[-n_val:]
    train_subjects_fit = train_subjects[:-n_val]

    fit_loader = DataLoader(
        VolumetricDataset(train_subjects_fit),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(
        VolumetricDataset(val_subjects),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    best_val_loss = float('inf')
    no_improve = 0
    history = []
    best_ckpt = os.path.join(output_dir, 'best_model.pt')

    print(f"\n{'='*60}")
    print(f"Fold {fold}  |  {len(train_subjects)} train  {len(test_subjects)} test  "
          f"(val from train: {n_val})")
    print(f"Device: {device}  max_time: {max_time:.0f} months")
    print(f"{'='*60}")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = run_epoch(model, fit_loader, optimizer, device, train=True)
        vl_loss, vl_mae = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history.append({'epoch': epoch,
                        'train_loss': tr_loss, 'val_loss': vl_loss,
                        'train_mae': tr_mae,   'val_mae': vl_mae})

        print(f"Epoch {epoch:4d}/{args.max_epochs} | "
              f"train MSE={tr_loss:.4f} MAE={tr_mae:.4f} | "
              f"val MSE={vl_loss:.4f} MAE={vl_mae:.4f} | {elapsed:.1f}s")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(output_dir, 'history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    # ---- Evaluate on test set using best checkpoint ----
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    test_loader = DataLoader(
        VolumetricDataset(test_subjects),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    preds_by_subj, trues_by_subj = [], []
    all_pred_flat, all_true_flat = [], []

    with torch.no_grad():
        for X, Y, lengths in test_loader:
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            pred = model(X, lengths)
            for i, l in enumerate(lengths):
                l = l.item()
                p = pred[i, :l].cpu().numpy()
                t = Y[i, :l].cpu().numpy()
                preds_by_subj.append(p)
                trues_by_subj.append(t)
                all_pred_flat.append(p)
                all_true_flat.append(t)

    all_pred = np.concatenate(all_pred_flat, axis=0)   # (N_visits, 145)
    all_true = np.concatenate(all_true_flat, axis=0)

    overall_mae = float(np.abs(all_pred - all_true).mean())
    overall_mse = float(((all_pred - all_true) ** 2).mean())
    per_region_mae = np.abs(all_pred - all_true).mean(axis=0)    # (145,)
    per_region_mse = ((all_pred - all_true) ** 2).mean(axis=0)   # (145,)

    mae_per_tp, mse_per_tp, tp_counts = compute_per_timepoint_metrics(
        preds_by_subj, trues_by_subj)

    # Save arrays
    np.save(os.path.join(output_dir, 'test_predictions.npy'), all_pred)
    np.save(os.path.join(output_dir, 'test_targets.npy'), all_true)
    np.save(os.path.join(output_dir, 'per_timepoint_mae.npy'), mae_per_tp)
    np.save(os.path.join(output_dir, 'per_timepoint_mse.npy'), mse_per_tp)
    np.save(os.path.join(output_dir, 'per_region_mae.npy'), per_region_mae)
    np.save(os.path.join(output_dir, 'per_region_mse.npy'), per_region_mse)

    # Summary metrics
    metrics = {
        'fold': fold,
        'n_train': len(train_subjects),
        'n_test': len(test_subjects),
        'best_val_mse': float(best_val_loss),
        'overall_mae': overall_mae,
        'overall_mse': overall_mse,
        'per_region_mae_mean': float(per_region_mae.mean()),
        'per_region_mae_min':  float(per_region_mae.min()),
        'per_region_mae_max':  float(per_region_mae.max()),
        'per_region_mse_mean': float(per_region_mse.mean()),
        'per_timepoint_mae': {
            str(t): float(v)
            for t, v in enumerate(mae_per_tp) if not np.isnan(v)
        },
        'per_timepoint_mse': {
            str(t): float(v)
            for t, v in enumerate(mse_per_tp) if not np.isnan(v)
        },
        'per_timepoint_n_subjects': {
            str(t): int(c) for t, c in enumerate(tp_counts) if c > 0
        },
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nFold {fold} test results:")
    print(f"  Overall MAE : {overall_mae:.4f}")
    print(f"  Overall MSE : {overall_mse:.4f}")
    print(f"  Per-region MAE — mean: {per_region_mae.mean():.4f}  "
          f"min: {per_region_mae.min():.4f}  max: {per_region_mae.max():.4f}")
    print(f"  Per-timepoint MAE (visit 0..{len(mae_per_tp)-1}):")
    for t, (mae_t, mse_t, cnt) in enumerate(zip(mae_per_tp, mse_per_tp, tp_counts)):
        if cnt > 0:
            print(f"    visit {t:2d}: MAE={mae_t:.4f}  MSE={mse_t:.4f}  "
                  f"(n={cnt} subjects)")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='5-fold cross-validation for VolumetricRNN')
    p.add_argument('--data', required=True,
                   help='Path to subjectsamples_longclean_dl_muse_allstudies.csv')
    p.add_argument('--fold_dir', default='.',
                   help='Directory containing the fold pkl files (default: .)')
    p.add_argument('--output_dir', default='results',
                   help='Root directory for results (default: results/)')
    p.add_argument('--fold', type=int, default=None,
                   help='Run a single fold 0-4 (default: run all 5)')

    # Model hyper-parameters
    p.add_argument('--rnn_type',    default='LSTM', choices=['LSTM', 'GRU'])
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--n_layers',    type=int, default=2)
    p.add_argument('--dropout',     type=float, default=0.3)

    # Training hyper-parameters
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--max_epochs',    type=int,   default=200)
    p.add_argument('--patience',      type=int,   default=20,
                   help='Early stopping patience (epochs)')

    p.add_argument('--device', default='auto',
                   help='"auto" selects GPU if available, else CPU')
    return p.parse_args()


def main():
    args = parse_args()
    # Fixed input/output sizes from the dataset
    args.input_size  = 151
    args.output_size = 145

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Loading all subjects from {args.data} ...")
    all_subjects = load_all_subjects(args.data)
    print(f"Loaded {len(all_subjects)} unique subjects")

    folds = [args.fold] if args.fold is not None else list(range(5))
    all_metrics = []

    for fold in folds:
        train_pkl = os.path.join(
            args.fold_dir, f'train_subject_allstudies_ids_dl_hmuse{fold}.pkl')
        test_pkl  = os.path.join(
            args.fold_dir, f'test_subject_allstudies_ids_dl_hmuse{fold}.pkl')

        with open(train_pkl, 'rb') as fh:
            train_ids = pickle.load(fh)
        with open(test_pkl, 'rb') as fh:
            test_ids = pickle.load(fh)

        train_subjects = filter_subjects(all_subjects, train_ids)
        test_subjects  = filter_subjects(all_subjects, test_ids)

        fold_dir = os.path.join(args.output_dir, f'fold_{fold}')

        # Save subject ID lists for reproducibility
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'subject_ids.json'), 'w') as fh:
            json.dump({'train': [str(i) for i in train_ids],
                       'test':  [str(i) for i in test_ids]}, fh, indent=2)

        metrics = run_fold(fold, train_subjects, test_subjects,
                           fold_dir, args, device)
        all_metrics.append(metrics)

    # Aggregate summary across folds
    if len(all_metrics) > 1:
        mae_vals = [m['overall_mae'] for m in all_metrics]
        mse_vals = [m['overall_mse'] for m in all_metrics]
        summary = {
            'folds_run': folds,
            'mae_per_fold': mae_vals,
            'mse_per_fold': mse_vals,
            'mae_mean': float(np.mean(mae_vals)),
            'mae_std':  float(np.std(mae_vals)),
            'mse_mean': float(np.mean(mse_vals)),
            'mse_std':  float(np.std(mse_vals)),
        }
        summary_path = os.path.join(args.output_dir, 'cv_summary.json')
        with open(summary_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n{'='*60}")
        print(f"5-fold CV summary")
        print(f"  MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
        print(f"  MSE: {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
        print(f"  Saved to {summary_path}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
