"""
Training and evaluation routines for the volumetric RNN.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import VolumetricDataset, collate_fn
from .model import VolumetricRNN


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores padded positions."""
    B, T, D = pred.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=pred.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    mask = mask.unsqueeze(-1).expand_as(pred)
    return nn.functional.mse_loss(pred[mask], target[mask])


def masked_mae(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> float:
    """MAE averaged over all valid (unpadded) positions — returned as a Python float."""
    B, T, D = pred.shape
    total_err, total_n = 0.0, 0
    for i, l in enumerate(lengths):
        l = l.item()
        err = (pred[i, :l] - target[i, :l]).abs().mean().item()
        total_err += err * l
        total_n += l
    return total_err / max(total_n, 1)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_mae, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for X, Y, lengths in loader:
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            pred = model(X, lengths)
            loss = masked_mse(pred, Y, lengths)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            total_mae += masked_mae(pred.detach(), Y, lengths)
            n_batches += 1
    return total_loss / n_batches, total_mae / n_batches


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    train_subjects,
    val_subjects,
    output_dir: str,
    input_size: int = 151,
    hidden_size: int = 256,
    output_size: int = 145,
    n_layers: int = 2,
    dropout: float = 0.3,
    rnn_type: str = 'LSTM',
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    num_workers: int = 0,
    device: str = 'cpu',
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    train_ds = VolumetricDataset(train_subjects)
    val_ds = VolumetricDataset(val_subjects)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    model = VolumetricRNN(input_size, hidden_size, output_size, n_layers, dropout, rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    print(f"Training {rnn_type} | hidden={hidden_size} layers={n_layers} | "
          f"{len(train_subjects)} train / {len(val_subjects)} val subjects")
    print(f"Device: {device}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'train_mae': train_mae, 'val_mae': val_mae})

        print(f"Epoch {epoch:4d}/{max_epochs} | "
              f"train MSE={train_loss:.4f} MAE={train_mae:.4f} | "
              f"val MSE={val_loss:.4f} MAE={val_mae:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MSE: {best_val_loss:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Evaluation on a held-out set
# ---------------------------------------------------------------------------

def evaluate(model_or_path, test_subjects, output_dir: str,
             input_size=151, hidden_size=256, output_size=145,
             n_layers=2, dropout=0.3, rnn_type='LSTM',
             batch_size=64, device='cpu'):
    """Load (or use) a model and report per-trajectory MAE on the test set."""
    device = torch.device(device)

    if isinstance(model_or_path, str):
        model = VolumetricRNN(input_size, hidden_size, output_size, n_layers, dropout, rnn_type)
        model.load_state_dict(torch.load(model_or_path, map_location=device))
    else:
        model = model_or_path
    model.to(device).eval()

    test_ds = VolumetricDataset(test_subjects)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    all_pred, all_true = [], []
    with torch.no_grad():
        for X, Y, lengths in test_loader:
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            pred = model(X, lengths)
            for i, l in enumerate(lengths):
                l = l.item()
                all_pred.append(pred[i, :l].cpu().numpy())
                all_true.append(Y[i, :l].cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)   # (N_visits, 145)
    all_true = np.concatenate(all_true, axis=0)

    overall_mae = np.abs(all_pred - all_true).mean()
    per_region_mae = np.abs(all_pred - all_true).mean(axis=0)   # (145,)

    print(f"\nTest set  |  overall MAE: {overall_mae:.4f}")
    print(f"Per-region MAE — min: {per_region_mae.min():.4f}  "
          f"max: {per_region_mae.max():.4f}  mean: {per_region_mae.mean():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_predictions.npy'), all_pred)
    np.save(os.path.join(output_dir, 'test_targets.npy'), all_true)
    np.save(os.path.join(output_dir, 'per_region_mae.npy'), per_region_mae)

    metrics = {'overall_mae': float(overall_mae),
               'per_region_mae_mean': float(per_region_mae.mean()),
               'per_region_mae_min': float(per_region_mae.min()),
               'per_region_mae_max': float(per_region_mae.max())}
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics
