"""
Entry point for training the volumetric trajectory RNN.

Usage
-----
# Train with defaults (auto-detects GPU if available)
python train_volumetric.py --data subjectsamples_longclean_dl_muse_allstudies.csv

# Custom hyperparameters
python train_volumetric.py \
    --data subjectsamples_longclean_dl_muse_allstudies.csv \
    --hidden_size 512 \
    --n_layers 3 \
    --dropout 0.4 \
    --lr 5e-4 \
    --batch_size 32 \
    --max_epochs 300 \
    --output_dir results/run1

# Use GRU instead of LSTM
python train_volumetric.py --data ... --rnn_type GRU
"""

import argparse
import json
import os
import torch

from cbig.VolumetricRNN.dataset import load_subjects, normalize_time, split_subjects
from cbig.VolumetricRNN.train import train, evaluate


def parse_args():
    p = argparse.ArgumentParser(description='Train volumetric trajectory RNN')
    p.add_argument('--data', required=True,
                   help='Path to subjectsamples_longclean_dl_muse_allstudies.csv')
    p.add_argument('--output_dir', default='results/volumetric_rnn',
                   help='Directory to save model, history, and predictions')
    p.add_argument('--rnn_type', default='LSTM', choices=['LSTM', 'GRU'])
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (epochs)')
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   help='"auto" picks GPU if available, else CPU')
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Loading data from {args.data} ...")
    subjects = load_subjects(args.data)
    print(f"Loaded {len(subjects)} subjects")

    train_s, val_s, test_s = split_subjects(
        subjects, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test subjects")

    # Normalize time using training set max
    train_s, max_time = normalize_time(train_s)
    val_s, _ = normalize_time(val_s, max_time=max_time)
    test_s, _ = normalize_time(test_s, max_time=max_time)
    print(f"Time normalized by max={max_time:.0f} months")

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    config['device_used'] = device
    config['max_time'] = float(max_time)
    config['n_train'] = len(train_s)
    config['n_val'] = len(val_s)
    config['n_test'] = len(test_s)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    model, history = train(
        train_subjects=train_s,
        val_subjects=val_s,
        output_dir=args.output_dir,
        input_size=151,
        hidden_size=args.hidden_size,
        output_size=145,
        n_layers=args.n_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        device=device,
    )

    # Evaluate on test set using best checkpoint
    best_ckpt = os.path.join(args.output_dir, 'best_model.pt')
    metrics = evaluate(
        model_or_path=best_ckpt,
        test_subjects=test_s,
        output_dir=args.output_dir,
        input_size=151,
        hidden_size=args.hidden_size,
        output_size=145,
        n_layers=args.n_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        device=device,
    )

    print("\nDone. Results saved to:", args.output_dir)
    print("Test metrics:", metrics)


if __name__ == '__main__':
    main()
