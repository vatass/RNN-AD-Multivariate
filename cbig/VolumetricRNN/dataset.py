"""
Data loading and batching for the volumetric trajectory prediction task.

Each row in the CSV corresponds to one visit for one subject:
  PTID  : subject identifier
  X     : list of 151 floats — [145 baseline volumes, Sex, Age, ?, ?, DX, months_since_baseline]
  Y     : list of 145 floats — brain volumes at this visit (z-scored)

Subjects are grouped into sequences ordered by time (months_since_baseline).
"""

import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_subjects(csv_path):
    """Return list of (X_seq, Y_seq) arrays, one per subject, sorted by time."""
    df = pd.read_csv(csv_path)
    df['X_parsed'] = df['X'].apply(ast.literal_eval)
    df['Y_parsed'] = df['Y'].apply(ast.literal_eval)
    df['time'] = df['X_parsed'].apply(lambda x: x[-1])

    subjects = []
    for _, group in df.groupby('PTID'):
        group = group.sort_values('time')
        X_seq = np.array(group['X_parsed'].tolist(), dtype=np.float32)
        Y_seq = np.array(group['Y_parsed'].tolist(), dtype=np.float32)
        subjects.append((X_seq, Y_seq))
    return subjects


def normalize_time(subjects, max_time=None):
    """
    Normalize the last feature in X (months) to [0, 1].
    If max_time is None it is computed from the given subjects (use training set).
    Returns (normalized_subjects, max_time).
    """
    if max_time is None:
        max_time = max(float(xs[-1, -1]) for xs, _ in subjects)
    normalized = []
    for xs, ys in subjects:
        xs = xs.copy()
        xs[:, -1] = xs[:, -1] / max_time
        normalized.append((xs, ys))
    return normalized, max_time


def split_subjects(subjects, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split subjects into train / val / test by subject ID (not by visit)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(subjects))
    n_test = max(1, int(len(subjects) * test_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    test = [subjects[i] for i in idx[:n_test]]
    val = [subjects[i] for i in idx[n_test:n_test + n_val]]
    train = [subjects[i] for i in idx[n_test + n_val:]]
    return train, val, test


class VolumetricDataset(Dataset):
    def __init__(self, subjects):
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        xs, ys = self.subjects[idx]
        return torch.from_numpy(xs), torch.from_numpy(ys)


def collate_fn(batch):
    """Pad variable-length sequences and return lengths for packing."""
    X_list, Y_list = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in X_list], dtype=torch.long)
    X_padded = pad_sequence(X_list, batch_first=True)   # (B, T_max, 151)
    Y_padded = pad_sequence(Y_list, batch_first=True)   # (B, T_max, 145)
    return X_padded, Y_padded, lengths
