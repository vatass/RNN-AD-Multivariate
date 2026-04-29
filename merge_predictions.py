"""
Merge predicted ROI trajectories from the RNN and MLP models into one CSV.

Each subject appears in exactly one test fold (5-fold CV), so pooling all
folds gives coverage of every subject.  The script verifies that both models
produce predictions for the same set of subjects before writing the output.

Expected directory layout
--------------------------
<rnn_dir>/
    fold_0/test_predictions.npy   (N_visits, 145)
    fold_0/test_targets.npy       (N_visits, 145)
    fold_0/test_ptids.json        [ptid, ptid, ...]  — one entry per visit row
    fold_1/ ... fold_4/ (same structure)

<mlp_dir>/
    fold_0/test_predictions.npy
    fold_0/test_targets.npy
    fold_0/test_ptids.json
    fold_1/ ... fold_4/

Output CSV columns
------------------
    PTID
    visit_idx          — 0-based visit index within subject
    ROI_1_rnn_pred, ROI_1_mlp_pred, ROI_1_real
    ROI_2_rnn_pred, ROI_2_mlp_pred, ROI_2_real
    ...
    ROI_145_rnn_pred, ROI_145_mlp_pred, ROI_145_real

Usage
-----
python merge_predictions.py \
    --rnn_dir results/rnn \
    --mlp_dir results/mlp \
    --out     results/merged_predictions.csv

python merge_predictions.py \
    --rnn_dir results/rnn \
    --mlp_dir results/mlp \
    --out     results/merged_predictions.csv \
    --n_folds 5 \
    --n_rois  145
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


N_ROIS_DEFAULT = 145
N_FOLDS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_fold(results_dir, fold, model_label):
    """Load predictions, targets and per-visit PTIDs for one fold of one model."""
    fold_dir = os.path.join(results_dir, 'fold_%d' % fold)

    pred_path  = os.path.join(fold_dir, 'test_predictions.npy')
    true_path  = os.path.join(fold_dir, 'test_targets.npy')
    ptids_path = os.path.join(fold_dir, 'test_ptids.json')

    missing = [p for p in (pred_path, true_path, ptids_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            '[%s fold %d] Missing files:\n  %s\n'
            'Run the training script first.' % (model_label, fold, '\n  '.join(missing)))

    preds = np.load(pred_path)   # (N_visits, 145)
    trues = np.load(true_path)   # (N_visits, 145)
    with open(ptids_path) as fh:
        ptids = json.load(fh)    # list of length N_visits

    if len(ptids) != preds.shape[0]:
        raise ValueError(
            '[%s fold %d] test_ptids.json has %d entries but '
            'test_predictions.npy has %d rows.'
            % (model_label, fold, len(ptids), preds.shape[0]))

    return preds, trues, ptids


def pool_folds(results_dir, n_folds, model_label):
    """
    Pool all folds into per-subject lists of (predictions, targets).

    Returns dict:  {ptid -> {'pred': list_of_visit_arrays,
                             'true': list_of_visit_arrays}}
    """
    by_subj = defaultdict(lambda: {'pred': [], 'true': []})
    subjects_seen = set()

    for fold in range(n_folds):
        preds, trues, ptids = load_fold(results_dir, fold, model_label)

        # Verify no subject appears in more than one fold
        fold_subjects = set(ptids)
        overlap = fold_subjects & subjects_seen
        if overlap:
            print('WARNING [%s]: subjects appear in multiple folds: %s'
                  % (model_label, overlap), file=sys.stderr)
        subjects_seen |= fold_subjects

        for visit_idx, ptid in enumerate(ptids):
            by_subj[ptid]['pred'].append(preds[visit_idx])
            by_subj[ptid]['true'].append(trues[visit_idx])

    return by_subj


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_coverage(rnn_subj, mlp_subj):
    rnn_set = set(rnn_subj.keys())
    mlp_set = set(mlp_subj.keys())

    only_rnn = rnn_set - mlp_set
    only_mlp = mlp_set - rnn_set

    if only_rnn:
        print('WARNING: %d subjects have RNN predictions but no MLP predictions.'
              % len(only_rnn), file=sys.stderr)
    if only_mlp:
        print('WARNING: %d subjects have MLP predictions but no RNN predictions.'
              % len(only_mlp), file=sys.stderr)

    common = rnn_set & mlp_set
    print('Subjects with predictions from both models: %d' % len(common))
    print('  RNN only : %d' % len(only_rnn))
    print('  MLP only : %d' % len(only_mlp))
    return common


# ---------------------------------------------------------------------------
# CSV assembly
# ---------------------------------------------------------------------------

def build_csv(rnn_subj, mlp_subj, common_subjects, n_rois, out_path):
    # Build column names
    roi_cols = []
    for r in range(1, n_rois + 1):
        roi_cols += ['ROI_%d_rnn_pred' % r,
                     'ROI_%d_mlp_pred' % r,
                     'ROI_%d_real'     % r]

    header = ['PTID', 'visit_idx'] + roi_cols

    n_rows = 0
    with open(out_path, 'w') as fh:
        fh.write(','.join(header) + '\n')

        for ptid in sorted(common_subjects, key=lambda x: str(x)):
            rnn_preds = rnn_subj[ptid]['pred']
            mlp_preds = mlp_subj[ptid]['pred']
            rnn_trues = rnn_subj[ptid]['true']
            mlp_trues = mlp_subj[ptid]['true']

            n_visits = len(rnn_preds)
            if len(mlp_preds) != n_visits:
                print('WARNING: subject %s has %d RNN visits but %d MLP visits — '
                      'using min.' % (ptid, n_visits, len(mlp_preds)),
                      file=sys.stderr)
                n_visits = min(n_visits, len(mlp_preds))

            for v in range(n_visits):
                rp = rnn_preds[v]
                mp = mlp_preds[v]
                # use RNN ground truth (both models train on same targets)
                gt = rnn_trues[v]

                row = [str(ptid), str(v)]
                for r in range(n_rois):
                    row += ['%.6f' % rp[r],
                            '%.6f' % mp[r],
                            '%.6f' % gt[r]]

                fh.write(','.join(row) + '\n')
                n_rows += 1

    return n_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Merge RNN and MLP predicted ROI trajectories into one CSV')
    parser.add_argument('--rnn_dir', required=True,
                        help='Root output directory from train_5fold_cv.py '
                             '(contains fold_0/ … fold_4/)')
    parser.add_argument('--mlp_dir', required=True,
                        help='Root output directory from train_mlp.py '
                             '(contains fold_0/ … fold_4/)')
    parser.add_argument('--out', required=True,
                        help='Path for the merged output CSV')
    parser.add_argument('--n_folds', type=int, default=N_FOLDS_DEFAULT)
    parser.add_argument('--n_rois',  type=int, default=N_ROIS_DEFAULT)
    args = parser.parse_args()

    print('Loading RNN predictions from', args.rnn_dir)
    rnn_subj = pool_folds(args.rnn_dir, args.n_folds, 'RNN')

    print('Loading MLP predictions from', args.mlp_dir)
    mlp_subj = pool_folds(args.mlp_dir, args.n_folds, 'MLP')

    common = verify_coverage(rnn_subj, mlp_subj)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print('Writing merged CSV to', args.out)
    n_rows = build_csv(rnn_subj, mlp_subj, common, args.n_rois, args.out)

    print('Done — %d rows written (%d subjects x visits).' % (n_rows, len(common)))


if __name__ == '__main__':
    main()
