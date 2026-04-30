# svdk_regression_multitask.py
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import pandas as pd
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from functions import process_temporal_singletask_data
import json
import pickle
import argparse
import os
import matplotlib.pyplot as plt 
import datetime
# Define Regions ROIs
import json
from collections import OrderedDict
torch.set_default_dtype(torch.float64)
import math
# --- 1) Define ROIs per region ---
REGION_ROIS = OrderedDict({
    0: ("Limbic system", [
        "MUSE_Volume_100","MUSE_Volume_101","MUSE_Volume_116","MUSE_Volume_117","MUSE_Volume_138",
        "MUSE_Volume_139","MUSE_Volume_166","MUSE_Volume_167","MUSE_Volume_170","MUSE_Volume_171",
    ]),
    1: ("Parietal lobe", [
        "MUSE_Volume_85","MUSE_Volume_86","MUSE_Volume_106","MUSE_Volume_107","MUSE_Volume_148","MUSE_Volume_149",
        "MUSE_Volume_168","MUSE_Volume_169","MUSE_Volume_176","MUSE_Volume_177","MUSE_Volume_194","MUSE_Volume_195",
        "MUSE_Volume_198","MUSE_Volume_199",
    ]),
    2: ("Ventricular system", [
        "MUSE_Volume_4","MUSE_Volume_11", "MUSE_Volume_49","MUSE_Volume_50","MUSE_Volume_51","MUSE_Volume_52",
    ]),
    3: ("Temporal lobe", [
        "MUSE_Volume_87","MUSE_Volume_88","MUSE_Volume_122","MUSE_Volume_123",
        "MUSE_Volume_132","MUSE_Volume_133","MUSE_Volume_154","MUSE_Volume_155","MUSE_Volume_180","MUSE_Volume_181",
        "MUSE_Volume_184","MUSE_Volume_185","MUSE_Volume_200","MUSE_Volume_201","MUSE_Volume_202","MUSE_Volume_203",
        "MUSE_Volume_206", "MUSE_Volume_207"
    ]),
    4: ("Occipital lobe", [
        "MUSE_Volume_83","MUSE_Volume_84","MUSE_Volume_108","MUSE_Volume_109","MUSE_Volume_114","MUSE_Volume_115",
        "MUSE_Volume_128","MUSE_Volume_129","MUSE_Volume_134","MUSE_Volume_135","MUSE_Volume_144","MUSE_Volume_145",
        "MUSE_Volume_156","MUSE_Volume_157","MUSE_Volume_160","MUSE_Volume_161","MUSE_Volume_196","MUSE_Volume_197",
    ]),
    5: ("Frontal lobe", [
        "MUSE_Volume_81", "MUSE_Volume_82", "MUSE_Volume_102", "MUSE_Volume_103", "MUSE_Volume_104", "MUSE_Volume_105",
        "MUSE_Volume_112", "MUSE_Volume_113", "MUSE_Volume_118", "MUSE_Volume_119", "MUSE_Volume_120", "MUSE_Volume_121",
        "MUSE_Volume_124", "MUSE_Volume_125", "MUSE_Volume_136", "MUSE_Volume_137", "MUSE_Volume_140", "MUSE_Volume_141", 
        "MUSE_Volume_142", "MUSE_Volume_143", "MUSE_Volume_146", "MUSE_Volume_147", "MUSE_Volume_150", "MUSE_Volume_151",
        "MUSE_Volume_152", "MUSE_Volume_153", "MUSE_Volume_162", "MUSE_Volume_163", "MUSE_Volume_164", "MUSE_Volume_165",
        "MUSE_Volume_172", "MUSE_Volume_173", "MUSE_Volume_174", "MUSE_Volume_175", "MUSE_Volume_178", "MUSE_Volume_179",
        "MUSE_Volume_182", "MUSE_Volume_183", "MUSE_Volume_186", "MUSE_Volume_187", "MUSE_Volume_190", "MUSE_Volume_191",
        "MUSE_Volume_192", "MUSE_Volume_193", "MUSE_Volume_204", "MUSE_Volume_205"
    ])
})
def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional but nice for full reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_region_y_indices_and_names(mode: int, roi_to_idx: dict):
    if mode not in REGION_ROIS:
        raise ValueError(f"Mode {mode} not supported")
    
    region_name, rois = REGION_ROIS[mode]
    roi_ids = []
    for roi in rois:
        if isinstance(roi, (list, tuple)):
            iterable = roi
        else:
            iterable = [roi]
        
        for r in iterable:
            r = str(r).strip()
            if r.startswith("MUSE_Volume_"):
                r = r.split("_")[-1]
            roi_ids.append(r)

    missing = [rid for rid in roi_ids if rid not in roi_to_idx]
    print(roi_to_idx)
    if missing:
        raise KeyError(f"These ROIS are missing: {missing}")
    
    idxs = [int(roi_to_idx[rid]) for rid in roi_ids]

    task_names = [f"ROI_{rid}" for rid in roi_ids]
    
    return region_name, idxs, task_names

def select_region_targets(Y: np.ndarray, mode: int, roi_to_idx: dict):

    if Y.ndim != 2 or Y.shape[1] != 145:
        raise ValueError(f"Expected Y shape (n,145). Got {Y.shape}")
    idxs = get_region_y_indices(mode, roi_to_idx)
    return Y[:, idxs]


# Step 1: Define the CognitiveDataset class
class CognitiveDataset(Dataset):
    def __init__(self, inputs, targets, subject_ids):
        """
        inputs: NumPy array of input features, shape (num_samples, input_dim)
        targets: NumPy array of target values, shape (num_samples,)
        subject_ids: List or array of subject IDs, length num_samples
        """
        assert len(inputs) == len(targets) == len(subject_ids), "Inputs, targets, and subject_ids must have the same length."

        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)
        self.subject_ids = subject_ids  # List or array of subject IDs

        # Create a mapping from subject ID to indices
        self.subject_to_indices = {}
        for idx, subject_id in enumerate(self.subject_ids):
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.subject_ids[idx]

# Step 2: Define the FeatureExtractor class
# Separate Processing and Addition
# Feature Extractor: Latent Concatenation of Time and Imaging Features
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))

class FeatureExtractorLatentConcatenation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractorLatentConcatenation, self).__init__()
        # Assuming the last feature is time
        self.imaging_dim = input_dim - 1
        self.time_dim = 1
        self.hidden_dim = hidden_dim

        # Layers for imaging features
        self.imaging_fc = nn.Sequential(
            nn.Linear(self.imaging_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Layers for time feature
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

        # Final layers after concatenation
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # You can add more layers if needed
        )

    def forward(self, x):
        x_imaging = x[:, :-1]  # All features except the last one
        x_time = x[:, -1].unsqueeze(1)  # The last feature (time)

        # Process imaging features
        imaging_features = self.imaging_fc(x_imaging)

        # Process time feature
        time_features = self.time_fc(x_time)

        # Concatenate imaging and time features
        combined_features = torch.cat([imaging_features, time_features], dim=1)

        # Final processing
        output = self.fc_combined(combined_features)
        return output

class RegressionNNLatentConcatenation(nn.Module):
    def __init__(self, feature_extractor, output_dim=4):
        super(RegressionNNLatentConcatenation, self).__init__()
        self.feature_extractor = feature_extractor
    
        # The output dimension of feature_extractor is hidden_dim
        self.fc_out = nn.Linear(feature_extractor.hidden_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc_out(features)
        return output


# Step 3: Define the Multitask GP Regression Model
class MultitaskDeepKernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, feature_extractor):
        batch_shape = torch.Size([num_tasks])  # For multitask GPs
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=batch_shape
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_variational_strategy, num_tasks=num_tasks
        )
        super(MultitaskDeepKernelGPModel, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor  # Shared feature extractor
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 5: Define the GPModelWrapper
class GPModelWrapper(nn.Module):
    def __init__(self, gp_model, likelihood):
        super(GPModelWrapper, self).__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood

    def forward(self, x):
        return self.gp_model(x)

# Step 6: SubjectBatchSampler remains the same
class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        batch = []
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            batch.extend(indices)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class TestSubjectBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.subject_ids = list(dataset.subject_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_ids)
        for subject_id in self.subject_ids:
            indices = self.dataset.subject_to_indices[subject_id]
            yield indices  # Yield all indices for the current subject

    def __len__(self):
        return len(self.subject_ids)

# Step 7: Define the select_inducing_points function
def select_inducing_points(train_x, train_subject_ids, selected_subject_ids=None, num_points_per_subject=3, device='cuda'):
    """
    Selects inducing points by sampling observations from selected subjects.

    Parameters:
    - train_x: NumPy array of shape (num_samples, input_dim)
    - train_subject_ids: List or array of subject IDs corresponding to each row in train_x
    - selected_subject_ids: List of subject IDs from which to select inducing points (optional)
    - num_points_per_subject: Number of inducing points to select per subject

    Returns:
    - inducing_points: Torch tensor of selected inducing points
    """
    import pandas as pd
    import numpy as np

    # Determine the expected number of features
    expected_num_features = train_x.shape[1]

    # Create a DataFrame for easier manipulation
    data = pd.DataFrame(train_x)
    data['PTID'] = train_subject_ids

    inducing_points_list = []

    if selected_subject_ids is not None:
        # Filter data to include only selected subjects
        data_selected = data[data['PTID'].isin(selected_subject_ids)]
    else:
        data_selected = data

    # Group by subject
    grouped = data_selected.groupby('PTID')

    for subject_id, group in grouped:
        # Sort the group by temporal variable (assuming it's the last column minus one)
        temporal_col_index = expected_num_features - 1  # Adjust index if necessary
        group_sorted = group.sort_values(by=group.columns[temporal_col_index])
        num_observations = group_sorted.shape[0]

        if num_observations >= num_points_per_subject:
            # Select earliest, median, and latest time points
            indices = [0, num_observations // 2, num_observations - 1]
            selected_points = group_sorted.iloc[indices]
        else:
            # If less than num_points_per_subject observations, select all
            selected_points = group_sorted

        # Drop 'PTID' column and convert to values
        selected_values = selected_points.drop('PTID', axis=1).values

        if selected_values.size > 0:
            if selected_values.shape[1] != expected_num_features:
                print(f"Warning: Subject {subject_id} has unexpected number of features: {selected_values.shape[1]} (expected {expected_num_features}).")
                continue  # Skip this subject or handle as appropriate

            # Ensure numerical data type
            selected_values = selected_values.astype(np.float64)
            inducing_points_list.append(selected_values)
        else:
            print(f"Warning: No inducing points for subject {subject_id}.")

    if inducing_points_list:
        inducing_points_array = np.vstack(inducing_points_list)
        inducing_points = torch.tensor(inducing_points_array, dtype=torch.float64)
        inducing_points = inducing_points.to(device)
    else:
        raise ValueError("No inducing points were selected. Check your data and selection criteria.")

    return inducing_points

def load_and_preprocess_region_based_data(folder, file, train_ids, test_ids, mode: int):
    f = open('/home/cbica/Desktop/LongGPClustering/roi_to_idx.json')
    roi_to_idx = json.load(f)

    index_to_roi = {v: k for k, v in roi_to_idx.items()}
 
    datasamples = pd.read_csv('/home/cbica/Desktop/DKGP/data/subjectsamples_longclean_dl_muse_allstudies.csv')

    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    test_x  = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y  = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].tolist()
    corresponding_test_ids  = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].tolist()

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids
    )

    train_x = train_x.numpy()
    train_y = train_y.numpy()  # (n_train, 145)
    test_x  = test_x.numpy()
    test_y  = test_y.numpy()   # (n_test, 145)

    region_name, idxs, task_names = get_region_y_indices_and_names(mode, roi_to_idx)
    train_y_region = train_y[:, idxs]
    test_y_region = test_y[:, idxs]

    print("NaNs per task:", np.isnan(train_y_region).sum(axis=0))
    print("std per task:", train_y_region.std(axis=0))
    print("mix/max per task:", train_y_region.min(axis=0), train_y_region.max(axis=0))

    y_mean = train_y_region.mean(axis=0, keepdims=True)
    y_std = train_y_region.std(axis=0, keepdims=True)

    num_outputs = len(task_names)

    print("Shape of train_x:", train_x.shape)
    print("Shape of train_y:", train_y_region.shape)

    return train_x, train_y_region, test_x, test_y_region,  corresponding_train_ids, corresponding_test_ids, num_outputs, task_names, region_name


def collate_fn(batch):
    # 'batch' is a list of samples, where each sample is a tuple (input, target, subject_id)
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]  # Keep as list or convert to tensor if numeric
    
    return inputs, targets, subject_ids

def _mse_mae_per_task(actuals_list, preds_list):
    mse_per_task = []
    mae_per_task = []
    for k in range(len(actuals_list)):
        mse_per_task.append(mean_squared_error(actuals_list[k], preds_list[k]))
        mae_per_task.append(mean_absolute_error(actuals_list[k], preds_list[k]))
    return mse_per_task, mae_per_task, float(np.mean(mse_per_task)), float(np.mean(mae_per_task))
    
#Plot trajectories
def _safe_filename(s:str) -> str:
    s = str(s)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)

def _ensure_task_plot_dirs(root_dir:str, num_tasks:int):
    task_dirs=[]
    os.makedirs(root_dir, exist_ok=True)
    for k in range(num_tasks):
        d = os.path.join(root_dir, f"task{k+1}")
        os.makedirs(d, exist_ok=True)
        task_dirs.append(d)
    return task_dirs

def _transpose_task_matrix(x, num_tasks:int):
    if x is None:
        return None
    if x.ndim != 2:
        return x
    if x.shape[0] == num_tasks and x.shape[1] != num_tasks:
        return x.T 
    return x

def save_subject_trajectory_plots(
    subject_id,
    t_np,
    y_true_np,
    y_pred_np,
    y_var_np,
    task_dirs,
    task_names=None,
    dpi=300,
    save_csv=True
    ):

    order = np.argsort(t_np)
    t = t_np[order]
    y_true = y_true_np[order, :]
    y_pred = y_pred_np[order, :]
    y_var = y_var_np[order, :] if y_var_np is not None else None

    subj_safe = _safe_filename(subject_id)
    num_tasks = y_true.shape[1]

    for k in range(num_tasks):
        fig = plt.figure(figsize=(7,4))

        plt.plot(t, y_true[:, k], marker='o', linewidth=2, label="Ground Truth")

        plt.plot(t, y_pred[:, k], marker='x', linestyle="--", linewidth=2, label="Predicted Trajectory")

        #Uncertainty 
        y_std = None
        y_lower = None
        y_upper = None
        if y_var is not None:
            y_std = np.sqrt(np.maximum(y_var[:, k], 0.0))
            y_lower = y_pred[:, k] - 2.0 * y_std
            y_upper = y_pred[:, k] + 2.0 * y_std
            plt.fill_between(t, y_lower, y_upper, alpha=0.2)

        task_label = (task_names[k] if task_names is not None else f"Task {k+1}")
        plt.title(f"Subject {subject_id} | {task_label}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        #Save
        out_png = os.path.join(task_dirs[k], f"{subj_safe}_task{k+1}_csv")
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)

        if save_csv:
            out_csv = os.path.join(task_dirs[k], f"{subj_safe}_task{k+1}.csv")
            if y_std is None:
                df = pd.DataFrame({
                    "time": t,
                    "y_true": y_true[:, k],
                    "y_pred": y_pred[:, k],
                })
            else:
                df = pd.DataFrame({
                    "time": t,
                    "y_true": y_true[:, k],
                    "y_pred": y_pred[:, k],
                    "y_std": y_std,
                    "y_lower_2std": y_lower,
                    "y_upper_2std": y_upper,
                })
            df.to_csv(out_csv, index=False)

# Added metrics
def gaussian_nll_per_task(y_true, mean, var, eps=1e-9):
    var = np.maximum(var, eps)
    nll = 0.5 * (np.log(2.0 * math.pi * var) + ((y_true - mean) ** 2) / var)
    nll_per_task = np.mean(nll, axis=0)
    return nll_per_task, float(np.mean(nll_per_task))

def coverage_and_width_per_task(y_true, mean, var, z=1.96, eps=1e-9):
    std = np.sqrt(np.maximum(var, eps))
    lower = mean - z * std
    upper = mean + z * std
    within = (y_true >= lower) & (y_true <= upper)
    coverage_per_task = 100.0 * np.mean(within, axis=0)
    width_per_task = np.mean(upper - lower, axis=0)
    return coverage_per_task, width_per_task, float(np.mean(coverage_per_task)), float(np.mean(width_per_task))

# Step 8: Main function
def main():
    #set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 0
    parser = argparse.ArgumentParser(description='Multi-output GP Regression and Classification for Neurodegeneration Prediction')
    # Data Parameters
    parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa')
    parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_mmse_dlmuse_allstudies")
    parser.add_argument("--folder", type=int, default=2)
    parser.add_argument("--sigma", type=float, nargs="+", default=None, help="Per-task monotonic direction. 1 for decreasing, -1 for increasing")
    parser.add_argument("--lambda_val", type=float, default=0.0, help="Monotonic penalty, all tasks are trained with the same penalty")
    parser.add_argument("--mode", type=int, default=0, help="Mode for brain region training")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for network")
    parser.add_argument("--points", type=int, default=3, help="Points per subject")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for pre-training")
    parser.add_argument("--heldout", type=int, default=-1, help="Type of heldout study")
    args = parser.parse_args()
    expID = args.experimentID
    file = args.file
    folder = args.folder
    lambda_val = args.lambda_val
    mode = args.mode
    heldout = args.heldout

    # Load train and test IDs
    train_ids, test_ids = [], []

    heldout_study = ['ADNI', 'BLSA', 'AIBL', 'CARDIA', 'HABS', 'OASIS', 'PENN', 'PreventAD', 'WRAP'] 

    if heldout == -1:
        print("Loading all studies...")
        with (open("/home/cbica/Desktop/DKGP/data/train_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try:
                    train_ids.append(pickle.load(openfile))
                except EOFError:
                    break 

        with (open("/home/cbica/Desktop/DKGP/data/test_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
            while True:
                try:
                    test_ids.append(pickle.load(openfile))
                except EOFError:
                    break
    else:
        print("Loading heldout study {}...".format(heldout_study[heldout]))
        with (open("/home/cbica/Desktop/LongGPClustering/data1/train_subject_allstudies_ids_hmuse_" + heldout_study[heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try:
                    train_ids.append(pickle.load(openfile))
                except EOFError:
                    break 

        with (open("/home/cbica/Desktop/LongGPClustering/data1/test_subject_allstudies_ids_hmuse_" + heldout_study[heldout] +  ".pkl", "rb")) as openfile:
            while True:
                try:
                    test_ids.append(pickle.load(openfile))
                except EOFError:
                    break

    train_ids = train_ids[0]
    test_ids = test_ids[0]

    print('Train IDs:', len(train_ids))
    print('Test IDs:', len(test_ids))

    # Load and preprocess data
    train_x, train_y, test_x, test_y, \
    corresponding_train_ids, corresponding_test_ids, \
    num_outputs, task_names, region_name = load_and_preprocess_region_based_data(
        folder=folder,
        file=file,
        train_ids=train_ids,
        test_ids=test_ids,
        mode=mode
    )
    temporal_index = -1

   # Split baseline and temporal features
    train_x_baseline = train_x[:, :-1]
    train_x_time = train_x[:, temporal_index].reshape(-1, 1)
    test_x_baseline = test_x[:, :-1]
    test_x_time = test_x[:, temporal_index].reshape(-1, 1)

    # Standardize temporal variable
    # scaler_time = StandardScaler()
    # train_x_time = scaler_time.fit_transform(train_x_time)
    # test_x_time = scaler_time.transform(test_x_time)

    # Combine features back
    train_x = np.hstack((train_x_baseline, train_x_time))
    test_x = np.hstack((test_x_baseline, test_x_time))

    # Convert data tensors
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)
    test_x = torch.tensor(test_x, dtype=torch.float64)
    test_y = torch.tensor(test_y, dtype=torch.float64)


    print("Train x shape :", train_x.shape)
    print("Train y shape :", train_y.shape)

    #Define monotonicity hyper-parameters
    num_tasks = num_outputs
    if mode == 2:
        sigma = torch.tensor([-1] * num_outputs, dtype=torch.float64, device=device)
    else:
        sigma = torch.tensor([1] * num_outputs, dtype=torch.float64, device=device)

    lambda_penalty = torch.tensor([lambda_val] * num_outputs, dtype=torch.float64, device=device)
    # Create datasets
    train_dataset = CognitiveDataset(inputs=train_x, targets=train_y, subject_ids=corresponding_train_ids)
    test_dataset = CognitiveDataset(inputs=test_x, targets=test_y, subject_ids=corresponding_test_ids)

    batch_size = 16  # Adjust as needed
    train_sampler = SubjectBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_subject_sampler = TestSubjectBatchSampler(test_dataset, shuffle=False)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, pin_memory=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_subject_sampler,
        collate_fn=collate_fn, pin_memory=False, num_workers=0
)

    # Determine input dimension
    input_dim = train_x.shape[1]
    hidden_dim = args.hidden_dim # Adjust as needed

    # Determine input dimension
    # =======================================
    # Step 1: Train the Deep Regression Model
    # =======================================
    # Initialize the model
    feature_extractor = FeatureExtractorLatentConcatenation(input_dim, hidden_dim)
    model = RegressionNNLatentConcatenation(feature_extractor, output_dim=num_outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop for deep regression model
    num_epochs = args.epochs # Adjust as needed
    total_regression_loss = [] 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        total_regression_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Regression Loss: {epoch_loss:.4f}")

    # Save the feature extractor
    output_file = "./multitask_trials"

    os.makedirs(output_file, exist_ok=True)
    print(f"Output directory {output_file} created")

    MAX_SUBJECT_PLOTS_PER_TASK = 20
    plots_saved_per_task = [0] * num_outputs
    plots_root = os.path.join(output_file, f"test_trajectories_mode{mode}_{lambda_val}_{region_name.replace(' ', '_')}")
    task_plots_dirs = _ensure_task_plot_dirs(plots_root, num_outputs)

    from pathlib import Path
    monotonicity_results = Path(f"{output_file}/results.txt")
    monotonicity_results.touch(exist_ok=True)
    torch.save(feature_extractor.state_dict(), '{}/multitask_feature_extractor_latentconcatenation.pth'.format(output_file))

    # Visualize the training loss
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(total_regression_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.grid(True)
    # plt.show()
    # plt.savefig('{}/regression_training_loss.png'.format(output_file),  dpi=300)
    # plt.savefig('{}/regression_training_loss.svg'.format(output_file),  dpi=300)

    # =======================================
    # Step 2: Load Feature Extractor for GP Model
    # =======================================
    # Re-initialize the feature extractor and load the saved parameters
    #Create output folder for runs
    

    feature_extractor_gp = FeatureExtractorLatentConcatenation(input_dim, hidden_dim).to(device)
    feature_extractor_gp.load_state_dict(torch.load('{}/multitask_feature_extractor_latentconcatenation.pth'.format(output_file)))
    feature_extractor_gp = feature_extractor_gp.double()
    feature_extractor_gp.eval()
    for p in feature_extractor_gp.parameters():
        p.requires_grad = False

    # Prepare the inducing points
    unique_train_subject_ids = list(set(corresponding_train_ids))
    selected_subject_ids = random.sample(unique_train_subject_ids, 128)  # Adjust the number as needed
    inducing_points = select_inducing_points(train_x, corresponding_train_ids, selected_subject_ids=selected_subject_ids, num_points_per_subject=args.points, device='cuda')
    # Ensure inducing points are in torch.float64
    inducing_points = inducing_points.double().to(device)
    # Initialize GP Regression Model and Likelihood
    gp_regression_model = MultitaskDeepKernelGPModel(inducing_points, num_tasks=num_outputs, feature_extractor=feature_extractor_gp)
    regression_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs)


    # Convert models and likelihoods to double precision
    gp_regression_model = gp_regression_model.double().to(device)
    regression_likelihood = regression_likelihood.double().to(device)
    model_wrapper = GPModelWrapper(gp_regression_model, regression_likelihood).to(device)
    # Define loss functions
    mll_regression = gpytorch.mlls.VariationalELBO(regression_likelihood, gp_regression_model, num_data=len(train_dataset), combine_terms = True)


    # Set up the optimizer
    optimizer = torch.optim.Adam([
        {'params': regression_likelihood.parameters()},
        {'params': gp_regression_model.mean_module.parameters()},
        {'params': gp_regression_model.covar_module.parameters()},
        {'params': gp_regression_model.variational_strategy.parameters()},
    ], lr=1e-3)

    print(any(p.requires_grad for p in feature_extractor_gp.parameters()))

    # Setup Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=0.5  
    )

    # Training Loops
    num_epochs = 150

    for epoch in range(num_epochs):
        model_wrapper.train()
        regression_likelihood.train()
        running_loss = 0.0

        for inputs, targets, subject_ids in train_loader:
            if inputs.shape[0] < 2:
                continue   # ← skip batch of size 0 or 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            
            #gp_regression_output = model_wrapper(inputs)
            # t = inputs[:, -1]
            # order = torch.argsort(t)

            # inputs = inputs[order]
            # targets = targets[order]
            # Regression Loss
            f = model_wrapper(inputs)
            pred = regression_likelihood(f)
            mean_pred = pred.mean
            loss_regression = -mll_regression(f, targets)
            # t = inputs[:, -1]
            # order = torch.argsort(t)
            # mu = mean_pred[order]
            # delta = mu[1:] - mu[:-1]
            # pen = torch.relu(sigma.view(1, -1) * delta).mean(dim=0)
            
            # Multi Subject Calculations
            total_penalty = 0.0
            if lambda_val > 0:
                unique_sids = set(subject_ids)

                delta = mean_pred[1:] - mean_pred[:-1]

                # mask transitions where subject changes
                same_subject = torch.tensor(
                    [subject_ids[i] == subject_ids[i-1] for i in range(1, len(subject_ids))],
                    device=inputs.device
                )
                delta = delta[same_subject]

                pen = torch.relu(sigma.view(1, -1) * delta).mean(dim=0)

                total_penalty = (lambda_penalty * pen).sum()


            total_loss = loss_regression + total_penalty
            #print(total_loss)
            total_loss.backward()

            #torch.nn.utils.clip_grad_norm_(gp_regression_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += total_loss.item() 

        epoch_loss = running_loss / len(train_loader)
        #scheduler.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Total Loss: {epoch_loss:.4f} "
            #f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    # Evaluation
    def is_monotonic(sequence, sig):
        seq = np.asarray(sequence)
        if seq.size <= 1:
            return True
        if sig < 0:
            return np.all(np.diff(seq) >= 0)
        else:
            return np.all(np.diff(seq) <= 0)

        
    model_wrapper.eval()
    regression_likelihood.eval()
    with torch.no_grad():
        regression_predictions = [[] for _ in range(num_outputs)]
        regression_actuals = [[] for _ in range(num_outputs)]

        all_means = []
        all_vars = []
        all_trues = []
        all_subjects = []

        mono_subject_ok = [0] * num_outputs
        mono_subject_total = 0

        mono_sample_ok = [0] * num_outputs
        mono_sample_total = 0

        for inputs, targets, subject_ids in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            subject_id = subject_ids[0]

            # Regression Predictions
            gp_regression_output = model_wrapper(inputs)    
            pred_regression = regression_likelihood(gp_regression_output)
            mean_pred = pred_regression.mean  # Shape: [batch_size, num_outputs]
            var_pred = pred_regression.variance

            mean_np = _transpose_task_matrix(mean_pred.detach().cpu().numpy(), num_outputs)
            var_np = _transpose_task_matrix(var_pred.detach().cpu().numpy(), num_outputs)

            t_np = inputs[:, -1].detach().cpu().numpy()
            y_true_np = targets.detach().cpu().numpy()

            can_plot_any = (MAX_SUBJECT_PLOTS_PER_TASK is None) or any(
                plots_saved_per_task[k] < MAX_SUBJECT_PLOTS_PER_TASK for k in range(num_outputs)
            )

            if can_plot_any:
                if MAX_SUBJECT_PLOTS_PER_TASK is None:
                    save_subject_trajectory_plots(
                        subject_id=subject_id,
                        t_np=t_np,
                        y_true_np=y_true_np,
                        y_pred_np=mean_np,
                        y_var_np=var_np,
                        task_dirs=task_plots_dirs,
                        task_names=task_names
                    )

                    for k in range(num_outputs):
                        plots_saved_per_task[k]+=1

                else:
                    save_subject_trajectory_plots(
                        subject_id=subject_id,
                        t_np=t_np,
                        y_true_np=y_true_np,
                        y_pred_np=mean_np,
                        y_var_np=var_np,
                        task_dirs=task_plots_dirs,
                        task_names=task_names
                    )
                    for k in range(num_outputs):
                        if plots_saved_per_task[k] < MAX_SUBJECT_PLOTS_PER_TASK:
                            plots_saved_per_task[k]+=1

            for i in range(num_outputs):
                regression_predictions[i].extend(mean_pred[:, i].cpu().numpy())
                regression_actuals[i].extend(targets[:, i].cpu().numpy())
            
            all_means.append(mean_pred.detach().cpu().numpy())
            all_vars.append(var_pred.detach().cpu().numpy())
            all_trues.append(targets.detach().cpu().numpy())
            all_subjects.append(subject_ids)

            #Check monotnicity
            t = inputs[:, -1].detach().cpu().numpy()
            order = np.argsort(t)

            mono_subject_total += 1
            for k in range(num_outputs):
                seq_k = mean_pred[:, k].detach().cpu().numpy()[order]
                if is_monotonic(seq_k, sigma[k]):
                    mono_subject_ok[k] += 1

    
    with torch.no_grad():
        for inputs, targets, subject_ids in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            f = model_wrapper(inputs)
            pred = regression_likelihood(f)
            mean_pred = pred.mean
            t = inputs[:, -1]
            order = torch.argsort(t)
            mu = mean_pred[order]
            delta = mu[1:] - mu[:-1]
            mono_sample_total += delta.shape[0]
            for k in range(num_outputs):
                if sigma[k].item() < 0:
                    mono_sample_ok[k] += (delta[:, k] >= 0).sum().item()
                else:
                    mono_sample_ok[k] += (delta[:, k] <= 0).sum().item()

    Y_mean = np.vstack(all_means)
    Y_var = np.vstack(all_vars)
    Y_true = np.vstack(all_trues)

    nll_per_task, nll_mean = gaussian_nll_per_task(Y_true, Y_mean, Y_var)

    cov_per_task, width_per_task, cov_mean, width_mean = coverage_and_width_per_task(Y_true, Y_mean, Y_var, z=1.96)

    mse_per_task, mae_per_task, mse_mean, mae_mean = _mse_mae_per_task(
        regression_actuals, regression_predictions
    )

    output_file = "./multitask_trials"
    from pathlib import Path
    monotonicity_results = Path(f"{output_file}/results.txt")
    monotonicity_results.touch(exist_ok=True)

    with monotonicity_results.open("a", encoding="utf-8") as f:
        print(str(datetime.datetime.now())+'\n', file=f)
        print("\n=== Multitask Evaluation ===", file=f)
        print(f"Heldout Study: {heldout_study[heldout]}", file=f)
        print(f"Num tasks: {num_outputs}", file=f)
        print(f"Sigma per task: {sigma}", file=f)
        print(f"Lambda penalty value: {lambda_val}", file=f)
        print(f"Mode: {mode}", file=f)
        print(f"Mean Test MSE (avg across tasks): {mse_mean:.4f}", file=f)
        print(f"Mean Test MAE (avg across tasks): {mae_mean:.4f}", file=f)
        print(f"Mean Test NLL (avg across tasks): {nll_mean:.4f}", file=f)
        print(f"Mean Test Coverage% (avg across tasks): {cov_mean:.2f}", file=f)
        print(f"Mean Test Interval Width (avg across task): {width_mean:.4f}", file=f)

        for k in range(num_outputs):
            subj_pct = 100.0 * mono_subject_ok[k] / max(mono_subject_total, 1)
            samp_pct = 100.0 * mono_sample_ok[k] / max(mono_sample_total, 1)

            print(
                f"Task {k+1}: Test MSE={mse_per_task[k]:.4f}, Test Mae={mae_per_task[k]:.4f}, "
                f"NLL={nll_per_task[k]:.4f}, Coverage%={cov_per_task[k]:.4f}, Width={width_per_task[k]:.4f}, "
                f"Monotonic subjects={subj_pct:.2f}%, Monotonic samples(df/dt)={samp_pct:.2f}%", file=f
            )

    print(f"\nMean Test MSE (avg across tasks): {mse_mean:.4f}")
    print(f"Mean Test MAE (avg across tasks): {mae_mean:.4f}")

    # Optionally, save the models
    save_file = "./multitask_trials/ckpts/"
    os.makedirs(save_file, exist_ok=True)
    torch.save(
        gp_regression_model.state_dict(),
        save_file + 'gp_regression_model_{}_{}.pth'.format(mode, lambda_val)
    )

    torch.save(
        regression_likelihood.state_dict(),
        save_file + 'regression_likelihood_{}_{}.pth'.format(mode, lambda_val)
    )

if __name__ == "__main__":
    main()
