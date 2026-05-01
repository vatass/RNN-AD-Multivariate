import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from pandas.core.indexes.base import default_index 
import sys
import torch
import gpytorch
from functions import *
import pickle
from exactgpmodels import SingleTaskDeepKernelNonLinear, SingleTaskDeepKernel
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import wandb
import time 
import json
import math 
sns.set_style("white", {'axes.grid' : False})
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior

# Plot Controls 
# sns.set_theme(context="paper",style="whitegrid", rc={'axes.grid' : True, 'font.serif': 'Times New Roman'})
# wandb.init(project="HMUSEDeepSingleTask", entity="vtassop", save_code=True)
parser = argparse.ArgumentParser(description='Temporal Deep Kernel Single Task GP model for a single HMUSE Roi')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='allstudies') # 1adni normally
parser.add_argument("--exp", help="Indicates the modality", default='')
parser.add_argument("--kfoldID", help="Identifier for the Kfold IDs", default="missingdasae")
parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_hmuse_allstudies")
parser.add_argument("--covariates", help="String with the Covariates used as Condition in the Model", default="Diagnosis-Age-Sex-APOE4_Alleles-Education_Years")
## Kernel Parameters ##
parser.add_argument('--kernel_choice', help='Indicates the selection of GP kernel', default='RBF', type=str)
parser.add_argument('--mean', help='Selection of Mean function', default='Constant', type=str)
## Deep Kernel Parameters 
parser.add_argument("--depth", help='indicates the depth of the Deep Kernel', default='')
parser.add_argument("--activ", help='indicates the activation function', default='relu')
parser.add_argument("--dropout", help='indicates the dropout rate', default=0.2, type=float)
## Training Parameters
parser.add_argument("--iterations", help="Epochs", default=200)
parser.add_argument("--optimizer", help='Optimizer', default='adam')
parser.add_argument("--learning_rate", help='Learning Rate', type=float, default=0.02)    # 0.01844 is in hmuse rois 
parser.add_argument("--task", help='Task id', type=str, default="MUSE")  # Right Hippocampus 
parser.add_argument("--roi_idx", type=int, default=17)
# Personalization # 
parser.add_argument("--personalization", type=str, default=False)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--pers_lr", type=float, help='Pers LR', default=0.01844) # 0.3510
####
# Feature Representation 
# H_MUSE features
# Clinical_Features = Diagnosis, Age, Sex, APOE4 Alleles, Education Years
# Genomic 
# Follow-Up = {HMUSE}{Time}
# Time 
####

t0= time.time()
args = parser.parse_args()
history = int(args.history)
personalization = args.personalization
gpu_id = int(args.gpuid) 
exp = args.exp 
iterations = int(args.iterations)
covariates = args.covariates.split("-")
kfoldID = args.kfoldID
file = args.file 
expID = args.experimentID 
depth = args.depth 
activ = args.activ 
dropout = args.dropout
task = args.task 
kernel = args.kernel_choice
mean = args.mean
roi_idx = args.roi_idx

text_task = task 

mae_TempGP_list, mae_TempGP_list_comp = [], []  
population_results = {'id' : [], 'kfold': [], 'score': [], 'upper': [], 'lower': [], 'variance': [],  'y': [], 'time': []}
population_mae_kfold = {'kfold': [], 'mae': [], 'mse': [], 'rmse': [], 'R2': [],  'interval': [], 'coverage': []}
population_metrics_per_subject = {'kfold': [], 'id': [], 'mae_per_subject': [], 'wes_per_subject': [], 'interval': [], 'coverage': [] }

train_population_results = {'id':[], 'score': [], 'upper': [], 'lower': [], 'y': [], 'time': [], 'kfold': [] } 
train_population_mae_kfold = {'mae': [], 'kfold': [], 'interval': [], 'coverage': []}

if personalization: 
    personalization_results = {'id': [], 'kfold': [], 'score': [], 'upper': [], 'lower': [], 'variance': [],  'y': [], 'time': [], 'history_points': [], 'model': []}
    person_metrics_history = {'history': [], 'ae': [], 'se': [], 'coverage': [] ,  'model':[], 'interval': [], 'id': [], 'time': [], 'kfold': [] }
    person_mean_metrics = {'id': [], 'history': [], 'mae': [], 'mse': [], 'r2': [], 'rmse': [], 'model': [], 'kfold': [], 'coverage': [], 'interval': []}
    person_metrics_per_subject = {'kfold': [], 'id': [], 'mae_per_subject': [], 'wes_per_subject': [], 'history': [] }

# Configuration of W&B
# wandb.config = args
mae_MTGP_list, coverage_MTGP_list, interval_MTGP_list = [], [], [] 

datasamples = pd.read_csv('./subjectsamples_longclean_dl_muse_allstudies.csv')
longitudinal_covariates = pd.read_csv("../LongitudinalDiffusion/longitudinal_covariates_allstudies.csv")
subject_ids = list(datasamples['PTID'].unique()) 

# wandb.config['Subjects'] = len(subject_ids) 

f = open('../LongGPClustering/roi_to_idx.json')
roi_to_idx = json.load(f)

for fold in range(5): 
    print('FOLD::', fold)
    train_ids, test_ids = [], []     

    with (open("train_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    # with (open("./data"+str(folder)+"/test_subject_ids_hmuse_" + kfoldID + str(fold) + ".pkl", "rb")) as openfile:
    with (open("test_subject_allstudies_ids_dl_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:

        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break

    train_ids = train_ids[0]
    test_ids = test_ids[0]


    ### Use the first 200 subjects in order to extract the correct alphas
    # val_ids = train_ids[:200]

    print('Train IDs', len(train_ids))
    print('Test IDs', len(test_ids))
    print() 

    for t in test_ids: 
        if t in train_ids: 
            raise ValueError('Test Samples belong to the train!')

    ### SET UP THE TRAIN/TEST DATA FOR THE NON LINEAR DEEP KERNEL GP REGRESSION ### 
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']    
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    
    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list() 
    assert len(corresponding_test_ids) == test_x.shape[0]
    assert len(corresponding_train_ids) == train_x.shape[0]

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    # val_x, val_y, val_x, val_y = process_temporal_singletask_data(train_x=val_x, train_y=val_y, test_x=val_x, test_y=val_y, test_ids=test_ids)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpu_id) 
        train_y = train_y.cuda(gpu_id)#.squeeze()
        test_x = test_x.cuda(gpu_id) 
        test_y = test_y.cuda(gpu_id)#.squeeze() 
        # val_x = val_x.cuda(gpu_id)
        # val_y = val_y.cuda(gpu_id)

    print('Train Data', train_x.shape)
    print('Targets', train_y.shape)
    print('Test Data', test_x.shape)
    print('Targets', test_y.shape)
# roi_idx is the index from 0 to 144.
    test_y = test_y[:, roi_idx]
    train_y = train_y[:, roi_idx]

    print(train_y.shape, test_y.shape)
    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    train_y = train_y.contiguous()
    test_y  = test_y.contiguous()

    print("---- NUMERICAL CHECKS ----")
    print("train_x NaNs:", torch.isnan(train_x).sum().item())
    print("train_x Infs:", torch.isinf(train_x).sum().item())
    print("test_x  NaNs:", torch.isnan(test_x).sum().item())
    print("test_x  Infs:", torch.isinf(test_x).sum().item())

    print("train_y NaNs:", torch.isnan(train_y).sum().item())
    print("train_y Infs:", torch.isinf(train_y).sum().item())
    print("test_y  NaNs:", torch.isnan(test_y).sum().item())
    print("test_y  Infs:", torch.isinf(test_y).sum().item())

    print("\n---- MEMORY LAYOUT ----")
    print("train_x contiguous:", train_x.is_contiguous())
    print("test_x  contiguous:", test_x.is_contiguous())
    print("train_y contiguous:", train_y.is_contiguous())
    print("test_y  contiguous:", test_y.is_contiguous())

    print("\n---- DTYPES ----")
    print("train_x dtype:", train_x.dtype)
    print("test_x  dtype:", test_x.dtype)
    print("train_y dtype:", train_y.dtype)
    print("test_y  dtype:", test_y.dtype)


    #### DEFINE GP MODEL #### 
    # depth = [(train_x.shape[1], train_x.shape[1] * 2 ), (train_x.shape[1]*2, int(train_x.shape[1]/2))]
    depth = [(train_x.shape[1], int(train_x.shape[1]/2))]
    dr = args.dropout
    activ = 'relu'

    lat_dim = int(train_x.shape[1]/2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
     pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None) 

    if torch.cuda.is_available(): 
        likelihood = likelihood.cuda(gpu_id) 
        deepkernelmodel = deepkernelmodel.cuda(gpu_id)

    training_iterations  =  iterations 
        
    # set up train mode 
    deepkernelmodel.feature_extractor.train()
    deepkernelmodel.train()
    deepkernelmodel.likelihood.train()

    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam([
        {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.covar_module.parameters(), 'lr': args.learning_rate },
        {'params': deepkernelmodel.mean_module.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.likelihood.parameters(),  'lr': args.learning_rate} ], weight_decay=0.1) ## try more reg 
  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50, 70], gamma=0.1)

    train_loss, val_loss = [], [] 
    for i in tqdm(range(training_iterations)):
        deepkernelmodel.train()
        likelihood.train()
        optimizer.zero_grad()
        output = deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        # wandb.log({"mll_train_loss": loss})
        train_loss.append(loss.item())
        optimizer.step()


    # Set into eval mode
    deepkernelmodel.eval()
    likelihood.eval()

    # Make Predictions in Test 
    total_differences, total_derivatives = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        f_preds = deepkernelmodel(test_x)
        y_preds = likelihood(f_preds)

        mean = y_preds.mean
        variance = y_preds.variance   ### What variance is this? Does it contain population variance?

        lower, upper = y_preds.confidence_region()

    mae_pop, ae_pop = mae(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
    mse_pop, rmse_pop, se_pop = mse(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())  
    rsq = R2(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy()) 

    # wandb.log({"test_mae": mae_pop})

    coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=mean.cpu().detach().numpy(), groundtruth=test_y.cpu().detach().numpy(),
     intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()])  

    coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 

    # uncertainty eval
    acp = np.mean(coverage)
    ncp = acp - 0.5 

    # get the 50% width 
    posterior_std = y_preds.variance.sqrt()  # Standard deviation at each test point
    inverse_width = 1/(upper.cpu().detach().numpy()-lower.cpu().detach().numpy())

    mae_per_subject, wmae_per_subject, coverage_per_subject, interval_per_subject = calculate_errors(predictions=mean.cpu().detach().numpy(), upper_tensor=upper, lower_tensor=lower,    
                                ids=corresponding_test_ids, 
                                real_values=test_y.cpu().detach().numpy(),
                                real_tensor= test_y, 
                                weights=inverse_width.tolist())

    for key in mae_per_subject: 
        population_metrics_per_subject['kfold'].append(fold) 
        population_metrics_per_subject['id'].append(key) 
        population_metrics_per_subject['mae_per_subject'].append(mae_per_subject[key]) 
        population_metrics_per_subject['wes_per_subject'].append(wmae_per_subject[key])
        population_metrics_per_subject['interval'].append(coverage_per_subject[key])
        population_metrics_per_subject['coverage'].append(interval_per_subject[key])

    # wandb.log({"test_interval": mean_interval_width, "test_coverage": np.mean(coverage)})

    mae_MTGP_list.append(mae_pop)
    coverage_MTGP_list.append(np.mean(coverage))
    interval_MTGP_list.append(mean_interval_width)

    population_results['id'].extend(corresponding_test_ids) 
    population_results['kfold'].extend([fold for c in range(len(corresponding_test_ids))])
    population_results['score'].extend(mean.cpu().detach().numpy().tolist())
    population_results['lower'].extend(lower.cpu().detach().numpy().tolist())
    population_results['upper'].extend(upper.cpu().detach().numpy().tolist()) 
    population_results['variance'].extend(variance.cpu().detach().numpy().tolist()) 
    population_results['y'].extend(test_y.cpu().detach().numpy().tolist())
  
    time_ = test_x[:, -1].cpu().detach().numpy().tolist() 
    population_results['time'].extend(time_) 

    population_mae_kfold['mae'].append(mae_pop)
    population_mae_kfold['mse'].append(mse_pop)
    population_mae_kfold['rmse'].append(rmse_pop)
    population_mae_kfold['R2'].append(rsq)
    population_mae_kfold['kfold'].append(fold)
    population_mae_kfold['coverage'].append(np.mean(coverage))
    population_mae_kfold['interval'].append(mean_interval_width)

    ### Save the Population Hyperparams ###
    ## Extract the Population Model Parameters
    population_hyperparams = {} 
    ## Transfer the GP hyperparams 
    print('==== Population Hyperparams ====')
    for param_name, param in deepkernelmodel.named_parameters():
        if not param_name.startswith('feature'): 
            population_hyperparams[param_name] = param
        else: 
            print(param_name, param)
    print(population_hyperparams)


    # Save model
    # save_model(deepkernelmodel, optimizer, likelihood, filename='./manuscript1/models/population_deep_kernel_gp_'+ str(roi_idx) + '_' + str(fold)+'.pth', train_x=train_x, train_y=train_y)


    ### Save the population Hyperparams ###
    population_hyperparams = {} 
    ## Transfer the GP hyperparams 
    print('==== Population Hyperparams ====')
    for param_name, param in deepkernelmodel.named_parameters():
        if not param_name.startswith('feature'): 
            population_hyperparams[param_name] = param
        else: 
            print(param_name, param)
    print(population_hyperparams)

    # TODO: Run for some of them. 
    # Save the Weights for Interpretability 
    # Get the weights of the feature extractor 
    feature_weights = deepkernelmodel.feature_extractor.state_dict()
    # Save the weights 
    task = 'MUSE_' + str(roi_idx)
    # torch.save(feature_weights, './results/dkgp/feature_weights_'+ str(task) + '_' + str(fold)+'.pth')


task = 'MUSE_' + str(roi_idx)
population_mae_kfold_df = pd.DataFrame(data=population_mae_kfold)
population_mae_kfold_df.to_csv('./results/dkgp/singletask_unharmonized_dlmuse_'+ str(task) + '_dkgp_mae_kfold_'+ expID+'.csv')

population_results_df = pd.DataFrame(data=population_results)
population_results_df.to_csv('./results/dkgp/singletask_unharmonized_dlmuse' + str(task) + '_dkgp_population_'+ expID+'.csv')

population_metrics_per_subject_df = pd.DataFrame(data=population_metrics_per_subject)
population_metrics_per_subject_df.to_csv('./results/dkgp/singletask_unharmonized_dlmuse_' + str(task) + '_dkgp_mae_per_subject_kfold_' + expID + '.csv')

print('#### Evaluation of the Singletask Deep Kernel Temporal GP for ROI ' + str(task) + '###')
print('POPULATION GP')
print('mean:', np.mean(mae_MTGP_list, axis=0))
print('var:', np.var(mae_MTGP_list, axis=0))
print('Interval', np.mean(interval_MTGP_list), np.var(interval_MTGP_list))
print('Coverage', np.mean(coverage_MTGP_list), np.var(coverage_MTGP_list))
# wandb.log({'mean MAE': np.mean(np.mean(mae_MTGP_list, axis=0)), "mean Interval": np.mean(np.mean(interval_MTGP_list, axis=0)), "mean Coverage": np.mean(np.mean(coverage_MTGP_list, axis=0))})

if personalization: 
    print("### Evaluation of the Personalization ###")
    print('mean:', np.mean(person_mean_metrics['mae'], axis=0))
    print('var:', np.var(person_mean_metrics['mae'], axis=0))
    print('Interval', np.mean(person_metrics_history['interval']), np.var(person_metrics_history['interval']))
    print('Coverage', np.mean(person_metrics_history['coverage']), np.var(person_metrics_history['coverage']))
    # wandb.log({'Person mean MAE':np.mean(person_mean_metrics['mae'], axis=0), "Person mean Interval": np.mean(person_metrics_history['interval'], axis=0), "Person mean Coverage": np.mean(person_metrics_history['coverage'], axis=0)})

t1 = time.time() - t0 
print("Time elapsed: ", t1) 



