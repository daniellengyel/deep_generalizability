import numpy as np
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader
import sys

import re

from .training_metrics import * 
from .utils import * 
from ..nets import Nets
from ..utils import *

import itertools

from scipy.stats import linregress, kendalltau
from sklearn.neighbors import LocalOutlierFactor

def get_outlier_filter(x_data, y_data):
    n_neighbors = 3
    if len(x_data) < 5:
        return np.array([True]*len(x_data)) # They are all outliers since we don't have enough datapoints
    combined_data = np.concatenate([x_data.reshape(len(x_data), 1), y_data.reshape(len(y_data), 1)], axis=1)
    try:
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        outlier_filter = clf.fit_predict(combined_data) == 1
    except: 
        return np.array([True]*len(x_data))
    return outlier_filter
    
def linregress_outliers(x_data, y_data, remove_outliers=True):
    if len(x_data) == 0:
        return 0, 0, 0, None, None
    x_data, y_data = np.array(x_data), np.array(y_data)

    if remove_outliers:
        outlier_filter = get_outlier_filter(x_data, y_data)
        x_data, y_data = x_data[outlier_filter], y_data[outlier_filter]

    return linregress(x_data, y_data)

def get_corr_array(experiment_folder, X_data_filter_f, Y_data_f, use_correct_filter):
    all_steps = get_exp_steps(experiment_folder)
    all_steps = np.array([list(v.keys()) for v in all_steps.values()]).reshape(-1)
    all_steps = sorted(list(set(all_steps)))
    
    exp_cfgs = load_configs(experiment_folder)
    exp_ids = exp_cfgs.index
    
    res_dict = {}
    
    for step in all_steps:
        print(step)
   
        X_data_filter = X_data_filter_f(step)
        Y_data = Y_data_f(step)
        
        for exp_id in exp_ids:
            if exp_id not in res_dict:
                res_dict[exp_id] = {}
            
            for model_idx in range(exp_cfgs.loc[exp_id]["num_nets"]):
                model_idx = str(model_idx)
                if model_idx not in res_dict[exp_id]:
                    res_dict[exp_id][model_idx] = {}
                    res_dict[exp_id][model_idx]["acc"] = {} 
                    if use_correct_filter:
                        res_dict[exp_id][model_idx]["correct_r_value"] = {}
                        res_dict[exp_id][model_idx]["incorrect_r_value"] = {}
                    else:
                        res_dict[exp_id][model_idx]["r_value"] = {}
                    
                curr_X_data, correct_filter = X_data_filter[exp_id][model_idx]
                curr_Y_data = np.array(Y_data[exp_id][model_idx])
                if use_correct_filter:
                    slope, intercept, correct_r_value, _, _ = linregress_outliers(curr_X_data[correct_filter], curr_Y_data[correct_filter])
                    res_dict[exp_id][model_idx]["correct_r_value"][step] = correct_r_value
                    print(slope)

                    slope, intercept, incorrect_r_value, _, _ = linregress_outliers(curr_X_data[~correct_filter], curr_Y_data[~correct_filter])
                    res_dict[exp_id][model_idx]["incorrect_r_value"][step] = incorrect_r_value
                else:
                    slope, intercept, r_value, _, _ = linregress_outliers(curr_X_data, curr_Y_data)
                    res_dict[exp_id][model_idx]["r_value"][step] = r_value
                res_dict[exp_id][model_idx]["acc"][step] =  sum(1*correct_filter)/float(len(correct_filter))
    return res_dict


if __name__ == "__main__":
                    
    X_data_filter_f = lambda step: get_exp_linear_loss_trace(experiment_folder, step=step, seed=0, device=None, num_datapoints=1000, on_test_set=False, should_cache=False)
    Y_data_f = lambda step:  load_cached_data(experiment_folder, "point_traces", step=step)[0]
    use_correct_filter = True
    c = get_corr_array(experiment_folder, X_data_filter_f, Y_data_f, use_correct_filter)
