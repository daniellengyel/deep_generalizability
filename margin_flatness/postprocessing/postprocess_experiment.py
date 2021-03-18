import numpy as np
import pandas as pd

from .utils import tensorboard_to_dict, get_data_for_experiment
from ..utils import *
from ..save_load import load_all_cached_meta_data, cache_data, load_configs, get_models
from ..data_getters import *
from ..data_getters import get_random_data_subset
from ..training_utils import get_criterion

from .sharpness import get_affine_trace, sample_average_flatness_pointwise, get_point_traces
from .model_related import get_point_loss, get_models_loss_acc
from .robustness import get_inp_out_jacobian_points, get_margins

import yaml, os, sys, re, time
from tqdm import tqdm


import torch

import pickle



# iterate through runs
def get_runs(experiment_folder, names):
    run_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        run_file_name = files[0] if ("tfevents" in files[0]) else files[1]
        curr_dir = os.path.basename(root)
        print(root)
        try:
            run_dir[curr_dir] = tensorboard_to_dict(os.path.join(root, run_file_name), names)
            cache_data(experiment_folder, "runs", run_dir)
        except:
            print("Error for this run.")

    return run_dir

# Ok not to have explicit seed since we are using the whole dataset.
def get_exp_loss_acc(experiment_folder, step, seed=0, train_datapoints=-1, test_datapoints=-1, device=None):
    print("Get loss acc")
    # init
    loss_dict = {}
    acc_dict = {}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)

    if test_datapoints == -1:
        num_test_datapoints = len(test_data)
    test_data = get_random_data_subset(test_data, num_datapoints=num_test_datapoints, seed=seed)

    if train_datapoints == -1:
        num_train_datapoints = len(train_data)
    train_data = get_random_data_subset(train_data, num_datapoints=num_train_datapoints, seed=seed)

    cfgs = load_configs(experiment_folder)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        loss_type = cfgs.loc[exp_name]["criterion"]
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        loss_dict[exp_name], acc_dict[exp_name] = get_models_loss_acc(models_dict, train_data, test_data, criterion, loss_type,
                                                                      device=device)
        # cache data
        cache_data(experiment_folder, "loss", loss_dict, step=step)
        cache_data(experiment_folder, "acc", acc_dict, step=step)

    return loss_dict, acc_dict




def compute_on_experiment(experiment_folder, name, step, seed, num_datapoints, on_test_set, device, verbose=False):

    meta_dict = {"seed": seed, "num_datapoints": num_datapoints, "on_test_set": on_test_set, "step": step}
    cached_results = check_if_already_computed(experiment_folder, name, step, meta_dict)

    if cached_results is not None:
        return cached_results

    results_dict = {}
    cfgs = load_configs(experiment_folder)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    # iterate through models
    for exp_name, curr_path in tqdm(exp_models_path_generator(experiment_folder), disable=(not verbose)):

        criterion = get_criterion(cfgs.loc[exp_name])
        loss_type = cfgs.loc[exp_name]["criterion"]

        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue

        results_dict[exp_name] = helper_compute_on_experiment(name, models_dict, data, seed, criterion, loss_type, device=device) # potentially change how we do loss_type and criterion

    # cache data
    cache_data(experiment_folder, name, results_dict, meta_dict, step=step, time_stamp=True)

    return results_dict


def helper_compute_on_experiment(name, models_dict, data, seed, criterion, loss_type, device=None):
    """robustness: [inp_out_jacobian, softmax_margins, output_margins, affine_upperbound_margins]
    flatness: [affine_trace, point_traces, sample_average_flatness_pointwise]"""

    if name == "inp_out_jacobian":
        return get_inp_out_jacobian_points(models_dict, data, device=device)

    elif name == "sample_average_flatness_pointwise":
        return sample_average_flatness_pointwise(models_dict, data, criterion, 100, 0.1, seed=seed)

    elif name == "affine_trace":
        return get_affine_trace(models_dict, data, loss_type, device=device)

    elif name == "point_traces":
        return get_point_traces(models_dict, data, criterion, device=device, seed=seed)

    elif name == "softmax_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=False, softmax_outputs=True, seed=seed)

    elif name == "output_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=False, softmax_outputs=False, seed=seed)

    elif name == "affine_upperbound_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=True, softmax_outputs=True, seed=seed)

    elif name == "point_loss":
        return get_point_loss(models_dict, data, loss_type, device=device)


def check_if_already_computed(experiment_folder, name, step, meta_data):

    all_cached_meta_data = load_all_cached_meta_data(experiment_folder, name, step)
    if all_cached_meta_data is None:
        return None

    for time_stamp in all_cached_meta_data:
        if meta_data == all_cached_meta_data[time_stamp]:
            return load_cached_data(experiment_folder, name, step, time_stamp)[0]
    
    return None