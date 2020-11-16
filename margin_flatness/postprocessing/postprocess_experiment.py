import numpy as np
import pandas as pd

from .utils import *
from ..utils import *
from ..save_load import *
from ..data_getters import *
from ..training_utils import *

from .sharpness_measures import *
from .model_related import * 
from .margins import *

import yaml, os, sys, re

import torch

import pickle

def get_all_steps_f(experiment_folder, f):
    """f is meant to be a function of step and will compute metrics and other things solely dependend on the current step. """
    all_steps = get_exp_steps(experiment_folder)
    all_steps = np.array([list(v.keys()) for v in all_steps.values()]).reshape(-1)
    all_steps = set(all_steps)

    for step in all_steps:
        f(step)

def get_data_for_experiment(experiment_path):
    all_nets = ["SimpleNet", "LinearNet", "BatchNormSimpleNet", "KeskarC3", "LeNet"]
    
    cfs = load_configs(experiment_path)
    if "data_name" in cfs.iloc[0]:
        data_name = cfs.iloc[0]["data_name"]
    else:
        data_name = experiment_path.split("/")[-2]

    assert cfs.iloc[0]["net_name"] in all_nets
    vectorized = cfs.ilwoc[0]["net_name"] in ["SimpleNet", "LinearNet", "BatchNormSimpleNet"]
    reduce_train_per = cfs.iloc[0]["reduce_train_per"]
    seed = cfs.iloc[0]["seed"]
    meta = cfs.iloc[0]["data_meta"]
    train_data, test_data = get_data(data_name, vectorized=vectorized, reduce_train_per=reduce_train_per, seed=seed, meta=meta)
    return train_data, test_data

# +++ process experiment results +++
def tb_to_dict(path_to_events_file, names):
    tb_dict = {}  # step, breakdown by / and _

    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            t_split = re.split('/+|_+', v.tag)
            if t_split[0] in names:
                tmp_dict = tb_dict
                t_split = [e.step] + t_split
                for i in range(len(t_split) - 1):
                    s = t_split[i]
                    if s not in tmp_dict:
                        tmp_dict[s] = {}
                        tmp_dict = tmp_dict[s]
                    else:
                        tmp_dict = tmp_dict[s]
                tmp_dict[t_split[-1]] = v.simple_value
    return tb_dict


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
            run_dir[curr_dir] = tb_to_dict(os.path.join(root, run_file_name), names)
            cache_data(experiment_folder, "runs", run_dir)
        except:
            print("Error for this run.")

    return run_dir


def get_exp_final_distances(experiment_folder, device=None):
    # init
    dist_dict = {}

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):

        beginning_models_dict = get_models(curr_path, 0, device)
        final_models_dict = get_models(curr_path, -1, device)
        dist_dict[exp_name] = get_models_final_distances(beginning_models_dict, final_models_dict)

        # cache data
        cache_data(experiment_folder, "dist", dist_dict)

    return dist_dict


def get_exp_tsne(experiment_folder, step):
    # init
    tsne_dict = {}

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        tsne_dict[exp_name] = get_models_tsne(models_dict)

        # cache data
        cache_data(experiment_folder, "tsne", tsne_dict)

    return tsne_dict

# Fix seed? 
def get_exp_grad(experiment_folder, step, use_gpu=False):
    # init
    grad_dict = {}

    # get data
    cfgs = load_configs(experiment_folder)
    train_data, test_data = get_data_for_experiment(experiment_folder)
    data = get_random_data_subset(train_data, num_datapoints=1000, seed=0)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        grad_dict[exp_name] = get_models_grad(models_dict, data, criterion, device=None)

        # cache data
        cache_data(experiment_folder, "grad", grad_dict)

    return grad_dict

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


# get eigenvalues of specific model folder.
def get_exp_eig(experiment_folder, step, num_eigenthings=5, device=None, only_vals=True):
    # init
    eigenvalue_dict = {}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    cfgs = load_configs(experiment_folder)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        eigenvalue_dict[exp_name] = get_models_eig(models_dict, train_loader, test_loader, criterion, num_eigenthings,
                                                   full_dataset=True, device=device, only_vals=only_vals, seed=seed)

        # cache data
        cache_data(experiment_folder, "eig", eigenvalue_dict, step=step)

    return eigenvalue_dict


def get_exp_trace(experiment_folder, step, seed=0, device=None):
    # init
    trace_dict = {}
    meta_dict = {"seed": seed}
    cfgs = load_configs(experiment_folder)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        
        trace_dict[exp_name] = get_models_trace(models_dict, train_loader, criterion, full_dataset=False, verbose=True,
                                                device=device, seed=seed)

        # cache data
        cache_data(experiment_folder, "trace", trace_dict, step=step)

    return trace_dict

def get_exp_point_loss(experiment_folder, step=-1, seed=0, device=None, num_datapoints=50, on_test_set=False, should_cache=False):
    results_dict = {}
    meta_dict = {"seed": seed}
    cfgs = load_configs(experiment_folder)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        loss_type = cfgs.loc[exp_name]["criterion"]
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        results_dict[exp_name] = get_point_loss_filters(models_dict, data, loss_type, device=device)

        # cache data
        if should_cache:
            cache_data(experiment_folder, "point_loss", results_dict, meta_dict, step=step)

    return results_dict

def get_exp_margins(experiment_folder, get_upperbound=False, softmax_outputs=False, step=-1, seed=0, device=None, num_datapoints=50, on_test_set=False, should_cache=False):
    margins_dict = {}
    meta_dict = {"seed": seed}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):

        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        a = time.time()
        margins_dict[exp_name] = get_margins_filters(models_dict, data, device=device, get_upperbound=get_upperbound, softmax_outputs=softmax_outputs, seed=seed)
        print(time.time() - a)
        # cache data
        if should_cache:
            cache_data(experiment_folder, "margins", margins_dict, meta_dict, step=step)

    return margins_dict

STORED_DATA = None

def get_exp_point_traces(experiment_folder, step, seed, device=None, num_datapoints=1000, on_test_set=False, should_cache=False):
    global STORED_DATA
    traces_dict = {}
    meta_dict = {"seed": seed}

    # get data
    if STORED_DATA is None:
        train_data, test_data = get_data_for_experiment(experiment_folder)
        if on_test_set:
            data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
        else:
            data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)
        STORED_DATA = data 
    else:
        data = STORED_DATA

    cfgs = load_configs(experiment_folder)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue

        # if exp_name != "1605478459.1644921":
        #     continue

        a = time.time()
        traces_dict[exp_name] = get_point_traces(models_dict, data, criterion, device=device, seed=seed)
        print(time.time() - a)
        # cache data
        if should_cache:
            cache_data(experiment_folder, "point_traces", traces_dict, meta_dict, step=step)

    return traces_dict

def get_exp_point_eig_density_traces(experiment_folder, step, seed, device=None, num_datapoints=1000, on_test_set=False, should_cache=False):
    traces_dict = {}
    meta_dict = {"seed": seed}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)
    
    cfgs = load_configs(experiment_folder)

    set_seed(seed)
    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device=device)
        if models_dict is None:
            continue
        traces_dict[exp_name] = get_point_eig_density_traces(models_dict, data, criterion, device=device, seed=seed)

        # cache data
        if should_cache:
            cache_data(experiment_folder, "point_eig_density_traces", traces_dict, meta_dict, step=step)

    return traces_dict

def get_exp_point_eig_density(experiment_folder, step, seed, device=None, num_datapoints=1000, on_test_set=False, should_cache=False):
    eig_density_dict = {}
    meta_dict = {"seed": seed}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    cfgs = load_configs(experiment_folder)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        criterion = get_criterion(cfgs.loc[exp_name])
        models_dict = get_models(curr_path, step, device=device)
        if models_dict is None:
            continue

        eig_density_dict[exp_name] = get_point_eig_density(models_dict, data, criterion, device=device, seed=seed)

        # cache data
        if should_cache:
            cache_data(experiment_folder, "point_eig_density", eig_density_dict, meta_dict, step=step)

    return eig_density_dict

def get_exp_linear_loss_trace(experiment_folder, step=-1, seed=0, device=None, num_datapoints=50, on_test_set=False, should_cache=False):
    results_dict = {}
    meta_dict = {"seed": seed}
    cfgs = load_configs(experiment_folder)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        loss_type = cfgs.loc[exp_name]["criterion"]
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        results_dict[exp_name] = get_linear_loss_trace(models_dict, data, loss_type, device=device)

        # cache data
        if should_cache:
            cache_data(experiment_folder, "linear_{}_trace".format(loss_type), results_dict, meta_dict, step=step)

    return results_dict

def get_exp_inp_out_jacobian(experiment_folder, step=-1, seed=0, device=None, num_datapoints=50, on_test_set=False, should_cache=False):
    results_dict = {}
    meta_dict = {"seed": seed}
    cfgs = load_configs(experiment_folder)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        loss_type = cfgs.loc[exp_name]["criterion"]
        models_dict = get_models(curr_path, step, device)
        if models_dict is None:
            continue
        results_dict[exp_name] = get_inp_out_jacobian(models_dict, data, loss_type, device=device)

        # cache data
        if should_cache:
            cache_data(experiment_folder, "inp_out_jacobian".format(loss_type), results_dict, meta_dict, step=step)

    return results_dict

