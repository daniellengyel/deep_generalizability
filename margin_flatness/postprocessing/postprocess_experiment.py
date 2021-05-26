import numpy as np
import pandas as pd

from .utils import tensorboard_to_dict, get_data_for_experiment
from ..utils import *
from ..save_load import load_all_cached_meta_data, cache_data, load_configs, get_models, exp_models_path_generator
from ..data_getters import *
from ..data_getters import get_random_data_subset
from ..training_utils import get_criterion
from ..nets.Nets import NormOutputNet

from .sharpness import get_affine_trace, sample_average_flatness_pointwise, get_point_traces, get_point_unit_traces
from .model_related import get_point_loss, get_models_loss_acc
from .robustness import get_inp_out_jacobian_points, get_margins, get_max_output

import yaml, os, sys, re, time
from tqdm import tqdm

from ray import tune
import ray

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
def get_exp_loss_acc(experiment_folder, step, seed=0, num_train_datapoints=-1, num_test_datapoints=-1, device=None, check_cache=False):
    meta_dict = {"seed": seed, "num_train_datapoints": num_train_datapoints, "num_test_datapoints": num_test_datapoints, "step": step}

    if check_cache:
        acc_results = check_if_already_computed(experiment_folder, "acc", step, meta_dict)
        loss_results = check_if_already_computed(experiment_folder, "acc", step, meta_dict)

        if acc_results is not None and loss_results is not None:
            print("Got cached results.")
            return loss_results, acc_results

    print("Get loss acc")
    # init
    loss_dict = {}
    acc_dict = {}

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)

    if num_test_datapoints == -1:
        num_test_datapoints = len(test_data)
    test_data = get_random_data_subset(test_data, num_datapoints=num_test_datapoints, seed=seed)

    if num_train_datapoints == -1:
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
        cache_data(experiment_folder, "loss", loss_dict, meta_dict, step=step, create_time_stamp=False)
        cache_data(experiment_folder, "acc", acc_dict, meta_dict, step=step, create_time_stamp=False)

    return loss_dict, acc_dict




def compute_on_experiment(experiment_folder, name, exp_ids, step, seed, num_datapoints, on_test_set, device, verbose=False, check_cache=True, meta=None):

    meta_dict = {"seed": seed, "num_datapoints": num_datapoints, "on_test_set": on_test_set, "step": step}
    meta_dict["meta"] = meta

    if check_cache:
        cached_results = check_if_already_computed(experiment_folder, name, step, meta_dict)

        if cached_results is not None:
            print("Got cached results.")
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
        if (not exp_ids is None) and (exp_name not in exp_ids):
            continue
        if meta_dict["meta"] is None or meta_dict["meta"]["criterion"] is None:
            loss_type = cfgs.loc[exp_name]["criterion"]
        else:
            loss_type = meta_dict["meta"]["criterion"]
        criterion = get_criterion(loss_type=loss_type)

        models_dict = get_models(curr_path, step, device)
        if (meta_dict["meta"] is not None) and ("normalize_output" in meta_dict["meta"]) and (meta_dict["meta"]["normalize_output"]):
            for k, m in models_dict.items():
                models_dict[k] = NormOutputNet(m)

        if models_dict is None:
            continue

        results_dict[exp_name] = helper_compute_on_experiment(name, models_dict, data, seed, criterion, loss_type, device=device, meta=meta_dict['meta']) # potentially change how we do loss_type and criterion

    # cache data
    cache_data(experiment_folder, name, results_dict, meta_dict, step=step, create_time_stamp=True)

    return results_dict

# iterate through models
def get_model_results(exp_id_path, experiment_folder, cfgs, step, name, data, seed, time_stamp, meta_dict, on_gpu=False):    
    exp_name, curr_path = exp_id_path["exp_paths"]
    if on_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")
    
    cfg = cfgs.loc[exp_name]

    if meta_dict["meta"] is None or meta_dict["meta"]["criterion"] is None:
        loss_type = cfg["criterion"]
    else:
        loss_type = meta_dict["meta"]["criterion"]
    criterion = get_criterion(loss_type=loss_type)


    models_dict = get_models(curr_path, step, device)
    if models_dict is None:
        return

    if ("meta" in meta_dict) and ("normalize_output" in meta_dict["meta"]) and (meta_dict["meta"]["normalize_output"]):
        for k, m in models_dict.items():
            models_dict[k] = NormOutputNet(m)

    results_dict = {}
    # results_dict[exp_name] = exp_name

    results_dict[exp_name] = helper_compute_on_experiment(name, models_dict, data, seed, criterion, loss_type, device=device, meta=meta_dict['meta']) # potentially change how we do loss_type and criterion

    # cache data
    cache_data(experiment_folder, name, results_dict, meta_dict, step=step, create_time_stamp=False, costum_time_stamp=time_stamp, sub_name=exp_name)

    

def multi_compute_on_experiment(experiment_folder, name, exp_ids_paths, step, seed, num_datapoints, on_test_set, num_cpus, num_gpus=0, verbose=False, meta=None):
    meta_dict = {"seed": seed, "num_datapoints": num_datapoints, "on_test_set": on_test_set, "step": step}
    meta_dict["meta"] = meta

    cfgs = load_configs(experiment_folder).loc[[exp_id[0] for exp_id in exp_ids_paths]]
    exp_name_paths = {"exp_paths": tune.grid_search(exp_ids_paths)}
    time_stamp = get_time_stamp(micro_second=True)

    # get data
    train_data, test_data = get_data_for_experiment(experiment_folder)
    if on_test_set:
        data = get_random_data_subset(test_data, num_datapoints=num_datapoints, seed=seed)
    else:
        data = get_random_data_subset(train_data, num_datapoints=num_datapoints, seed=seed)

    if len(exp_ids_paths) == 1:
        get_model_results({"exp_paths": exp_ids_paths[0]}, experiment_folder, cfgs, step, name, data, seed, time_stamp, meta_dict)
        return

    ray.shutdown()
    ray.init(_temp_dir='/rds/general/user/dl2119/ephemeral', num_cpus=num_cpus, num_gpus=num_gpus)

    if num_gpus > 0:
        tune.run(tune.with_parameters(lambda exp_id_path, data: get_model_results(exp_id_path, experiment_folder, cfgs, step, name, data, seed, time_stamp, meta_dict, on_gpu=True), data=data), config=exp_name_paths, resources_per_trial={'gpu': 1})
    else:
        tune.run(tune.with_parameters(lambda exp_id_path, data: get_model_results(exp_id_path, experiment_folder, cfgs, step, name, data, seed, time_stamp, meta_dict), data=data), config=exp_name_paths)
    
    return time_stamp


def helper_compute_on_experiment(name, models_dict, data, seed, criterion, loss_type, device=None, meta=None):
    """robustness: [inp_out_jacobian, softmax_margins, output_margins, affine_upperbound_margins, point_loss, point_unit_loss]
    flatness: [affine_trace, point_traces, point_unit_traces, sample_average_flatness_pointwise]"""
    
    if name == "inp_out_jacobian":
        return get_inp_out_jacobian_points(models_dict, data, device=device)

    elif name == "sample_average_flatness_pointwise":
        return sample_average_flatness_pointwise(models_dict, data, criterion, meta, seed=seed)

    elif name == "affine_trace":
        return get_affine_trace(models_dict, data, loss_type, device=device)

    elif name == "point_traces":
        return get_point_traces(models_dict, data, criterion, device=device, seed=seed)

    elif name == "point_unit_traces":
        return get_point_unit_traces(models_dict, data, criterion, device=device, seed=seed)

    elif name == "softmax_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=False, softmax_outputs=True, seed=seed)

    elif name == "output_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=False, softmax_outputs=False, seed=seed)

    elif name == "point_output":
        return get_max_output(models_dict, data, device=device, get_upperbound=False, softmax_outputs=False, seed=seed)

    elif name == "affine_upperbound_margins":
        return get_margins(models_dict, data, device=device, get_upperbound=True, softmax_outputs=True, seed=seed)

    elif name == "point_loss":
        return get_point_loss(models_dict, data, loss_type, device=device)
        
    elif name == "point_unit_loss":
        return get_point_loss(models_dict, data, loss_type, device=device, unit_output=True)

    elif name == "model_loss_acc":
        return get_models_loss_acc(models_dict, data, criterion, loss_type, device=device)

def check_if_already_computed(experiment_folder, name, step, meta_data):

    all_cached_meta_data = load_all_cached_meta_data(experiment_folder, name, step)
    if all_cached_meta_data is None:
        return None

    for time_stamp in all_cached_meta_data:
        if meta_data == all_cached_meta_data[time_stamp]:
            return load_cached_data(experiment_folder, name, step, time_stamp)[0]
    
    return None