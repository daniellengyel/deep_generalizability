import numpy as np
import pandas as pd
import torch
import json

from .utils import get_time_stamp
from .training_utils import get_nets

import yaml, os, sys, re, copy

import pickle

def get_exp_steps(experiment_folder):
    exp_steps = {}
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        exp_steps[exp_name] = get_all_steps(curr_path)
    return exp_steps

def get_all_steps(steps_dir):
    step_dict = {}
    for root, dirs, files in os.walk(steps_dir):
        for step_dir in dirs:
            name_split_underscore = step_dir.split("_")
            if len(name_split_underscore) == 1:
                continue
            step_dict[int(name_split_underscore[1])] = step_dir
    return step_dict


def get_models(model_folder_path, step, device=None):
    if step == -1:
        all_steps = get_all_steps(model_folder_path)
        step = int(max(all_steps.keys(), key=lambda x: int(x)))

    model_path = os.path.join(model_folder_path, "step_{}".format(step))
    if not os.path.exists(model_path):
        return None

    models_dict = {}
    for root, dirs, files in os.walk(model_path):
        for model_file_name in files:
            model_idx = model_file_name.split("_")[1].split(".")[0]
            model = load_model(os.path.join(root, model_file_name), device)
            models_dict[model_idx] = model

    return models_dict


def get_all_models(experiment_folder, step, device=None):
    models_dict = {}
    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        # try:
        models_dict[exp_name] = get_models(curr_path, step, device)
        # except:
        #     continue
    return models_dict


def cache_data(
    experiment_folder, name, data, meta_dict, step, create_time_stamp, costum_time_stamp=None, sub_name=None
):
    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    
    cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    if costum_time_stamp is not None:
        cache_folder = os.path.join(cache_folder, costum_time_stamp)
    elif create_time_stamp:
        curr_cache_folder = os.path.join(cache_folder, get_time_stamp(), get_time_stamp(micro_second=True))
        # Practically we should never have two caches with the same time_stamp
        while os.path.exists(curr_cache_folder):
            curr_cache_folder = os.path.join(cache_folder, get_time_stamp(micro_second=True))
        cache_folder = curr_cache_folder
    if sub_name is not None:
        cache_folder = os.path.join(cache_folder, sub_name)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    with open(os.path.join(cache_folder, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    if meta_dict is not None:
        with open(os.path.join(cache_folder, "meta.yml"), "w") as f:
            yaml.dump(meta_dict, f)

def join_cached_sub_data(experiment_folder, name, step):
    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    list_time_subfolders_with_paths = [f.path for f in os.scandir(cache_folder) if f.is_dir()]

    # for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
    res_dict = {}
    cached_meta_data = None
    for curr_time in os.listdir(cache_folder):
        curr_time_dir = os.path.join(cache_folder, curr_time)
        for exp_name in os.listdir(curr_time_dir):
            curr_dir = os.path.join(curr_time_dir, exp_name)
            if not os.path.isdir(curr_dir):
                continue
            cached_data_path = os.path.join(curr_dir, "data.pkl")
            cached_meta_path = os.path.join(curr_dir, "meta.yml")
            if os.path.isfile(cached_meta_path):
                with open(cached_meta_path, "rb") as f:
                    cached_meta_data = yaml.load(f)
            else:
                print("No meta data for exp: {}".format(exp_name))
                cached_meta_data = {}
            
            chached_meta_data_key = json.dumps(cached_meta_data, sort_keys=True)
            if chached_meta_data_key not in res_dict:
                res_dict[chached_meta_data_key] = [{}, cached_meta_data]

            if os.path.isfile(cached_data_path):
                with open(cached_data_path, "rb") as f:
                    cached_data = pickle.load(f)
                for k, v in cached_data.items():
                    res_dict[chached_meta_data_key][0][k] = v
            else:
                cached_data = None

    for v in res_dict.values():
        curr_dict = v[0]
        cached_meta_data = v[1]
        cache_data(experiment_folder, name, curr_dict, cached_meta_data, step, create_time_stamp=False, costum_time_stamp="{}-joined".format(get_time_stamp(micro_second=True)))

    # return cached_data, cached_meta_data

def load_cached_data(experiment_folder, name, step, time_stamp=None):
    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    
    cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    if time_stamp is not None:
        cache_folder = os.path.join(cache_folder, time_stamp)

    cached_data_path = os.path.join(cache_folder, "data.pkl")
    if os.path.isfile(cached_data_path):
        with open(cached_data_path, "rb") as f:
            cached_data = pickle.load(f)
    else:
        cached_data = None

    cached_meta_path = os.path.join(cache_folder, "meta.yml")
    if os.path.isfile(cached_meta_path):
        with open(cached_meta_path, "rb") as f:
            cached_meta_data = yaml.load(f)
    else:
        cached_meta_data = None

    return cached_data, cached_meta_data

def load_all_cached_meta_data(experiment_folder, name, step):
    all_cached_meta_data = {}

    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    
    cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    if not os.path.isdir(cache_folder):
        return None

    for time_stamp in os.listdir(cache_folder):
        if "DS_Store" in time_stamp:
            continue        
        
        curr_cache_folder = os.path.join(cache_folder, time_stamp)

        cached_meta_path = os.path.join(curr_cache_folder, "meta.yml")
        if os.path.isfile(cached_meta_path):
            with open(cached_meta_path, "rb") as f:
                cached_meta_data = yaml.load(f)
        else:
            cached_meta_data = None
        
        all_cached_meta_data[time_stamp] = cached_meta_data

    return all_cached_meta_data   


def get_cached_data_list(experiment_folder):
    pass

def exp_models_path_generator(experiment_folder):
    for curr_dir in os.listdir("{}/models".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/models".format(experiment_folder), curr_dir)
        yield curr_dir, root


def save_models(models, model_name, model_params, experiment_root, curr_exp_name, step, optimizers):
    models_path = os.path.join(experiment_root, "models", curr_exp_name, "step_{}".format(step))
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    for idx_model in range(len(models)):
         torch.save({'model_name': model_name,
                     'model_params': model_params,
                     'model_state_dict': models[idx_model].state_dict(),
                     'optimizer_state_dict': optimizers[idx_model].state_dict()}
                    , os.path.join(models_path, "model_{}.pt".format(idx_model)))

def load_model(PATH, device=None):
    if device is None:
        device = torch.device('cpu')
    meta_data = torch.load(PATH, map_location=device)
    model = get_nets(meta_data["model_name"], meta_data["model_params"], num_nets=1, device=device)[0]
    model.load_state_dict(meta_data['model_state_dict'])
    model.eval()
    return model

def load_configs(experiment_folder):
    config_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "config.yml"), "rb") as f:
            config = yaml.load(f)
        config_dir[curr_dir] = config
        config_dir[curr_dir]["net_params"] = tuple(config_dir[curr_dir]["net_params"])
        if ("softmax_adaptive" in config_dir[curr_dir]) and (
        isinstance(config_dir[curr_dir]["softmax_adaptive"], list)):
            config_dir[curr_dir]["softmax_adaptive"] = tuple(config_dir[curr_dir]["softmax_adaptive"])

    return pd.DataFrame(config_dir).T