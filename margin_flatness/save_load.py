import numpy as np
import pandas as pd
import torch

from .utils import *
from .training_utils import get_nets

import yaml, os, sys, re, copy
from .nets.Nets import *

from .data_getters import *

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


def get_all_models(experiment_folder, step):
    models_dict = {}
    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        try:
            models_dict[exp_name] = get_models(curr_path, step)
        except:
            continue
    return models_dict


def cache_data(experiment_folder, name, data, meta_dict=None, step=None):
    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    if step is not None:
        cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    with open(os.path.join(cache_folder, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    if meta_dict is not None:
        with open(os.path.join(cache_folder, "meta.yml"), "w") as f:
            yaml.dump(meta_dict, f)

def load_cached_data(experiment_folder, name, step=None):
    """Available names: ["tsne", "runs", "trace", "acc", "dist", "loss", "grad", "eig"]"""
    cache_folder = os.path.join(experiment_folder, "postprocessing", name)
    if step is not None:
        cache_folder = os.path.join(cache_folder, "step_{}".format(step))

    cached_data_path = os.path.join(cache_folder, "data.pkl")
    if os.path.isfile(cached_data_path):
        with open(cached_data_path, "rb") as f:
            cached_data = pickle.load(f)
    else:
        cached_data =  None
    
    cached_meta_path = os.path.join(cache_folder, "meta.yaml")
    if os.path.isfile(cached_meta_path):
        with open(cached_meta_path, "rb") as f:
            cached_meta_data = yaml.load(f)
    else:
        cached_meta_data  = None

    return cached_data, cached_meta_data

def get_cached_data_list(experiment_folder):
    pass

def exp_models_path_generator(experiment_folder):
    for curr_dir in os.listdir("{}/models".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/models".format(experiment_folder), curr_dir)
        yield curr_dir, root


def save_models(models, model_name, model_params, experiment_root, curr_exp_name, step):
    models_path = os.path.join(experiment_root, "models", curr_exp_name, "step_{}".format(step))
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    for idx_model in range(len(models)):
         torch.save({'model_name': model_name,
                     'model_params': model_params,
                     'model_state_dict': models[idx_model].state_dict()}
                    , os.path.join(models_path, "model_{}.pt".format(idx_model)))

def load_model(PATH, device=None):
    if device is None:
        device = torch.device('cpu')
    meta_data = torch.load(PATH, map_location=device)
    model = get_nets(meta_data["model_name"], meta_data["model_params"], num_nets=1, device=device)[0]
    model.load_state_dict(meta_data['model_state_dict'])
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