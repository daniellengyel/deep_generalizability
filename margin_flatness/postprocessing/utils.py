import numpy as np
import pandas as pd
import torch

from ..utils import *

import yaml, os, sys, re

from ..data_getters import get_data, get_exp_steps
from ..save_load import load_configs

import pickle

def get_data_for_experiment(experiment_path): # potentially store data? Have this be a class which caches the last data loaded? 
    all_nets = ["SimpleNet", "LinearNet", "BatchNormSimpleNet", "KeskarC3", "LeNet"]
    
    cfs = load_configs(experiment_path)
    if "data_name" in cfs.iloc[0]:
        data_name = cfs.iloc[0]["data_name"]
    else:
        data_name = experiment_path.split("/")[-2]

    assert cfs.iloc[0]["net_name"] in all_nets
    vectorized = cfs.iloc[0]["net_name"] in ["SimpleNet", "LinearNet", "BatchNormSimpleNet"]
    reduce_train_per = cfs.iloc[0]["reduce_train_per"]
    seed = cfs.iloc[0]["seed"]
    meta = cfs.iloc[0]["data_meta"]
    train_data, test_data = get_data(data_name, vectorized=vectorized, reduce_train_per=reduce_train_per, seed=seed, meta=meta)
    return train_data, test_data


def non_constant_cols(df):
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] != a[1:]).any(0)


def get_hp(cfs):
    filter_cols = non_constant_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict

def get_all_steps_f(experiment_folder, f):
    """f is meant to be a function of step and will compute metrics and other things solely dependend on the current step. It is also responsible for saving, so nothing is returned."""
    all_steps = get_exp_steps(experiment_folder)
    all_steps = np.array([list(v.keys()) for v in all_steps.values()]).reshape(-1)
    all_steps = set(all_steps)

    for step in all_steps:
        f(step)

# +++ process tensorboard results +++
def tensorboard_to_dict(path_to_events_file, names):
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