import numpy as np
import pandas as pd
import sys


import re

from .utils import *
from .lineages import *

import itertools


def get_runs_arr(runs, var_name="Kish", exp_ids=None, running_average_gamma=1, is_mean=False):
    all_arrs = []
    val_steps = []

    for i in runs:

        curr_arr = None
        curr_val_steps = None
        if (exp_ids is not None) and (i not in exp_ids):
            continue

        for step in sorted(runs[i], key=lambda x: int(x)):
            try:

                # going down the tree with node names given by var_name.split("/")
                curr_dict = runs[i][step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]
                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed

                    to_append = np.array([[curr_dict["net"][str(nn)] for nn in range(num_nets)]])
                    if is_mean:
                        to_append = np.mean(to_append).reshape(1, -1)
                else:
                    to_append = np.array([curr_dict[""]])

                if curr_arr is None:
                    curr_arr = to_append
                else:
                    to_append = curr_arr[-1] * (1 - running_average_gamma) + running_average_gamma * to_append
                    curr_arr = np.concatenate((curr_arr, to_append), axis=0)

                if curr_val_steps is None:
                    curr_val_steps = [step]
                else:
                    curr_val_steps.append(step)
            except:
                pass
                # print("No {} for step {}".format(var_name, step))

        all_arrs.append(curr_arr)
        val_steps.append(curr_val_steps)

    return val_steps, np.array(all_arrs)

def get_metric_at_training_step(runs, var_name, step, exp_ids=None):
    exp_res = {}

    for exp_id in runs:

        if (exp_ids is not None) and (exp_id not in exp_ids):
            continue

        if step == -1:
            try:
                last_step = max(runs[exp_id], key=lambda x: int(x))
                curr_step = last_step
            except:
                continue
        else:
            curr_step = step

        stop_trying = False
        res = None
        while not stop_trying:
            try:
                # going down the tree with node names given by var_name.split("/")
                curr_dict = runs[exp_id][curr_step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]

                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed
                    res = np.array([curr_dict["net"][str(nn)] for nn in range(num_nets)])
                    exp_res[exp_id] = res
                else:
                    res = curr_dict[""]
                    exp_res[exp_id] = res

                stop_trying = True
            except:
                if (step == -1) and (curr_step > last_step - 5):
                    curr_step -= 1
                else:
                    stop_trying = True
        if res is None:
            print("No {} for step {}".format(var_name, step))
            exp_res[exp_id] = None

    return exp_res

def get_filtered_training_metrics(exp_dict, exp_ids, metric_name, path_aggregator=None):
    """Will apply lineages if a proper resampling array exists. Only returns results for arrays corresponding
    to the lineage of the spawn particle. 
    Assumes every experiement has same number of steps"""
    exp_sampling_arr = exp_dict["resampling_idxs"]
    exp_runs = exp_dict["runs"]

    y_arr = []
    x_arr = []

    for i, exp_id in enumerate(exp_ids):

        sampling_arr = exp_dict["resampling_idxs"][exp_id]
        x_vals, y_vals = get_runs_arr(exp_dict, metric_name, exp_ids, is_mean=False)

        if len(sampling_arr) <= 2:
            Ys = y_vals[i].T
        else:
            resampling_arr = np.array([sampling_arr[str(i)] for i in range(len(sampling_arr))])[1:-1] # Note, each element is the parent particle that was chosen. So the lineages are shifted to align with the values.

            curr_lineage, curr_assignments = find_lineages(resampling_arr)
            Ys = get_linages_vals(curr_lineage,  y_vals[i], x_arr=x_vals[i]).values()
        
        # TODO do selector stuff here
        if path_aggregator is not None:
            Ys = np.array([np.mean(Ys, axis=0)])

        y_arr.append(Ys)
        x_arr.append(np.array([x_vals[0] for _ in range(len(Ys))]))

    y_arr = np.array(y_arr)
    x_arr = np.array(x_arr)
    y_arr = np.concatenate(y_arr, axis=0)
    return y_arr, x_arr


