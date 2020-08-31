import numpy as np
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpl
from cycler import cycler
import torch
from torch.utils.data import DataLoader
import sys

import re

from .postprocessing import *
from .training_metrics import * 
from .utils import * 
from ..nets import Nets
from ..utils import *

import itertools


def get_end_stats(exp_folder):
    
    runs, _ = load_cached_data(exp_folder, "runs")
    trace, _ = load_cached_data(exp_folder, "trace", step=-1) # assume the trace i get is from the end.
    acc, _ = load_cached_data(exp_folder, "acc", step=-1)
    loss, _ = load_cached_data(exp_folder, "loss", step=-1)
    dist, _ = load_cached_data(exp_folder, "dist")

    stats_dict = {}
    configs = load_configs(exp_folder)
    for exp_id in configs.index:

        num_nets = configs.loc[exp_id]["num_nets"]
        if runs is not None:
            num_steps = max(runs[exp_id], key=lambda x: int(x)) - 1

        stats_dict[str(exp_id)] = {}

        if loss is not None:
            Loss_train_list = [loss[exp_id][str(nn)][0] for nn in range(num_nets)]
            Loss_test_list = [loss[exp_id][str(nn)][1] for nn in range(num_nets)]

            stats_dict[str(exp_id)]["Loss Test Mean"] = np.mean(Loss_test_list)
            stats_dict[str(exp_id)]["Loss Test Max"] = np.max(Loss_test_list)
            stats_dict[str(exp_id)]["Loss Test Min"] = np.min(Loss_test_list)

            stats_dict[str(exp_id)]["Loss Train Mean"] = np.mean(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Max"] = np.max(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Min"] = np.min(Loss_train_list)

        if acc is not None:
            Acc_train_list = [acc[exp_id][str(nn)][0] for nn in range(num_nets)]
            Acc_test_list = [acc[exp_id][str(nn)][1] for nn in range(num_nets)]

            stats_dict[str(exp_id)]["Acc Train Mean"] = np.mean(Acc_train_list)
            stats_dict[str(exp_id)]["Acc Train Max"] = np.max(Acc_train_list)
            stats_dict[str(exp_id)]["Acc Train Min"] = np.min(Acc_train_list)

            stats_dict[str(exp_id)]["Acc Test Mean"] = np.mean(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Max"] = np.max(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Min"] = np.min(Acc_test_list)

            stats_dict[str(exp_id)]["Acc Gap Mean"] = stats_dict[str(exp_id)]["Acc Test Mean"] - \
                                                    stats_dict[str(exp_id)]["Acc Train Mean"]

        if dist is not None:
            stats_dict[str(exp_id)]["Dist Mean"] = np.mean(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Max"] = np.max(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Min"] = np.min(dist[exp_id])
        
        if runs is not None:
            norm_list = np.array([runs[exp_id][num_steps]["Norm"]["net"][str(nn)] for nn in range(num_nets)])
            stats_dict[str(exp_id)]["Norm Mean"] = np.mean(norm_list)
            stats_dict[str(exp_id)]["Norm Max"] = np.max(norm_list)
            stats_dict[str(exp_id)]["Norm Min"] = np.min(norm_list)

        if trace is not None:
            Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            stats_dict[str(exp_id)]["Trace Mean"] = np.mean(Trace_list)
            stats_dict[str(exp_id)]["Trace Mean Std"] = np.mean(Trace_std_list)
            stats_dict[str(exp_id)]["Trace Max"] = np.max(Trace_list)
            stats_dict[str(exp_id)]["Trace Min"] = np.min(Trace_list)

    stats_pd = pd.DataFrame(stats_dict).T

    # append hyperparameters to DataFrame
    cfs_hp = get_hp(configs)
    cfs_hp_df = configs[list(cfs_hp.keys())]
    stats_pd = pd.concat([stats_pd, cfs_hp_df], axis=1)

    return stats_pd


def _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds, save_location=None):
    if len(plots_names) > 1:
        plt.legend(tuple(plots),
                   plots_names,
                   scatterpoints=1,
                   loc='best',
                   ncol=3,
                   )
    elif len(plots) > 1:
        plt.legend(loc='upper right')

    config = {"axes.titlesize" : 24,
                "axes.labelsize" : 22,
                "lines.linewidth" : 3,
                "lines.markersize" : 10,
                "xtick.labelsize" : 18,
                "ytick.labelsize" : 18,
                'grid.linestyle': 'dashed',
                'legend.fontsize': 22,
            }
    plt.grid(True)

    plt.rcParams.update(config) 

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["figure.figsize"] = (13,9)


    plt.xlabel(X_axis_name, )
    plt.ylabel(Y_axis_name, )

    if X_axis_bounds is not None:
        plt.xlim(X_axis_bounds)
    if Y_axis_bounds is not None:
        plt.ylim(Y_axis_bounds)

    color_selector = np.arange(12)
    color_selector[0] = 3
    color_selector[1] = 1
    color_selector[3] = 0
    color = plt.cm.tab20(color_selector)
    mpl.rcParams['axes.prop_cycle'] = cycler('color', color) # plt.cm.Set3(np.arange(12))))
    if save_location is not None:
        plt.savefig(save_location + ".png")#, format='eps')
    plt.show()


def id_selection_from_hyperparameters_generator(cfs, filter_seperate, filter_not_seperate):
    if filter_seperate is None:
        filter_seperate = []
    if filter_not_seperate is None:
        filter_not_seperate = []

    if (filter_not_seperate == []) and (filter_seperate == []):
        yield list(cfs.index), "all"
        yield None, None
        return 

    unique_seperate_filter_dict = {f: list(set(cfs[f])) for f in filter_seperate}
    unique_seperate_filter_keys = list(unique_seperate_filter_dict.keys())

    unique_not_seperate_filter_dict = {f: list(set(cfs[f])) for f in filter_not_seperate}
    unique_not_seperate_filter_keys = list(unique_not_seperate_filter_dict.keys())

    unique_all_filter_keys = unique_seperate_filter_keys + unique_not_seperate_filter_keys

    for s_comb in itertools.product(*unique_seperate_filter_dict.values()):

        for ns_comb in itertools.product(*unique_not_seperate_filter_dict.values()):
            comb = s_comb + ns_comb

            exp_ids = list(cfs[(cfs[unique_all_filter_keys] == comb).to_numpy().all(1)].index)

            yield exp_ids, comb
        
        yield None, s_comb


def get_mod_idx(arr, mod):
    if mod == "all":
        return list(range(len(arr)))

    if mod == "max":
        idx = np.argmax(arr)
    else:
        idx = np.argmin(arr)
    return [idx]


def cached_data_selector(experiment_folder, metric_name):
    # name_mod_split = axis_name.split(":")
    # if len(name_mod_split) == 2:
    #     name, mod = name_mod_split
    # else:
    #     name, mod = axis_name, None

    # name_split = name.split(" ")

    # tmp_cache = {}

    # get the relevant data i think. I would like to also determine for which step

    metric_data, cached_meta_data = load_cached_data(exp_folder, metric_name)

    def helper(exp_id):
        if name_split[0] == "eigs":
            eigs = exp_dict["stuff"]["eig"][exp_id][str(nn_idx)]
            if name_split[1] == "min":
                return min(eigs)
            else:
                return max(eigs)
        elif name_split[0] in exp_dict["stuff"]["configs"].loc[exp_id]:
            return exp_dict["stuff"]["configs"].loc[exp_id][name_split[0]]
        elif name_split[0] == "grad":
            return exp_dict["stuff"]["grad"][exp_id][str(nn_idx)]
        elif name_split[0] == "trace":
            return np.mean(exp_dict["stuff"]["trace"][exp_id][str(nn_idx)])
        elif name_split[0] == "gap":
            if name_split[1] == "acc":
                d = exp_dict["stuff"]["acc"][exp_id][str(nn_idx)]
            else:
                d = exp_dict["stuff"]["loss"][exp_id][str(nn_idx)]
            return d[1] - d[0]

        else:
            xs = exp_dict["stuff"][name_split[0]][exp_id][str(nn_idx)]
            if name_split[1] == "train":
                return xs[0]
            else:
                return xs[1]

    return helper





def hp_data_func_plot(experiment_folder, data_func, X_axis_name, Y_axis_name, plot_name, filter_seperate=None, filter_not_seperate=None,
                 save_exp_path=None, X_axis_bounds=None, Y_axis_bounds=None, pre_filtered_exp_ids=None):
    plots = []
    plots_names = []

    cfs = load_configs(experiment_folder)

    for exp_ids, comb in id_selection_from_hyperparameters_generator(cfs, filter_seperate, filter_not_seperate):
        if (exp_ids is None):
            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}_{}".format(plot_name, X_axis_name, Y_axis_name.replace("/", "-"), str(comb)))
            else:
                save_location = None

            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    save_location=save_location)

            plots = []
            plots_names = []
        else:
            if pre_filtered_exp_ids is not None:
                exp_ids = list(set(exp_ids) & set(pre_filtered_exp_ids))

            if len(exp_ids) == 0:
                continue

            plots, plots_names = data_func(exp_ids, plots, plots_names, comb)


# ++++ functions for hp_data_func_plot ++++
def margin_trace_correct_incorrect_plot(margins_filters, point_traces, use_correct_filter=False):
    def xy_func(exp_ids, plots, plots_names, comb=None):
        x_correct = []
        y_correct = []

        x_incorrect = []
        y_incorrect = []
        for exp_id in exp_ids:
            for model_idx in margins_filters[exp_id].keys():

                curr_point_margins, correct_filters = margins_filters[exp_id][model_idx]
                curr_point_traces = np.array(point_traces[exp_id][model_idx])
                if use_correct_filter:
                    x_correct.append(curr_point_margins[correct_filters])
                    y_correct.append(curr_point_traces[correct_filters])

                    x_incorrect.append(curr_point_margins[~correct_filters])
                    y_incorrect.append(curr_point_traces[~correct_filters])
                else:
                    x_correct.append(curr_point_margins)
                    y_correct.append(curr_point_traces)
        if use_correct_filter:
            x_correct = np.concatenate(x_correct, axis=0)
            y_correct = np.concatenate(y_correct, axis=0)

            x_incorrect = np.concatenate(x_incorrect, axis=0)
            y_incorrect = np.concatenate(y_incorrect, axis=0)

            if len(x_correct) > 0:
                plots.append(plt.scatter(x_correct.T, y_correct.T))
                plots_names.append("Correct, {}".format(comb))

            if len(x_incorrect) > 0:
                plots.append(plt.scatter(x_incorrect.T, y_incorrect.T))
                plots_names.append("Incorrect, {}".format(comb))
        else:
            x_data = np.concatenate(x_correct, axis=0)
            y_data = np.concatenate(y_correct, axis=0)
            
            plots.append(plt.scatter(x_data.T, y_data.T))
            plots_names.append("All, {}".format(comb))
        
        return plots, plots_names
        
    return xy_func

def margins_correct_incorrect_hist_plot(margins_filters, correct_filter=False):
    def xy_func(exp_ids, plots, plots_names, comb):
        x_correct = []
        x_incorrect = []
        
        for exp_id in exp_ids:
            for model_idx in margins_filters[exp_id].keys():

                margins, correct_filters = margins_filters[exp_id][model_idx]
                if correct_filter:
                    x_correct.append(margins[correct_filters])
                    x_incorrect.append(margins[~correct_filters])
                else:
                    x_correct.append(margins)
                    
        if correct_filter:
            x_correct = np.concatenate(x_correct, axis=0)
            x_incorrect = np.concatenate(x_incorrect, axis=0)
        
            if len(x_correct) > 0:
                plots.append(plt.hist(x_correct, bins=100, histtype='step', label="Correct, {}".format(comb)))

            if len(x_incorrect) > 0:
                plots.append(plt.hist(x_incorrect, bins=100, histtype='step', label="Incorrect, {}".format(comb)))
        else:
            x_data = np.concatenate(x_correct, axis=0)
            plots.append(plt.hist(x_data, bins=100, histtype='step', label=comb))

        return plots, plots_names
        
    return xy_func


def plot_stats(stats_pd, X_axis_name, Y_axis_name):

    def xy_func(exp_ids, plots, plots_names, comb):
        x_correct = []
        x_incorrect = []
        
        filter_pd = stats_pd.loc[exp_ids]

        x_values = filter_pd[X_axis_name].to_numpy()
        y_values = filter_pd[Y_axis_name].to_numpy()

        plots.append(plt.scatter(x_values, y_values))
        plots_names.append(comb)

        return plots, plots_names
        
    return xy_func

def timeseries_plot(exp_folder, metric_name):
    runs_data, cached_meta_data = load_cached_data(exp_folder, "runs")

    def xy_func(exp_ids, plots, plots_names, comb):

        x_arr, y_arr = get_timeseries_training_metrics(exp_ids, metric_name, runs_data=runs_data, path_aggregator=None)
        plots.append(plt.plot(x_arr.T, y_arr.T)[0])

        return plots, plots_names
    return xy_func

def simple_data_scatter_plot(X_data, Y_data):

    def xy_func(exp_ids, plots, plots_names, comb):

        x = np.concatenate(np.array([[v for v in X_data[exp_id].values()] for exp_id in exp_ids]), axis=0)
        y = np.concatenate(np.array([[v for v in Y_data[exp_id].values()] for exp_id in exp_ids]), axis=0)

        if plot_type == "scatter":
            plots.append(plt.scatter(x.T, y.T))
        else: 
            plots.append(plt.plot(x.T, y.T)[0])
        
        plots_names.append(comb)
        return plots, plots_names
    return xy_func


def metric_scatter_plot(experiment_folder, X_metric_name, Y_metric_name, X_modifier=None, Y_modifier=None):
    # get the metric selector

    X_selector = get_selector_mod(experiment_folder, X_metric_name)
    Y_selector = get_selector_mod(experiment_folder, Y_metric_name)

    assert not ((X_modifier is not None) and (Y_modifier is not None))    
    assert (X_modifier is not None) or (Y_modifier is not None)


    def xy_func(exp_ids, plots, plots_names, comb):
        # just return what the selector gives i suppose

        x_vals = []
        y_vals = []
        for exp_id in exp_ids:
            num_nets = exp_dict["stuff"]["configs"].loc[exp_id]["num_nets"]
            Xs = [X_selector(exp_id, i) for i in range(num_nets)]
            Ys = [Y_selector(exp_id, i) for i in range(num_nets)]

            if X_mod is not None:
                nn_idxs = get_mod_idx(Xs, X_modifier)
            else:
                nn_idxs = get_mod_idx(Ys, Y_modifier)

            if (len(nn_idxs) == 0):
                continue

            x_vals.append([X_selector(exp_id, i) for i in nn_idxs])
            y_vals.append([Y_selector(exp_id, i) for i in nn_idxs])

        return np.array(x_vals).reshape(-1), np.array(y_vals).reshape(-1)
