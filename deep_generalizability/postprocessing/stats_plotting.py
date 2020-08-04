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
from .lineages import *
from .utils import * 
from ..nets import Nets
from ..utils import *

import itertools


def get_end_stats(exp_dict):
    stuff = exp_dict["stuff"]
    stats_dict = {}

    if "runs" in stuff:
        runs = stuff["runs"]

    else:
        runs = None

    if "trace" in stuff:
        trace = stuff["trace"]  # assume the trace i get is from the end.
    else:
        trace = None

    if "acc" in stuff:
        accs = stuff["acc"]
    else:
        accs = None

    if "loss" in stuff:
        loss = stuff["loss"]
    else:
        loss = None

    if "dist" in stuff:
        dist = stuff["dist"]
    else:
        dist = None

    configs = stuff["configs"]
    for exp_id in configs.index:

        num_nets = configs.loc[exp_id]["num_nets"]
        if runs is not None:
            try:
                num_steps = max(runs[exp_id], key=lambda x: int(x)) - 1
            except:
                continue

        try:
            stats_dict[str(exp_id)] = {}

            if loss is None:
                Loss_train_list = [runs[exp_id][num_steps]["Loss"]["train"]["net"][str(nn)] for nn in range(num_nets)]
            else:
                Loss_train_list = [loss[exp_id][str(nn)][0] for nn in range(num_nets)]
                Loss_test_list = [loss[exp_id][str(nn)][1] for nn in range(num_nets)]

                stats_dict[str(exp_id)]["Loss Test Mean"] = np.mean(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Test Max"] = np.max(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Test Min"] = np.min(Loss_test_list)

            stats_dict[str(exp_id)]["Loss Train Mean"] = np.mean(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Max"] = np.max(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Min"] = np.min(Loss_train_list)

            if accs is None:
                Acc_test_list = [runs[exp_id][num_steps]["Accuracy"]["net"][str(nn)] for nn in range(num_nets)]
            else:
                Acc_train_list = [accs[exp_id][str(nn)][0] for nn in range(num_nets)]
                Acc_test_list = [accs[exp_id][str(nn)][1] for nn in range(num_nets)]

                stats_dict[str(exp_id)]["Acc Train Mean"] = np.mean(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Train Max"] = np.max(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Train Min"] = np.min(Acc_train_list)

            stats_dict[str(exp_id)]["Acc Test Mean"] = np.mean(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Max"] = np.max(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Min"] = np.min(Acc_test_list)

            if accs is not None:
                stats_dict[str(exp_id)]["Gap Mean"] = stats_dict[str(exp_id)]["Acc Test Mean"] - \
                                                      stats_dict[str(exp_id)]["Acc Train Mean"]

            if ("runs" in stuff) and ("resampling_idxs" in exp_dict):
                # get mean total path weigth
                Y_axis_name = "Potential/curr"

                x_vals, y_vals = get_runs_arr(exp_dict, Y_axis_name, exp_ids=[exp_id], is_mean=False)

                try:
                    sampling_arr = exp_dict["resampling_idxs"][exp_id]
                    resampling_arr = np.array([sampling_arr[str(i)] for i in range(len(sampling_arr))])[1:-1]

                    curr_lineage, curr_assignments = find_lineages(resampling_arr)
                    Ys = get_linages_vals(curr_lineage, y_vals[0])
                except:
                    Ys = y_vals[0].T
                    print("Did not use lineages for {}".format(exp_id))
                stats_dict[str(exp_id)]["Path Weight Sum"] = np.mean(np.sum(Ys, axis=1))

        except:
            print("Error: No stats for {}".format(exp_id))

        if dist is not None:
            stats_dict[str(exp_id)]["Dist Mean"] = np.mean(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Max"] = np.max(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Min"] = np.min(dist[exp_id])
        else:
            print("YO")
            norm_list = np.array([runs[exp_id][num_steps]["Norm"]["net"][str(nn)] for nn in range(num_nets)])

            stats_dict[str(exp_id)]["Norm Mean"] = np.mean(norm_list)
            stats_dict[str(exp_id)]["Norm Max"] = np.max(norm_list)
            stats_dict[str(exp_id)]["Norm Min"] = np.min(norm_list)

        if trace is not None:

            try:
                Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
                Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
                stats_dict[str(exp_id)]["Trace Mean"] = np.mean(Trace_list)
                stats_dict[str(exp_id)]["Trace Mean Std"] = np.mean(Trace_std_list)
                stats_dict[str(exp_id)]["Trace Max"] = np.max(Trace_list)
                stats_dict[str(exp_id)]["Trace Min"] = np.min(Trace_list)

                # print(dict_key)
                # print(trace[exp_id][str(0)])
                # print()

                # stats_dict[str(exp_id)]["Train Loss/Trace Correlation"] = get_correlation(Loss_train_list, Trace_list)
                # stats_dict[str(exp_id)]["Test Acc/Trace Correlation"] = get_correlation(Acc_test_list, Trace_list)
            except:
                print("Error: No trace for {}".format(exp_id))

    #         print("Mean Loss: {:.4f}".format(np.mean(Loss_list)))
    #         print("Mean Trace: {:.4f}".format(np.mean(Trace_list)))
    #         print("Mean Acc: {:.4f}".format(np.mean(Acc_list)))

    #         print("Loss/Trace Correlation: {:.4f}".format(get_correlation(Loss_list, Trace_list)))
    #         print("Acc/Trace Correlation: {:.4f}".format(get_correlation(Acc_list, Trace_list)))

    #         print("")

    stats_pd = pd.DataFrame(stats_dict).T

    cfs_hp = get_hp(configs)
    cfs_hp_df = configs[list(cfs_hp.keys())]
    stats_pd = pd.concat([stats_pd, cfs_hp_df], axis=1)

    return stats_pd


def _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds, save_location=None):
    print(len(plots_names))
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


def plot_stats(stats_pd, X_axis_name, Y_axis_name, Z_axis_name=None, filter_seperate=None, filter_not_seperate=None, save_exp_path=None, X_axis_bounds=None,
               Y_axis_bounds=None, X_axis_display_name=None, Y_axis_display_name=None):
    plots = []
    plots_names = []

    if (filter_seperate is None) and (filter_not_seperate is None):
        x_values = stats_pd[X_axis_name].to_numpy()
        y_values = stats_pd[Y_axis_name].to_numpy()

        plots.append(plt.scatter(x_values, y_values))
        plots_names.append("Plot all")
        if save_exp_path is not None:
            save_location = os.path.join(save_exp_path,
                                         "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str("all")))
        else:
            save_location = None

        _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
              X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name,
              save_location=save_location)
    else:
        if (("lr_bs_ratio" in filter_seperate) or ("lr_bs_ratio" in filter_not_seperate)) or (
                X_axis_name == "lr_bs_ratio") or (
                Y_axis_name == "lr_bs_ratio"):
            stats_pd["lr_bs_ratio"] = stats_pd["learning_rate"] / stats_pd["batch_train_size"]

        for exp_ids in id_selection_from_hyperparameters_generator(cfs, filter_seperate, filter_not_seperate):

            filter_pd = stats_pd[(stats_pd[unique_all_filter_keys] == comb).to_numpy().all(1)]

            x_values = filter_pd[X_axis_name].to_numpy()
            y_values = filter_pd[Y_axis_name].to_numpy()
            if Z_axis_name is not None:
                y_values = filter_pd[Y_axis_name].to_numpy()

            plots.append(plt.scatter(x_values, y_values))
            plots_names.append(comb)

            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str(s_comb)))
            else:
                save_location = None

            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name, save_location=save_location)

            plots = []
            plots_names = []


def get_mod_idx(arr, mod):
    if mod == "all":
        return list(range(len(arr)))

    if mod == "max":
        idx = np.argmax(arr)
    else:
        idx = np.argmin(arr)
    return [idx]


def get_selector_mod(exp_dict, axis_name):
    name_mod_split = axis_name.split(":")
    if len(name_mod_split) == 2:
        name, mod = name_mod_split
    else:
        name, mod = axis_name, None

    name_split = name.split(" ")

    tmp_cache = {}

    def helper(exp_id, nn_idx):
        if name_split[0] ==  "path":
            if exp_id not in tmp_cache:
                # get mean total path weigth
                Y_axis_name = "Potential/curr"

                x_vals, y_vals = get_runs_arr(exp_dict, Y_axis_name, exp_ids=[exp_id], is_mean=False)
                print(y_vals.shape)
                try:
                    sampling_arr = exp_dict["resampling_idxs"][exp_id]
                    resampling_arr = np.array([sampling_arr[str(i)] for i in range(len(sampling_arr))])[1:-1]

                    curr_lineage, curr_assignments = find_lineages(resampling_arr)
                    inverted_curr_assignments = {}
                    for k, v in curr_assignments.items():
                        for vv in v:
                            inverted_curr_assignments[vv] = k
                    Ys = get_linages_vals(curr_lineage, y_vals[0])
                    sum_ys = [np.sum(Ys[inverted_curr_assignments[nn]]) + y_vals[0][-1][nn] for nn in range(y_vals.shape[2])]

                except:
                    Ys = y_vals[0].T
                    # print("Did not use lineages for {}".format(exp_id))
                    sum_ys = [np.sum(Ys[nn]) for nn in range(y_vals.shape[2])]

                tmp_cache[exp_id] = sum_ys

            else:
                return tmp_cache[exp_id][nn_idx]
        #     stats_dict[str(exp_id)]["Path Weight Sum"] = np.mean(np.sum(Ys, axis=1))
        elif name_split[0] == "eigs":
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

    return helper, mod


def get_plot_special(exp_dict, exp_ids, X_axis_name, Y_axis_name):
    X_selector, X_mod = get_selector_mod(exp_dict, X_axis_name)
    Y_selector, Y_mod = get_selector_mod(exp_dict, Y_axis_name)

    assert not ((X_mod is not None) and (Y_mod is not None))
    assert (X_mod is not None) or (Y_mod is not None)

    x_vals = []
    y_vals = []
    for exp_id in exp_ids:
        num_nets = exp_dict["stuff"]["configs"].loc[exp_id]["num_nets"]
        Xs = [X_selector(exp_id, i) for i in range(num_nets)]
        Ys = [Y_selector(exp_id, i) for i in range(num_nets)]

        if X_mod is not None:
            nn_idxs = get_mod_idx(Xs, X_mod)
        else:
            nn_idxs = get_mod_idx(Ys, Y_mod)

        if (len(nn_idxs) == 0):
            continue

        x_vals.append([X_selector(exp_id, i) for i in nn_idxs])
        y_vals.append([Y_selector(exp_id, i) for i in nn_idxs])

    return np.array(x_vals).reshape(-1), np.array(y_vals).reshape(-1)


def plot_special(exp_dict, X_axis_name, Y_axis_name, filter_seperate=None, filter_not_seperate=None,
                 save_exp_path=None, X_axis_bounds=None, Y_axis_bounds=None, pre_filtered_exp_ids=None, is_mean=False,
                 X_axis_display_name=None, Y_axis_display_name=None):
    plots = []
    plots_names = []

    cfs = exp_dict["stuff"]["configs"]


    for exp_ids, comb in id_selection_from_hyperparameters_generator(cfs, filter_seperate, filter_not_seperate):
        if (exp_ids is None) and (comb is None):
            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str(s_comb)))
            else:
                save_location = None

            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name, save_location=save_location)

            plots = []
            plots_names = []
        else:
            if pre_filtered_exp_ids is not None:
                exp_ids = list(set(exp_ids) & set(pre_filtered_exp_ids))

            if len(exp_ids) == 0:
                continue

            if X_axis_name == "time":
                y_arr, x_arr = get_filtered_training_metrics(exp_dict, exp_ids, Y_axis_name, path_aggregator=None)
                
                plots.append(plt.plot(x_arr.T, y_arr.T)[0])

            else:
                x_vals, y_vals = get_plot_special(exp_dict, exp_ids, X_axis_name, Y_axis_name)

                # print("Correlation for {} {}/{}: {}".format(comb, X_axis_name, Y_axis_name,
                #                                             get_correlation(x_vals, y_vals)))

                plots.append(plt.scatter(x_vals, y_vals))
                plots_names.append(comb)
    
def hp_plot(experiment_folder, X_data, Y_data, X_axis_name, Y_axis_name, filter_seperate=None, filter_not_seperate=None,
                 save_exp_path=None, X_axis_bounds=None, Y_axis_bounds=None, pre_filtered_exp_ids=None, plot_type="scatter"):
    plots = []
    plots_names = []

    cfs = load_configs(experiment_folder)

    for exp_ids, comb in id_selection_from_hyperparameters_generator(cfs, filter_seperate, filter_not_seperate):
        if (exp_ids is None):
            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str(comb)))
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
            x = np.concatenate(np.array([[v for v in X_data[exp_id].values()] for exp_id in exp_ids]), axis=0)
            y = np.concatenate(np.array([[v for v in Y_data[exp_id].values()] for exp_id in exp_ids]), axis=0)

            if plot_type == "scatter":
                plots.append(plt.scatter(x.T, y.T))
            else: 
                plots.append(plt.plot(x.T, y.T)[0])
            plots_names.append(comb)

def hp_xy_func_plot(experiment_folder, xy_func, X_axis_name, Y_axis_name, plot_name, filter_seperate=None, filter_not_seperate=None,
                 save_exp_path=None, X_axis_bounds=None, Y_axis_bounds=None, pre_filtered_exp_ids=None, plot_type="scatter"):
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

            plots, plots_names = xy_func(exp_ids, plots, plots_names, comb)
