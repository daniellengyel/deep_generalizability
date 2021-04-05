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
import scipy

import re

from .training_metrics import * 
from .utils import * 
from ..nets import Nets
from ..utils import *
from .correlation import *
from ..save_load import load_cached_data, load_configs

import itertools

from scipy.stats import linregress
from sklearn.neighbors import LocalOutlierFactor


import seaborn as sns

COLORS = plt.cm.tab20(np.arange(20))

CORRECT_COLOR_IDX = 3
INCORRECT_COLOR_IDX = 1


def get_end_stats(exp_folder, step=-1, with_min_max=False):
    
    trace, _ = None, None # load_cached_data(exp_folder, "point_traces", step=step) # assume the trace i get is from the end.
    acc, _ = load_cached_data(exp_folder, "acc", step=step)
    loss, _ = load_cached_data(exp_folder, "loss", step=step)

    stats_dict = {}
    configs = load_configs(exp_folder)
    for exp_id in configs.index:

        num_nets = configs.loc[exp_id]["num_nets"]

        stats_dict[str(exp_id)] = {}

        if loss is not None:
            Loss_train_list = [loss[exp_id][str(nn)][0] for nn in range(num_nets)]
            Loss_test_list = [loss[exp_id][str(nn)][1] for nn in range(num_nets)]

            stats_dict[str(exp_id)]["Loss Test Mean"] = np.mean(Loss_test_list)
            stats_dict[str(exp_id)]["Loss Train Mean"] = np.mean(Loss_train_list)

            if with_min_max:
                stats_dict[str(exp_id)]["Loss Test Max"] = np.max(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Test Min"] = np.min(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Train Max"] = np.max(Loss_train_list)
                stats_dict[str(exp_id)]["Loss Train Min"] = np.min(Loss_train_list)

        if acc is not None:
            Acc_train_list = [acc[exp_id][str(nn)][0] for nn in range(num_nets)]
            Acc_test_list = [acc[exp_id][str(nn)][1] for nn in range(num_nets)]

            stats_dict[str(exp_id)]["Acc Train Mean"] = np.mean(Acc_train_list)
            stats_dict[str(exp_id)]["Acc Test Mean"] = np.mean(Acc_test_list)
            
            if  with_min_max:
                stats_dict[str(exp_id)]["Acc Train Max"] = np.max(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Train Min"] = np.min(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Test Max"] = np.max(Acc_test_list)
                stats_dict[str(exp_id)]["Acc Test Min"] = np.min(Acc_test_list)

            stats_dict[str(exp_id)]["Acc Gap Mean"] = stats_dict[str(exp_id)]["Acc Test Mean"] - \
                                                    stats_dict[str(exp_id)]["Acc Train Mean"]

        if trace is not None:
            Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            stats_dict[str(exp_id)]["Trace Mean"] = np.mean(Trace_list)
            stats_dict[str(exp_id)]["Trace Mean Std"] = np.mean(Trace_std_list)

            if with_min_max:
                stats_dict[str(exp_id)]["Trace Max"] = np.max(Trace_list)
                stats_dict[str(exp_id)]["Trace Min"] = np.min(Trace_list)

    stats_pd = pd.DataFrame.from_dict(stats_dict, orient="index")

    # append hyperparameters to DataFrame
    cfs_hp = get_hp(configs)
    cfs_hp_df = configs[list(cfs_hp.keys())]
    stats_pd = pd.concat([stats_pd, cfs_hp_df], axis=1)

    return stats_pd


def costum_plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds, save_location=None):
    if len(plots_names) > 1:
        plt.legend(tuple(plots),
                   plots_names,
                   scatterpoints=1,
                   loc='best',
                   ncol=2,
                   )
    elif len(plots) > 1:
        plt.legend(loc='best')

    config = {"axes.titlesize" : 24,
                "axes.labelsize" : 22,
                "lines.linewidth" : 3,
                "lines.markersize" : 5,
                "xtick.labelsize" : 18,
                "ytick.labelsize" : 18,
                'grid.linestyle': 'dashed',
                'legend.fontsize': 22,
            }
    plt.grid(True)

    plt.rcParams.update(config) 

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["figure.figsize"] = (14,8)


    plt.xlabel(X_axis_name, )
    plt.ylabel(Y_axis_name, )

    if X_axis_bounds is not None:
        plt.xlim(X_axis_bounds)
    if Y_axis_bounds is not None:
        plt.ylim(Y_axis_bounds)

    # color_selector = np.arange(12)
    # color_selector[0] = 3
    # color_selector[1] = 1
    # color_selector[3] = 0
    # color = plt.cm.tab20(color_selector)
    # mpl.rcParams['axes.prop_cycle'] = cycler('color', color) # plt.cm.Set3(np.arange(12))))
    if save_location is not None:
        plt.savefig(save_location + ".png", dpi=300, bbox_inches = "tight", )#, format='eps')
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

            costum_plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    save_location=save_location)

            plots = []
            plots_names = []
        else:
            if pre_filtered_exp_ids is not None:
                exp_ids = list(set(exp_ids) & set(pre_filtered_exp_ids))

            if len(exp_ids) == 0:
                continue

            plots, plots_names = data_func(exp_ids, plots, plots_names, comb)

def plot_regression(x_data, y_data, color, remove_outliers=True):
    min_x = np.min(x_data)
    max_x = np.max(x_data)

    slope, intercept, r_value, _, _ = linregress_outliers(x_data, y_data, remove_outliers=remove_outliers)

    plot_corr, = plt.plot([min_x, max_x], [slope*min_x + intercept, max_x*slope + intercept], color=color)
    
    if remove_outliers:
        outlier_filter = get_outlier_filter(x_data, y_data)
        plt.scatter(x_data[~outlier_filter], y_data[~outlier_filter], color=color, marker="x", s=120)

    plot_name = "rvalue: {:.2f}".format(r_value)
    return plot_corr, plot_name

# ++++ functions for hp_data_func_plot ++++
def margin_trace_correct_incorrect_plot(margins_filters, point_traces, use_correct_filter=False, draw_correlation=False):
    def xy_func(exp_ids, plots, plots_names, comb=None):
        x_correct = []
        y_correct = []

        x_incorrect = []
        y_incorrect = []
        print(exp_ids)
        for exp_id in exp_ids:
            for model_idx in margins_filters[exp_id].keys():

                curr_point_margins, correct_filters = margins_filters[exp_id][model_idx]
                curr_point_traces = np.array(point_traces[exp_id][model_idx])
                if use_correct_filter:
                    x_correct.append(curr_point_margins[correct_filters])
                    y_correct.append(curr_point_traces[correct_filters])
                    print(np.mean(curr_point_traces[correct_filters]))

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
                plots.append(plt.scatter(x_correct.T, y_correct.T, color=COLORS[CORRECT_COLOR_IDX]))
                plots_names.append("Correct")

                if draw_correlation:
                    plot_corr, plot_name = plot_regression(x_correct, y_correct, color=COLORS[CORRECT_COLOR_IDX])
                    plots.append(plot_corr)
                    plots_names.append("Correct, {}".format(plot_name))

            if len(x_incorrect) > 0:
                plots.append(plt.scatter(x_incorrect.T, y_incorrect.T, color=COLORS[INCORRECT_COLOR_IDX]))
                plots_names.append("Incorrect")

                if draw_correlation:
                    plot_corr, plot_name = plot_regression(x_incorrect, y_incorrect, color=COLORS[INCORRECT_COLOR_IDX])
                    plots.append(plot_corr)
                    plots_names.append("Incorrect, {}".format(plot_name))
        else:
            x_data = np.concatenate(x_correct, axis=0)
            y_data = np.concatenate(y_correct, axis=0)

            min_x = 0 #np.min(np.min(x_data), np.min(x_data))
            max_x = max(np.max(x_data), np.max(x_data))
            
            plots.append(plt.scatter(x_data.T, y_data.T, color=COLORS[CORRECT_COLOR_IDX]))
            plots_names.append("All, {}".format(comb))

            if draw_correlation:
                plot_corr, plot_name = plot_regression(x_data, y_data, color=COLORS[CORRECT_COLOR_IDX])
                plots.append(plot_corr)
                plots_names.append(plot_name)
        
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

