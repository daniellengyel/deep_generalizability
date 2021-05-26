import numpy as np
import pandas as pd
import math 

import margin_flatness.postprocessing.postprocess_experiment as mf_post
import margin_flatness

import yaml, os, sys, re

import torch

import pickle


def main():
    # save analysis processsing

    root_folder = os.environ["PATH_TO_DEEP_FOLDER"]
    data_name = "KMNIST"

    # Job specific 
    ReLUexps = [
        "May22_20-49-07_cx3-3-13.cx3.hpc.ic.ac.uk",
        "May22_20-49-07_cx3-7-1.cx3.hpc.ic.ac.uk",
        "May22_20-49-08_cx3-7-3.cx3.hpc.ic.ac.uk",
        "May22_20-51-16_cx3-4-2.cx3.hpc.ic.ac.uk",
        "May22_20-51-16_cx3-5-26.cx3.hpc.ic.ac.uk",
        "May22_20-53-25_cx3-3-29.cx3.hpc.ic.ac.uk"
    ]
    try:
        TOTAL_ARRAYS = int(os.environ["TOTAL_ARRAYS"])
        ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1 
    except:
        ARRAY_INDEX = 1
        TOTAL_ARRAYS = 1
        
    # exp =  "LeNet_short" #
    exp = "w128_l8_ReLU_long"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

    # init torch
    is_gpu = False
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    # if ARRAY_INDEX == 1:

    # margin_flatness.save_load.join_cached_sub_data(experiment_folder, "model_loss_acc", -1)
    # margin_flatness.save_load.join_cached_sub_data(experiment_folder, "output_margins", -1)

    # # TODO Exp ids should be set here. 
    meta = {"criterion": "cross-entropy", "normalize_output": True}
    exp_ids = list(margin_flatness.save_load.exp_models_path_generator(experiment_folder))

    start_idx = math.ceil(len(exp_ids) / TOTAL_ARRAYS) * ARRAY_INDEX
    end_idx = math.ceil(len(exp_ids) / TOTAL_ARRAYS) * (ARRAY_INDEX + 1)
    exp_ids_curr = exp_ids[start_idx:end_idx]
    print(start_idx)
    print(end_idx)
    print(len(exp_ids_curr))
    print(len(exp_ids))

    num_cpus = 32
    num_gpus = 1

    # exp_ids = [("1621270824.0509984", os.path.join(experiment_folder, "models", "1621270824.0509984"))]
    # mf_post.multi_compute_on_experiment(experiment_folder, "model_loss_acc", exp_ids_curr, step=-1, seed=0, num_datapoints=-1, on_test_set=False, num_cpus=num_cpus, num_gpus=num_gpus, verbose=True, meta=None)
    # mf_post.multi_compute_on_experiment(experiment_folder, "model_loss_acc", exp_ids_curr, step=-1, seed=0, num_datapoints=-1, on_test_set=True, num_cpus=num_cpus, num_gpus=num_gpus, verbose=True, meta=None)

    # mf_post.multi_compute_on_experiment(experiment_folder, "output_margins", exp_ids_curr, step=-1, seed=0, num_datapoints=1000, on_test_set=False, num_cpus=num_cpus, num_gpus=0, verbose=True, meta=None)
    # mf_post.multi_compute_on_experiment(experiment_folder, "point_loss", exp_ids_curr, step=-1, seed=0, num_datapoints=1000, on_test_set=False, num_cpus=num_cpus, num_gpus=0, verbose=True, meta=meta)
    # mf_post.multi_compute_on_experiment(experiment_folder, "point_traces", exp_ids_curr, step=-1, seed=0, num_datapoints=1000, on_test_set=False, num_cpus=num_cpus, num_gpus=0, verbose=True, meta=meta)
    mf_post.multi_compute_on_experiment(experiment_folder, "inp_out_jacobian", exp_ids_curr, step=-1, seed=0, num_datapoints=1000, on_test_set=False, num_cpus=num_cpus, num_gpus=0, verbose=True, meta=None)


    # mf_post.get_exp_loss_acc(experiment_folder, step=-1, seed=0, num_train_datapoints=5000, num_test_datapoints=5000, device=device)
    # print(margin_flatness.postprocessing.stats_plotting.get_end_stats(experiment_folder, step=-1, with_min_max=False))
    

    # mf_post.compute_on_experiment(experiment_folder, "point_loss", -1, 0, 100, on_test_set=False, device=None, verbose=True, check_cache=False, meta=meta)
    # mf_post.compute_on_experiment(experiment_folder, "point_traces", -1, 0, 100, on_test_set=False, device=None, verbose=True, check_cache=False, meta=meta)




if __name__ == "__main__":
    main()
