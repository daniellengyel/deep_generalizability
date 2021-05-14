import numpy as np
import pandas as pd

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
    ARRAY_INDEX = 0 # int(os.environ["PBS_ARRAY_INDEX"]) - 1    
    ReLUexps = ["May13_15-55-36_cx3-6-12.cx3.hpc.ic.ac.uk", "May13_15-58-23_cx3-5-23.cx3.hpc.ic.ac.uk", "May13_16-04-19_cx3-5-4.cx3.hpc.ic.ac.uk"]
    exp = ReLUexps[ARRAY_INDEX]
    # exp = "May13_16-04-19_cx3-6-16.cx3.hpc.ic.ac.uk"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

    # init torch
    is_gpu = False
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")


    
    a = mf_post.multi_compute_on_experiment(experiment_folder, "point_traces", step=-1, seed=0, num_datapoints=1000, on_test_set=False, num_cpus=4, num_gpus=0, verbose=False, meta=None)
    print(a)
    # mf_post.get_exp_loss_acc(experiment_folder, step=-1, seed=0, num_train_datapoints=1000, num_test_datapoints=1000, device=None)
    # print(margin_flatness.postprocessing.stats_plotting.get_end_stats(experiment_folder, step=-1, with_min_max=False))
    
    meta = {"N": 100, "delta": 0.0015, "criterion": "cross-entropy"}

    # mf_post.compute_on_experiment(experiment_folder, "point_loss", -1, 0, 100, on_test_set=False, device=None, verbose=True, check_cache=False, meta=meta)
    # mf_post.compute_on_experiment(experiment_folder, "point_traces", -1, 0, 100, on_test_set=False, device=None, verbose=True, check_cache=False, meta=meta)




if __name__ == "__main__":
    main()
