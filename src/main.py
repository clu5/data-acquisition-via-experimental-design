import argparse
import json
import math
import os
import pickle
import time
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from opendataval import dataval
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (AME, DVRL, BetaShapley, DataBanzhaf, DataOob,
                                 DataShapley, InfluenceFunction,
                                 InfluenceSubsample, KNNShapley, LavaEvaluator,
                                 LeaveOneOut, RandomEvaluator,
                                 RobustVolumeShapley)
from opendataval.model import RegressionSkLearnWrapper
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from torchvision.datasets import MNIST
from torchvision.models import efficientnet_b1, resnet50
from torchvision.transforms import (CenterCrop, Compose, Lambda, Resize,
                                    ToTensor)
from tqdm import tqdm

import convex
import frank_wolfe
import utils


def main(args):
    data = utils.get_data(
        dataset=args.dataset,
        num_buyer=args.num_buyers * args.batch_size,
        num_seller=args.num_seller,
        num_val=args.num_val,
        dim=args.num_dim,
        noise_level=args.noise_level,
        random_state=args.random_seed,
        cost_range=args.cost_range,
        cost_func=args.cost_func,
    )

    x_s = data["X_sell"].astype(np.single)
    y_s = data["y_sell"].astype(np.single)
    x_val = data["X_val"].astype(np.single)
    y_val = data["y_val"].astype(np.single)
    x_b = data["X_buy"].astype(np.single)
    y_b = data["y_buy"].astype(np.single)
    costs = data.get("costs_sell")
    index_s = data['index_sell']
    index_b = data['index_buy']
    index_v = data['index_val']

    print(f"{x_s.shape = }".center(40, "="))
    print(f"{x_b.shape = }".center(40, "="))
    if costs is not None:
        print(f"{costs.shape = }".center(40, "="))

    errors = defaultdict(list)
    runtimes = defaultdict(list)
    weights = defaultdict(list)

    eval_range = list(range(1, 30, 1)) + list(
        range(30, args.max_eval_range, args.eval_step)
    )

    # loop over each test point in buyer
    for i, j in tqdm(enumerate(range(0, x_b.shape[0], args.batch_size))):
        x_test = x_b[j : j + args.batch_size]
        y_test = y_b[j : j + args.batch_size]

        err_kwargs = dict(
            x_test=x_test, y_test=y_test, x_s=x_s, y_s=y_s, eval_range=eval_range
        )


        if costs is not None:
            error_func = utils.get_error_under_budget
            err_kwargs["costs"] = costs
        else:
            error_func = utils.get_error_fixed
            err_kwargs["return_list"] = True

        w_baselines = utils.get_baseline_values(
            x_s,
            y_s,
            x_val,
            y_val,
            x_val,
            y_val,
            baselines=args.baselines,
            baseline_kwargs={
                "DataShapley": {"mc_epochs": 100, "models_per_iteration": 10},
                "DataBanzhaf": {"mc_epochs": 100, "models_per_iteration": 10},
                "BetaShapley": {"mc_epochs": 100, "models_per_iteration": 10},
                "DataOob": {"num_models": 500},
                "InfluenceSubsample": {"num_models": 500},
            },
        )

        os_start = time.perf_counter()
        w_os = frank_wolfe.one_step(x_s, x_test)
        os_end = time.perf_counter()
        runtimes["Ours (single step)"].append(os_end - os_start)
        weights["Ours (single step)"].append(w_os)

        fw_start = time.perf_counter()
        res_fw = frank_wolfe.design_selection(
            x_s,
            y_s,
            x_test,
            y_test,
            num_select=10,
            num_iters=args.num_iters,
            alpha=None,
            recompute_interval=0,
            line_search=True,
            costs=costs,
            reg_lambda=args.reg_lambda,
        )
        fw_end = time.perf_counter()
        w_fw = res_fw["weights"]
        runtimes["Ours (multi-step)"].append(fw_end - fw_start)
        weights["Ours (multi-step)"].append(w_fw)

        errors["Ours (multi-step)"].append(error_func(w=w_fw, **err_kwargs))
        errors["Ours (single step)"].append(error_func(w=w_os, **err_kwargs))

        values, times = w_baselines
        for k, v in values.items():
            errors[k].append(error_func(w=v, **err_kwargs))
            weights[k].append(v)

        w_rand = np.random.permutation(len(x_s))
        errors["Random"].append(error_func(w=w_rand, **err_kwargs))
        weights["Random"].append(w_rand)

        for k, v in times.items():
            runtimes[k].append(v)

        results = dict(errors=errors, eval_range=eval_range, runtimes=runtimes)

        if i % 25 == 0:
            save_results(args=args, results=results)
            plot_results(args=args, results=results)

        print(f"round {i} done".center(40, "="))

    if not args.skip_save:
        with open(args.result_dir / f"{args.save_name}-data.pkl", "wb") as f:
            pickle.dump(data, f)

        with open(args.result_dir / f"{args.save_name}-weights.pkl", "wb") as f:
            pickle.dump(weights, f)

    save_results(args=args, results=results)
    plot_results(args=args, results=results)

    return results

def save_results(args, results):
    for k, v in vars(args).items():
        if k not in results:
            if isinstance(v, Path):
                v = str(v)
            results[k] = v
        else:
            print(f"Found {k} in results. Skipping.")

    result_path = args.result_dir / f"{args.save_name}-results.json"

    with open(result_path, "w") as f:
        json.dump(results, f, default=float)
    print(f"Results saved to {result_path}".center(80, "="))


def plot_results(args, results):
    figure_path = args.figure_dir / f"{args.save_name}-plot.png"
    if args.cost_range is not None:
        utils.plot_errors_under_budget(results, figure_path)
    else:
        utils.plot_errors_fixed(results, figure_path)
    print(f"Plot saved to {figure_path}".center(80, "="))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--figure_dir", default="../figures")
    parser.add_argument("--result_dir", default="../results")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument(
        "-d",
        "--dataset",
        default="gaussian",
        choices=[
            "gaussian",
            "mimic",
            "bone",
            "fitzpatrick",
            "drug",
        ],
        type=str,
        help="dataset to run experiment on",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="number of test points to optimize at once",
    )
    parser.add_argument(
        "--num_buyers",
        default=30,
        type=int,
        help="number of test buyer points used in experimental design",
    )
    parser.add_argument(
        "--num_seller",
        default=1000,
        type=int,
        help="number of seller points used in experimental design",
    )
    parser.add_argument(
        "--num_val",
        default=100,
        type=int,
        help="number of validation points for baselines",
    )
    parser.add_argument(
        "--num_dim",
        default=100,
        type=int,
        help="dimensionality of the data samples",
    )
    parser.add_argument(
        "--num_iters",
        default=500,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "--max_eval_range",
        default=150,
        type=int,
        help="max number training points to select for evaluation",
    )
    parser.add_argument(
        "--eval_step",
        default=5,
        type=int,
        help="evaluation interval",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[
            # "AME",
            # "BetaShapley",
            # "DataBanzhaf",
            "DataOob",
            # "DataShapley",
            "DVRL",
            # "InfluenceSubsample",
            "KNNShapley",
            "LavaEvaluator",
            # "LeaveOneOut",
            "RandomEvaluator",
            # "RobustVolumeShapley",
        ],
        type=str,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument("--debug", action="store_true", help="Turn on debugging mode")
    parser.add_argument(
        "--skip_save", action="store_true", help="Don't save weights or data"
    )
    parser.add_argument(
        "--cost_range",
        nargs="*",
        default=None,
        help="""
        Choose range of costs to sample uniformly.
        E.g. costs=[1, 2, 3, 9] will randomly set each seller data point
        to one of these costs and apply the cost_func during optimization.
        If set to None, no cost is applied during optimization.
        Default is None.
        """,
    )
    parser.add_argument(
        "--cost_func",
        default="linear",
        choices=["linear", "squared", "square_root"],
        type=str,
        help="Choose cost function to apply.",
    )
    parser.add_argument(
        "--noise_level",
        default=0.1,
        type=float,
        help="level of noise to add for cost",
    )
    parser.add_argument(
        "--reg_lambda",
        default=0.0,
        type=float,
        help="Regularization initial inverse information matrix to be identity. Should be between 0 and 1.",
    )
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    args.figure_dir = Path(args.figure_dir)
    args.result_dir = Path(args.result_dir)
    args.figure_dir.mkdir(exist_ok=True, parents=True)
    args.result_dir.mkdir(exist_ok=True, parents=True)
    if args.debug:
        args.save_name = "debug"
        args.num_buyers = 2
        args.num_seller = 100
        args.num_val = 5
        args.num_dim = 10
        args.num_iters = 10
        args.baselines = ["KNNShapley", "LavaEvaluator"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        save_name = f"{timestamp}-{args.dataset}-{args.num_seller}"
        if args.cost_range is not None:
            save_name += f"-{args.cost_func}"
        if args.exp_name is not None:
            save_name += f"-{args.exp_name}"
        args.save_name = save_name

    start = time.perf_counter()
    results = main(args)
    end = time.perf_counter()

    print(f"Total runtime {end-start:.0f} seconds".center(80, "="))
