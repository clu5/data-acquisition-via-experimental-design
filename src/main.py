import argparse
from datetime import datetime
import pickle
import json
import math
import os
import time
import uuid
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict
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

import frank_wolfe
import utils
import convex


def main(args):
    data = utils.get_data(
        dataset=args.dataset,
        num_buyer=args.num_buyers,
        num_seller=args.num_seller,
        num_val=args.num_val,
        dim=args.num_dim,
        noise_level=0.1,
    )

    x_s = data["X_sell"].astype(np.single)
    y_s = data["y_sell"]
    x_val = data["X_val"].astype(np.single)
    y_val = data["y_val"]
    x_b = data["X_buy"].astype(np.single)
    coef = data.get("coef")
    y_b = (data["X_buy"] @ coef) if coef is not None else data["y_buy"]
    data ["y_b"] = y_b
    with open(args.result_dir / f'{args.uuid}-data.pkl', 'wb') as f:
        pickle.dump(data, f)

    errors = defaultdict(list)
    runtimes = defaultdict(list)

    eval_range = list(range(2, 30, 1)) + list(range(20, 150, 5))

    # loop over each test point in buyer
    for j in tqdm(range(0, x_b.shape[0])):
        x_test = x_b[j : j + 1]
        y_test = y_b[j : j + 1]

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
        )
        w_fw = res_fw["weights"]
        w_os = frank_wolfe.one_step(x_s, x_test)

        errors['Ours (multi-step)'].append([utils.get_error(x_test, y_test, x_s, y_s, w_fw, k) for k in eval_range])
        errors['Ours (single step)'].append([utils.get_error(x_test, y_test, x_s, y_s, w_os, k) for k in eval_range])

        w_baselines = utils.get_baseline_values(
            x_s, y_s, x_val, y_val, x_val, y_val,
            baselines=args.baselines,
            baseline_kwargs={
                'DataShapley': {'mc_epochs': 100, 'models_per_iteration': 10},
                'DataBanzhaf': {'mc_epochs': 100, 'models_per_iteration': 10},
                'BetaShapley': {'mc_epochs': 100, 'models_per_iteration': 10},
                'DataOob': {'num_models': 100},
                'KNNShapley': {},
                'LavaEvaluator': {},
                'DVRL': {'rl_epochs': 100},
                'InfluenceSubsample': {'num_models': 100},
                'LeaveOneOut': {},
            },

        )
        values, times = w_baselines

        for k, v in values.items():
            errors[k].append([utils.get_error(x_test, y_test, x_s, y_s, v, k) for k in eval_range])

        w_rand = np.random.permutation(len(x_s))
        errors['Random'].append([utils.get_error(x_test, y_test, x_s, y_s, w_rand, k) for k in eval_range])

        for k, v in times.items():
            runtimes[k].append(v)

        print(f'round {j} done'.center(40, '='))

    return dict(errors=errors, runtimes=runtimes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--figure_dir", default="../figures")
    parser.add_argument("--result_dir", default="../results")
    parser.add_argument("--dataset", default="gaussian",
        choices=["gaussian", "news", "bone"],
        type=str,
        help="dataset to run experiment on",
    )
    parser.add_argument(
        "--num_buyers",
        default=10,
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
        default=1000,
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
        "--baselines",
        nargs="+",
        default=[
            #"AME",
            #"BetaShapley",
            #"DataBanzhaf",
            #"DataOob",
            #"DataShapley",
            #"DVRL",
            #"InfluenceSubsample",
            "KNNShapley",
            "LavaEvaluator",
            #"LeaveOneOut",
            #"RandomEvaluator",
            #"RobustVolumeShapley",
        ],
        type=str,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    args.figure_dir = Path(args.figure_dir)
    args.result_dir = Path(args.result_dir)
    args.figure_dir.mkdir(exist_ok=True, parents=True)
    args.result_dir.mkdir(exist_ok=True, parents=True)
    print(type(args.figure_dir))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    random_uuid = uuid.uuid4()
    args.uuid = f"{timestamp}"
    start = time.perf_counter()
    result = main(args)
    end = time.perf_counter()
    print(f"Total runtime {end-start:.0f} seconds".center(40, "="))
    with open(args.result_dir / f"{uuid}-results.json", "w") as f:
        json.dump(result, f)
    print(f"Results saved to {args.result_dir/{uuid}-results.json}".center(40, "="))
