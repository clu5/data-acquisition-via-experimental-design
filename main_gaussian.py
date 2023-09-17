import argparse
import json
import math
import os
import time
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
                                 DataShapley, KNNShapley, LavaEvaluator,
                                 LeaveOneOut, RandomEvaluator,
                                 RobustVolumeShapley)
from opendataval.model import RegressionSkLearnWrapper
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression
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


def get_data(
    scale_data=False,
    cluster=False,
    random_seed=0,
    num_seller=10000,
    num_buyer=1000,
    dim=1000,
    noise_level=10,
    val_split=0.1,
):
    random_state = check_random_state(random_seed)

    # Generate some random seller data
    X_sell = np.random.normal(size=(num_seller, dim))
    X_sell /= np.linalg.norm(X_sell, axis=1, keepdims=True)  # normalize data

    # generate true coefficients
    beta_true = np.random.exponential(scale=1, size=dim)
    beta_true *= np.sign(np.random.random(size=dim))

    y_sell = X_sell @ beta_true + noise_level * np.random.randn(num_seller)

    # Generate some random buyer data
    X_buy = np.random.normal(size=[num_buyer, dim])
    X_buy /= np.linalg.norm(X_buy, axis=1, keepdims=True)  # normalize data
    y_buy = X_buy @ beta_true

    if scale_data:
        MMS = MinMaxScaler()
        X_sell = MMS.fit_transform(X_sell)
        y_sell = MMS.fit_transform(y_sell)
        X_buy = MMS.fit_transform(X_buy)
        y_buy = MMS.fit_transform(y_buy)

    if cluster:
        KM = KMeans(n_clusters=3, init="k-means++")
        KM.fit(X_buy)
        cluster_indices = KM.labels_

        cluster_1 = cluster_indices == 0
        cluster_2 = cluster_indices == 1
        cluster_3 = cluster_indices == 2

        x3_train, x3_test, y3_train, y3_test = train_test_split(
            x[cluster_3],
            y[cluster_3],
            test_size=test_frac,
            random_state=random_state,
        )

        x2_train, x2_val, y2_train, y2_val = train_test_split(
            x[cluster_2],
            y[cluster_2],
            test_size=num_validation / x[cluster_2].shape[0],
            random_state=random_state,
        )

        train_x = torch.cat([x[cluster_1], x2_train, x3_train])
        train_y = torch.cat([y[cluster_1], y2_train, y3_train])

        val_x = x2_val
        val_y = y2_val

        test_x = x3_test
        test_y = y3_test

    else:
        X_sell, X_val, y_sell, y_val = train_test_split(
            X_sell,
            y_sell,
            test_size=val_split,
            random_state=random_state,
        )

    return X_sell, y_sell, X_val, y_val, X_buy, y_buy


def get_baseline_values(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    metric=mean_squared_error,
    random_state=0,
    baselines=["DataOob"],
):
    fetcher = DataFetcher.from_data_splits(
        train_x, train_y, val_x, val_y, test_x, test_y, one_hot=False
    )
    model = RegressionSkLearnWrapper(LinearRegression)
    baseline_values = {
        k: getattr(dataval, k)(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
        for k in baselines
    }
    return baseline_values


def evaluate_subset(train_subset, train_x, train_y, test_x, test_y):
    if isinstance(train_x, torch.Tensor):
        train_x = train_x.numpy()
        train_y = train_y.numpy()
        test_x = test_x.numpy()
        test_y = test_y.numpy()
    LR = LinearRegression()
    LR.fit(train_x[train_subset], train_y[train_subset])
    pred = LR.predict(test_x)
    error = mean_squared_error(test_y, pred)
    return error


def design_selection(
    seller_data,
    buyer_data,
    num_select=100,
    num_iters=100,
    alpha=0.1,
    line_search=False,
):
    X_sell, y_sell = seller_data
    X_buy, y_buy = buyer_data

    # initialize seller weights
    n_sell = X_sell.shape[0]
    weights = np.ones(n_sell) / n_sell

    def scale(cov):
        cov -= cov.min()
        return cov / (cov.max() - cov.min())

    # inverse covariance matrix
    inv_cov = np.linalg.pinv(X_sell.T @ X_sell)

    # experimental design loss i.e. E[X_buy.T @ inv_cov @ X]
    loss = frank_wolfe.compute_exp_design_loss(X_buy, inv_cov)

    # track losses and errors
    losses = {}
    errors = {}

    for i in tqdm(range(num_iters)):
        # Pick coordinate with largest gradient to update
        neg_grad = frank_wolfe.compute_neg_gradient(X_sell, X_buy, inv_cov)
        update_coord = np.argmax(neg_grad)

        # Step size
        if line_search:
            alpha, loss = frank_wolfe.opt_step_size(
                X_sell[update_coord], X_buy, inv_cov, loss
            )
        else:
            alpha = 0.1
            # alpha = 2 / (3 + i)

        # Update weight vector
        weights *= 1 - alpha  # shrink weights by 1 - alpha
        weights[update_coord] += alpha  # increase magnitude of picked coordinate

        # Update inverse covariance matrix
        inv_cov /= 1 - alpha  # Update with respect to weights shrinking
        inv_cov = frank_wolfe.sherman_morrison_update_inverse(  # update with respect to picked coordinate increasing
            inv_cov,
            alpha * X_sell[update_coord, :],
            X_sell[update_coord, :],
        )
        inv_cov = scale(
            inv_cov
        )  # rescale inverse covariance matrix to be between 0 and 1
        selected_seller_indices = np.unique(
            np.random.choice(np.arange(weights.shape[0]), size=num_select, p=weights)
        )
        results = frank_wolfe.evaluate_indices(
            X_sell,
            y_sell,
            X_buy,
            y_buy,
            selected_seller_indices,
            inverse_covariance=inv_cov,
        )
        losses[i] = results["exp_loss"]
        errors[i] = results["mse_error"]
    return dict(losses=losses, errors=errors, weights=weights)


def main(args):
    np.random.seed(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    X_sell, y_sell, X_val, y_val, X_buy, y_buy = get_data(
        cluster=args.cluster,
        scale_data=args.scale_data,
        random_seed=args.seed,
        num_seller=1000 if args.debug else args.num_seller,
        num_buyer=max(args.num_buyers),
        dim=100 if args.debug else args.num_dim,
        noise_level=10,
        val_split=0.1,
    )

    # our method (experimental design loss)
    design_values = {}
    design_errors = {}
    design_losses = {}
    for num_buyer in args.num_buyers:
        print(
            f"Starting experimental design with {num_buyer} buyer points".center(
                40, "-"
            )
        )
        exp_start = time.perf_counter()
        seller_data = X_sell, y_sell
        buyer_data = X_buy[:num_buyer], y_buy[:num_buyer]
        design_results = design_selection(
            seller_data,
            buyer_data,
            num_select=100,
            num_iters=10 if args.debug else args.num_iters,
            alpha=0.1,
            line_search=False,
        )
        design_values[num_buyer] = design_results["weights"]
        design_errors[num_buyer] = design_results["errors"]
        design_losses[num_buyer] = design_results["losses"]

        exp_end = time.perf_counter()
        print(f"Finished in {exp_end-exp_start:.0f} seconds".center(40, "-"))

    print(f"Starting baselines".center(40, "-"))
    base_start = time.perf_counter()
    # other data valuation baselines
    baseline_values = get_baseline_values(
        X_sell,
        y_sell,
        X_val,
        y_val,
        X_buy,
        y_buy,
        random_state=args.seed,
        baselines=args.baselines,
    )
    base_end = time.perf_counter()
    print(f"Finished valuations in {base_end-base_start:.0f} seconds".center(40, "-"))

    # Start evaluations on test set (buyer data)
    eval_range = [
        int(k)
        for k in np.concatenate(
            [
                np.arange(2, 20),
                np.arange(20, min(200, X_sell.shape[0]), 10),
                np.arange(200, min(1000, X_sell.shape[0]), 50),
            ]
        )
    ]

    baseline_evals = {}
    for baseline, values in baseline_values.items():
        baseline_evals[baseline] = {
            k: evaluate_subset(
                values.argsort()[-k:],
                X_sell,
                y_sell,
                X_buy,
                y_buy,
            )
            for k in eval_range
        }
    print(f"Finished baseline evaluations".center(40, "-"))

    random_trials = range(5)  # average mse over random samples of weights
    design_evals = {}
    for num_buy, weights in design_values.items():
        design_evals[f"exp_design_{num_buy}"] = {
            k: np.mean(
                [
                    evaluate_subset(
                        np.unique(
                            np.random.choice(
                                np.arange(weights.shape[0]), size=k, p=weights
                            )
                        ),
                        X_sell,
                        y_sell,
                        X_buy,
                        y_buy,
                    )
                    for _ in random_trials
                ]
            )
            for k in eval_range
        }
    print(f"Finished design evaluations".center(40, "-"))

    # One-step baseline
    inv_cov = np.linalg.inv(X_sell.T @ X_sell)
    one_step_values = np.mean((X_sell @ inv_cov @ X_buy.T) ** 2, axis=1).argsort()
    design_values["one_step"] = one_step_values
    design_evals["one_step"] = {
        k: evaluate_subset(one_step_values[-k:], X_sell, y_sell, X_buy, y_buy)
        for k in eval_range
    }

    # Random seller baseline
    random_seller_values = np.random.permutation(X_sell.shape[0])
    design_evals["random_seller"] = {
        k: np.mean(
            [
                evaluate_subset(random_seller_values[:k], X_sell, y_sell, X_buy, y_buy)
                for _ in random_trials
            ]
        )
        for k in eval_range
    }

    # Random buyer baseline
    random_buyer_values = np.random.permutation(X_sell.shape[0])
    design_evals["random_buyer"] = {
        k: np.mean(
            [
                evaluate_subset(random_buyer_values[:k], X_sell, y_sell, X_buy, y_buy)
                for _ in random_trials
            ]
        )
        for k in eval_range
    }

    # Save data values and evaluations
    all_values = {
        "exp_design": {k: np.array(v).tolist() for k, v in design_values.items()},
        "baselines": {k: np.array(v).tolist() for k, v in baseline_values.items()},
    }
    all_evals = {
        "exp_design": {k: np.array(v).tolist() for k, v in design_evals.items()},
        "baselines": {k: np.array(v).tolist() for k, v in baseline_evals.items()},
    }

    save_name = "gaussian"
    if args.cluster:
        save_name += "cluster"

    save_name += f"dim_{args.num_dim}"

    save_name += str(args.seed)

    if args.debug:
        save_name = "debug"

    with open(results_dir / f"{save_name}-design-losses.json", "w") as fh:
        json.dump(design_losses, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-design-errors.json", "w") as fh:
        json.dump(design_errors, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-values.json", "w") as fh:
        json.dump(all_values, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-evals.json", "w") as fh:
        json.dump(all_evals, fh, default=float, indent=4)

    print("Experiment completed".center(40, "="))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main_gaussian.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization on synthetically generated data",
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--cluster", action="store_true", help="use non IID validation set"
    )
    parser.add_argument("--scale_data", action="store_true", help="standardize data")
    parser.add_argument(
        "--num_buyers",
        nargs="+",
        default=[10, 100, 1000],
        type=list,
        help="number of buyer points used in experimental design",
    )
    parser.add_argument(
        "--num_seller",
        default=100000,
        type=int,
        help="number of seller points used in experimental design",
    )
    parser.add_argument(
        "--num_dim",
        default=1000,
        type=int,
        help="dimensionality of the data samples",
    )
    parser.add_argument(
        "--num_iters",
        default=100,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        nargs="+",
        default=["DataOob", "BetaShapley", "RandomEvaluator"],
        type=list,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1,
        type=int,
        help="random seed",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)
