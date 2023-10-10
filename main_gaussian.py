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


def get_data(
    scale_data=False,
    cluster=False,
    random_seed=0,
    num_seller=10000,
    num_buyer=1000,
    num_val=1000,
    dim=1000,
    noise_level=10,
    val_split=0.1,
    buyer_subset=False,
    seller_subset=False,
    return_beta=False,
    exponential=False,
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
    if exponential:
        X_buy = np.random.exponential(size=[num_buyer, dim])
    else:
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
        n_clusters = 5
        KM = KMeans(n_clusters=n_clusters, init="k-means++")
        KM.fit(X_buy)
        buyer_clusters = KM.labels_

        X_buy = X_buy[buyer_clusters == 0]
        y_buy = y_buy[buyer_clusters == 0]

        X_sell, X_val, y_sell, y_val = train_test_split(
            X_sell,
            y_sell,
            test_size=val_split,
            random_state=random_state,
        )

        val_clusters = KM.predict(X_val)
        X_val = X_val[val_clusters == 1][:num_val]
        y_val = y_val[val_clusters == 1][:num_val]

    else:
        X_sell, X_val, y_sell, y_val = train_test_split(
            X_sell,
            y_sell,
            test_size=val_split,
            random_state=random_state,
        )
        X_val = X_val[:num_val]
        y_val = y_val[:num_val]

    if buyer_subset:
        # print('buyer subset'.center(40, '='))
        X_buy = X_sell[:num_buyer]
        y_buy = y_sell[:num_buyer]
        assert np.allclose(X_sell[0], X_buy[0])

    if seller_subset:
        # print('seller subset'.center(40, '='))
        X_sell = np.concatenate([X_buy, X_sell])
        y_sell = np.concatenate([y_buy, y_sell])
        assert np.allclose(X_sell[0], X_buy[0])

    if return_beta:
        return X_sell, y_sell, X_val, y_val, X_buy, y_buy, beta_true
    else:
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
    baseline_kwargs={"DataOob": {"num_models": 100}},
    use_ridge=False,
):
    fetcher = DataFetcher.from_data_splits(
        train_x, train_y, val_x, val_y, test_x, test_y, one_hot=False
    )
    if use_ridge:
        model = RegressionSkLearnWrapper(Ridge)
    else:
        model = RegressionSkLearnWrapper(LinearRegression)

    kwargs = defaultdict(dict)
    for b in baselines:
        kwargs[b]["random_state"] = random_state
        if b in baseline_kwargs:
            for k, v in baseline_kwargs[b].items():
                kwargs[b][k] = v

    baseline_values = {
        b: getattr(dataval, b)(**kwargs[b])
        .train(fetcher=fetcher, pred_model=model)
        .data_values
        for b in baselines
    }
    return baseline_values


def evaluate_subset(train_subset, train_x, train_y, test_x, test_y):
    # if isinstance(train_x, torch.Tensor):
    #     train_x = train_x.numpy()
    #     train_y = train_y.numpy()
    #     test_x = test_x.numpy()
    #     test_y = test_y.numpy()
    # LR = LinearRegression()
    # LR.fit(train_x[train_subset], train_y[train_subset])
    # pred = LR.predict(test_x)
    # error = mean_squared_error(test_y, pred)
    coef = frank_wolfe.least_norm_linear_regression(train_x[train_subset], train_y[train_subset])
    error = frank_wolfe.MSE(test_x, test_y, coef)
    return error


def design_selection(
    seller_data,
    buyer_data,
    num_select=100,
    num_iters=100,
    alpha=0.1,
    line_search=False,
    scale_cov=True,
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
    inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)
    # inv_cov = np.linalg.pinv(np.cov(X_sell.T))

    if scale_cov:
        inv_cov = scale(
            inv_cov
        )  # rescale inverse covariance matrix to be between 0 and 1

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
        # if line_search:
        #     alpha, loss = frank_wolfe.opt_step_size(
        #         X_sell[update_coord], X_buy, inv_cov, loss
        #     )
        # else:
        #     alpha = 0.01
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

        if scale_cov:
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

    save_name = "gaussian"
    if args.cluster:
        save_name += "_cluster"

    if args.buyer_subset:
        save_name += f"_buyer_subset"

    if args.seller_subset:
        save_name += f"_seller_subset"

    save_name += f"_dim_{args.num_dim}"

    save_name += f"_alpha_{args.alpha}"

    save_name += f"_{args.seed}"

    if args.debug:
        save_name = "debug"

    X_sell, y_sell, X_val, y_val, X_buy, y_buy = get_data(
        cluster=args.cluster,
        scale_data=args.scale_data,
        random_seed=args.seed,
        num_seller=1000 if args.debug else args.num_seller,
        num_buyer=(5 if args.cluster else 1) * max(args.num_buyers),
        dim=100 if args.debug else args.num_dim,
        noise_level=10,
        val_split=0.1,
        seller_subset=args.seller_subset,
        buyer_subset=args.buyer_subset,
        
    )
    print(f"{X_sell.shape=}")
    print(f"{X_val.shape=}")
    print(f"{X_buy.shape=}")

    np.save(results_dir / f"{save_name}_X_sell.npy", X_sell)
    np.save(results_dir / f"{save_name}_y_sell.npy", y_sell)
    np.save(results_dir / f"{save_name}_X_val.npy", X_val)
    np.save(results_dir / f"{save_name}_y_val.npy", y_val)
    np.save(results_dir / f"{save_name}_X_buy.npy", X_buy)
    np.save(results_dir / f"{save_name}_y_buy.npy", y_buy)

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
            num_iters=25 if args.debug else args.num_iters,
            alpha=args.alpha,
            line_search=False,
        )
        design_values[num_buyer] = design_results["weights"]
        design_errors[num_buyer] = design_results["errors"]
        design_losses[num_buyer] = design_results["losses"]

        exp_end = time.perf_counter()
        print(f"Finished in {exp_end-exp_start:.0f} seconds".center(40, "-"))

    # other data valuation baselines
    print(f"Starting baselines".center(40, "-"))
    base_start = time.perf_counter()
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

    random_trials = range(10)  # average mse over random samples of weights
    design_evals = {}
    for num_buy, weights in design_values.items():
        design_evals[f"exp_design_{num_buy}"] = {
            k: np.mean([evaluate_subset(
                np.unique(np.random.choice(np.arange(weights.shape[0]), size=k, p=weights)),
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
    one_step_values = np.mean((X_sell @ inv_cov @ X_buy.T) ** 2, axis=1)
    design_values["one_step"] = one_step_values
    design_evals["one_step"] = {
        k: evaluate_subset(one_step_values.argsort()[-k:], X_sell, y_sell, X_buy, y_buy)
        for k in eval_range
    }

    # Random seller baseline
    random_seller_values = np.random.permutation(X_sell.shape[0])
    design_evals["random_seller"] = random_seller_values
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
    design_evals["random_buyer"] = random_buyer_values
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

    with open(results_dir / f"{save_name}-design-losses.json", "w") as fh:
        json.dump(design_losses, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-design-errors.json", "w") as fh:
        json.dump(design_errors, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-values.json", "w") as fh:
        json.dump(all_values, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-evals.json", "w") as fh:
        json.dump(all_evals, fh, default=float, indent=4)

    print("Experiment completed".center(40, "="))

# TODO additional experiments
# buyer is subset of seller data / some of seller data is buyer data repeated
# vary buyers dataset between 5-100; vary k as 1-5x the buyer dataset
# increase number of clusters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main_gaussian.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization on synthetically generated data",
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--cluster", action="store_true", help="use non IID validation set"
    )
    parser.add_argument("--buyer_subset", action="store_true", help="buyer is subset of seller data")
    parser.add_argument("--seller_subset", action="store_true", help="seller data contains subsets of the buyer data")
    parser.add_argument("--scale_data", action="store_true", help="standardize data")
    parser.add_argument(
        "--num_buyers",
        nargs="+",
        default=[1, 5, 25, 50, 75, 100],
        # default=[5, 25, 50],
        type=list,
        help="number of buyer points used in experimental design",
    )
    parser.add_argument(
        "--num_seller",
        default=10000,
        type=int,
        help="number of seller points used in experimental design",
    )
    parser.add_argument(
        "--num_val",
        default=1000,
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
        default=200,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help="optimization step size",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        nargs="+",
        default=["DataOob", "RandomEvaluator"],
        # default=["DataOob", "BetaShapley", "RandomEvaluator"],
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
