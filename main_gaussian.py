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
                                 DataShapley, InfluenceFunction, InfluenceSubsample,
                                 KNNShapley, LavaEvaluator,
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
    noise_level=1,
    val_split=0.1,
    buyer_subset=False,
    num_seller_subset=0,
    return_beta=False,
    exponential=False,
    student_df=1,
    n_clusters=30,
    bone_data=False,
    cost_fn=None,
    costs=None,
):
    random_state = check_random_state(random_seed)

    val_split = num_val / (num_val + num_seller)
    num_seller += num_val

    if cluster:
        num_buyer *= n_clusters

    if bone_data:
        img = torch.load('bone-features.pt')
        img = img.to(torch.double).numpy()
        lab = torch.load('bone-labels.pt').numpy()
        X_sell, X_buy, y_sell, y_buy = train_test_split(img, lab, test_size=num_buyer, random_state=random_state)
        X_sell = X_sell[:num_seller]
        y_sell = y_sell[:num_seller]
    else:
        # Generate some random seller data
        X_sell = np.random.normal(size=(num_seller, dim))
        X_sell /= np.linalg.norm(X_sell, axis=1, keepdims=True)  # normalize data

        # generate true coefficients
        beta_true = np.random.exponential(scale=1, size=dim)
        beta_true *= np.sign(np.random.random(size=dim))

        if cost_fn is not None and costs is not None:
            X_sell *= cost_fn(costs)
            
            y_sell = X_sell @ beta_true + noise_level * np.random.randn(num_seller)
        else:
            y_sell = X_sell @ beta_true + noise_level * np.random.randn(num_seller)

        # Generate some random buyer data
        if exponential:
            X_buy = np.random.exponential(size=[num_buyer, dim])
        elif student_df:
            X_buy = np.random.standard_t(df=student_df, size=[num_buyer, dim])
        else:
            X_buy = np.random.normal(size=[num_buyer, dim])
        X_buy /= np.linalg.norm(X_buy, axis=1, keepdims=True)  # normalize data
        y_buy = X_buy @ beta_true
        # y_buy = X_buy @ beta_true + noise_level * np.random.randn(num_buyer)

    if scale_data:
        MMS = MinMaxScaler()
        #X_sell = MMS.fit_transform(X_sell)
        y_sell = MMS.fit_transform(y_sell.reshape(-1, 1)).squeeze()
        #X_buy = MMS.fit_transform(X_buy)
        y_buy = MMS.fit_transform(y_buy. reshape(-1, 1)).squeeze()

    if cluster:
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

        #val_clusters = KM.predict(X_val)
        #X_val = X_val[val_clusters == 1][:num_val]
        #y_val = y_val[val_clusters == 1][:num_val]
        X_val = X_val[:num_val]
        y_val = y_val[:num_val]

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

    if num_seller_subset > 0:
        # print('seller subset'.center(40, '='))
        X_sell = np.concatenate([np.tile(X_buy, (num_seller_subset, 1)), X_sell])
        if bone_data:
            y_sell = np.concatenate([y_buy, y_sell])
        else:
            noisy_y_buy = np.tile(y_buy, num_seller_subset) + noise_level * np.random.randn(num_seller_subset * num_buyer)
            y_sell = np.concatenate([noisy_y_buy, y_sell])
        assert np.allclose(X_sell[0], X_buy[0])

    if bone_data:
        beta_true = None
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

    baseline_values = {}
    baseline_runtimes = {}
    for b in baselines:
        start_time = time.perf_counter()
        print(b.center(40, '-'))
        baseline_values[b] = getattr(dataval, b)(**kwargs[b]).train(fetcher=fetcher, pred_model=model).data_values
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'\tTIME: {runtime:.0f}')
        baseline_runtimes[b] = runtime
    return baseline_values, baseline_runtimes 


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
    scale_cov=False,
    compute_inverse=False,
    shrink=True,
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
    # inv_cov = np.eye(X_sell.shape[1])

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
        if shrink:
            weights *= 1 - alpha  # shrink weights by 1 - alpha
        weights[update_coord] += alpha  # increase magnitude of picked coordinate

        if compute_inverse:
            inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)
        else:
            # Update inverse covariance matrix
            if shrink:
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

        selected_seller_indices = np.random.choice(np.arange(weights.shape[0]), size=num_select, p=weights/weights.sum())
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

    if args.bone_data:
        save_name = "bone"
        args.num_dim = 1000
    else:
        save_name = "gaussian"
    if args.cluster:
        save_name += "_cluster"

    if args.buyer_subset:
        save_name += f"_buyer_subset"

    if args.num_seller_subset > 0:
        save_name += f"_seller_subset_{args.num_seller_subset}"

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
        noise_level=args.noise_level,
        val_split=0.1,
        num_seller_subset=args.num_seller_subset,
        buyer_subset=args.buyer_subset,
        bone_data=args.bone_data,
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
    design_runtimes = {}
    for num_buyer in args.num_buyers:
        print(
            f"Starting experimental design with {num_buyer} buyer points".center(
                40, "-"
            )
        )
        seller_data = X_sell, y_sell
        buyer_data = X_buy[:num_buyer], y_buy[:num_buyer]
        start_time = time.perf_counter()
        design_results = design_selection(
            seller_data,
            buyer_data,
            num_select=100,
            num_iters=25 if args.debug else args.num_iters,
            alpha=args.alpha,
            line_search=False,
        )
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"Finished in {runtime:.0f} seconds".center(40, "-"))
        design_values[num_buyer] = design_results["weights"]
        design_errors[num_buyer] = design_results["errors"]
        design_losses[num_buyer] = design_results["losses"]
        design_runtimes[f'exp_design_{num_buyer}'] = runtime


    # other data valuation baselines
    print(f"Starting baselines".center(40, "-"))
    base_start = time.perf_counter()
    baseline_values, baseline_runtimes = get_baseline_values(
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
                np.arange(2, 50),
                np.arange(50, min(200, X_sell.shape[0]), 10),
                np.arange(200, min(1000, X_sell.shape[0]), 50),
            ]
        )
    ]

    baseline_evals = {}
    for baseline, values in baseline_values.items():
        exp_start = time.perf_counter()
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

    random_trials = range(10)  # average mse over random samples of weights
    design_evals = {}
    for num_buy, weights in design_values.items():
        design_evals[f"exp_design_{num_buy}"] = {
            k: np.mean([evaluate_subset(
                np.random.choice(np.arange(weights.shape[0]), size=k, p=weights, replace=False),
                # np.unique(np.random.choice(np.arange(weights.shape[0]), size=k, p=weights)),
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
    inv_cov = np.linalg.pinv(X_sell.T @ X_sell)
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
    all_values = {}
    for k, v in design_values.items():
        all_values[k] = np.array(v).tolist() 
    for k, v in baseline_values.items():
        all_values[k] = np.array(v).tolist() 
    all_evals = {}
    for k, v in design_evals.items():
        all_evals[k] = np.array(v).tolist() 
    for k, v in baseline_evals.items():
        all_evals[k] = np.array(v).tolist() 
    all_runtimes = {}
    for k, v in design_runtimes.items():
        all_runtimes[k] = v
    for k, v in baseline_runtimes.items():
        all_runtimes[k] = v

    with open(results_dir / f"{save_name}-design-losses.json", "w") as fh:
        json.dump(design_losses, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-design-errors.json", "w") as fh:
        json.dump(design_errors, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-values.json", "w") as fh:
        json.dump(all_values, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-evals.json", "w") as fh:
        json.dump(all_evals, fh, default=float, indent=4)

    with open(results_dir / f"{save_name}-runtimes.json", "w") as fh:
        json.dump(all_runtimes, fh, default=float, indent=4)
        
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
    parser.add_argument("--buyer_subset", action="store_true", help="buyer is subset of seller data")
    parser.add_argument("--num_seller_subset", default=0, type=int, help="seller data contains subsets of the buyer data repeated this many times")
    parser.add_argument("--scale_data", action="store_true", help="standardize data")
    parser.add_argument(
        "--num_buyers",
        nargs="+",
        # default=[1, 5, 25, 50, 75, 100],
        default=[1, 5, 10],
        type=list,
        help="number of buyer points used in experimental design",
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
        default=100,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "--alpha",
        default=0.1,
        type=float,
        help="optimization step size",
    )
    parser.add_argument(
        "-b",
        "--baselines",
        nargs="+",
        # default=["DataOob", "RandomEvaluator"],
        default=[
            "AME",
            "BetaShapley",
            "DataBanzhaf",
            "DataOob",
            #"DataShapley",
            "DVRL",
            #"InfluenceSubsample",
            "KNNShapley",
            "LavaEvaluator",
            "LeaveOneOut",
            "RandomEvaluator",
            #"RobustVolumeShapley",
        ],
        type=str,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--bone_data", action="store_true", help="use bone image data"
    )
    parser.add_argument(
        "--noise_level",
        default=1,
        type=float,
        help="label noise",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)
