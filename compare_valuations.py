import argparse
import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (AME, DVRL, BetaShapley, DataBanzhaf, DataOob,
                                 DataShapley, InfluenceFunctionEval,
                                 KNNShapley, LavaEvaluator, LeaveOneOut,
                                 RandomEvaluator, RobustVolumeShapley)
from opendataval.model import RegressionSkLearnWrapper
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from torchvision.datasets import MNIST
from torchvision.models import efficientnet_b1, resnet18
from torchvision.transforms import Resize
from tqdm import tqdm

from design import Valuator


def get_data(
    dataset,
    data_dir,
    scale_data=False,
    num_image_features=100,
    num_image_samples=5000,
    cluster=False,
    random_seed=0,
    num_validation=100,
):
    random_state = check_random_state(random_seed)

    if dataset == "mnist":
        mnist_train = MNIST(root=data_dir, train=True)
        mnist_test = MNIST(root=data_dir, train=False)
        model = efficientnet_b1(weights="IMAGENET1K_V1").cuda()

        resize = Resize((64, 64))
        train_data = resize(mnist_train.data / 255).unsqueeze(1).repeat(1, 3, 1, 1)
        test_data = resize(mnist_test.data / 255).unsqueeze(1).repeat(1, 3, 1, 1)

        make_loader = lambda x, batch_size=32: torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=batch_size
        )
        dev_x = torch.cat(
            [model(x[0].cuda()).detach().cpu() for x in tqdm(make_loader(train_data))]
        )
        test_x = torch.cat(
            [model(x[0].cuda()).detach().cpu() for x in tqdm(make_loader(test_data))]
        )

        dev_y = mnist_train.targets.float()
        test_y = mnist_test.targets.float()

        M, D = dev_x.shape
        random_samples = np.random.choice(np.arange(M), num_image_samples * (2 if cluster else 1))
        random_features = np.random.choice(np.arange(D), num_image_features)

        dev_x = dev_x[:, random_features][random_samples]
        dev_y = dev_y[random_samples]
        test_x = test_x[:, random_features]

        if scale_data:
            MMS = MinMaxScaler()
            dev_x = MMS.fit_transform(dev_x)
            dev_y = MMS.fit_transform(dev_y[:, None]).flatten()
            test_x = MMS.fit_transform(test_x)
            test_y = MMS.fit_transform(test_y[:, None]).flatten()
            print(f'{dev_x.sum()=}')
            print(f'{test_x.sum()=}')

    else:
        if dataset == "synthetic":
            x, y = make_regression(
                n_samples=1000,
                n_features=30,
                n_informative=5,
                random_state=random_state,
                noise=1,
            )
        elif dataset == "diabetes":
            data = load_diabetes()
            x = data["data"]
            y = data["target"]
            x = np.delete(x, 1, 1)  # exclude binary feature to prevent singluar matrix
        elif dataset == "housing":
            column_names = [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ]
            df = pd.read_csv(
                data_dir / "housing.csv",
                header=None,
                delimiter=r"\s+",
                names=column_names,
            )
            exclude = ["CHAS"]
            df = df.drop(
                columns=exclude
            )  # exclude binary feature to prevent singluar matrix
            x = df.iloc[:, :12].values
            y = df.MEDV.values
        elif dataset == "wine":
            df = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
            x = df.iloc[:, :11].values
            y = df.quality.values.astype(float)
        elif dataset == "fires":
            df = pd.read_csv(data_dir / "forestfires.csv")
            df.month = pd.get_dummies(df.month).values.argmax(1)
            df.day = pd.get_dummies(df.day).values.argmax(1)
            df.drop(columns=["month", "day"], inplace=True)
            x = df.iloc[:, :10].values
            y = df.area.values
        else:
            raise ValueError(f"{dataset} not found")

        if scale_data:
            MMS = MinMaxScaler()
            x = MMS.fit_transform(x)
            y = MMS.fit_transform(y[:, None]).flatten()

        dev_x, test_x, dev_y, test_y = train_test_split(
            x,
            y,
            test_size=0.5,
            random_state=random_state,
        )

    if cluster:
        KM = KMeans(n_clusters=3, init="k-means++")
        KM.fit(np.concatenate([dev_x, test_x]))
        cluster_mask = KM.labels_

        val_mask = (KM.predict(dev_x) == 0)
        val_x = dev_x[val_mask]
        val_y = dev_y[val_mask]

        train_x = dev_x[~val_mask]
        train_y = dev_y[~val_mask]

        test_mask = KM.predict(test_x) == 1
        test_x = test_x[test_mask]
        test_y = test_y[test_mask]

    else:
        train_x, val_x, train_y, val_y = train_test_split(
            dev_x,
            dev_y,
            test_size=0.2,
            random_state=random_state,
        )

    val_x = val_x[:num_validation]
    val_y = val_y[:num_validation]

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    )


def get_values(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    metric=mean_squared_error,
    random_state=0,
):
    fetcher = DataFetcher.from_data_splits(
        train_x, train_y, val_x, val_y, test_x, test_y, one_hot=False
    )
    model = RegressionSkLearnWrapper(LinearRegression)

    ame_values = (
        AME(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    banz_values = (
        DataBanzhaf(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    oob_values = (
        DataOob(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    shap_values = (
        DataShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    beta_values = (
        BetaShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    loo_values = (
        LeaveOneOut(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    dvrl_values = (
        DVRL(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    lava_values = (
        LavaEvaluator(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    influence_values = (
        InfluenceFunctionEval(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    knn_values = (
        KNNShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    robust_values = (
        RobustVolumeShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    random_values = (
        RandomEvaluator(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )

    return {
        "AME": ame_values,
        "Banzhaf": banz_values,
        "OOB": oob_values,
        "Shapley": shap_values,
        "Beta Shapley": beta_values,
        "Robust Shapley": robust_values,
        "KNN Shapley": knn_values,
        "LOO": loo_values,
        "DVRL": dvrl_values,
        "LAVA": lava_values,
        "Influence": influence_values,
        "Random": random_values,
    }


def evaluate_subset(train_values, train_x, train_y, test_x, test_y, k=10):
    if isinstance(train_x, torch.Tensor):
        train_x = train_x.numpy()
    if isinstance(train_y, torch.Tensor):
        train_y = train_y.numpy()
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.numpy()
    if isinstance(test_y, torch.Tensor):
        test_y = test_y.numpy()
    subset = train_values.argsort()[:-k:-1]
    # subset = train_values.argsort()[:k]
    # print(subset)
    LR = LinearRegression()
    LR.fit(train_x[subset], train_y[subset])
    pred = LR.predict(test_x)
    error = mean_squared_error(test_y, pred)
    return error


def main(args):
    np.random.seed(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    print(args.dataset.center(40, "="))

    data_dir = Path(args.data_dir)
    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    ) = get_data(
        args.dataset,
        data_dir,
        cluster=args.cluster,
        scale_data=args.scale_data,
        random_seed=args.seed,
        num_validation=args.num_validation,
    )
    print(f"{train_x.shape=}")
    print(f"{val_x.shape=}")
    print(f"{test_x.shape=}")

    # other data valuation baselines
    values = get_values(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        random_state=args.seed,
    )

    # our method (experimental design)
    V = Valuator()
    for num_buyer in args.num_buyers:
        design_values = V.optimize(test_x[:num_buyer], train_x)
        values[f"Design-{num_buyer}"] = design_values

    num_features = train_x.shape[1]

    if args.dataset == "mnist":
        subsets = list(range(num_features, 200, 5))
    elif args.dataset == "synthetic":
        subsets = list(range(num_features, train_x.shape[0], 1))
    else:
        subsets = list(range(num_features, train_x.shape[0], 5))


    errors = {}
    for k, v in values.items():
        errors[k] = {
            s: evaluate_subset(v, train_x, train_y, test_x, test_y, k=s)
            for s in subsets
        }

    # validation set baseline
    num_val = val_x.shape[0]
    val_values = np.random.permutation(num_val)  # dummy values
    errors["Validation baseline"] = {
        s: evaluate_subset(val_values, val_x, val_y, test_x, test_y, k=s)
        for s in range(num_features, num_val)
    }

    # test set baseline
    num_test = test_x.shape[0]
    test_values = np.random.permutation(num_test)  # dummy values
    errors["Test baseline"] = {
        s: evaluate_subset(test_values, test_x, test_y, test_x, test_y, k=s)
        for s in range(num_features, num_test)
    }

    with open(
        results_dir
        / f"values-{args.dataset}-{'non-iid' if args.cluster else 'iid'}-{args.seed}.json",
        "w",
    ) as f:
        json.dump(
            {k: np.asarray(v).tolist() for k, v in values.items()},
            f,
            default=float,
            indent=4,
        )

    with open(
        results_dir
        / f"errors-{args.dataset}-{'non-iid' if args.cluster else 'iid'}-{args.seed}.json",
        "w",
    ) as f:
        json.dump(errors, f, default=float, indent=4)

    print(f"{args.dataset} completed".center(40, "="))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compare_valutions.py",
        description="Runs data valuation experiment",
        epilog="Compare data valuation methods",
    )
    parser.add_argument("--dataset", default="diabetes")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--results_dir", default="new_results")
    parser.add_argument(
        "--cluster", action="store_true", help="use non IID validation set"
    )
    parser.add_argument("--scale_data", action="store_true", help="standardize data")
    parser.add_argument(
        "-nb",
        "--num_buyers",
        nargs='+',
        default=[1, 5, 25, 50],
        type=list,
        help="number of buyer points used in experimental design",
    )
    # parser.add_argument(
    #     "-nt",
    #     "--num_train",
    #     default=10000,
    #     help="number of training points used in valuation",
    # )
    parser.add_argument(
        "-nv",
        "--num_validation",
        default=50,
        type=int,
        help="number of validation points used in valuation",
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
