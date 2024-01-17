import argparse
import json
import math
import os
import sys

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
from design import Valuator
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (AME, DVRL, BetaShapley, DataBanzhaf, DataOob,
                                 DataShapley, InfluenceFunctionEval,
                                 KNNShapley, LavaEvaluator, LeaveOneOut,
                                 RandomEvaluator, RobustVolumeShapley)
from opendataval.model import RegressionSkLearnWrapper
from PIL import Image
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


def get_data(
    dataset,
    data_dir,
    scale_data=False,
    num_image_features=30,
    num_image_samples=1000,
    random_seed=0,
    num_validation=100,
    test_frac=0.5,
    num_sellers=10,
    attributes_per_buyer=[1, 2, 3, 4],
):
    random_state = check_random_state(random_seed)

    if dataset == "mnist":
        mnist = MNIST(root=data_dir, train=True)
        model = efficientnet_b1(weights="IMAGENET1K_V1").cuda()

        resize = Resize((64, 64))
        images = (
            resize(mnist.data / 255).unsqueeze(1).repeat(1, 3, 1, 1)[:num_image_samples]
        )

        make_loader = lambda x, batch_size=32: torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=batch_size
        )
        x = torch.cat(
            [model(x_i[0].cuda()).detach().cpu() for x_i in tqdm(make_loader(images))]
        )
        y = mnist.targets.float()[:num_image_samples]

        M, D = x.shape
        # random_samples = np.random.choice(np.arange(M), num_image_samples)
        random_features = np.random.choice(np.arange(D), num_image_features)

        x = x[:, random_features]
        # x = x[:, random_features][random_samples]
        # y = y[random_samples]

    elif dataset == "bone-age":
        if (data_dir / "bone-age-resnet50-features.pt").exists():
            x = torch.load(data_dir / "bone-age-resnet50-features.pt")[
                :num_image_samples
            ]
            y = torch.load(data_dir / "bone-age-labels.pt")[:num_image_samples]
            print("bone age embeddings found. Using existing")
        else:
            bone_dir = data_dir / "bone-age"
            bone_images = list((bone_dir / "boneage-training-dataset").glob("*.png"))[
                :num_image_samples
            ]
            bone_transforms = Compose(
                [
                    Resize(size=224),
                    CenterCrop(224),
                    ToTensor(),
                    Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            )
            model = resnet50(pretrained=True).cuda()
            bone_features = []
            with torch.inference_mode():
                for k in tqdm(range(len(bone_images))):
                    image = bone_transforms(Image.open(bone_images[k]))
                    features = model(image[None].cuda()).cpu()
                    bone_features.append(features)
            del model
            torch.cuda.empty_cache()
            bone_df = pd.read_csv(bone_dir / "train.csv")
            get_label = lambda k: bone_df.loc[
                bone_df["id"] == int(bone_images[k].stem), "boneage"
            ].values[0]
            x = torch.cat(bone_features)
            y = torch.tensor(list(map(get_label, range(len(bone_images)))))
            torch.save(x, data_dir / "bone-age-resnet50-features.pt")
            torch.save(y, data_dir / "bone-age-labels.pt")

        M, D = x.shape
        # random_samples = np.random.choice(np.arange(M), num_image_samples)
        random_features = np.random.choice(np.arange(D), num_image_features)

        x = x[:, random_features]
        # x = x[:, random_features][random_samples]
        # y = y[random_samples]

    # else:  # tabular data
    elif dataset == "synthetic":
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

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    # estimate global coefficients
    LR = LinearRegression()
    LR.fit(x, y)

    # coefficients
    m = LR.coef_

    # bias
    b = LR.intercept_

    dev_x, test_x = train_test_split(
        x,
        test_size=test_frac,
        random_state=random_state,
    )

    train_x, val_x = train_test_split(
        dev_x,
        test_size=num_validation / dev_x.shape[0],
        random_state=random_state,
    )

    # number of total attributes
    M = train_x.shape[1]

    # number of attributes per seller
    M_i = max(1, M // num_sellers)

    # number of total samples
    N = train_x.shape[0]

    # number of samples per seller
    N_i = N // num_sellers

    print(
        f"{train_x.shape=}",
        f"{num_sellers=}",
        f"{M=}",
        f"{M_i=}",
        f"{N=}",
        f"{N_i=}",
        sep="\n",
    )

    pad = lambda x, padding: torch.nn.functional.pad(
        torch.tensor(x).float(), padding, "constant", 0
    )

    seller_x = torch.cat(
        [
            pad(
                train_x[N_i * i : N_i * (i + 1), M_i * i : M_i * (i + 1)],
                [M_i * i, M - (M_i * (i + 1))],
            )
            for i in range(num_sellers)
        ]
    ).float()

    # add remaining attributes
    if num_sellers * M_i < M:
        r = M - num_sellers * M_i
        seller_x = torch.cat([seller_x, pad(train_x[-N_i:, -r:], (M - r, 0))])

    assert seller_x.mean(0).all(), "found zeros in seller matrix"

    seller_y = torch.tensor(seller_x @ m + b).float()

    buyer_xs = {}
    buyer_ys = {}

    for j, num_attributes in enumerate(attributes_per_buyer):
        k = f"buyer_{j}"
        buyer_xs[k] = pad(test_x[:, :num_attributes], [0, M - num_attributes])
        buyer_ys[k] = torch.tensor(buyer_xs[k] @ m + b).float()

    val_xs = {}
    val_ys = {}

    for j, num_attributes in enumerate(attributes_per_buyer):
        k = f"val_{j}"
        val_xs[k] = pad(val_x[:, :num_attributes], [0, M - num_attributes])
        val_ys[k] = torch.tensor(val_xs[k] @ m + b).float()

    return seller_x.float(), seller_y.float(), val_xs, val_ys, buyer_xs, buyer_ys


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

    print("AME")
    ame_values = (
        AME(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("BANZHAF")
    banz_values = (
        DataBanzhaf(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("OOB")
    oob_values = (
        DataOob(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("SHAP")
    shap_values = (
        DataShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("BETA")
    beta_values = (
        BetaShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("LOO")
    loo_values = (
        LeaveOneOut(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("DVRL")
    dvrl_values = (
        DVRL(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("LAVA")
    lava_values = (
        LavaEvaluator(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("INFLUENCE")
    influence_values = (
        InfluenceFunctionEval(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("KNN")
    knn_values = (
        KNNShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("ROBUST")
    robust_values = (
        RobustVolumeShapley(random_state=random_state)
        .train(fetcher=fetcher, pred_model=model)
        .data_values
    )
    print("RANDOM")
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
        seller_x,
        seller_y,
        val_xs,
        val_ys,
        buyer_xs,
        buyer_ys,
    ) = get_data(
        args.dataset,
        data_dir,
        scale_data=args.scale_data,
        random_seed=args.seed,
        num_validation=args.num_validation,
        test_frac=args.test_frac,
        num_sellers=args.num_sellers,
        attributes_per_buyer=args.attributes_per_buyer,
    )
    print(f"{args.dataset} loaded".center(40, "-"))
    print(f"{seller_x.shape=}")
    print(f"{buyer_xs['buyer_0'].shape=}")
    print(f"{val_xs['val_0'].shape=}")

    for buyer, num_attributes in tqdm(enumerate(args.attributes_per_buyer)):
        print(f"{buyer=} {num_attributes}".center(40, "-"))
        vk = f"val_{buyer}"
        bk = f"buyer_{buyer}"

        # our method (experimental design)
        condition_number = np.linalg.cond(seller_x.T @ seller_x)
        print(f"{condition_number=}")
        assert (
            condition_number < 1 / sys.float_info.epsilon
        ), f"matrix is singular: {condition_number=}\n{seller_x.mean(0)}\n{seller_x}"
        design_values = {}
        V = Valuator()
        for num_buyer in tqdm(args.num_buyers):
            print(
                num_buyer,
                f"{buyer_xs[bk][:num_buyer].shape=}",
                f"seller={seller_x.shape}",
            )
            buyer_data = buyer_xs[f"buyer_{buyer}"][:num_buyer]
            design_values[f"optimal design w/ {num_buyer}"] = V.optimize(
                buyer_data, seller_x
            )

        # other data valuation baselines
        values = get_values(
            seller_x,
            seller_y,
            val_xs[vk],
            val_ys[vk],
            buyer_xs[bk],
            buyer_ys[bk],
            random_state=args.seed,
        )
        print(f"finished valuations".center(40, "-"))
        for k, v in design_values.items():
            values[k] = v

        num_features = seller_x.shape[1]

        if args.dataset in ("mnist", "bone-age"):
            subsets = list(range(num_features, 200, 5))
        elif args.dataset == "synthetic":
            subsets = list(range(num_features, seller_x.shape[0], 1))
        else:
            subsets = list(range(num_features, seller_x.shape[0], 5))

        errors = {}
        for k, v in values.items():
            errors[k] = {
                s: evaluate_subset(
                    v, seller_x, seller_y, buyer_xs[bk], buyer_ys[bk], k=s
                )
                for s in subsets
            }

        # validation set baseline
        num_val = val_xs[vk].shape[0]
        val_values = np.random.permutation(num_val)  # dummy values
        errors["Validation baseline"] = {
            s: evaluate_subset(
                val_values, val_xs[vk], val_ys[vk], buyer_xs[bk], buyer_ys[bk], k=s
            )
            for s in range(num_features, num_val)
        }

        # test set baseline
        num_test = buyer_xs[bk].shape[0]
        test_values = np.random.permutation(num_test)  # dummy values
        errors["Test baseline"] = {
            s: evaluate_subset(
                test_values, buyer_xs[bk], buyer_ys[bk], buyer_xs[bk], buyer_ys[bk], k=s
            )
            for s in range(num_features, num_test)
        }

        with open(
            results_dir / f"{args.dataset}-{buyer=}-attribute-values-{args.seed}.json",
            "w",
        ) as f:
            json.dump(
                {k: np.asarray(v).tolist() for k, v in values.items()},
                f,
                default=float,
                indent=4,
            )

        with open(
            results_dir / f"{args.dataset}-{buyer=}-attribute-errors-{args.seed}.json",
            "w",
        ) as f:
            json.dump(errors, f, default=float, indent=4)

    print(f"experiment for {args.dataset} complete :)".center(40, "="))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="attribute_subset.py",
        description="Runs attribute subset experiment",
        epilog="Compare data valuation methods for seller selection",
    )
    parser.add_argument("--dataset", default="diabetes")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--scale_data", action="store_true", help="standardize data")
    parser.add_argument(
        "-nb",
        "--num_buyers",
        nargs="+",
        default=[1, 5, 25, 50],
        type=list,
        help="number of buyer points used in experimental design",
    )
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
    parser.add_argument(
        "-t",
        "--test_frac",
        default=0.5,
        type=float,
        help="fraction of data to use for testing",
    )
    parser.add_argument(
        "-ns",
        "--num_sellers",
        default=5,
        type=int,
        help="number of sellers to partition attributes between (splits evenly)",
    )
    parser.add_argument(
        "--attributes_per_buyer",
        nargs="+",
        default=[1, 2, 3],
        type=list,
        help="multiplier for number of attributes each buyer has",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)
