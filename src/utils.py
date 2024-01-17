import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import clip
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_gaussian_data(num_samples=100, dim=10, noise=0.1, costs=None):
    X = np.random.normal(size=(num_samples, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if costs is not None:
        X *= costs
    coef = np.random.exponential(scale=1, size=dim)
    coef *= np.sign(np.random.uniform(low=-1, high=1, size=dim))
    y = X @ coef + noise * np.random.randn(num_samples)
    return dict(X=X, y=y, coef=coef, noise=noise, dim=dim, costs=costs)


def get_news_data(num_samples=100, dim=10, data_dir="csvs", scale=True):
    df = pd.read_csv(Path(data_dir) / "OnlineNewsPopularity.csv")
    X = df.iloc[:, 1 : dim + 1 - 1].values
    y = df.iloc[:, -1].values.astype(float)
    if scale:
        X = MinMaxScaler().fit_transform(X)
        # y = (y - y.min()) / (y.max() - y.min())
    # coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    coef = np.linalg.pinv(X).dot(y)
    return dict(X=X[:num_samples], y=y[:num_samples], coef=coef)


def embed_images(img_paths, model_name="clip", device="cpu"):
    match model_name:
        case "clip":
            model, preprocess = clip.load("ViT-B/32", device=device)
            inference_func = model.encode_image
        case "resnet":
            model = resnet50(pretrained=True).to(device)
            preprocess = Compose(
                [
                    Resize(size=224),
                    CenterCrop(224),
                    ToTensor(),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                ]
            )
            inference_func = model.forward
        case _:
            raise Exception("Model not found")

    embeddings = []
    with torch.inference_mode():
        for img_path in tqdm(img_paths):
            img = Image.open(img_path)
            embedding = inference_func(preprocess(img)[None].to(device))
            embeddings.append(embedding.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(embeddings)


def get_bone_data(
    num_samples,
    img_dir="../../data/bone-age/boneage-training-dataset",
    csv_path="../../data/bone-age/train.csv",
    recompute_embeddings=False,
    embedding_path="../bone_age_embeddings.pt",
    device='cuda',
):
    if recompute_embeddings or not Path(embedding_path).exists():
        img_dict = {int(p.stem): p for p in Path(img_dir).glob("*.png")}
        df = pd.read_csv(csv_path)
        img_paths = []
        labels = []
        for i, r in df.iterrows():
            img_paths.append(img_dict[r.id])
            labels.append(r.boneage)
        embeddings = embed_images(img_paths, device=device).numpy()
        labels = torch.tensor(labels).numpy()
        torch.save(dict(embeddings=embeddings, labels=labels), embedding_path)

    embed_dict = torch.load(embedding_path)
    embeddings = embed_dict["embeddings"]
    labels = embed_dict["labels"]

    return dict(X=embeddings[:num_samples], y=labels[:num_samples])


def get_data(
    dataset="gaussian",
    scale_data=False,
    cluster=False,
    random_seed=0,
    num_seller=10000,
    num_buyer=100,
    num_val=100,
    dim=100,
    noise_level=1,
    buyer_subset=False,
    num_seller_subset=0,
    return_beta=False,
    exponential=False,
    n_clusters=30,
    cost_fn=None,
    costs=None,
    recompute_embeddings=False,
):
    total_samples = num_seller + num_buyer + num_val
    match dataset:
        case "gaussian":
            data = get_gaussian_data(total_samples, dim=dim, noise=noise_level)
        case "news":
            data = get_news_data(total_samples, dim=dim)
        case "bone":
            data = get_bone_data(total_samples, recompute_embeddings=recompute_embeddings)
        case _:
            raise Exception("Dataset not found")

    X = data["X"]
    y = data["y"]
    coef = data.get("coef")

    X_dev, X_buy, y_dev, y_buy = train_test_split(
        X, y, test_size=num_buyer, random_state=random_seed
    )
    X_sell, X_val, y_sell, y_val = train_test_split(
        X_dev, y_dev, test_size=num_val, random_state=random_seed
    )
    if coef is not None:
        y_buy = X_buy @ coef

    return dict(
        X_sell=X_sell,
        y_sell=y_sell,
        X_val=X_val,
        y_val=y_val,
        X_buy=X_buy,
        y_buy=y_buy,
        coef=coef,
    )


def get_error(x_b, y_b, x_s, y_s, w, k=10, use_sklearn=False):
    s = w.argsort()[::-1][:k]
    x_k = x_s[s]
    y_k = y_s[s]

    if use_sklearn:
        LR = LinearRegression(fit_intercept=False)
        LR.fit(x_k, y_k)
        y_hat = LR.predict(x_b)
    else:
        beta_k = np.linalg.pinv(x_k) @ y_k
        y_hat = x_b @ beta_k

    return mean_squared_error(y_b, y_hat)


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
        print(b.center(40, "-"))
        baseline_values[b] = (
            getattr(dataval, b)(**kwargs[b])
            .train(fetcher=fetcher, pred_model=model)
            .data_values
        )
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"\tTIME: {runtime:.0f}")
        baseline_runtimes[b] = runtime
    return baseline_values, baseline_runtimes
