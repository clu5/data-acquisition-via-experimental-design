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
from transformers import GPT2Tokenizer, GPT2Model

from tqdm import tqdm


def get_gaussian_data(num_samples=100, dim=10, noise=0.1, costs=None):
    X = np.random.normal(size=(num_samples, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if costs is not None:
        X *= costs
    coef = np.random.exponential(scale=1, size=dim)
    coef *= np.sign(np.random.uniform(low=-1, high=1, size=dim))
    y = X @ coef + noise * np.random.randn(num_samples)
    return dict(X=X, y=y, coef=coef, noise=noise, dim=dim, costs=costs)


def get_mimic_data(
    num_samples,
    data_dir,
    csv_path="mimic-los-data.csv",
    scale=True,
):
    df = pd.read_csv(data_dir / csv_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    if scale:
        X = MinMaxScaler().fit_transform(X)
    coef = np.linalg.pinv(X).dot(y)
    return dict(X=X[:num_samples], y=y[:num_samples], coef=coef)


def embed_images(img_paths, model_name="clip", device="cpu"):
    match model_name:
        case "clip":
            model, preprocess = clip.load("ViT-B/32", device=device)
            inference_func = model.encode_image
        case "resnet":
            model = resnet18(pretrained=True).to(device)
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

def get_fitzpatrick_data(
    num_samples,
    data_dir,
    img_dir="fitzpatrick17k/images",
    csv_path="fitzpatrick17k/fitzpatrick-mod.csv",
    recompute_embeddings=False,
    embedding_path="fitzpatrick17k/fitzpatrick_embeddings.pt",
    device="cuda",
    model_name="clip",
):
    data_dir = Path(data_dir)
    embedding_path = Path(embedding_path)
    embedding_name = f"{embedding_path.stem}_{model_name}{embedding_path.suffix}"
    embedding_path = embedding_path.parent / embedding_name
    print(f'{data_dir=}')
    print(f'{embedding_path=}')
    if recompute_embeddings or not (data_dir / embedding_path).exists():
        print(f'No embeddings found at: {data_dir / embedding_path}. Creating new embeddings...')
        img_dict = {p.stem: p for p in Path(data_dir / img_dir).glob("*.jpg")}
        df = pd.read_csv(data_dir / csv_path)
        img_paths = []
        labels = []
        for k, v in img_dict.items():
            if k in df.md5hash.values:
                img_paths.append(v)
                labels.append(df[df.md5hash == k].aggregated_fitzpatrick_scale.values[0])
        embeddings = embed_images(img_paths, device=device, model_name=model_name).numpy()
        labels = torch.tensor(labels).numpy()
        torch.save(
            dict(embeddings=embeddings, labels=labels), data_dir / embedding_path
        )

    embed_dict = torch.load(data_dir / embedding_path)
    embeddings = embed_dict["embeddings"]
    labels = embed_dict["labels"]

    return dict(X=embeddings[:num_samples], y=labels[:num_samples])


def get_bone_data(
    num_samples,
    data_dir,
    img_dir="bone-age/boneage-training-dataset",
    csv_path="bone-age/train.csv",
    recompute_embeddings=False,
    embedding_path="bone-age/bone_age_embeddings.pt",
    device="cuda",
    model_name="clip",
):
    data_dir = Path(data_dir)
    embedding_path = Path(embedding_path)
    embedding_name = f"{embedding_path.stem}_{model_name}{embedding_path.suffix}"
    embedding_path = embedding_path.parent / embedding_name
    if recompute_embeddings or not (data_dir / embedding_path).exists():
        print(f'No embeddings found at: {data_dir / embedding_path}. Creating new embeddings...')
        img_dict = {int(p.stem): p for p in Path(data_dir / img_dir).glob("*.png")}
        df = pd.read_csv(data_dir / csv_path)
        img_paths = []
        labels = []
        for i, r in df.iterrows():
            img_paths.append(img_dict[r.id])
            labels.append(r.boneage)
        embeddings = embed_images(img_paths, device=device, model_name=model_name).numpy()
        labels = torch.tensor(labels).numpy()
        torch.save(
            dict(embeddings=embeddings, labels=labels), data_dir / embedding_path
        )

    embed_dict = torch.load(data_dir / embedding_path)
    embeddings = embed_dict["embeddings"]
    labels = embed_dict["labels"]

    return dict(X=embeddings[:num_samples], y=labels[:num_samples])


def embed_text(text_inputs:list[str], model_name='gpt2', max_length=4096, device='cuda'):
    match model_name:
        case "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2Model.from_pretrained(model_name).to(device)
        case _:
            raise Exception("Model not found")
    embeddings = []
    for x in tqdm(text_inputs):
        inputs = tokenizer(x, return_tensors="pt", max_length=max_length, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(
            outputs.last_hidden_state.mean(dim=1).cpu()
        )
    return torch.cat(embeddings)

def get_drug_data(
    num_samples,
    data_dir,
    csv_path="druglib/druglib.csv",
    recompute_embeddings=False,
    embedding_path="druglib/druglib_embeddings.pt",
    device="cuda",
    model_name="gpt2",
    max_length=4096,
):
    data_dir = Path(data_dir)
    embedding_path = Path(embedding_path)
    embedding_name = f"{embedding_path.stem}_{model_name}{embedding_path.suffix}"
    embedding_path = embedding_path.parent / embedding_name
    if recompute_embeddings or not (data_dir / embedding_path).exists():
        print(f'No embeddings found at: {data_dir / embedding_path}. Creating new embeddings...')
        df = pd.read_csv(data_dir / csv_path)
        reviews = []
        labels = []
        for i, r in tqdm(df.iterrows()):
            x = f'Benefits: {r.benefitsReview}\nSide effects: {r.sideEffectsReview}\nComments: {r.commentsReview}'
            if len(x) > max_length:
                continue
            reviews.append(x)
            labels.append(r.rating)
        embeddings = embed_text(reviews, device=device).numpy()
        labels = torch.tensor(labels).numpy()
        torch.save(
            dict(embeddings=embeddings, labels=labels), data_dir / embedding_path
        )

    embed_dict = torch.load(data_dir / embedding_path)
    embeddings = embed_dict["embeddings"]
    labels = embed_dict["labels"]

    return dict(X=embeddings[:num_samples], y=labels[:num_samples])



def split_data(num_buyer=1, num_val=10, random_state=0, X=None, y=None, costs=None):
    assert X is not None, "X is missing"
    assert y is not None, "y is missing"
    if costs is None:
        X_dev, X_buy, y_dev, y_buy = train_test_split(
            X,
            y,
            test_size=num_buyer,
            random_state=random_state,
        )
        X_sell, X_val, y_sell, y_val = train_test_split(
            X_dev,
            y_dev,
            test_size=num_val,
            random_state=random_state,
        )
        return dict(
            X_sell=X_sell,
            y_sell=y_sell,
            X_buy=X_buy,
            y_buy=y_buy,
            X_val=X_val,
            y_val=y_val,
        )
    else:
        X_dev, X_buy, y_dev, y_buy, costs_dev, costs_buy = train_test_split(
            X,
            y,
            costs,
            test_size=num_buyer,
            random_state=random_state,
        )
        X_sell, X_val, y_sell, y_val, costs_sell, costs_val = train_test_split(
            X_dev,
            y_dev,
            costs_dev,
            test_size=num_val,
            random_state=random_state,
        )
        return dict(
            X_sell=X_sell,
            y_sell=y_sell,
            costs_sell=costs_sell,
            X_buy=X_buy,
            y_buy=y_buy,
            costs_buy=costs_buy,
            X_val=X_val,
            y_val=y_val,
            costs_val=costs_val,
        )


def get_cost_function(cost_func, bias=0):
    match cost_func:
        case "square_root":
            return lambda c: c**0.5 + bias
        case "linear":
            return lambda c: c**1.0 + bias
        case "squared":
            return lambda c: c**2.0 + bias
        case _:
            raise Exception(f"{_} not supported")


def get_data(
    dataset="gaussian",
    data_dir="../../data",
    random_state=0,
    num_seller=10000,
    num_buyer=100,
    num_val=100,
    dim=100,
    noise_level=1,
    cost_range=None,
    cost_func="linear",
    recompute_embeddings=False,
):
    total_samples = num_seller + num_buyer + num_val
    data_dir = Path(data_dir)
    match dataset:
        case "gaussian":
            data = get_gaussian_data(total_samples, dim=dim, noise=noise_level)
        case "mimic":
            data = get_mimic_data(total_samples, data_dir=data_dir)
        case "fitzpatrick":
            data = get_fitzpatrick_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "bone":
            data = get_bone_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "drug":
            data = get_drug_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='gpt2',
            )
        case _:
            raise Exception("Dataset not found")

    X = data["X"]
    y = data["y"]
    coef = data.get("coef")

    if cost_range is None:
        ret = split_data(num_buyer, num_val, random_state=random_state, X=X, y=y)
    else:
        costs = np.random.choice(cost_range, size=X.shape[0]).astype(np.single)
        ret = split_data(
            num_buyer, num_val, random_state=random_state, X=X, y=y, costs=costs
        )

    ret["coef"] = coef
    ret["cost_range"] = cost_range
    ret["cost_func"] = cost_func

    match dataset, cost_range:
        case "gaussian", None:  # gaussian, no costs
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, None:  # not gaussian, no costs
            pass
        case "gaussian", _:  # gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(type(ret["costs_sell"]))
            ret["y_sell"] = (
                np.einsum("i,ij->ij", h(ret["costs_sell"]), ret["X_sell"]) @ coef + e
            )
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, _:  #  not gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(f'{e[:10].round(2)=}', e.mean())
            print(f'{ret["y_sell"][:10]}   {ret["y_sell"].mean()=}')
            print(f'{h(ret["costs_sell"][:10])=}')
            e *= ret["y_sell"].mean() / h(ret["costs_sell"])
            print(f'{e[:10].round(2)=}', e.mean())
            ret["y_sell"] = ret["y_sell"] + e
            print(f'{ret["y_sell"].mean()=}')

    return ret


def get_error_fixed(
    x_test,
    y_test,
    x_s,
    y_s,
    w,
    eval_range=range(1, 10),
    use_sklearn=False,
    return_list=False,
):
    sorted_w = w.argsort()[::-1]

    errors = {}
    for k in eval_range:
        selected = sorted_w[:k]
        x_k = x_s[selected]
        y_k = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_k, y_k)
            y_hat = LR.predict(x_test)
        else:
            beta_k = np.linalg.pinv(x_k) @ y_k
            y_hat = x_test @ beta_k

        errors[k] = mean_squared_error(y_test, y_hat)

    return list(errors.values()) if return_list else errors


def get_error_under_budget(
    x_test,
    y_test,
    x_s,
    y_s,
    w,
    costs=None,
    eval_range=range(1, 10),
    use_sklearn=False,
    return_list=False,
):
    assert costs is not None, "Missing costs"
    sorted_w = w.argsort()[::-1]
    cum_cost = np.cumsum(costs[sorted_w])

    errors = {}
    for budget in eval_range:
        under_budget_index = np.searchsorted(cum_cost, budget, side="left")

        # Could not find any points under budget constraint
        if under_budget_index == 0:
            continue

        selected = sorted_w[:under_budget_index]
        x_budget = x_s[selected]
        y_budget = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_budget, y_budget)
            y_hat = LR.predict(x_test)
        else:
            beta_budget = np.linalg.pinv(x_budget) @ y_budget
            y_hat = x_test @ beta_budget

        errors[budget] = mean_squared_error(y_test, y_hat)

    # Remove keys with values under budget
    # errors = {k: v for k, v in errors.items() if v is not None}
    return list(errors.values()) if return_list else errors


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


def plot_errors_fixed(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    errors = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(errors.items()):
        err = np.array(v)
        quantiles.append(np.quantile(err, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case _:
                k = k

        match k:
            case k if "Ours" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"
        plt.plot(
            eval_range,
            err.mean(0).squeeze(),
            label=k,
            marker=marker,
            ls=ls,
            lw=lw,
            ms=ms,
        )

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Number of Datapoints selected", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")


def plot_errors_under_budget(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    error_under_budgets = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(error_under_budgets.items()):
        error_per_budget = defaultdict(list)
        for v_i in v:
            for b, e in v_i.items():
                error_per_budget[b].append(e)

        budgets = []
        errors = []
        for b, e in dict(sorted(error_per_budget.items())).items():
            budgets.append(b)
            errors.append(np.mean(e))

        quantiles.append(np.quantile(errors, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case _:
                k = k

        match k:
            case k if "Ours" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"

        plt.plot(budgets, errors, label=k, marker=marker, ls=ls, lw=lw, ms=ms)

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Budget", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")
