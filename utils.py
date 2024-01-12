import argparse
import json
import math
import os
import time
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



def get_gaussian_data(num=100, dim=10, noise=0.1, costs=None):
    X = np.random.normal(size=(num, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if costs is not None:
        X *= costs
    coef = np.random.exponential(scale=1, size=dim)
    coef *= np.sign(np.random.uniform(low=-1, high=1, size=dim))
    y = X @ coef + noise * np.random.randn(num)
    return dict(X=X, y=y, coef=coef, noise=noise, dim=dim, costs=costs)


def get_news_data(num=100, dim=10, data_dir='csvs', scale=True):
    df = pd.read_csv(Path(data_dir)/'OnlineNewsPopularity.csv')
    X = df.iloc[:, 1:dim+1-1].values
    y = df.iloc[:, -1].values.astype(float)
    if scale:
        X = MinMaxScaler().fit_transform(X)
        # y = (y - y.min()) / (y.max() - y.min())
    # coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    coef = np.linalg.pinv(X).dot(y)
    return dict(X=X[:num], y=y[:num], coef=coef)

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
    # data = get_gaussian_data(num_seller + num_buyer + num_val, dim=dim, noise=noise_level)
    data = get_news_data(num_seller + num_buyer + num_val, dim=dim)
    X = data['X']
    y = data['y']
    coef = data['coef']
    
    X_dev, X_buy, y_dev, _ = train_test_split(X, y, test_size=num_buyer, random_state=random_seed)
    X_sell, X_val, y_sell, y_val = train_test_split(X_dev, y_dev, test_size=num_val, random_state=random_seed)
    
    return X_sell, y_sell, X_val, y_val, X_buy, X_buy @ coef, coef

    # match dataset
    # case "gaussian"
    # case "bone"
    
    # random_state = check_random_state(random_seed)

    # val_split = num_val / (num_val + num_seller)
    # num_seller += num_val

    # if cluster:
    #     num_buyer *= n_clusters

    # if bone_data:
    #     img = torch.load('bone-features.pt')
    #     img = img.to(torch.double).numpy()
    #     lab = torch.load('bone-labels.pt').numpy()
    #     X_sell, X_buy, y_sell, y_buy = train_test_split(img, lab, test_size=num_buyer, random_state=random_state)
    #     X_sell = X_sell[:num_seller]
    #     y_sell = y_sell[:num_seller]
    # else:
    #     # Generate some random seller data
    #     X_sell = np.random.normal(size=(num_seller, dim))
    #     X_sell /= np.linalg.norm(X_sell, axis=1, keepdims=True)  # normalize data

    #     # generate true coefficients
    #     beta_true = np.random.exponential(scale=1, size=dim)
    #     beta_true *= np.sign(np.random.random(size=dim))

    #     if cost_fn is not None and costs is not None:
    #         X_sell *= cost_fn(costs)
            
    #         y_sell = X_sell @ beta_true + noise_level * np.random.randn(num_seller)
    #     else:
    #         y_sell = X_sell @ beta_true + noise_level * np.random.randn(num_seller)

    #     # Generate some random buyer data
    #     if exponential:
    #         X_buy = np.random.exponential(size=[num_buyer, dim])
    #     elif student_df:
    #         X_buy = np.random.standard_t(df=student_df, size=[num_buyer, dim])
    #     else:
    #         X_buy = np.random.normal(size=[num_buyer, dim])
    #     X_buy /= np.linalg.norm(X_buy, axis=1, keepdims=True)  # normalize data
    #     y_buy = X_buy @ beta_true
    #     # y_buy = X_buy @ beta_true + noise_level * np.random.randn(num_buyer)

    # if scale_data:
    #     MMS = MinMaxScaler()
    #     #X_sell = MMS.fit_transform(X_sell)
    #     y_sell = MMS.fit_transform(y_sell.reshape(-1, 1)).squeeze()
    #     #X_buy = MMS.fit_transform(X_buy)
    #     y_buy = MMS.fit_transform(y_buy. reshape(-1, 1)).squeeze()

    # if cluster:
    #     KM = KMeans(n_clusters=n_clusters, init="k-means++")
    #     KM.fit(X_buy)
    #     buyer_clusters = KM.labels_

    #     X_buy = X_buy[buyer_clusters == 0]
    #     y_buy = y_buy[buyer_clusters == 0]

    #     X_sell, X_val, y_sell, y_val = train_test_split(
    #         X_sell,
    #         y_sell,
    #         test_size=val_split,
    #         random_state=random_state,
    #     )

    #     #val_clusters = KM.predict(X_val)
    #     #X_val = X_val[val_clusters == 1][:num_val]
    #     #y_val = y_val[val_clusters == 1][:num_val]
    #     X_val = X_val[:num_val]
    #     y_val = y_val[:num_val]

    # else:
    #     X_sell, X_val, y_sell, y_val = train_test_split(
    #         X_sell,
    #         y_sell,
    #         test_size=val_split,
    #         random_state=random_state,
    #     )
    #     X_val = X_val[:num_val]
    #     y_val = y_val[:num_val]

    # if buyer_subset:
    #     # print('buyer subset'.center(40, '='))
    #     X_buy = X_sell[:num_buyer]
    #     y_buy = y_sell[:num_buyer]
    #     assert np.allclose(X_sell[0], X_buy[0])

    # if num_seller_subset > 0:
    #     # print('seller subset'.center(40, '='))
    #     X_sell = np.concatenate([np.tile(X_buy, (num_seller_subset, 1)), X_sell])
    #     if bone_data:
    #         y_sell = np.concatenate([y_buy, y_sell])
    #     else:
    #         noisy_y_buy = np.tile(y_buy, num_seller_subset) + noise_level * np.random.randn(num_seller_subset * num_buyer)
    #         y_sell = np.concatenate([noisy_y_buy, y_sell])
    #     assert np.allclose(X_sell[0], X_buy[0])

    # if bone_data:
    #     beta_true = None
    # if return_beta:
    #     return X_sell, y_sell, X_val, y_val, X_buy, y_buy, beta_true
    # else:
    #     return X_sell, y_sell, X_val, y_val, X_buy, y_buy
