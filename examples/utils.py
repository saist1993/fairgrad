import os
import numpy as np
import pandas as pd
from pathlib import Path
from random import shuffle


folder_location = Path("dataset")


def normalize(x):
    # scale to [-1, 1]
    x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    return x_


def get_celeb_data():
    """

    Acknowledgements
    Original data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Download the csv file from - https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    Returns
    -------

    """
    try:
        df = pd.read_csv(folder_location / Path("celebA/list_attr_celeba.csv"), sep=";")
    except FileNotFoundError:
        print(
            "please download csv file from kaggle "
            "- https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and save it in dataset/celebA/"
        )

    df = df.rename(columns={"Male": "sex"})

    s = -1 * df["sex"]
    y = df["Smiling"]
    df = df.drop(columns=["sex", "Smiling", "picture_ID"])

    X = df.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    X = X[:, (X != 0).any(axis=0)]
    y = ((y + 1) / 2).astype(int)
    s = ((s + 1) / 2).astype(int)

    return X, y, s
