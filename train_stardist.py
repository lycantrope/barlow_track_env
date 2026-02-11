import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

import CellTracker.stardistwrapper as sdw
import h5py
import hdf5plugin  # noqa: F401, This module provide shared library of hdf filter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from csbdeep.utils import normalize
from stardist import (
    fill_label_holes,
)
from tqdm import tqdm


def plot_max_projection(im: np.ndarray, axis=0):
    im_max = np.max(im, axis=axis)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.NullLocator())
    ax.imshow(im_max)
    fig.tight_layout()


def imread(hdf_path: os.PathLike) -> np.ndarray:
    with h5py.File(hdf_path, mode="r") as handler:
        data = np.asarray(handler["data"])
    return data


def glob(path: str) -> Iterator[Path]:
    # This function mimic the glob.glob
    path_obj = Path(path)
    return path_obj.parent.glob(path_obj.name)


def load_training_images(
    path_train_images: str,
    path_train_labels: str,
):
    """Load images for training StarDist3DCustom"""

    X = sorted(glob(path_train_images), key=lambda x: x.name)
    Y = sorted(glob(path_train_labels), key=lambda x: x.name)
    assert len(X) > 0 and len(Y) > 0, "Error: No images found in either X or Y."
    assert all(
        Path(x).name == Path(y).name for x, y in zip(X, Y)
    ), "Error: Filenames in X and Y do not match."
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0, 1, 2)  # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print(
            "Normalizing image channels %s."
            % ("jointly" if axis_norm is None or 3 in axis_norm else "independently")
        )
        sys.stdout.flush()

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    if len(X) == 1:
        print(
            "Warning: only one training data was provided! It will be used for both training and validation purposes!"
        )
        X = [X[0], X[0]]
        Y = [Y[0], Y[0]]
    rng = np.random.RandomState()
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print(
        f"""number of images:{len(X):>4d}
- training:{len(X_trn):>10d}       
- validation:{len(X_val):>8d}
X[0].shape={X[0].shape}   
"""
    )
    return X, Y, X_trn, Y_trn, X_val, Y_val, n_channel


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)

    cfg = parser.parse_args()

    X, Y, X_trn, Y_trn, X_val, Y_val, n_channel = load_training_images(
        cfg.image,
        cfg.label,
    )

    model_name = "stardist_model"
    model = sdw.configure(Y, n_channel, model_name=model_name)

    model.train(
        X_trn,
        Y_trn,
        validation_data=(X_val, Y_val),
        augmenter=sdw.augmenter,
        epochs=cfg.epochs,
    )


if __name__ == "__main__":
    train()
