from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List

import CellTracker.stardistwrapper as sdw
import h5py
import hdf5plugin  # noqa: F401, This module provide shared library of hdf filter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from CellTracker.stardist3dcustom import StarDist3DCustom
from csbdeep.utils import normalize
from stardist import (
    Rays_GoldenSpiral,
    calculate_extents,
    fill_label_holes,
    gputools_available,
)
from stardist.models import Config3D
from stardist.utils import _normalize_grid
from tqdm import tqdm

STARDIST_MODELS = "stardist_models"

UP_LIMIT = 400000


def print_dict(my_dict: dict):
    for key in my_dict.keys():
        print(f"{key}: {my_dict[key]}")


def stardist_configure(
    Y: List[np.ndarray],
    n_channel: int,
    up_limit: int = UP_LIMIT,
    model_name: str = "stardist",
    basedir: str = STARDIST_MODELS,
):
    # extents = calculate_extents(Y)
    # anisotropy = tuple(np.max(extents) / extents)
    model_home = Path.home() / basedir / model_name
    config_path = model_home / "config.json"

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    if not config_path.is_file():
        anisotropy = (1.0, 1.0, 1.0)
        print("empirical anisotropy of labeled objects = %s" % str(anisotropy))

        # 96 is a good default choice (see 1_data.ipynb)
        n_rays = 96

        # Predict on subsampled grid for increased efficiency and larger field of view
        grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

        # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
        rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

        # Set train_patch_size which should
        # 1. match anisotropy and under a predefined limitation
        a, b, c = anisotropy
        train_patch_size = np.cbrt(up_limit * a * b * c) / np.array([a, b, c])
        # 2. less than the image size
        up_limit_xyz = Y[0].shape[0], np.min(Y[0].shape[1:3]), np.min(Y[0].shape[1:3])
        scaling = np.min(np.asarray(up_limit_xyz) / train_patch_size)
        if scaling < 1:
            train_patch_size = train_patch_size * scaling
        # 3. can be divided by div_by (related to unet architecture)
        # Increase unet_n_depth from 2 to 3
        unet_n_depth = 2  #
        grid_norm = _normalize_grid(grid, 3)
        unet_pool = (2, 2, 2)
        div_by = tuple(p**unet_n_depth * g for p, g in zip(unet_pool, grid_norm))
        print(f"div_by={div_by}")
        train_patch_size = [int(d * (i // d)) for i, d in zip(train_patch_size, div_by)]
        # 4. size of x and y should be the same (since augmentation will flip x-y axes)
        train_patch_size[1] = train_patch_size[2] = min(train_patch_size[1:])
        conf = Config3D(
            rays=rays,
            grid=grid,
            anisotropy=anisotropy,
            use_gpu=use_gpu,
            n_channel_in=n_channel,
            # adjust for your data below (make patch size as large as possible)
            train_patch_size=train_patch_size,
            train_batch_size=2,
            # Increase U-Net depth from 2 to 3
            unet_n_depth=unet_n_depth,
            # use sigmoid in last layer
            unet_last_activation="relu",
        )
        assert (
            conf.unet_n_depth == unet_n_depth
        ), f"{conf.unet_n_depth} != {unet_n_depth}"
        assert conf.grid == grid_norm, f"{conf.grid} != {grid_norm}"
        assert conf.unet_pool == unet_pool, f"{conf.unet_pool} != {unet_pool}"

    else:
        conf = Config3D(**json.load(config_path.open("r")))

    print_dict(vars(conf))
    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist3DCustom(config=conf, name=model_name, basedir=basedir)
    if (model_home / "weights_manual.keras").is_file():
        model.keras_model.load_weights(str(model_home / "weights_manual.keras"))
        print("Load model from weights_manual.keras")
    elif (model_home / "weights_best.h5").is_file():
        model.keras_model.load_weights(
            str(model_home / "weights_manual.keras"),
            by_name=True,
            skip_mismatch=True,
        )
        print("Load model from weights_best.h5")

    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap("ZYX"))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print(
            "WARNING: median object size larger than field of view of the neural network."
        )

    return model


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
    path_train_images: List[Path],
    path_train_labels: List[Path],
):
    """Load images for training StarDist3DCustom"""
    assert (
        len(path_train_images) > 0 and len(path_train_labels) > 0
    ), "Error: No images found in either X or Y."
    assert all(
        Path(x).name == Path(y).name
        for x, y in zip(path_train_images, path_train_labels)
    ), "Error: Filenames in X and Y do not match."
    X = list(map(imread, path_train_images))
    Y = list(map(imread, path_train_labels))
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


def imwrite(hdf_path: os.PathLike, data: np.ndarray) -> np.ndarray:
    with h5py.File(hdf_path, mode="w") as handler:
        handler.create_dataset("data", data=data)


def train_and_inference():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="stardist_3d")

    cfg = parser.parse_args()

    path_train_images = sorted(glob(cfg.image), key=lambda x: x.name)
    path_train_labels = sorted(glob(cfg.label), key=lambda x: x.name)

    X, Y, X_trn, Y_trn, X_val, Y_val, n_channel = load_training_images(
        path_train_images,
        path_train_labels,
    )

    model_name = cfg.model_name
    model = stardist_configure(Y, n_channel, model_name=model_name)

    model.train(
        X_trn,
        Y_trn,
        validation_data=(X_val, Y_val),
        augmenter=sdw.augmenter,
        epochs=cfg.epochs,
    )

    # Save final model as keras files
    model.keras_model.save(
        str(Path.home() / STARDIST_MODELS / model_name / "weights_manual.keras")
    )
    model.optimize_thresholds(X_val, Y_val)

    # make inference of all data
    output = path_train_images[0].parent.parent / "pred"
    output.mkdir(exist_ok=True, parents=True)

    for im_p, img in tqdm(zip(path_train_images, X)):
        # normalizing images (stardist function)
        (labels, details), prob_map = model.predict_instances(img)
        imwrite(output / f"pred_{model_name}_{im_p.name}", labels.astype("u2"))

    # Save final model as keras files
    model.keras_model.save(
        str(Path.home() / STARDIST_MODELS / model_name / "weights_manual.keras")
    )


if __name__ == "__main__":
    train_and_inference()
