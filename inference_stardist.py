import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import CellTracker.stardistwrapper as sdw
import h5py
import hdf5plugin  # noqa: F401, This module provide shared library of hdf filter
import numpy as np
from csbdeep.utils import normalize
from tqdm import tqdm


def imread(hdf_path: os.PathLike) -> np.ndarray:
    with h5py.File(hdf_path, mode="r") as handler:
        data = np.asarray(handler["data"])
    return data


def imwrite(hdf_path: os.PathLike, data: np.ndarray) -> np.ndarray:
    with h5py.File(hdf_path, mode="w") as handler:
        handler.create_dataset("data", data=data)


def glob(path: str) -> Iterator[Path]:
    # This function mimic the glob.glob
    path_obj = Path(path)
    return path_obj.parent.glob(path_obj.name)


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str)

    cfg = parser.parse_args()
    model_path = Path(cfg.model_path)
    image_path = cfg.image
    output = cfg.output

    config_path = model_path.joinpath("config.json")

    if not config_path.is_file():
        parser.error(f"model_path is not a stardist folder: {cfg.model_path}")

    if output is None:
        output = Path(image_path).parent / "pred"
    else:
        output = Path(output)

    output.mkdir(exist_ok=True)

    images = sorted(glob(image_path), key=lambda x: x.name)

    config = sdw.Config3D(**json.load(config_path.open("r")))
    model = sdw.StarDist3DCustom(
        config=config,
        name=model_path.name,
        basedir=model_path.parent.as_posix(),
    )

    axis_norm = (0, 1, 2)

    for im_p in tqdm(images):
        img = imread(im_p)

        # normalizing images (stardist function)
        img_norm = normalize(img, 1, 99.8, axis=axis_norm)

        (labels, details), prob_map = model.predict_instances(img_norm)
        imwrite(output / im_p.name, labels)

    print("finish")


if __name__ == "__main__":
    inference()
