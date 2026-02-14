import argparse
import os

import hdf5plugin
from wbfm.pipeline.project_initialization import preprocess_fluorescence_data
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


def preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--continue_from_frame", default=0)
    parser.add_argument("--to_zip_zarr_using_7z", default=True, type=bool)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    to_zip_zarr_using_7z = args.to_zip_zarr_using_7z

    DEBUG = args.DEBUG

    if not DEBUG:
        using_monkeypatch()

    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger("step_0c.log")

    cfg.config["project_path"] = project_path

    preprocess_fluorescence_data(cfg, to_zip_zarr_using_7z, DEBUG)


if __name__ == "__main__":
    preprocess()
