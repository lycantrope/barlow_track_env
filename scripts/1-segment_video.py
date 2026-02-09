"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""

import argparse
import os
from pathlib import Path

# Experiment tracking
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.segmentation.util.utils_pipeline import (
    segment_video_using_config_2d,
    segment_video_using_config_3d,
)

SEGMENTS_FACTORY = {
    "2d": segment_video_using_config_2d,
    "3d": segment_video_using_config_3d,
}

ENVS = {
    "NUMEXPR_MAX_THREADS": "56",  # For windows workstation
    # Set environment variables to (try to) deal with rare blosc decompression errors
    "BLOSC_NOLOCK": "1",
    "BLOSC_NTHREADS": "1",
    # Tensorflow has memory flushing problems, so disallow gpu
    # For unknown reason, the inference using GPU is much slower than multithreading CPU.
    # We disable the GPU.
    "CUDA_VISIBLE_DEVICES": "-1",
}

os.environ.update(ENVS)


def segment_video():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--continue_from_frame", default=None)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()
    args.project_path = str(Path(args.project_path).as_posix())
    project_path = args.project_path
    continue_from_frame = args.continue_from_frame
    DEBUG = args.DEBUG

    if not DEBUG:
        using_monkeypatch()

    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_1.log")
    check_all_needed_data_for_step(project_cfg, 1)

    segment_cfg = project_cfg.get_segmentation_config()
    preprocessing_cfg = project_cfg.get_preprocessing_config()

    mode = segment_cfg.config["segmentation_type"]
    opt = {
        "preprocessing_cfg": preprocessing_cfg,
        "segment_cfg": segment_cfg,
        "project_cfg": project_cfg,
        "continue_from_frame": continue_from_frame,
        "DEBUG": DEBUG,
    }

    try:
        func = SEGMENTS_FACTORY[mode]
    except KeyError:
        raise ValueError(
            f"Unknown segmentation_type; expected '2d' or '3d' instead of {mode}"
        )
    with safe_cd(project_path):
        func(**opt)


if __name__ == "__main__":
    segment_video()
