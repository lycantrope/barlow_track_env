"""
The top level function for producing training data via feature-based tracking
"""

import argparse

from wbfm.pipeline.tracklets import build_frame_objects_using_config
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import (
    ModularProjectConfig,
    update_path_to_segmentation_in_config,
)
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step


def produce_training_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--only_calculate_desynced", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    DEBUG = args.DEBUG
    only_calculate_desynced = args.only_calculate_desynced

    if not DEBUG:
        using_monkeypatch()

    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_2a.log")
    check_all_needed_data_for_step(project_cfg, 2)

    train_cfg = update_path_to_segmentation_in_config(project_cfg)
    train_cfg.update_self_on_disk()

    build_frame_objects_using_config(
        project_cfg,
        train_cfg,
        only_calculate_desynced=only_calculate_desynced,
        DEBUG=DEBUG,
    )


if __name__ == "__main__":
    produce_training_data()
