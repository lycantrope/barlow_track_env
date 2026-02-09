"""
The top level function for producing training data via feature-based tracking
"""

import argparse

from wbfm.pipeline.tracklets import postprocess_matches_to_tracklets_using_config
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


def produce_training_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    DEBUG = args.DEBUG

    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_2b.log")

    train_cfg = project_cfg.get_training_config()
    segmentation_config = project_cfg.get_segmentation_config()

    if not DEBUG:
        using_monkeypatch()

    postprocess_matches_to_tracklets_using_config(
        project_cfg,
        segmentation_config,
        train_cfg,
        DEBUG=DEBUG,
    )


if __name__ == "__main__":
    produce_training_data()
