"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import argparse

from wbfm.pipeline.tracking import match_tracks_and_tracklets_using_config
from wbfm.pipeline.tracklets import consolidate_tracklets_using_config
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


def combine_tracks():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    DEBUG = args.DEBUG

    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_3b.log")

    if not DEBUG:
        using_monkeypatch()

    match_tracks_and_tracklets_using_config(project_cfg, DEBUG=DEBUG)

    training_cfg = project_cfg.get_training_config()
    z_threshold = training_cfg.config["pairwise_matching_params"].get(
        "z_threshold", 2.5
    )
    consolidate_tracklets_using_config(
        project_cfg,
        z_threshold=z_threshold,
        DEBUG=DEBUG,
    )


if __name__ == "__main__":
    combine_tracks()
