# Experiment tracking
import argparse

from wbfm.pipeline.tracking import track_using_using_config
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    DEBUG = args.DEBUG

    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_3a.log")
    project_dir = project_cfg.project_dir

    check_all_needed_data_for_step(project_cfg, 2)

    if not DEBUG:
        using_monkeypatch()

    with safe_cd(project_dir):
        tracklet_cfg = project_cfg.get_training_config()
        use_barlow_network = tracklet_cfg.config["tracker_params"].get(
            "use_barlow_network", False
        )
        track_using_using_config(
            project_cfg,
            use_superglue_tracker=not use_barlow_network,
            DEBUG=DEBUG,
        )


if __name__ == "__main__":
    main()
