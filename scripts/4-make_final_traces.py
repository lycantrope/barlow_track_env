"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import argparse

from wbfm.pipeline.traces import full_step_4_make_traces_from_config

# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path", required=True)
    parser.add_argument("--match_using_indices", action="store_false")
    parser.add_argument("--allow_hybrid_loading", action="store_false")
    parser.add_argument("--allow_only_global_tracker", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")

    args = parser.parse_args()

    project_path = args.project_path
    DEBUG = args.DEBUG
    allow_only_global_tracker = args.allow_only_global_tracker
    match_using_indices = args.match_using_indices
    allow_hybrid_loading = args.allow_hybrid_loading
    project_cfg = ModularProjectConfig(project_path)
    project_cfg.setup_logger("step_4.log")

    if not DEBUG:
        using_monkeypatch()

    full_step_4_make_traces_from_config(
        project_cfg,
        allow_only_global_tracker=allow_only_global_tracker,
        allow_hybrid_loading=allow_hybrid_loading,
        match_using_indices=match_using_indices,
        DEBUG=DEBUG,
    )


if __name__ == "__main__":
    main()
