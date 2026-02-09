import argparse
import logging
from pathlib import Path

from wbfm.pipeline.project_initialization import build_project_structure_from_nwb_file
from wbfm.utils.external.custom_errors import MissingAnalysisError
from wbfm.utils.nwb.utils_nwb_unpack import unpack_nwb_to_project_structure
from wbfm.utils.segmentation.util.utils_metadata import recalculate_metadata_from_config

logger = logging.getLogger(__name__)


def main():
    """
    Example:
    python 0a-create_new_project_from_nwb.py with
        project_dir='/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms'
        nwb_file='/path/to/your/data.nwb'

    See also wbfm/scripts/0a-create_new_project.py
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--nwb_file")
    parser.add_argument("--experimenter", default="")
    parser.add_argument("--task_name", default="")
    parser.add_argument("--copy_nwb_file", action="store_true")
    parser.add_argument("--unpack_nwb", action="store_true")

    args = parser.parse_args()

    cfg = vars(args)
    cfg["project_dir"] = str(Path(cfg["project_dir"]).as_posix())
    cfg["nwb_file"] = str(Path(cfg["nwb_file"]).as_posix())

    project_fname = build_project_structure_from_nwb_file(
        cfg,
        cfg["nwb_file"],
        cfg["copy_nwb_file"],
    )

    if cfg["unpack_nwb"]:
        unpack_nwb_to_project_structure(project_fname)
        try:
            recalculate_metadata_from_config(
                project_fname, name_mode="neuron", allow_hybrid_loading=True
            )
        except MissingAnalysisError:
            logger.warning(
                "Could not recalculate metadata after unpacking NWB file; this may be because segmentation has not yet been run."
            )

    logger.info(f"Successfully created new project at: {project_fname}")


if __name__ == "__main__":
    main()
