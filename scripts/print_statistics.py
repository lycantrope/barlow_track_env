import argparse

from wbfm.utils.projects.finished_project_data import print_project_statistics
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project_status import print_sacred_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print current project status')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    args = parser.parse_args()
    project_path = args.project_path
    project_config = ModularProjectConfig(project_path, log_to_file=False)

    print_project_statistics(project_config)
