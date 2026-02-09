import argparse

from wbfm.utils.projects.utils_project_status import print_sacred_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print current project status')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    args = parser.parse_args()
    project_path = args.project_path

    print_sacred_log(project_path)
