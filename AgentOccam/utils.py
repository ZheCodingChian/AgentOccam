import os
import json

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
EVALUATOR_DIR = os.path.join(CURRENT_DIR, "config_files")
TRAJECTORY_DIR = os.path.join(CURRENT_DIR, "trajectories")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
HOMEPAGE_URL = "localhost:4399"


def load_json_obj_from_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
