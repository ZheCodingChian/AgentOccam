import os
import time
import re
import argparse
import json
import yaml
import random
import shutil

from AgentOccam.env import WebArenaEnvironmentWrapper
from AgentOccam.AgentOccam import AgentOccam
from AgentOccam.prompts import AgentOccam_prompt


class DotDict(dict):
    """Dot notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def log_run(log_file, log_data, summary_file, summary_data):
    """Save trajectory and summary"""
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    import csv
    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)


def run():
    parser = argparse.ArgumentParser(
        description="Run AgentOccam web agent on tasks"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML config file path"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))

    if config.logging:
        if config.logname:
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))

    random.seed(42)

    config_file_list = []
    task_ids = config.env.task_ids

    if hasattr(config.env, "relative_task_dir") and config.env.relative_task_dir is not None:
        relative_task_dir = config.env.relative_task_dir
    else:
        relative_task_dir = "tasks"

    if task_ids == "all" or task_ids == ["all"]:
        task_ids = [filename[:-len(".json")] for filename in os.listdir(f"config_files/{relative_task_dir}") if filename.endswith(".json")]

    for task_id in task_ids:
        config_file_list.append(f"config_files/{relative_task_dir}/{task_id}.json")

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    def agent_init():
        return AgentOccam(
            prompt_dict={k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
            config=config.agent,
        )

    for config_file in config_file_list:
        with open(config_file, "r") as f:
            task_config = json.load(f)
            print(f"\n{'='*80}")
            print(f"Task {task_config['task_id']}: {task_config['intent'][:100]}...")
            print(f"{'='*80}")

        if config.logging and os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
            print(f"Skipping {task_config['task_id']} (already completed).")
            continue

        env = WebArenaEnvironmentWrapper(
            config_file=config_file,
            max_browser_rows=config.env.max_browser_rows,
            max_steps=config.max_steps,
            slow_mo=1,
            observation_type="accessibility_tree",
            current_viewport_only=current_viewport_only,
            viewport_size={"width": 1920, "height": 1080},
            headless=config.env.headless,
            global_config=config
        )

        agent = agent_init()
        objective = env.get_objective()
        status = agent.act(objective=objective, env=env)
        env.close()

        if config.logging:
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.actor.model,
                "type": "AgentOccam",
                "trajectory": agent.get_trajectory(),
            }
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model,
                "type": "AgentOccam",
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1) if "/" in log_file else log_file,
            }
            if status:
                summary_data.update(status)

            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=os.path.join(dstdir, "summary.csv"),
                summary_data=summary_data,
            )

            print(f"\n✓ Trajectory saved to: {log_file}")
            print(f"Status: {status}")


if __name__ == "__main__":
    run()
