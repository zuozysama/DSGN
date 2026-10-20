import argparse
import colorsys
import os
import re
import sys
import json
import numpy as np
import glob
import multiprocessing
from multiprocessing import Pool, cpu_count
from datetime import datetime
from matplotlib import pyplot as plt

sys.path.insert(0, "../")

from visualization import env_visualizer_cus
from utils import logger as logger
from config_manager import ConfigManager


def generate_distinct_colors(num_colors):
    """Generate high-contrast colors for visualization."""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.0
    for i in range(num_colors):
        h += golden_ratio_conjugate
        h %= 1.0
        # Avoid deep blue background
        if 0.55 < h < 0.75:
            h += 0.25
            h %= 1.0
        s = 0.5 + (i % 2) * 0.5
        v = 0.85 + (i % 3) * 0.05
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return colors


def ensure_visualization_config_loaded():
    """Load the base env config needed by MarineEnv before replaying JSON trajectories."""
    config_manager = ConfigManager()
    if config_manager.get_config().get("env"):
        return

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(project_root, "config", "exp_config_cus.yaml")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Base visualization config not found: {config_file_path}")

    config_manager.load_config(config_file_path)


def _worker_wrapper(args):
    """Run  worker wrapper."""
    return _run_visualization_process(*args)


def _run_visualization_process(json_file_path, save_video=False):
    """Run  run visualization process."""

    plt.switch_backend('Agg')
    ensure_visualization_config_loaded()

    if not os.path.exists(json_file_path):
        print(f"[Error] File not found: {json_file_path}")
        return

    try:
        with open(json_file_path, "r") as f:
            exp_data = json.load(f)

        # print(f"Processing: {os.path.basename(json_file_path)}")

        # Extract model names used in the experiment

        metadata = exp_data.get("metadata", {})
        model_name = metadata.get("model", "UnknownModel")
        scenario = metadata.get("scenario", "UnknownScenario")
        ep_id = metadata.get("episode_idx", "0")


        episode_log = {}
        if "episode_log" in exp_data:
            episode_log["episode_log"] = exp_data["episode_log"]

            if "milestones" in exp_data["episode_log"]:
                episode_log["milestones"] = exp_data["episode_log"]["milestones"]

        colors = generate_distinct_colors(120)

        curr_scenario_model_name_episode_traj = exp_data.get("trajectories", [])
        if not curr_scenario_model_name_episode_traj:
            print(f"[Skip] No trajectory data in {os.path.basename(json_file_path)}")
            return


        current_dir = os.path.dirname(json_file_path)
        viz_root_dir = os.path.join(current_dir, "Visualizations")
        os.makedirs(viz_root_dir, exist_ok=True)

        # Create sub-folder for this specific model

        current_model_name = model_name
        model_viz_dir = os.path.join(viz_root_dir, current_model_name)
        os.makedirs(model_viz_dir, exist_ok=True)

        file_basename = f"sched_{scenario}_ep_{ep_id}"
        image_path = os.path.join(model_viz_dir, f"{file_basename}.pdf")
        video_path = (
            os.path.join(model_viz_dir, f"{file_basename}.mp4") if save_video else None
        )

        print(f"[{os.getpid()}] Visualizing -> {file_basename}")

        # Init Visualizer
        ev = env_visualizer_cus.EnvVisualizer(video_plots=save_video)
        ev.agent_name = current_model_name

        # Reset env with saved config

        if "episode_env_init_config" in exp_data:
            config_data = exp_data["episode_env_init_config"]
            ev.env.reset_with_eval_config(config_data)
        else:
            ev.env.reset()

        ev.init_visualize()

        # Play & Save
        ev.play_episode(
            trajectories=curr_scenario_model_name_episode_traj,
            colors=colors,
            save_video_path=video_path,
            save_image_path=image_path,
            episode_log=episode_log
        )

        # Cleanup to prevent memory leaks
        plt.close("all")
        del ev

    except Exception as e:
        print(f"[Error] Failed to process {json_file_path}: {e}")
        # import traceback
        # traceback.print_exc()


def run_visualization_process(target_dir, save_video=False):
    """Run run visualization process."""
    if not os.path.exists(target_dir):
        logger.error(f"Directory not found: {target_dir}")
        return


    search_pattern = os.path.join(target_dir, "*.json")
    all_files = glob.glob(search_pattern)


    target_files = [f for f in all_files if "final_results" not in f and "_ep" in f]

    total_files = len(target_files)
    if total_files == 0:
        logger.warning(f"No trajectory files found in {target_dir}")
        return

    print(f"Found {total_files} trajectory files. Starting parallel processing...")


    tasks = [(f, save_video) for f in target_files]


    num_processes = min(16, max(1, int(cpu_count() * 0.1)))
    print(f"Using {num_processes} CPU cores safely to avoid I/O bottlenecks.")

    start_time = datetime.now()


    # ctx = multiprocessing.get_context('spawn')
    with Pool(processes=num_processes) as pool:
        for i, _ in enumerate(pool.imap_unordered(_worker_wrapper, tasks)):
            if i % 5 == 0:
                print(f"Progress: {i}/{total_files} done...", end='\r')

    end_time = datetime.now()
    print(f"\nVisualization completed! Time taken: {end_time - start_time}")


def has_trajectory_jsons(target_dir):
    """Return True when a directory directly contains episode trajectory JSON files."""
    return bool(glob.glob(os.path.join(target_dir, "*_ep*.json")))


def discover_visualization_targets(path):
    """
    Match run_pipeline_cus.py visualization discovery.

    Accepts a single episode JSON, a group directory with *_ep*.json files,
    or a pipeline experiment root containing group subdirectories.
    """
    if os.path.isfile(path):
        if path.endswith(".json") and "final_results" not in os.path.basename(path):
            return [path]
        return []

    if not os.path.isdir(path):
        return []

    group_dirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item != "Statistics_Plots":
            if has_trajectory_jsons(item_path):
                group_dirs.append(item_path)

    if not group_dirs and has_trajectory_jsons(path):
        group_dirs.append(path)

    return sorted(group_dirs)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to an episode JSON, a group folder, or a pipeline experiment root",
    )
    parser.add_argument("-v", "--video", action="store_true")
    args = parser.parse_args()

    targets = discover_visualization_targets(args.path)

    if not targets:
        logger.warning(
            f"No trajectory JSON files found in {args.path}. "
            "Expected *_ep*.json files directly or inside group subdirectories."
        )
        sys.exit(0)

    print("=" * 60)
    print("Running Visualization")
    print(f"Input Path: {args.path}")
    print(f"Detected {len(targets)} visualization target(s).")
    print("=" * 60)

    for target in targets:
        if os.path.isfile(target):
            print(f"\n--- Rendering Single Episode: {os.path.basename(target)} ---")
            _run_visualization_process(target, save_video=args.video)
        else:
            print(f"\n--- Rendering Visualization for Group: [{os.path.basename(target)}] ---")
            run_visualization_process(target, args.video)

    print("\nVisualization completed successfully.")


if __name__ == "__main__":
    multiprocessing.freeze_support() if hasattr(multiprocessing, "freeze_support") else None
    main()
