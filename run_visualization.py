import argparse
import colorsys
import os
import sys
import json
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
from datetime import datetime
from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, "../")

from visualization import env_visualizer
from utils import logger as logger
from config_manager import ConfigManager


SCALE_CONFIG_FILES = {
    "small": "exp_config_small_scale.yaml",
    "medium": "exp_config_medium_scale.yaml",
    "large": "exp_config_large_scale.yaml",
}

CONFIG_INDEX_TO_SCALE = {
    "0": "small",
    "1": "medium",
    "2": "large",
    "3": "large",
    "-1": "large",
}


def infer_scale_from_path(path):
    """Infer experiment scale from the pipeline output directory name."""
    normalized = os.path.normpath(path).lower()
    parts = normalized.split(os.sep)
    for part in reversed(parts):
        tokens = part.replace("-", "_").split("_")
        for scale in SCALE_CONFIG_FILES:
            if scale in tokens:
                return scale
    return None


def resolve_config_file(target_dir, config=None):
    """Resolve the visualization config file for standalone replay."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(project_root, "config")

    if config:
        config_key = str(config).strip()
        config_key_lower = config_key.lower()
        scale = CONFIG_INDEX_TO_SCALE.get(config_key_lower, config_key_lower)

        if scale in SCALE_CONFIG_FILES:
            return os.path.join(config_dir, SCALE_CONFIG_FILES[scale])

        if config_key_lower.endswith((".yaml", ".yml", ".json")):
            if os.path.isabs(config_key):
                return config_key
            return os.path.join(project_root, config_key)

        raise ValueError(
            f"Unknown config '{config}'. Use small, medium, large, 0, 1, 2/3, or a config file path."
        )

    scale = infer_scale_from_path(target_dir)
    if scale is None:
        raise ValueError(
            "Could not infer scale from experiment path. Please pass --config small|medium|large."
        )

    return os.path.join(config_dir, SCALE_CONFIG_FILES[scale])


def ensure_visualization_config_loaded(config_file):
    """Load the env config needed by MarineEnv before visualizing trajectories."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Visualization config file not found: {config_file}")

    config_manager = ConfigManager()
    loaded = config_manager.get_config()
    if loaded.get("env") and loaded.get("_visualization_config_file") == config_file:
        return

    config_manager.load_config(config_file)
    config_manager.set("_visualization_config_file", config_file)


def generate_distinct_colors(num_colors):
    """Generate high-contrast colors for visualization."""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.0
    for i in range(num_colors):
        h += golden_ratio_conjugate
        h %= 1.0
        # Avoid deep blue background
        if 0.55 < h < 0.75: h += 0.25; h %= 1.0
        s = 0.5 + (i % 2) * 0.5
        v = 0.85 + (i % 3) * 0.05
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return colors

def _worker_wrapper(args):
    """Wrap the worker entry point for multiprocessing execution."""
    return _render_single_episode(*args)

def _render_single_episode(viz_root_dir, model_name, schedule_id, ep_id, trajectory, config_data, save_video, colors, config_file):
    """Render one saved episode into figures and optional video."""
    # Force a non-interactive backend inside worker processes.
    plt.switch_backend('Agg')
    ensure_visualization_config_loaded(config_file)

    try:
        # Keep each model's outputs in its own folder for easier browsing.
        model_viz_dir = os.path.join(viz_root_dir, model_name)
        os.makedirs(model_viz_dir, exist_ok=True)

        file_basename = f"sched_{schedule_id}_ep_{ep_id}"
        image_path = os.path.join(model_viz_dir, f"{file_basename}.pdf")
        video_path = os.path.join(model_viz_dir, f"{file_basename}.mp4") if save_video else None

        # Rebuild the environment from the saved evaluation config before replaying.
        ev = env_visualizer.EnvVisualizer(video_plots=save_video)
        ev.agent_name = model_name
        ev.env.reset_with_eval_config(config_data)
        ev.init_visualize()
        ev.play_episode(
            trajectories=trajectory,
            colors=colors,
            save_video_path=video_path,
            save_image_path=image_path
        )

        # Explicit cleanup helps when many episodes are rendered in parallel.
        plt.close('all')
        del ev

        return True, f"{model_name}/{file_basename}"

    except Exception as e:

        error_msg = f"Failed to process {model_name} sched_{schedule_id}_ep_{ep_id}: {str(e)}"
        return False, error_msg

def run_visualization_process(target_dir, save_video=False, config=None):
    """Replay saved experiment trajectories and render visual outputs.

    Args:
        target_dir: Experiment directory containing saved outputs.
        save_video: Whether to export videos in addition to figures.
        config: Optional visualization config override.
    """
    filename = "exp_results.json"
    file_path = os.path.join(target_dir, filename)

    if not os.path.exists(file_path):
        logger.error(f"Results file not found: {file_path}")
        return

    config_file = resolve_config_file(target_dir, config)
    ensure_visualization_config_loaded(config_file)
    logger.info(f"Visualization config: {config_file}")

    logger.info(f"Loading experiment data from {file_path}...")

    with open(file_path, "r") as f:
        exp_data = json.load(f)

    # The saved experiment file stores trajectories per schedule and per model.
    model_names = exp_data.get("names", ["Unknown"])
    all_trajs = exp_data.get("all_trajectories_exp", [])

    if not all_trajs:
        logger.warning("No trajectory data found in JSON. Skipping visualization.")
        return
    colors = generate_distinct_colors(120)

    # Render into a dedicated folder inside the experiment directory.
    viz_root_dir = os.path.join(target_dir, "Visualizations")
    os.makedirs(viz_root_dir, exist_ok=True)

    tasks = []

    for schedule_id in range(len(all_trajs)):
        for agent_id in range(len(all_trajs[schedule_id])):
            # Keep the saved model order when naming visualization outputs.
            current_model_name = model_names[agent_id] if agent_id < len(model_names) else f"Model_{agent_id}"
            episodes_list = all_trajs[schedule_id][agent_id]

            for ep_id in range(len(episodes_list)):
                trajectory = episodes_list[ep_id]
                config_data = exp_data["all_eval_configs_exp"][schedule_id][agent_id][ep_id]

                # Each task is self-contained so workers do not depend on shared state.
                tasks.append((
                    viz_root_dir,
                    current_model_name,
                    schedule_id,
                    ep_id,
                    trajectory,
                    config_data,
                    save_video,
                    colors,
                    config_file
                ))

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("No valid episodes found to process.")
        return

    # Drop the large JSON payload before the worker pool starts to reduce memory pressure.
    del exp_data
    del all_trajs
    import gc
    gc.collect()

    # Keep the worker count conservative because each process allocates a full renderer.
    num_processes = min(16, max(1, int(cpu_count() * 0.1)))
    print(f"Found {total_tasks} episodes. Starting parallel processing safely using {num_processes} CPU cores...")

    start_time = datetime.now()
    success_count = 0
    fail_count = 0

    # =====================================================================

    # =====================================================================
    if num_processes == 1:
        results_iterator = (_worker_wrapper(task) for task in tasks)

        if HAS_TQDM:
            for success, msg in tqdm(results_iterator, total=total_tasks, desc="Rendering"):
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(msg)
        else:
            for i, (success, msg) in enumerate(results_iterator):
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(msg)

                if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                    print(f"Progress: {i + 1}/{total_tasks} done...", end='\r')
    else:
        with Pool(processes=num_processes) as pool:

            results_iterator = pool.imap_unordered(_worker_wrapper, tasks)

            if HAS_TQDM:

                for success, msg in tqdm(results_iterator, total=total_tasks, desc="Rendering"):
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.error(msg)
            else:

                for i, (success, msg) in enumerate(results_iterator):
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.error(msg)

                    if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                        print(f"Progress: {i + 1}/{total_tasks} done...", end='\r')

    end_time = datetime.now()

    print("\n" + "="*50)
    print(f"Visualization Completed!")
    print(f"Time Taken  : {end_time - start_time}")
    print(f"Successful  : {success_count}")
    print(f"Failed      : {fail_count}")
    print(f"Output Dir  : {viz_root_dir}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to experiment folder")
    parser.add_argument("-v", "--video", action="store_true", help="Flag to save videos as MP4")
    parser.add_argument(
        "-C",
        "--config",
        type=str,
        default=None,
        help=(
            "Visualization config override. Use small, medium, large, 0, 1, 2/3, "
            "or a config file path. If omitted, inferred from the experiment folder name."
        ),
    )
    args = parser.parse_args()

    run_visualization_process(args.path, args.video, config=args.config)
