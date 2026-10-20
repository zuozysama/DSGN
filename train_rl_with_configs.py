import argparse
import itertools
import os
import sys
import traceback
import yaml
import wandb
import time
from datetime import datetime
from pathlib import Path

from config_manager import ConfigManager
from environment.env import MarineEnv
from policy.agent import Agent
from policy.trainer import Trainer
from thirdparty.APF import ApfAgent
from utils import logger as logger
from utils import save_project

sys.path.insert(0, "./thirdparty")

# Argument parser setup
parser = argparse.ArgumentParser(description="Train IQN model")

parser.add_argument(
    "-C", "--config-files", nargs='+', default=[
        "./config/config.yaml",
        "./config/config_dqn.yaml"
    ], help="Configuration files for training parameters"
)
parser.add_argument(
    "-R", "--index-configs", type=int, default=3,
    help="Specific the index of the config file in single process mode"
)
parser.add_argument(
    "-N", "--model-name", nargs='+', default=[
        "DSGN",  # match the config file: config.yaml
        "IQN",  # match the config file: config.yaml
        "TransIQN",  # match the config file: config.yaml
        "DQN",  # match the config file: config_dqn.yaml
        "DSGN-WO-BETA",
        "AVERAGE",
        "RULE_BASED",
        "RANDOM",
        "DSGN-SCALAR",
        "DSGN-SCALAR-WO-BETA"
    ], help="Name of the model"
)
parser.add_argument(
    "-I", "--model-index", type=int, required=True,
    help="Index of the model in the config file"
)
parser.add_argument(
    "-D", "--device", type=str, required=True,
    help="Device to run all subprocesses (e.g., 'cuda:0', 'cpu')"
)
parser.add_argument(
    "--project", type=str, default="DSGN_RL_Training",
    help="wandb project name"
)
parser.add_argument(
    "--group", type=str, default=None,
    help="wandb group name"
)
parser.add_argument(
    "-G", "--gating-type", type=str, choices=["vector", "scalar", "average", "random", "rule_based", "no_gating"],
    required=True, help="The gating type"
)


def initialize_wandb(config, config_file):
    """Initialize wandb with a specific directory."""
    process_id = os.getpid()
    project_root = os.path.dirname(os.path.abspath(__file__))
    # wandb 产物统一保存到数据盘 (优先 DSGN_SAVE_DIR), 避免系统盘被占满
    save_root = os.environ.get("DSGN_SAVE_DIR") or project_root
    wandb_dir = os.path.join(save_root, "wandb_logs")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = wandb_dir

    try:
        # Ensure the directory exists
        os.makedirs(wandb_dir, exist_ok=True)
        logger.info(f"Setting wandb directory to: {wandb_dir}")

        wandb.init(
            project=args.project,
            group=args.group,
            name=f"{args.gating_type}_trial_{os.path.basename(config_file)}_{os.getpid()}" + time.strftime("-%Y%m%d-%H:%M"),
            config=config,
            dir=wandb_dir,  # Explicitly set the directory for `wandb.init`
            tags=[
                os.path.basename(config_file),
                f"pid_{os.getpid()}",
                "IQN" if config.get("use_iqn", False) else "DQN",
                f"seed_{config.get('seed', 'unknown')}"
            ]
        )
        logger.info(f"Process {process_id} - WANDB initialized for config: {config_file}, file saved at: {wandb_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return False


def print_config(config, indent=0):
    """Print configuration dictionary."""
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{'  ' * indent}{key}:")
            print_config(value, indent + 1)
        else:
            logger.info(f"{'  ' * indent}{key}: {value}")


def initialize_config_manager(config_file):
    """Initialize and load configuration manager."""
    config_manager = ConfigManager()
    config_manager.load_config(config_file)
    return config_manager


def create_experiment_directory(params, model_name) -> Path:
    # Return a Path object instead of a string
    """Create the output directory for one training run."""
    process_id = os.getpid()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    params["training_time"] = timestamp
    exp_dir = Path(params["save_dir"]) / "TrainedModels" / model_name / f"training_{timestamp}_seed{params.get('seed', 'unknown')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Process {process_id} - Experiment directory created at: {exp_dir}")
    return exp_dir


def run_trial(device, config_file):
    """Execute a single training trial."""
    process_id = os.getpid()
    start_time = datetime.now()
    logger.info(
        f"\n🚀 Starting Trial\n"
        f"🔧 Process ID   : {process_id}\n"
        f"💻 Device       : {device}\n"
        f"🗂️ Config File : {config_file}\n"
    )

    try:
        # Load configuration
        config_manager = initialize_config_manager(config_file)
        params = config_manager.get_config()
        params["save_dir"] = os.environ.get("DSGN_SAVE_DIR") or os.path.dirname(os.path.abspath(__file__))
        params["gating_type"] = args.gating_type
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        model_name = args.model_name[args.model_index]

        # Print configuration
        # logger.info("==== Training Setup ====")
        # print_config(params)

        # Initialize wandb
        if not initialize_wandb(params, config_file):
            logger.warning("Proceeding without wandb logging")

        # Create experiment directory
        exp_dir = create_experiment_directory(params, model_name)

        # Save configuration
        config_file_path = os.path.join(exp_dir, "trial_config.yaml")
        with open(config_file_path, 'w') as file:
            yaml.safe_dump(params, file)

        if wandb.run:
            wandb.save(config_file_path, base_path=params["save_dir"])

        # Initialize environments
        train_env = MarineEnv(seed=params["seed"], schedule=params["training_schedule"])
        test_env = MarineEnv(seed=66)
        logger.info(f"Process {process_id} - Environments initialized")

        # Initialize agents
        assert (model_name == "DQN") ^ params["use_iqn"], "Model name and use_iqn should match"
        pursuer_agent = Agent(
            device=device, use_iqn=params["use_iqn"], seed=params["seed"] + 100,
            model_name=args.model_name[args.model_index],
            LR=float(config_manager.get("training.learning_rate")),
            gating_type=args.gating_type,
            gating_reg_coeff=float(config_manager.get("training.gating_reg_coeff"))
        )
        logger.info(f"Process {process_id} - Agents initialized with model: {pursuer_agent.model_name}")
        evader_agent = ApfAgent(train_env.evaders[0].a, train_env.evaders[0].w)
        logger.info(f"Process {process_id} - All agents initialized")

        if "load_model" in params:
            pursuer_agent.load_model(path=params["load_model"], device=device)
            logger.info(f"Process {process_id} - Model loaded")

        # Trainer setup

        trainer = Trainer(
            train_env=train_env, test_env=test_env,
            train_schedule=params["training_schedule"],
            eval_schedule=params["eval_schedule"],
            pursuer_agent=pursuer_agent, evader_agent=evader_agent,
            device=device,
            num_workers=8,
            seed=params["seed"]
        )

        trainer.save_eval_config(exp_dir)
        if wandb.run:
            wandb.save(os.path.join(exp_dir, "eval_config.yaml"), base_path=params["save_dir"])

        # Training
        trainer.learn(
            total_timesteps=params["total_timesteps"],
            save_freq=params["save_freq"],
            save_path=exp_dir
        )
        logger.info(f"Process {process_id} - Training finished")

        duration = datetime.now() - start_time
        logger.info(f"Training completed in {duration}")
        if wandb.run:
            wandb.log({"training_duration": duration.total_seconds()})

        return f"Process {process_id} completed successfully."

    except Exception as e:
        logger.error(f"Error in process {process_id}: {e}")
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    save_project.save_source_file()

    try:
        run_trial(args.device, args.config_files[args.index_configs])
    except Exception as e:
        logger.error(f"Error in single process mode: {e}")
