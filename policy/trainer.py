import copy
import json
import os
import time
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
import wandb

from config_manager import ConfigManager
from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent
from utils import logger as logger
from utils import log_gating_weights


from environment.vec_env import SubprocVecEnv, VecEnvCmd


def batch_obs_dict(obs_list, device):
    """Batch a list of observation dictionaries into tensors."""
    if len(obs_list) == 0:
        return None


    keys = obs_list[0].keys()
    batched_dict = {}

    for key in keys:
        val_list = [obs[key] for obs in obs_list]
        stacked_val = np.stack(val_list)


        if key in ["masks", "types"]:

            if key == "types":
                tensor_val = torch.tensor(stacked_val, dtype=torch.long, device=device)
            else:
                tensor_val = torch.tensor(
                    stacked_val, dtype=torch.float32, device=device
                )
        else:
            tensor_val = torch.tensor(stacked_val, dtype=torch.float32, device=device)

        batched_dict[key] = tensor_val

    return batched_dict


def make_env(seed, schedule, rank, enable_logging_worker0=True):
    """Create an environment factory for subprocess execution."""

    def _init():

        """Run  init."""
        is_logging = (rank == 0) and enable_logging_worker0


        env = MarineEnv(seed=seed, schedule=schedule, enable_logging=is_logging)
        return env

    return _init


class Trainer:
    """
    Trainer class for training and evaluating reinforcement learning agents in a specified environment.

    Attributes:
        train_env (MarineEnv): Training environment.
        test_env (MarineEnv): Evaluation environment.
        pursuer_agent (Agent): Pursuer agent.
        evader_agent (ApfAgent): Evader agent.
        eval_config (list): List of evaluation configurations.
        UPDATE_EVERY (int): Number of steps between model updates.
        learning_starts (int): Number of steps before learning begins.
        target_update_interval (int): Interval for updating the target network.
        exploration_fraction (float): Fraction of total steps for exploration decay.
        initial_eps (float): Initial exploration rate.
        final_eps (float): Final exploration rate.
        episode_max_length (int): Maximum length of each episode.
        current_timestep (int): Current timestep.
        learning_timestep (int): Learning timestep counter.
        eval_timesteps (list): List of evaluation timesteps.
        eval_actions (list): List of evaluation actions.
        eval_trajectories (list): List of evaluation trajectories.
        eval_rewards (list): List of evaluation rewards.
        eval_successes (list): List of evaluation successes.
        eval_times (list): List of evaluation times.
        eval_energies (list): List of evaluation energies.
        eval_obs (list): List of evaluation observations.
        eval_pursuers (list): List of evaluation pursuers.
        eval_evaders (list): List of evaluation evaders.
    """

    def __init__(
        self,
        train_env: SubprocVecEnv | MarineEnv,
        test_env: MarineEnv,
        train_schedule: dict,
        eval_schedule: dict,
        pursuer_agent: Agent = None,
        evader_agent: ApfAgent = None,
        episode_max_length: int = 3000,
        update_every: int = 4,
        learning_starts: int = 3000,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.25,
        initial_eps: float = 0.6,
        final_eps: float = 0.05,
        device: str = "cpu",
        num_workers: int = 8,
        seed: int = 9,
    ):
        """
        Initialize Trainer instance.

        Args:
            train_env (MarineEnv): Training environment.
            eval_schedule (dict): Evaluation schedule.
            pursuer_agent (Agent, optional): Pursuer agent. Defaults to None.
            evader_agent (ApfAgent, optional): Evader agent. Defaults to None.
            episode_max_length (int, optional): Maximum length of each episode. Defaults to 3000.
            update_every (int, optional): Number of steps between model updates. Defaults to 4.
            learning_starts (int, optional): Number of steps before learning begins. Defaults to 3000.
            target_update_interval (int, optional): Interval for updating the target network. Defaults to 10000.
            exploration_fraction (float, optional): Fraction of total steps for exploration decay. Defaults to 0.25.
            initial_eps (float, optional): Initial exploration rate. Defaults to 0.6.
            final_eps (float, optional): Final exploration rate. Defaults to 0.05.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device
        self.num_workers = num_workers


        if isinstance(train_env, SubprocVecEnv):
            self.train_env = train_env
        else:

            env_fns = [
                make_env(
                    seed + i * 100,
                    schedule=train_schedule,
                    rank=i,
                    enable_logging_worker0=True,
                )
                for i in range(num_workers)
            ]
            self.train_env = SubprocVecEnv(env_fns)

        self.test_env = test_env

        self.pursuer_agent = pursuer_agent
        self.evader_agent = evader_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)


        self.trainer_config = ConfigManager().get_instance()
        self.UPDATE_EVERY = self.trainer_config.get("training.update_every")
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.episode_max_length = self.trainer_config.get("training.episode_max_length")

        # Current timestep
        self.current_timestep = 0
        self.ep_num_counter = 0

    def create_eval_configs(self, eval_schedule: dict):
        """
        Create evaluation configurations.

        Args:
            eval_schedule (dict): Evaluation schedule.
        """
        self.eval_config.clear()
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                self.test_env.num_pursuers = eval_schedule["num_pursuers"][i]
                self.test_env.num_evaders = eval_schedule["num_evaders"][i]
                self.test_env.num_cores = eval_schedule["num_cores"][i]
                self.test_env.num_obs = eval_schedule["num_obstacles"][i]
                self.test_env.min_pursuer_evader_init_dis = eval_schedule[
                    "min_pursuer_evader_init_dis"
                ][i]
                self.test_env.reset()
                self.eval_config.append(self.test_env.episode_data())

    def save_eval_config(self, directory: Path):
        """
        Save evaluation configurations.

        Args:
            directory (Path): Directory to save configurations.
        """
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def save_latest_model(self, path):
        """Save the latest model snapshot."""
        if self.pursuer_agent:
            self.pursuer_agent.save_latest_model(path)

    def linear_eps(self, total_timesteps: int) -> float:
        """Compute a linearly decayed epsilon value."""
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def learn(
        self,
        total_timesteps: int,
        save_freq: int,
        save_path: Path,
        verbose: bool = True,
    ):
        """Run the main training loop.

        Args:
            total_timesteps: Number of environment steps to train for.
            save_dir: Directory used for checkpoints and logs.

        Returns:
            None.
        """
        logger.info(
            f"Main Process {os.getpid()} | 🚀 Starting Multi-Process Training with {self.num_workers} workers."
        )


        init_results = self.train_env.reset(
            options={"time_steps": self.current_timestep}
        )


        current_p_obs_list = [res[0] for res in init_results]
        current_e_obs_list = [res[1] for res in init_results]


        num_pursuers = len(current_p_obs_list[0])


        worker_ep_rewards = np.zeros((self.num_workers, num_pursuers))
        worker_ep_discounted_rewards = np.zeros((self.num_workers, num_pursuers))
        worker_ep_gamma_pows = np.ones((self.num_workers, num_pursuers))
        worker_ep_lengths = np.zeros(self.num_workers, dtype=int)


        worker_ep_deactivated_t = [[-1] * num_pursuers for _ in range(self.num_workers)]

        start_time = time.time()

        while self.current_timestep < total_timesteps:
            eps = self.linear_eps(total_timesteps)

            # ==========================================================
            # PART 1: Pursuer Batch Inference (GPU, Batched)
            # ==========================================================


            flat_pursuer_states = [
                s for worker_states in current_p_obs_list for s in worker_states
            ]


            valid_indices = []
            valid_states = []
            for idx, s in enumerate(flat_pursuer_states):
                if s is not None:
                    valid_indices.append(idx)
                    valid_states.append(s)


            flat_pursuer_actions = [None] * len(flat_pursuer_states)


            if len(valid_states) > 0:
                batch_input = batch_obs_dict(valid_states, self.device)

                with torch.no_grad():
                    if self.pursuer_agent.use_iqn:
                        batch_acts, _, _ = self.pursuer_agent.act(batch_input, eps)
                    else:
                        batch_acts, _ = self.pursuer_agent.act_dqn(batch_input, eps)


                for i, original_idx in enumerate(valid_indices):
                    flat_pursuer_actions[original_idx] = batch_acts[i]


            worker_pursuer_actions = []
            ptr = 0
            for w in range(self.num_workers):
                n_p = len(current_p_obs_list[w])
                worker_pursuer_actions.append(flat_pursuer_actions[ptr : ptr + n_p])
                ptr += n_p

            # ==========================================================
            # PART 2: Evader Inference (CPU, Loop)
            # ==========================================================
            worker_evader_actions = []
            for w in range(self.num_workers):
                current_e_obs = current_e_obs_list[w]
                actions = []
                for e_obs in current_e_obs:
                    if e_obs is None:
                        actions.append(None)
                    else:
                        act = self.evader_agent.act(e_obs)
                        actions.append(act)
                worker_evader_actions.append(actions)

            # ==========================================================
            # PART 3: Environment Step (Async)
            # ==========================================================
            actions_to_send = list(zip(worker_pursuer_actions, worker_evader_actions))
            self.train_env.step_async(actions_to_send)


            step_results = self.train_env.step_wait()

            # ==========================================================
            # PART 4: Buffer Store & Manual Reset Check (Robust Version)
            # ==========================================================

            next_input_p_obs_list = []
            next_input_e_obs_list = []
            resets_to_trigger = []

            for w in range(self.num_workers):
                (n_p_obs, n_e_obs), reward, done_flags, info_list = step_results[w]

                # ------------------------------------------------------

                # ------------------------------------------------------

                current_n_agents = len(reward)

                max_n_agents = worker_ep_rewards.shape[1]


                if current_n_agents > max_n_agents:
                    diff = current_n_agents - max_n_agents
                    logger.info(f"Main Process {os.getpid()} | 📈 Curriculum Level Up! Expanding stats matrix width: {max_n_agents} -> {current_n_agents}")


                    worker_ep_rewards = np.pad(worker_ep_rewards, ((0,0), (0, diff)), 'constant', constant_values=0)
                    worker_ep_discounted_rewards = np.pad(worker_ep_discounted_rewards, ((0,0), (0, diff)), 'constant', constant_values=0)
                    worker_ep_gamma_pows = np.pad(worker_ep_gamma_pows, ((0,0), (0, diff)), 'constant', constant_values=1)


                    for i in range(self.num_workers):
                        worker_ep_deactivated_t[i].extend([-1] * diff)


                    max_n_agents = current_n_agents

                # ------------------------------------------------------

                # ------------------------------------------------------
                worker_ep_lengths[w] += 1


                worker_ep_rewards[w, :current_n_agents] += reward
                worker_ep_discounted_rewards[w, :current_n_agents] += reward * worker_ep_gamma_pows[w, :current_n_agents]
                worker_ep_gamma_pows[w, :current_n_agents] *= self.pursuer_agent.GAMMA


                for i in range(current_n_agents):
                    if done_flags[i] and worker_ep_deactivated_t[w][i] == -1:
                        worker_ep_deactivated_t[w][i] = worker_ep_lengths[w]

                # ------------------------------------------------------

                # ------------------------------------------------------
                if info_list and isinstance(info_list[0], dict) and "capture_events" in info_list[0]:
                    events = info_list[0]["capture_events"]
                    if events:
                        for evt in events:
                            metrics = {
                                "global_step": self.current_timestep,
                                f"worker_{w}/capture_event/leader_id": evt["leader_id"],
                                f"worker_{w}/capture_event/evader_id": evt["evader_id"],
                                f"worker_{w}/capture_event/team_size": evt["team_size"],
                                f"worker_{w}/capture_event/max_angle_gap": evt["max_angle_gap"],
                                "global/capture_event/max_angle_gap": evt["max_angle_gap"],
                                "global/capture_event/team_size": evt["team_size"],
                                "global/capture_event/worker_id": w
                            }
                            wandb.log(metrics, commit=True)
                            if verbose and w == 0:
                                logger.info(f"Main Process {os.getpid()} | Worker {w} Capture 🎯! Gap: {evt['max_angle_gap']:.2f}")

                # ------------------------------------------------------

                # ------------------------------------------------------
                curr_p_obs = current_p_obs_list[w]
                curr_act = worker_pursuer_actions[w]


                for i, p_obs in enumerate(curr_p_obs):
                    if p_obs is None: continue
                    self.pursuer_agent.memory.add(
                        p_obs,
                        curr_act[i],
                        reward[i],
                        n_p_obs[i],
                        done_flags[i]
                    )

                # ------------------------------------------------------

                # ------------------------------------------------------
                env_info = info_list[0] if info_list else {}
                active_count = env_info.get("active_pursuers_count", 3)
                all_captured = env_info.get("all_captured", False)

                cond_timeout = worker_ep_lengths[w] >= self.episode_max_length
                should_reset = cond_timeout or all_captured or (active_count < 3)

                if should_reset:
                    # Log Summary
                    self.ep_num_counter += 1
                    if verbose and w == 0:
                        outcome_str = "Timeout | Max Length Reached"
                        if all_captured: outcome_str = "VICTORY | All Captured"
                        elif active_count < 3: outcome_str = f"DEFEAT | Too Few Pursuers ({active_count})"
                        elif any([i.get("collision") for i in info_list]): outcome_str = "CRASH"


                        self._print_worker0_summary(
                            ep_len=worker_ep_lengths[w],
                            ep_reward=worker_ep_rewards[w, :current_n_agents],
                            ep_discounted_reward=worker_ep_discounted_rewards[w, :current_n_agents],
                            info=info_list[0],
                            outcome=outcome_str
                        )

                    self.log_training_episode(
                        ep_length=worker_ep_lengths[w],
                        ep_num=self.ep_num_counter,
                        eps=eps,
                        ep_rewards=worker_ep_rewards[w, :current_n_agents],
                        ep_discounted_rewards=worker_ep_discounted_rewards[w, :current_n_agents],
                        infos=info_list,
                        ep_deactivated_t=worker_ep_deactivated_t[w]
                    )

                    resets_to_trigger.append(w)


                    worker_ep_lengths[w] = 0


                    worker_ep_rewards[w] = 0.0
                    worker_ep_discounted_rewards[w] = 0.0
                    worker_ep_gamma_pows[w] = 1.0


                    worker_ep_deactivated_t[w] = [-1] * max_n_agents

                    next_input_p_obs_list.append(None)
                    next_input_e_obs_list.append(None)

                else:
                    cleaned_next_obs = []
                    for k, obs in enumerate(n_p_obs):
                        if done_flags[k]: cleaned_next_obs.append(None)
                        else: cleaned_next_obs.append(obs)
                    next_input_p_obs_list.append(cleaned_next_obs)
                    next_input_e_obs_list.append(n_e_obs)

            # ==========================================================
            # PART 5: Execute Delayed Reset
            # ==========================================================
            if resets_to_trigger:
                reset_options = {"time_steps": self.current_timestep}
                reset_results = self.train_env.reset_indices(
                    resets_to_trigger, options=reset_options
                )

                for idx, w in enumerate(resets_to_trigger):
                    init_p, init_e = reset_results[idx]
                    next_input_p_obs_list[w] = init_p
                    next_input_e_obs_list[w] = init_e


            current_p_obs_list = next_input_p_obs_list
            current_e_obs_list = next_input_e_obs_list


            self.current_timestep += self.num_workers

            # ==========================================================
            # PART 6: Training Update
            # ==========================================================
            if self.current_timestep >= self.learning_starts:
                if self.pursuer_agent.memory.size > self.pursuer_agent.BATCH_SIZE:
                    samples_collected = self.num_workers
                    updates_needed = int(samples_collected / self.UPDATE_EVERY)
                    if updates_needed < 1:
                        updates_needed = 1

                    loss = 0.0
                    for _ in range(updates_needed):
                        loss = self.pursuer_agent.train(global_step=self.current_timestep)

                    # Check freq using gap logic
                    prev_timestep = self.current_timestep - self.num_workers
                    if (self.current_timestep // 200) > (prev_timestep // 200):
                        wandb.log({"train/loss": loss}, commit=False)

                # Target Update
                prev_timestep = self.current_timestep - self.num_workers
                if (self.current_timestep // self.target_update_interval) > (
                    prev_timestep // self.target_update_interval
                ):
                    self.pursuer_agent.soft_update()

                # Gating Weights Log
                if self.current_timestep % 1000 == 0:
                    if hasattr(self.pursuer_agent.policy_local, "attention_weights"):
                        log_gating_weights(
                            self.pursuer_agent.policy_local.attention_weights.fused,
                            self.current_timestep,
                        )

                # Save Model
                if (self.current_timestep // save_freq) > (prev_timestep // save_freq):
                    for agent in [self.pursuer_agent]:
                        if agent is None or not agent.training:
                            continue
                        agent.save_latest_model(save_path, filename="latest_model.pth")
                        checkpoint_path = save_path / f"checkpoint_{self.current_timestep}"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        agent.save_latest_model(checkpoint_path)
                    if verbose:
                        logger.info(f"Main Process {os.getpid()} | 💾 Model Checkpoint saved at step {self.current_timestep}")

            # Log FPS
            if self.current_timestep % 5000 == 0:
                fps = self.current_timestep / (time.time() - start_time)
                logger.info(f"Main Process {os.getpid()} | Timestep: {self.current_timestep} | FPS: {fps:.2f}")
                wandb.log({"system/fps": fps}, commit=True)

        logger.info("Main Process {os.getpid()} | Training Completed.")
        self.train_env.close()
        wandb.finish()

    def log_training_episode(
        self,
        ep_length,
        ep_num,
        eps,
        ep_rewards,
        ep_discounted_rewards,
        infos,
        ep_deactivated_t,
    ):
        """Log aggregate statistics for one training episode."""

        metrics = {
            "episode/length": ep_length,
            "episode/number": ep_num,
            "episode/exploration_rate": eps,
            "episode/mean_reward": np.mean(ep_rewards),
            "episode/mean_discounted_reward": np.mean(ep_discounted_rewards),
            "episode/min_reward": np.min(ep_rewards),
            "episode/max_reward": np.max(ep_rewards),
            "episode/total_timesteps": self.current_timestep,
        }


        num_active_pursuers = 0


        actual_agent_count = len(infos)

        for i in range(actual_agent_count):
            reward = ep_rewards[i]
            agent_info = infos[i]


            is_deactivated = agent_info.get("deactivated", False)
            is_collision = agent_info.get("collision", False)
            pos_x = agent_info.get("x", 0.0)
            pos_y = agent_info.get("y", 0.0)
            state_str = agent_info.get("state", "unknown")
            first_contact = agent_info.get("first_contact_step", None)

            if not is_deactivated:
                num_active_pursuers += 1


            death_time = ep_deactivated_t[i] if ep_deactivated_t[i] != -1 else ep_length


            agent_metrics = {
                f"pursuer_{i}/state": state_str,
                f"pursuer_{i}/reward": reward,
                f"pursuer_{i}/discounted_reward": ep_discounted_rewards[i],
                f"pursuer_{i}/survival_steps": death_time,
                f"pursuer_{i}/deactivated": is_deactivated,
                f"pursuer_{i}/collision": is_collision,
                f"pursuer_{i}/position_x": pos_x,
                f"pursuer_{i}/position_y": pos_y,
            }


            if first_contact is not None:
                agent_metrics[f"pursuer_{i}/first_contact_step"] = first_contact

            metrics.update(agent_metrics)


        all_captured = infos[0].get("all_captured", False)

        metrics.update({
            "episode_summary/active_pursuers": num_active_pursuers,
            "episode_summary/all_evaders_captured": all_captured,
            "episode_summary/global_timesteps": self.current_timestep,
        })


        wandb.log(metrics, commit=True)

    def _print_worker0_summary(self, ep_len, ep_reward, ep_discounted_reward, info, outcome):
        """Print a concise training summary from worker zero."""
        logger.info(f"Main Process {os.getpid()} | {'=' * 20} Worker 0 Episode Summary {'=' * 20}")
        logger.info(f"Main Process {os.getpid()} | ⏱️  Steps      : {ep_len}")
        logger.info(
            f"Main Process {os.getpid()} | 💰 Total Reward : {np.mean(ep_reward):.2f}"
        )
        logger.info(f"Main Process {os.getpid()} | Total Discounted Reward : {np.mean(ep_discounted_reward):.2f}")
        logger.info(f"Main Process {os.getpid()} | 🏁 Outcome      : {outcome}")


        # if 'reward_components' in info:


        logger.info(f"Main Process {os.getpid()} | {'=' * 60}\n")

    def evaluation(self):
        """
        Evaluate the agent's performance and record the time for each operation.
        """
        # Initialize data storage
        # actions_data = []
        # trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        # obs_data = []
        # pursuers_data = []
        # evaders_data = []

        for idx, config in enumerate(self.eval_config):

            logger.info(f"Process {os.getpid()} - Evaluating episode {idx}")

            # Record environment reset time
            state, _ = self.test_env.reset_with_eval_config(config)
            evader_states, _ = self.test_env.get_evaders_observation()

            # obs = [[copy.deepcopy(rob.perception.observed_obstacles)] for rob in self.test_env.pursuers]
            # pursuers = [[copy.deepcopy(rob.perception.observed_pursuers)] for rob in self.test_env.pursuers]
            # evaders = [[copy.deepcopy(rob.perception.observed_evaders)] for rob in self.test_env.pursuers]

            pursuer_num = len(self.test_env.pursuers)

            rewards = [0.0] * pursuer_num
            times = [0.0] * pursuer_num
            energies = [0.0] * pursuer_num
            end_episode = False
            length = 0

            while not end_episode:

                action = []
                for i, rob in enumerate(self.test_env.pursuers):
                    if rob.deactivated:
                        action.append(None)
                        continue

                    if self.pursuer_agent.use_iqn:
                        a, _, _ = self.pursuer_agent.act(state[i])
                    else:
                        a, _ = self.pursuer_agent.act_dqn(state[i])

                    action.append(a)

                # Record evader action selection time
                evader_action = []
                for j, evader in enumerate(self.test_env.evaders):
                    if evader.deactivated:
                        evader_action.append(None)
                        continue

                    a = self.evader_agent.act(evader_states[j])
                    evader_action.append(a)

                # Execute actions
                state, reward, done, info = self.test_env.step((action, evader_action))
                evader_states, _ = self.test_env.get_evaders_observation()

                for i, rob in enumerate(self.test_env.pursuers):
                    if rob.deactivated:
                        continue

                    rewards[i] += self.pursuer_agent.GAMMA**length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_action_energy_cost(action[i])
                    # obs[i].append(copy.deepcopy(rob.perception.observed_obstacles))
                    # pursuers[i].append(copy.deepcopy(rob.perception.observed_pursuers))
                    # evaders[i].append(copy.deepcopy(rob.perception.observed_evaders))

                    if rob.collision:
                        rob.deactivated = True

                end_episode = (
                    (length >= self.episode_max_length / 2)
                    or len(
                        [
                            puruer
                            for puruer in self.test_env.pursuers
                            if not puruer.deactivated
                        ]
                    )
                    < 3
                    or self.test_env.check_all_evader_is_captured()
                )

                if end_episode:
                    logger.info(
                        f"Episode ended at length {length}, "
                        f"len(pursuers) = {len([puruer for puruer in self.test_env.pursuers if not puruer.deactivated])}, "
                        f"check_all_evader_is_captured = {self.test_env.check_all_evader_is_captured()}"
                    )
                length += 1

            actions = []
            trajectories = []
            for rob in self.test_env.pursuers + self.test_env.evaders:
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))

            success = True if self.test_env.check_all_evader_is_captured() else False

            # actions_data.append(actions)
            # trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            # obs_data.append(obs)
            # pursuers_data.append(pursuers)
            # evaders_data.append(evaders)

        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data) / len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])

        logger.info(f"Process {os.getpid()} ++++++++ Evaluation Info ++++++++")
        logger.info(f"Process {os.getpid()} - Avg cumulative reward: {avg_r:.2f}")
        logger.info(f"Process {os.getpid()} - Success rate: {success_rate:.2f}")
        if avg_t is not None:
            logger.info(f"Process {os.getpid()} - Avg time: {avg_t:.2f}")
            logger.info(f"Process {os.getpid()} - Avg energy: {avg_e:.2f}")
        logger.info("++++++++ Evaluation Info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        # self.eval_actions.append(actions_data)
        # self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        # self.eval_obs.append(obs_data)
        # self.eval_pursuers.append(pursuers_data)
        # self.eval_evaders.append(evaders_data)
        eval_metrics = {
            "evaluation/avg_reward": avg_r,
            "evaluation/success_rate": success_rate,
            "evaluation/episode_count": len(self.eval_config),
            "evaluation/timestep": self.current_timestep,
        }

        if avg_t is not None:
            eval_metrics.update(
                {
                    "evaluation/avg_time": avg_t,
                    "evaluation/avg_energy": avg_e,
                }
            )

        wandb.log(eval_metrics)

    def save_evaluation(self, eval_log_path: Path):
        """
        Save evaluation results.

        Args:
            eval_log_path (Path): Path for evaluation logs.
        """
        filename = "evaluations.npz"

        # Convert irregular arrays to object arrays
        eval_data = {
            "timesteps": np.array(self.eval_timesteps, dtype=object),
            "actions": np.array(self.eval_actions, dtype=object),
            # "trajectories": np.array(self.eval_trajectories, dtype=object),
            "rewards": np.array(self.eval_rewards, dtype=object),
            "successes": np.array(self.eval_successes),
            "times": np.array(self.eval_times),
            "energies": np.array(self.eval_energies),
        }

        os.makedirs(eval_log_path, exist_ok=True)
        np.savez(os.path.join(eval_log_path, filename), **eval_data)
