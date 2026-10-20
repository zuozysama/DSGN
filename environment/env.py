import copy
import json
import os
from collections import defaultdict
from typing import Optional, Any, List, Tuple, Dict, Union

import gym
import numpy as np
from numpy import ndarray, dtype, floating, number, bool_
from scipy.spatial import KDTree
import wandb

import utils
from config_manager import ConfigManager
from robots.evader import Evader
from robots.pursuer import Pursuer
from utils import generate_centers
from utils import logger as logger
from .patrol_point import PatrolPoint


class Core:
    """
    Initialize a vortex core object.

    Attributes:
        x (np.ndarray): X-coordinates of vortex core.
        y (np.ndarray): Y-coordinates of vortex core.
        clockwise (bool): Vortex rotation direction (True for clockwise).
        Gamma (float): Circulation strength of vortex core.
    """

    def __init__(self, x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]],
                 clockwise: bool, gamma: float):
        """
        Args:
            x (np.ndarray): X-coordinates of vortex core.
            y (np.ndarray): Y-coordinates of vortex core.
            clockwise (bool): Vortex rotation direction.
            gamma (float): Circulation strength of vortex core.
        """
        self.x = x  # x coordinate of the vortex core
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise  # if the vortex direction is clockwise
        self.Gamma = gamma  # circulation strength of the vortex core


class Obstacle:
    """
    Initialize an obstacle object.

    Attributes:
        x (np.ndarray): X-coordinates of obstacle center.
        y (np.ndarray): Y-coordinates of obstacle center.
        r (float): Radius of obstacle.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, r: float):
        """
        Args:
            x (np.ndarray): X-coordinates of obstacle center.
            y (np.ndarray): Y-coordinates of obstacle center.
            r (float): Radius of obstacle.
        """
        self.x = x  # x coordinate of obstacle center
        self.y = y  # y coordinate of obstacle center
        self.r = r  # obstacle radius


class MarineEnv(gym.Env):
    r"""
    Marine environment class for reinforcement learning tasks.

    Simulates a marine environment with dynamic/static obstacles including vortex cores and static obstacles.
    Supports curriculum learning with environmental changes across multiple episodes.

    Attributes:
        sd (int): Random seed.
        rd (np.random.RandomState): Random number generator.
        width (int): Map x-axis dimension.
        height (int): Map y-axis dimension.
        vortex_core_radius (float): Vortex core radius.
        v_rel_max (float): Max relative velocity between opposing currents.
        p (float): Max allowed relative velocity between vortex cores.
        v_range (list): Velocity range at vortex edge.
        obs_r_range (list): Obstacle radius range.
        clear_r (float): Safe radius around robot start/goal positions.
        min_pursuer_evader_init_dis (float): Min start distance between pursuers and evaders.
        min_evader_evader_init_dis (float): Min start distance between evaders.
        timestep_penalty (float): Penalty per timestep.
        distance_reward (float): Distance reward value.
        collision_penalty (float): Collision penalty value.
        goal_reward (float): Goal achievement reward.
        num_cores (int): Number of vortex cores.
        num_obs (int): Number of obstacles.
        num_pursuers (int): Number of pursuers.
        num_evaders (int): Number of evaders.
        pursuers (List[Pursuer]): List of pursuers.
        evaders (List[Evader]): List of evaders.
        cores (list): List of vortex cores.
        core_centers (None): Vortex core positions.
        obstacles (list): List of obstacles.
        schedule (dict): Curriculum learning schedule.
        episode_time_steps (int): Current episode timesteps.
        episode_max_length (int): Max episode length.
        total_time_steps (int): Total training timesteps.
        observation_in_robot_frame (bool): Return observations in robot frame.
    """

    def __init__(self, seed: int = 0, schedule: dict = None, enable_logging=False):
        """
        Initialize marine environment object.

        Args:
            seed (int): Random seed. Default 0.
            schedule (dict): Curriculum learning schedule. Default None.
        """

        self.sd = seed
        self.rd = np.random.RandomState(seed)  # PRNG
        self.enable_logging = enable_logging

        # Parameter initialization
        # Map parameters
        self.config_env = ConfigManager().get_instance()
        self.width = self.config_env.get("env.width")  # x-axis dimension
        self.height = self.config_env.get("env.height")  # y-axis dimension
        self.arena_size = self.width

        # Vortex parameters
        self.vortex_core_radius = 0.5  # vortex core radius
        self.v_rel_max = 1.0  # max relative velocity between opposing currents
        self.p = 0.8  # max allowed relative velocity between vortex cores
        self.v_range = [5, 10]  # velocity range at vortex edge

        # Obstacle parameters
        self.obs_r_range = self.config_env.get("env.obs_r_range")  # obstacle radius range

        # Robot generation parameters
        self.clear_r = self.config_env.get("env.clear_r")  # safe zone around start positions
        self.min_pursuer_evader_init_dis = 15.0  # min pursuer-evader start distance
        self.min_evader_evader_init_dis = 10.0  # min evader-evader start distance

        # Reward/penalty parameters
        self.goal_reward = self.config_env.get("env.goal_reward")  # Encirclement Reward
        self.distance_reward = self.config_env.get("env.distance_reward")  # Pursuit Guidance Base
        self.timestep_penalty = self.config_env.get("env.timestep_penalty")

        # Penalties
        self.emergency_penalty = self.config_env.get("env.emergency_penalty")
        self.collision_penalty = self.config_env.get("env.collision_penalty")
        self.boundary_penalty = self.config_env.get("env.boundary_penalty")
        self.decay_factor = self.config_env.get("env.decay_factor")

        # Patrol rewards
        self.arrival_reward_scale = self.config_env.get("env.patrol_arrival_scale")
        self.shaping_reward_scale = self.config_env.get("env.patrol_shaping_scale")
        self.full_coverage_reward = self.config_env.get("env.patrol_full_coverage_reward")
        self.norm_factor = self.config_env.get("env.patrol_normalization_factor")

        # Distance thresholds
        self.capture_distance = self.config_env.get("env.capture_distance")
        self.related_distance = self.config_env.get("env.related_distance")
        self.safe_distance = 0.5 * self.capture_distance

        # Entity counts (Will be overwritten by schedule)
        self.num_cores = 0
        self.num_obs = 8
        self.num_pursuers = 3
        self.num_evaders = 0

        # Entity lists
        self.pursuers: List[Pursuer] = []
        self.evaders: List[Evader] = []
        self.pursuers.append(Pursuer(0))
        self.evaders.append(Evader(0))
        self.cores: List[Core] = []
        self.obstacles: List[Obstacle] = []
        self.core_centers = None

        # Curriculum learning
        self.schedule = schedule
        self.episode_time_steps = 0
        self.episode_max_length = self.config_env.get("training.episode_max_length")
        self.total_time_steps = 0

        # Observation settings
        self.observation_in_robot_frame = True
        self.last_step_observed_evader_ids = []
        self.last_pursuer_positions = {}

        # Cache visibility and motion state so rewards can compare against the previous step.
        self.last_step_observed_evader_ids = []
        self.last_pursuer_positions = {}

        # Precompute patrol anchors once and rebuild them on every reset.
        patrol_coords = generate_centers(self.width, self.height, fov_radius=20.0)

        self.patrol_points: list[PatrolPoint] = [
            PatrolPoint(
                (x, y), idx, threshold=2.0, normalization_factor=self.norm_factor
            )
            for idx, (x, y) in enumerate(patrol_coords)
        ]
        self.visited_set: set[int] = set()

        # Episode accumulators are reset on every environment reset.
        self.ep_reward_components_per_agent = []
        self.ep_collisions_count_per_agent = []


        self.first_contact_step = None

    def get_action_space_dimension(self) -> int:
        """
        Get pursuer action space dimension.

        Returns:
            int: Action space dimension.
        """
        return self.pursuers[0].compute_actions_dimension()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment.

        Resets environment according to curriculum schedule, including regenerating:
        pursuers, evaders, vortex cores, and obstacles.

        Args:
            seed (Optional[int]): Random seed. Default None.
            options (Optional[dict]): Reset options. Default None.
                - time_steps (int): Global training timestep from Trainer (for curriculum sync).

        Returns:
            list: Pursuer observations.
        """
        # The trainer passes the global step here so curriculum phases stay synchronized.
        if options and "time_steps" in options:


            self.total_time_steps = options["time_steps"]
            # logger.debug(f"Process {os.getpid()} synced timestep to {self.total_time_steps}")
        else: logger.info(f"Process {os.getpid()}, Global timestep is not synced by trainer!")

        # Select the active curriculum phase before rebuilding entities.
        current_evader_speed_ratio = 1.0  # Default full speed
        if self.schedule is not None:

            steps = self.schedule["time_steps"]
            diffs = np.array(steps) - self.total_time_steps


            idx = len(diffs[diffs <= 0]) - 1


            if idx < 0: idx = 0

            self.num_pursuers = self.schedule["num_pursuers"][idx]
            self.num_evaders = self.schedule["num_evaders"][idx]
            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_pursuer_evader_init_dis = self.schedule["min_pursuer_evader_init_dis"][idx]


            if "evader_speed_ratios" in self.schedule:
                current_evader_speed_ratio = self.schedule["evader_speed_ratios"][idx]


            if self.enable_logging:
                logger.info(f"Process {os.getpid()} | Global Step {self.total_time_steps} | Phase {idx} | "
                            f"P: {self.num_pursuers}, E: {self.num_evaders}, Speed: {current_evader_speed_ratio}")

        # Clear all runtime state before repopulating the map.
        self.episode_time_steps = 0
        self.first_contact_step = None

        self.cores.clear()
        self.obstacles.clear()
        self.pursuers.clear()
        self.evaders.clear()

        # Rebuild patrol points first so later obstacle placement can avoid blocking them.
        patrol_coords = generate_centers(self.width, self.height, fov_radius=20.0)

        self.patrol_points: list[PatrolPoint] = [
            PatrolPoint(
                (x, y), idx, threshold=2.0, normalization_factor=self.norm_factor
            )
            for idx, (x, y) in enumerate(patrol_coords)
        ]
        self.visited_set = set()
        self.last_pursuer_positions = {}

        # Sample evaders before pursuers so spacing checks can enforce the initial separation.
        num_evaders = 0
        iteration = 50000
        while num_evaders < self.num_evaders:
            evader_start = self.rd.uniform(low=15.0 * np.ones(2),
                                         high=np.array([self.width - 15.0, self.height - 15.0]))
            iteration -= 1
            if self.check_evader_start_position(evader_start=evader_start):
                evader = Evader(index=num_evaders)
                evader.start = evader_start

                evader.max_speed = evader.max_speed * current_evader_speed_ratio

                self.reset_robot(evader)
                self.evaders.append(evader)
                num_evaders += 1
            if iteration == 0:
                raise RuntimeError("Failed to generate Evaders")

        # Pursuers are generated after evaders so they can respect the scenario spacing rules.
        num_pursuers = 0
        iteration = 8000
        while num_pursuers < self.num_pursuers:
            pursuer_start = self.rd.uniform(low=2.0 * np.ones(2), high=np.array([self.width - 2.0, self.height - 2.0]))
            iteration -= 1
            if self.check_pursuer_start_position(pursuer_start):
                pursuer = Pursuer(index=num_pursuers, enable_logging=self.enable_logging)
                pursuer.start = pursuer_start
                self.reset_robot(pursuer)
                self.pursuers.append(pursuer)
                # Init last position
                self.last_pursuer_positions[num_pursuers] = np.array([pursuer.x, pursuer.y])
                num_pursuers += 1
            if iteration == 0:
                raise RuntimeError("Failed to generate Pursuers")

        # Generate flow-field cores first, then fill the remaining free space with obstacles.
        num_cores = self.num_cores
        if num_cores > 0:
            iteration = 8000
            while True:
                center = self.rd.uniform(low=np.zeros(2), high=np.array([self.width, self.height]))
                direction = self.rd.binomial(1, 0.5)
                v_edge = self.rd.uniform(low=self.v_range[0] * 0.5, high=self.v_range[1] * 0.5)
                gamma = 2 * np.pi * (self.vortex_core_radius * 0.5) * v_edge
                core = Core(center[0], center[1], bool(direction), gamma)

                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1

                if iteration == 0 or num_cores == 0:
                    if len(self.cores) == self.num_cores:
                        break
                    else:
                        logger.error(f"Process {os.getpid()} | Failed to generate Cores!")
                        raise RuntimeError("Failed to generate Cores")

        centers = None
        for core in self.cores:
            c = np.array([[core.x, core.y]])
            centers = c if centers is None else np.vstack((centers, c))
        if centers is not None:
            self.core_centers = KDTree(centers)

        # Obstacle generation also preserves patrol coverage by avoiding patrol anchors.
        num_obs = self.num_obs
        if num_obs > 0:
            iteration = 10000
            while True:
                center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
                r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])

                if self._is_obstacle_too_close_to_patrol(center[0], center[1], r):
                    iteration -= 1
                    continue

                obs = Obstacle(center[0], center[1], r)
                iteration -= 1


                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1

                if iteration == 0 or num_obs == 0:
                    if len(self.obstacles) == self.num_obs:
                        break
                    else:
                        logger.error(f"Process {os.getpid()} | Failed to generate Obstacles (Patrol Conflict?)!")
                        raise RuntimeError("Failed to generate Obstacles")

        # Build the first observation with team-level visibility already aggregated.
        raw_obs, collisions = self.get_pursuers_observations()


        global_seen_evaders = set()
        for p in self.pursuers:
            global_seen_evaders.update(p.perception.observed_evaders)


        final_obs = self.update_pursuers_observation_with_aggregated_evaders_ids(
            raw_obs, list(global_seen_evaders)
        )


        has_target = (len(global_seen_evaders) > 0)

        for obs in final_obs:
            if obs is not None:

                obs['has_active_target'] = np.array([1.0 if has_target else 0.0], dtype=np.float32)

        # Reset reward bookkeeping after the initial observation has been prepared.
        self.ep_reward_components_per_agent = [defaultdict(float) for _ in range(len(self.pursuers))]
        self.ep_collisions_count_per_agent = np.zeros(len(self.pursuers))


        return final_obs, collisions, False

    def step(self, actions: Tuple[List[int], List[int]]) -> tuple:
        """
        Execute one environment simulation step and update the environment.
        Flow: Update Physics -> Shared Perception -> Calc Rewards -> Termination

        Updates environment state based on pursuer and evader actions, calculates rewards,
        termination status, and additional information.

        Args:
            actions (Tuple[List[int], List[int]]): Tuple of pursuer and evader actions.

        Returns:
            Tuple:
                List[Any]: Pursuers' observations list
                List[float]: Reward values list
                List[bool]: Episode termination flags list
                List[dict]: Additional information dictionaries list
        """
        rewards = np.zeros(len(self.pursuers))
        pursuer_actions, evader_actions = actions
        pursuer_actions = utils.to_list(pursuer_actions)
        evader_actions = utils.to_list(evader_actions)

        # -----------------------------------------------------
        # 1. Physics Update (Execute Actions)
        # -----------------------------------------------------
        self.update_pursuers_states(pursuer_actions)
        self.update_evader_state(evader_actions)

        # Merge individual detections into a shared target list before computing rewards.
        raw_observations, collisions = self.get_pursuers_observations()


        global_seen_evader_ids = set()
        for pursuer in self.pursuers:
            if not pursuer.deactivated:
                global_seen_evader_ids.update(pursuer.perception.observed_evaders)


        final_observations = self.update_pursuers_observation_with_aggregated_evaders_ids(
            raw_observations, list(global_seen_evader_ids)
        )


        has_target = (len(global_seen_evader_ids) > 0)

        for obs in final_observations:
            if obs is not None:

                obs['has_active_target'] = np.array([1.0 if has_target else 0.0], dtype=np.float32)


        if self.first_contact_step is None and len(global_seen_evader_ids) > 0:
            self.first_contact_step = self.episode_time_steps
            if self.enable_logging:
                logger.info(f"Process {os.getpid()} | Evader first sighted at step {self.episode_time_steps}")

        # Distance buffers are reused by multiple reward terms in the same step.
        all_pursuers_distance_to_other_pursuers = self.get_distance_to_other_pursuers()
        all_pursuers_distance_to_obstacles = self.get_distance_to_obstacles()


        all_pursuers2evader_dis_after_move = self.get_distance_to_evaders(global_seen_evader_ids)

        # Compute patrol and pursuit rewards from the post-action state.
        emergency_penalty = self.emergency_reward(all_pursuers_distance_to_other_pursuers,
                                                  all_pursuers_distance_to_obstacles
                                                  )
        time_boundary_penalty = self.time_boundary_penalty()


        patrol_arrival_rew = self.patrol_arrival_reward()
        patrol_shaping_rew = self.patrol_shaping_reward()


        pursuing_guidance_rew = self.pursuing_guidance_reward(all_pursuers2evader_dis_after_move)

        # Limit over-allocation by counting how many pursuers commit to the same evader.
        target_allocation = defaultdict(list)

        for i in range(len(self.pursuers)):
            if self.pursuers[i].deactivated: continue

            dists = all_pursuers2evader_dis_after_move[i]
            if dists:
                min_d = min(dists)

                if min_d < self.related_distance:


                    target_idx = dists.index(min_d)
                    target_allocation[target_idx].append((i, min_d))


        redundant_agents = set()
        for t_idx, agents in target_allocation.items():

            if len(agents) > 3:

                agents.sort(key=lambda x: x[1])

                for _, agent_idx in agents[3:]:
                    redundant_agents.add(agent_idx)

        reaction_threshold = self.related_distance

        for i in range(len(self.pursuers)):
            dists = all_pursuers2evader_dis_after_move[i]
            is_engaging = False


            if dists:
                min_dist = min(dists)
                if min_dist < reaction_threshold:
                    is_engaging = True


            if i in redundant_agents:
                is_engaging = False


            if is_engaging:


                patrol_shaping_rew[i] = 0.0


            else:


                pursuing_guidance_rew[i] = 0.0


        rewards = (
                emergency_penalty +
                time_boundary_penalty +
                patrol_arrival_rew +
                patrol_shaping_rew +
                pursuing_guidance_rew
        )


        for i in range(len(self.pursuers)):
            self.ep_reward_components_per_agent[i]["emergency_penalty"] += emergency_penalty[i]
            self.ep_reward_components_per_agent[i]["time_boundary_penalty"] += time_boundary_penalty[i]
            self.ep_reward_components_per_agent[i]["patrol_arrival_reward"] += patrol_arrival_rew[i]
            self.ep_reward_components_per_agent[i]["patrol_shaping_reward"] += patrol_shaping_rew[i]
            self.ep_reward_components_per_agent[i]["pursuing_guidance_reward"] += pursuing_guidance_rew[i]
            self.ep_reward_components_per_agent[i]["reward"] += rewards[i]


        for i, p in enumerate(self.pursuers):
            self.last_pursuer_positions[i] = np.array([p.x, p.y])

        # -----------------------------------------------------
        # 5. Determine Termination & Encirclement Success
        # -----------------------------------------------------
        dones = [False] * len(self.pursuers)
        infos: list[dict[str, Union[str, list[dict]]]] = [{"state": "normal"} for _ in range(len(self.pursuers))]
        agent_events = [{"event": "none", "reward": 0.0}] * len(self.pursuers)


        capture_targets, capture_angles, num_pursuer_captures, capture_evader_ids, capture_events_data = self.get_capture_status_and_info()
        if capture_events_data: infos[0]["capture_events"] = capture_events_data


        is_full_coverage = self.check_full_coverage()


        is_all_captured = self.num_evaders > 0 and self.check_all_evader_is_captured()

        for idx, pursuer in enumerate(self.pursuers):
            infos[idx]["first_contact_step"] = self.first_contact_step
            # ---------------------------------------------------

            # ---------------------------------------------------
            if pursuer.deactivated:
                dones[idx] = True
                if pursuer.collision:
                    infos[idx] = {"state": "deactivated after collision"}
                elif is_all_captured:
                    infos[idx] = {"state": "all_evaders_captured"}
                else:
                    infos[idx] = {"state": "deactivated (unknown)"}
                continue

            # ---------------------------------------------------

            # ---------------------------------------------------
            if collisions[idx]:
                rewards[idx] += self.collision_penalty

                pursuer.deactivated = True
                dones[idx] = True
                infos[idx] = {"state": "collision"}
                self.ep_reward_components_per_agent[idx]["collision_penalty"] += self.collision_penalty
                self.ep_collisions_count_per_agent[idx] += 1
                continue

            # ---------------------------------------------------


            # ---------------------------------------------------
            if capture_targets[idx]:

                encirclement_rew = self.encirclement_reward(capture_angles[idx], num_pursuer_captures[idx])
                rewards[idx] += encirclement_rew


                infos[idx] = {"state": f"✌️capture evader {capture_evader_ids[idx]}"}
                pursuer.is_current_target_captured = False  # Reset flag

                self.ep_reward_components_per_agent[idx]["goal_reward"] += encirclement_rew
                agent_events[idx] = {"event": "capture", "reward": encirclement_rew}


            # ---------------------------------------------------

            # ---------------------------------------------------


            if is_all_captured:
                dones[idx] = True
                infos[idx] = {"state": "all_evaders_captured"}

                rewards[idx] += 50.0
                self.ep_reward_components_per_agent[idx]["goal_reward"] += 50.0
                continue


            if self.episode_time_steps >= self.episode_max_length:
                dones[idx] = True
                infos[idx] = {"state": "too long episode"}
                continue


            if self.num_evaders == 0 and is_full_coverage:
                dones[idx] = True
                rewards[idx] += 50.0
                infos[idx] = {"state": "success_full_coverage"}
                self.ep_reward_components_per_agent[idx]["success_reward"] = 50.0
                continue

        # Log to WandB
        self._log_to_wandb(self.total_time_steps, rewards, dones, infos, agent_events)

        self.episode_time_steps += 1
        self.total_time_steps += 1

        return final_observations, rewards, dones, infos

    def assign_robot_position(self, pursuer_position_list: List[np.ndarray], evader_position_list: List[np.ndarray],
                              pursuer_theta_list: List[float] = None, evader_theta_list: List[float] = None):
        """
        Assign robot positions.

        Args:
            pursuer_position_list (List[np.ndarray]): Pursuer positions.
            evader_position_list (List[np.ndarray]): Evader positions.
            pursuer_theta_list (List[float]): Pursuer orientations.
            evader_theta_list (List[float]): Evader orientations.
        """
        try:
            assert len(pursuer_position_list) == len(self.pursuers), ("Number of pursuers should be equal to number of "
                                                                      "pursuers!")
            assert len(evader_position_list) == len(self.evaders), ("Number of evaders should be equal to number of "
                                                                    "evaders!")
        except AssertionError as e:
            logger.info(f"Process {os.getpid()} len(pursuer_position_list): {len(pursuer_position_list)}, "
                        f"len(self.pursuer): {len(self.pursuers)}, "
                        f"len(evader_position_list): {len(evader_position_list)}, "
                        f"len(self.evaders): {len(self.evaders)}")
            logger.error(f"Process {os.getpid()} | AssertionError: {e}.")
            raise RuntimeError(f"Error: {e}")

        for i, _pursuer in enumerate(self.pursuers):
            _pursuer.start = pursuer_position_list[i]
            self.reset_robot(_pursuer)
        for i, _evader in enumerate(self.evaders):
            _evader.start = evader_position_list[i]
            self.reset_robot(_evader)
        if pursuer_theta_list is not None:
            for i, _pursuer in enumerate(self.pursuers):
                _pursuer.theta = pursuer_theta_list[i]
        if evader_theta_list is not None:
            for i, _evader in enumerate(self.evaders):
                _evader.theta = evader_theta_list[i]

    def reset_robot(self, rob: Pursuer | Evader):
        """
        Reset robot state.

        Resets the robot's initial state including position, velocity, and perception data.

        Args:
            rob (Pursuer | Evader): Robot instance to reset.
        """
        if rob.robot_type == 'pursuer':
            rob.is_current_target_captured = False
        rob.collision = False
        rob.deactivated = False
        rob.init_theta = self.rd.uniform(low=0.0, high=2 * np.pi)
        rob.init_speed = self.rd.uniform(low=0.0, high=rob.max_speed)
        current_v = self.get_current_velocity(rob.start[0], rob.start[1])
        rob.reset_state(current_velocity=current_v)

    def reset_coverage_grid(self):
        """
        Reset the coverage grid.

        Resets the coverage grid to its initial state.
        """
        # Patrol Points Initialization
        patrol_coords = generate_centers(self.width, self.height, fov_radius=20.0)

        self.patrol_points: list[PatrolPoint] = [
            PatrolPoint(
                (x, y), idx, threshold=2.0, normalization_factor=self.norm_factor
            )
            for idx, (x, y) in enumerate(patrol_coords)
        ]
        self.visited_set: set[int] = set()

    def check_all_pursuers_deactivated(self) -> bool:
        """
        Check if all pursuers are deactivated.

        Returns:
            bool: True if all pursuers are deactivated, False otherwise.
        """
        return all(rob.deactivated for rob in self.pursuers)

    def check_all_evaders_deactivated(self) -> bool:
        """
        Check if all evaders are deactivated.

        Returns:
            bool: True if all evaders are deactivated, False otherwise.
        """
        return all(rob.deactivated for rob in self.evaders)

    def check_all_evader_is_captured(self) -> bool:
        """
        Check if all evaders have been captured.

        Returns:
            bool: True if all evaders are captured, False otherwise.
        """
        return all(evader.deactivated for evader in self.evaders)

    def check_all_pursuers_chased(self) -> bool:
        """
        Check if all pursuers have been chased.return all(pursuer.is_pursuing for pursuer in self.pursuers)
        """
        return all(pursuer.is_pursuing for pursuer in self.pursuers)

    def check_all_evader_is_chased(self) -> bool:
        """
        Check if all evaders have been chased.
        Returns:
            bool: True if all evaders have been chased, False otherwise.
        """
        return all(evader.deactivated for evader in self.evaders)

    def check_full_coverage(self) -> bool:
        """Return whether every patrol point has been visited."""
        return len(self.visited_set) == len(self.patrol_points)

    def check_any_pursuers_collision(self) -> bool:
        """
        Check if any pursuer has collided.

        Returns:
            bool: True if any pursuer has collided, False otherwise.
        """
        return any(rob.collision for rob in self.pursuers)

    def check_any_evaders_collision(self) -> bool:
        """
        Check if any evader has collided.

        Returns:
            bool: True if any evader has collided, False otherwise.
        """
        return any(rob.collision for rob in self.evaders)

    def update_evader_state(self, evader_actions: List[int]):
        """
        Update evader states.

        Updates evader states including position, velocity and trajectory based on actions.

        Args:
            evader_actions (List[int]): List of actions for evaders.
        """
        for i, evader_action in enumerate(evader_actions):
            evader = self.evaders[i]

            if evader.deactivated:
                continue

            evader.action_history.append(evader_action)

            self.update_rob_state(evader, evader_action)

            self.update_pursuing_status()

            evader.trajectory.append([evader.x, evader.y, evader.theta, evader.speed, evader.velocity[0],
                                      evader.velocity[1]])

    def get_distance_to_evaders(self, global_seen_evader_ids: set[int] = None) -> List[
        Union[List[float], Optional[float]]]:
        """
        Get distances from pursuers to evaders.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to all evaders.
        """
        all_pursuers_distance_to_evaders = []
        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_evaders.append(None)
                continue


            pursuer_distance_to_evaders = [
                np.linalg.norm([pursuer.x - evader.x, pursuer.y - evader.y])
                for evader in self.evaders
                if evader.id in global_seen_evader_ids and not evader.deactivated
            ]
            all_pursuers_distance_to_evaders.append(pursuer_distance_to_evaders)
        assert len(all_pursuers_distance_to_evaders) == len(self.pursuers), ("Number of distances should be equal to "
                                                                             "number of pursuers")
        return all_pursuers_distance_to_evaders

    def get_distance_to_patrol_points(self) -> List[list[float] | None]:
        """
        Get distances between pursuers and patrol points.
        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to all patrol points.
        """
        all_pursuers_distance_to_patrol_points = []
        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_patrol_points.append(None)
                continue
            pursuer_distance_to_patrol_points = [
                np.linalg.norm([pursuer.x - patrol_point.position[0], pursuer.y - patrol_point.position[1]]) for
                patrol_point in self.patrol_points]
            all_pursuers_distance_to_patrol_points.append(pursuer_distance_to_patrol_points)
        assert len(all_pursuers_distance_to_patrol_points) == len(self.pursuers), (
            "Number of distances should be equal to number of patrol points")

        return all_pursuers_distance_to_patrol_points

    def get_distance_to_other_pursuers(self) -> List[Optional[float]]:
        """
        Get distances between pursuers.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to other pursuers.
        """
        all_pursuers_distance_to_other_pursuers = []

        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_other_pursuers.append(None)
                continue

            pursuer_distance_to_other_pursuers = [
                np.linalg.norm([pursuer.x - other_pursuer.x, pursuer.y - other_pursuer.y]) for other_pursuer in
                self.pursuers if other_pursuer is not pursuer]

            all_pursuers_distance_to_other_pursuers.append(np.min(pursuer_distance_to_other_pursuers))

        assert len(all_pursuers_distance_to_other_pursuers) == len(self.pursuers), (
            "Number of distances should be equal "
            "to number of pursuers")
        return all_pursuers_distance_to_other_pursuers

    def update_pursuers_states(self, pursuer_actions: List[int]):
        """
        Update pursuer states.

        Updates pursuer states including position, velocity and trajectory based on actions.

        Args:
            pursuer_actions (List[int]): List of actions for pursuers.
        """
        for i, pursuer_action in enumerate(pursuer_actions):
            pursuer = self.pursuers[i]

            if pursuer.deactivated:
                continue

            pursuer.action_history.append(pursuer_action)

            self.update_rob_state(pursuer, pursuer_action)

            pursuer.trajectory.append([pursuer.x, pursuer.y, pursuer.theta, pursuer.speed, pursuer.velocity[0],
                                       pursuer.velocity[1]])

    def get_distance_to_obstacles(self) -> List[Optional[float]]:
        """
        Get distances from pursuers to obstacles.

        Returns:
            List[Optional[float]]: List of minimum distances from each pursuer to obstacles.
        """
        all_pursuers_distance_to_obstacles = []

        for pursuer in self.pursuers:
            if pursuer.deactivated:
                all_pursuers_distance_to_obstacles.append(None)
                continue

            if not pursuer.perception.observation["statics"]:
                all_pursuers_distance_to_obstacles.append(np.inf)
            else:
                closet_obstacle = pursuer.perception.observation["statics"][0]
                pursuer_distance_to_obstacle = np.linalg.norm([closet_obstacle[0], closet_obstacle[1]])

                all_pursuers_distance_to_obstacles.append(pursuer_distance_to_obstacle)

        assert len(all_pursuers_distance_to_obstacles) == len(self.pursuers), ("Number of distances should be equal to "
                                                                               "number of pursuers")
        return all_pursuers_distance_to_obstacles

    def time_boundary_penalty(self) -> np.ndarray:
        """
        Returns:
            rewards (ndarray): List of reward values for each pursuer.
        """
        rewards = np.zeros(len(self.pursuers))
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated:
                rewards[i] = 0.0
                continue
            rewards[i] += self.timestep_penalty

            if pursuer.x < 0 or pursuer.x > self.arena_size or pursuer.y < 0 or pursuer.y > self.arena_size:
                rewards[i] += self.boundary_penalty
                # logger.info(
                #     f"Process {os.getpid()} | Pursuer {i} has reached the boundary at time step {self.episode_time_steps} at {pursuer.x, pursuer.y}.")
        return rewards

    def patrol_arrival_reward(self) -> np.ndarray:
        """Reward pursuers that newly reach a patrol point."""
        rewards = np.zeros(len(self.pursuers))

        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated: continue

            for patrol_point in self.patrol_points:

                if patrol_point.check_reached((pursuer.x, pursuer.y), self.episode_time_steps):

                    norm_idleness = patrol_point.get_norm_idleness(self.episode_time_steps)


                    reward = self.arrival_reward_scale * norm_idleness
                    rewards[i] += reward


                    patrol_point.update_visit(self.episode_time_steps)
                    self.visited_set.add(patrol_point.id)


                    if self.enable_logging:
                        logger.info(f"Process {os.getpid()}: Pursuer {i} visited Point {patrol_point.id}. "
                                     f"Idleness: {norm_idleness:.2f}, Reward: {reward:.2f}")


                    # break

        return rewards

    def patrol_shaping_reward(self) -> np.ndarray:
        """Shape patrol behavior toward the currently most urgent patrol point."""
        rewards = np.zeros(len(self.pursuers))


        target_point = max(self.patrol_points, key=lambda p: p.get_idleness(self.episode_time_steps))
        target_pos = np.array(target_point.position)

        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated: continue


            curr_pos = np.array([pursuer.x, pursuer.y])
            prev_pos = self.last_pursuer_positions.get(i, curr_pos)


            dist_curr = np.linalg.norm(curr_pos - target_pos)
            dist_prev = np.linalg.norm(prev_pos - target_pos)


            diff = (dist_prev - dist_curr)

            rewards[i] += self.shaping_reward_scale * diff

        return rewards

    def pursuing_guidance_reward(self, all_pursuers_dis_to_evaders) -> np.ndarray:
        """
        [Corrected] Difference Reward based on distance to nearest target.
        Formula: Reward = Scale * (Min_Dist_Prev - Min_Dist_Curr)

        Why this is better:
        - Prevents 'camping': Standing still yields 0 reward, resulting in net loss due to timestep penalty.
        - Forces movement: The only way to get positive reward is to actively close the gap.
        """
        rewards = np.zeros(len(self.pursuers))


        scale = self.distance_reward


        min_buffer_dist = self.pursuers[0].distance_capture * 0.5

        push_back_strength = 2.0

        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated: continue


            distances_curr = all_pursuers_dis_to_evaders[i]
            if not distances_curr: continue
            min_dist_curr = min(distances_curr)

            # ---------------------------------------------------------

            # ---------------------------------------------------------
            if min_dist_curr < min_buffer_dist:


                rewards[i] -= push_back_strength * (min_buffer_dist - min_dist_curr)


                continue

            # ---------------------------------------------------------

            # ---------------------------------------------------------


            prev_pos = self.last_pursuer_positions.get(i, np.array([pursuer.x, pursuer.y]))


            known_evader_ids = pursuer.perception.observed_evaders
            if not known_evader_ids: continue

            dists_prev_list = []
            for evader in self.evaders:

                if evader.id in known_evader_ids and not evader.deactivated:
                    d = np.linalg.norm(prev_pos - np.array([evader.x, evader.y]))
                    dists_prev_list.append(d)

            if not dists_prev_list: continue
            min_dist_prev = min(dists_prev_list)


            diff = min_dist_prev - min_dist_curr


            rewards[i] += scale * diff

        return rewards

    def emergency_reward(self, all_pursuers_distance_to_other_pursuers,
                         all_pursuers_distance_to_obstacles) -> np.ndarray:
        """
        Returns:
            rewards (ndarray): List of reward values for each pursuer.
        """
        rewards = np.zeros(len(self.pursuers))
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated:
                rewards[i] = 0.0
                continue

            dis_to_pursuer = all_pursuers_distance_to_other_pursuers[i]
            dis_to_obstacle = all_pursuers_distance_to_obstacles[i]

            if dis_to_pursuer < self.safe_distance or dis_to_obstacle < self.safe_distance:
                rewards[i] += self.emergency_penalty

        return rewards

    def encirclement_reward(self, capture_angle, num_pursuer_capture):
        """

        Args:
            capture_angle: The angle between the pursuer and the evader when the evader is captured by the pursuer.
            num_pursuer_capture: The number of pursuers that capture the evader.

        Returns: The reward for the evader.

        """
        decay_factor = self.compute_capture_reward_factor(
            capture_angles=capture_angle,
            pursuer_count=num_pursuer_capture
        )
        return self.goal_reward * decay_factor

    def count_nearby_pursuers(self, evader, distance_threshold):
        """
        Count pursuers within range of an evader.

        Args:
            evader: Evader object with x,y coordinates
            distance_threshold (float): Detection range threshold

        Returns:
            tuple: (count, mask) where count is number of nearby pursuers,
                   mask is boolean array indicating which pursuers are in range
        """
        count = 0
        mask = np.zeros(len(self.pursuers), dtype=bool)
        for i, pursuer in enumerate(self.pursuers):
            distance = np.linalg.norm(np.array([pursuer.x, pursuer.y]) - np.array([evader.x, evader.y]))
            if distance <= distance_threshold:
                count += 1
                mask[i] = True
        return count, mask

    def update_pursuing_status(self):
        """Update pursuers' pursuing status based on distance to nearest evader."""
        for pursuer in self.pursuers:
            distances = [np.linalg.norm(np.array([pursuer.x, pursuer.y]) - np.array([evader.x, evader.y])) for evader in
                         self.evaders if evader.id in pursuer.perception.observed_evaders]
            if not distances:
                pursuer.is_pursuing = False
            else:
                nearest_dis = np.min(distances)
                pursuer.is_pursuing = nearest_dis <= self.related_distance

    def update_rob_state(self, rob: Pursuer | Evader, rob_action: int):
        """
        Update robot state through multiple steps.

        Args:
            rob (Pursuer | Evader): Robot instance to update
            rob_action (int): Action index to execute
        """
        for _ in range(rob.N):
            current_velocity = self.get_current_velocity(rob.x, rob.y)
            rob.update_state(rob_action, current_velocity)

    def compute_capture_reward_factor(self, capture_angles: List[float], pursuer_count: int) -> float:
        """
        Calculate capture reward factor based on formation quality.

        Args:
            capture_angles (List[float]): Angles between capturing pursuers
            pursuer_count (int): Number of participating pursuers

        Returns:
            float: Capture reward multiplier
        """
        if not capture_angles or pursuer_count <= 0:
            return 0.0

        # Calculate angle distribution quality
        sigma = np.std(capture_angles)

        # Compute formation quality factor
        capture_reward_factor = ((2 * np.pi) / pursuer_count) * np.exp(-sigma)

        return capture_reward_factor

    def out_of_boundary(self) -> bool:
        """
        Check if any robot exceeds map boundaries.

        Returns:
            bool: True if any robot is out of bounds, False otherwise
        """
        x_out = any(rob.x < 0.0 or rob.x > self.width for rob in self.pursuers + self.evaders)
        y_out = any(rob.y < 0.0 or rob.y > self.height for rob in self.pursuers + self.evaders)
        return x_out or y_out

    def get_pursuers_observations(self) -> Tuple[List[Any], List[bool]]:
        """
        Retrieve observations for all pursuers.

        Returns:
            tuple:
                List[Any]: Pursuers' observation data
                List[bool]: Collision status list
        """
        observations = []
        collisions = []

        for pursuer in self.pursuers:
            observation, collision = pursuer.perception_output(self.obstacles,
                                                               self.pursuers,
                                                               self.evaders,
                                                               self.observation_in_robot_frame,
                                                               self.patrol_points,
                                                               self.episode_time_steps
                                                               )
            observations.append(observation)
            collisions.append(collision)

        return observations, collisions

    def update_pursuers_observation_with_aggregated_evaders_ids(self, original_observations, evader_ids):
        """
        Update pursuers' observation with aggregated evader IDs.
        Args:
            original_observations (list): List of pursuer observations
            evader_ids (list): List of aggregated evader IDs
        """
        updated_observations = []

        # Log change (Debug)
        if set(evader_ids) != set(self.last_step_observed_evader_ids):
            # logger.info(f"Global Known Evaders Changed: {evader_ids}")
            self.last_step_observed_evader_ids = list(evader_ids)

        for pursuer in self.pursuers:
            if original_observations[pursuer.id] is None:
                updated_observations.append(None)
                continue


            updated_observation = pursuer.update_aggregated_evader_perception(
                evader_ids,
                self.evaders,
                original_observation_dict=original_observations[pursuer.id]
            )
            updated_observations.append(updated_observation)

        return updated_observations

    def get_evaders_observation(self) -> Tuple[List[Any], List[bool]]:
        """
        Retrieve observations for all evaders.

        Returns:
            tuple:
                List[Any]: Evaders' observation data
                List[bool]: Collision status list
        """
        observations = []
        collisions = []

        for evader in self.evaders:
            observation, collision = evader.perception_output(obstacles=self.obstacles,
                                                              pursuers=self.pursuers,
                                                              evaders=self.evaders,
                                                              in_robot_frame=self.observation_in_robot_frame)
            observations.append(observation)
            collisions.append(collision)
        return observations, collisions

    def get_capture_status_and_info(self) -> Tuple[List[bool], List[List[float]], List[int], List[Optional[int]], List[Optional[dict]]]:
        """
        Retrieve detailed capture information.

        Returns:
            tuple:
                List[bool]: Capture success status
                List[List[float]]: Capture angle distributions
                List[int]: Number of participating pursuers
                List[Optional[int]]: Captured evader IDs
        """
        capture_targets = []
        capture_angles = []
        num_pursuer_captures = []
        capture_evader_ids = []
        capture_events_data = []

        for pursuer in self.pursuers:

            is_captured, capture_angle, num_pursuer_capture, capture_evader_id, capture_info = (
                pursuer.check_capture_current_target(pursuers=self.pursuers, evaders=self.evaders))

            capture_targets.append(is_captured)
            capture_angles.append(capture_angle)
            num_pursuer_captures.append(num_pursuer_capture)
            capture_evader_ids.append(capture_evader_id)

            if is_captured and capture_info is not None:
                capture_events_data.append(capture_info)


        for pursuer in self.pursuers:
            for e_id in pursuer.captured_evaderId_list:

                for evader in self.evaders:
                    if evader.id == e_id:
                        evader.deactivated = True

        return capture_targets, capture_angles, num_pursuer_captures, capture_evader_ids, capture_events_data

    def _log_to_wandb(self, global_step: int, per_agent_rewards: np.ndarray, per_agent_dones: list[bool],
                      per_agent_infos: list[dict], per_agent_events: list[dict]):
        """Log per-step and episode-level metrics to Weights and Biases."""

        if not self.enable_logging:
            return
        step_log: dict[str, Any] = {"total_time_steps": global_step}


        outcome_map = {
            "collision": -1,
            "deactivated after collision": -1,
            "deactivated (unknown)": -1,
            "too long episode": -2,
            "normal": 0,
            "success_full_coverage": 10,
            "all_evaders_captured": 20,
        }

        for idx in range(len(self.pursuers)):
            # 1. Step Reward
            step_log[f"step_metrics/agent_{idx}/step_reward"] = per_agent_rewards[idx]

            # 2. Sparse Events
            event_info = per_agent_events[idx]
            event_type = event_info.get("event", "none")
            if event_type != "none":
                step_log[f"events/agent_{idx}/{event_type}"] = 1
                step_log[f"events/agent_{idx}/{event_type}_reward"] = event_info.get("reward", 0.0)

            # 3. Episode Metrics
            if per_agent_dones[idx]:
                outcome_str = per_agent_infos[idx].get("state", "unknown")


                outcome_val = 0
                if outcome_str in outcome_map:
                    outcome_val = outcome_map[outcome_str]
                else:
                    if outcome_str.startswith("✌️capture"):
                        outcome_val = 1
                    elif outcome_str.startswith("all_evaders"):
                        outcome_val = 20
                    else:
                        outcome_val = 0

                episode_metrics = {
                    f"episode_metrics/agent_{idx}/episode_reward": self.ep_reward_components_per_agent[idx]["reward"],
                    f"episode_metrics/agent_{idx}/episode_length": self.episode_time_steps + 1,
                    f"episode_metrics/agent_{idx}/collisions": self.ep_collisions_count_per_agent[idx],
                    f"episode_metrics/agent_{idx}/outcome": outcome_val,
                }


                for key, value in self.ep_reward_components_per_agent[idx].items():
                    episode_metrics[f"episode_metrics/agent_{idx}/reward_components/{key}"] = value

                step_log.update(episode_metrics)


        if all(per_agent_dones):
            num_agents = len(self.pursuers)
            all_ep_rewards = [self.ep_reward_components_per_agent[i]["reward"] for i in range(num_agents)]
            all_ep_lengths = [self.episode_time_steps + 1] * num_agents
            all_ep_collisions = [self.ep_collisions_count_per_agent[i] for i in range(num_agents)]

            # Success Rate Calculation

            patrol_success = sum(1 for info in per_agent_infos if info.get("state") == "success_full_coverage")
            capture_success = sum(1 for info in per_agent_infos if info.get("state") == "all_evaders_captured")


            total_success = patrol_success + capture_success
            success_rate = total_success / num_agents if num_agents > 0 else 0.0

            step_log["episode_metrics/global/mean_episode_reward"] = np.mean(all_ep_rewards)
            step_log["episode_metrics/global/mean_episode_length"] = np.mean(all_ep_lengths)
            step_log["episode_metrics/global/total_collisions"] = np.sum(all_ep_collisions)
            step_log["episode_metrics/global/success_rate"] = success_rate

        # Skip when wandb is not initialized (e.g. --no_wandb evaluation runs).
        if wandb.run is None:
            return
        wandb.log(step_log, commit=True)

    def check_pursuer_start_position(self, pursuer_start: np.ndarray) -> bool:
        """
        Validate pursuer start position.

        Args:
            pursuer_start (np.ndarray): Proposed start coordinates [x, y]

        Returns:
            bool: True if position meets safety requirements
        """
        # Check distance to other pursuers
        for pursuer in self.pursuers:
            if np.linalg.norm(pursuer_start - pursuer.start) <= self.clear_r:
                return False

        # Check distance to evaders
        for evader in self.evaders:
            if np.linalg.norm(pursuer_start - evader.start) <= self.min_pursuer_evader_init_dis:
                return False

        return True

    def check_evader_start_position(self, evader_start: np.ndarray) -> bool:
        """
        Validate evader start position.

        Args:
            evader_start (np.ndarray): Proposed start coordinates [x, y]

        Returns:
            bool: True if position meets safety requirements
        """
        # Check distance to pursuers
        for pursuer in self.pursuers:
            if np.linalg.norm(pursuer.start - evader_start) <= self.min_pursuer_evader_init_dis:
                return False

        # Check distance to other evaders
        for evader in self.evaders:
            if np.linalg.norm(evader_start - evader.start) <= self.clear_r:
                return False

        return True

    def _is_obstacle_too_close_to_patrol(self, x: float, y: float, r: float, min_safe_dist: float = 6.0) -> bool:
        """Check whether an obstacle would block a patrol point."""
        if not self.patrol_points:
            return False

        for pp in self.patrol_points:


            dist = np.hypot(x - pp.position[0], y - pp.position[1])


            if dist < (r + min_safe_dist):
                return True

        return False

    def check_core(self, core_j: Core) -> bool:
        """
        Validate vortex core position.

        Ensures the vortex core is within map boundaries and doesn't overlap with:
        - Other vortex cores
        - Obstacles
        - Robot starting positions

        Args:
            core_j (Core): Vortex core instance to validate

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check map boundaries
        if core_j.x - self.vortex_core_radius < 0.0 or core_j.x + self.vortex_core_radius > self.width:
            return False
        if core_j.y - self.vortex_core_radius < 0.0 or core_j.y + self.vortex_core_radius > self.height:
            return False

        # Check distance to robot start positions
        for rob in self.pursuers + self.evaders:
            core_pos = np.array([core_j.x, core_j.y])
            dis_s = core_pos - rob.start
            if np.linalg.norm(dis_s) < self.vortex_core_radius + self.clear_r:
                return False

        # Check interactions with other vortex cores
        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx * dx + dy * dy)

            if core_i.clockwise == core_j.clockwise:
                # Same rotation direction - validate boundary speeds
                boundary_i = core_i.Gamma / (2 * np.pi * self.v_rel_max)
                boundary_j = core_j.Gamma / (2 * np.pi * self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                # Opposite rotation - validate velocity ratios
                gamma_l = max(core_i.Gamma, core_j.Gamma)
                gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = gamma_l / (2 * np.pi * (dis - 2 * self.vortex_core_radius))
                v_2 = gamma_s / (2 * np.pi * self.vortex_core_radius)
                if v_1 > self.p * v_2:
                    return False

        return True

    def check_obstacle(self, obs: Obstacle) -> bool:
        """
        Validate obstacle position.

        Ensures the obstacle is within map boundaries and doesn't overlap with:
        - Other obstacles
        - Vortex cores
        - Robot starting positions

        Args:
            obs (Obstacle): Obstacle instance to validate

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check map boundaries
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        # Check distance to robot start positions
        for rob in self.pursuers + self.evaders:
            obs_pos = np.array([obs.x, obs.y])
            dis_s = obs_pos - rob.start
            if np.linalg.norm(dis_s) < obs.r + self.clear_r:
                return False

        min_obstructions_dis = 5  # Minimum separation distance

        # Check distance to vortex cores
        for core in self.cores:
            dx = core.x - obs.x
            dy = core.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)
            if dis <= self.vortex_core_radius + obs.r + min_obstructions_dis:
                return False

        # Check distance to other obstacles
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)
            if dis <= obstacle.r + obs.r + min_obstructions_dis:
                return False

        return True

    def get_current_velocity(self, x: float, y: float) -> np.ndarray:
        """
        Calculate ocean current velocity at specified coordinates.

        Computes cumulative velocity from all vortex cores considering:
        - Distance from each core
        - Rotation direction (clockwise/counter-clockwise)
        - Circulation strength (Gamma)

        Args:
            x (float): X-coordinate
            y (float): Y-coordinate

        Returns:
            np.ndarray: Current velocity vector [vx, vy]
        """
        if len(self.cores) == 0:
            return np.zeros(2)

        # Query nearest vortex cores
        d, idx = self.core_centers.query(np.array([x, y]), k=len(self.cores))
        if isinstance(idx, np.int64):
            idx = [idx]

        v_radial_set = []
        v_velocity = np.zeros((2, 1))
        for i in list(idx):
            core = self.cores[i]
            v_radial = np.array([[core.x - x], [core.y - y]])

            # Avoid duplicate contributions
            for v in v_radial_set:
                project = np.transpose(v) @ v_radial
                if project[0, 0] > 0:
                    continue

            v_radial_set.append(v_radial)
            dis = np.linalg.norm(v_radial)
            v_radial /= dis
            if core.clockwise:
                rotation = np.array([[0., -1.], [1., 0.]])
            else:
                rotation = np.array([[0., 1.], [-1., 0.]])
            v_tangent = rotation @ v_radial
            speed = self.compute_speed(core.Gamma, float(dis))
            v_velocity += v_tangent * speed

        return np.array([v_velocity[0, 0], v_velocity[1, 0]])

    def compute_speed(self, gamma: float, d: float) -> float:
        """
        Calculate tangential velocity at distance d from vortex core.

        Velocity profile:
        - Linear increase within core radius
        - Inverse relationship beyond core radius

        Args:
            gamma (float): Circulation strength
            d (float): Distance from core center

        Returns:
            float: Tangential velocity magnitude
        """
        if d <= self.vortex_core_radius:
            return gamma / (2 * np.pi * self.vortex_core_radius * self.vortex_core_radius) * d
        else:
            return gamma / (2 * np.pi * d)

    def reset_with_eval_config(self, eval_config: dict):
        """
        Reset environment with evaluation configuration.

        Loads predefined environment parameters including:
        - Map dimensions
        - Vortex core properties
        - Obstacle placements
        - Robot initial states

        Args:
            eval_config (dict): Configuration dictionary containing:
                - env: Environment parameters
                - cores: Vortex core specifications
                - obstacles: Obstacle positions
                - pursuers/evaders: Robot configurations
        """
        self.episode_time_steps = 0

        # Load environment parameters
        env_config = eval_config["env"]
        self.sd = env_config["seed"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.vortex_core_radius = env_config["vortex_core_radius"]
        self.v_rel_max = env_config["v_rel_max"]
        self.p = env_config["p"]
        self.v_range = copy.deepcopy(env_config["v_range"])
        self.obs_r_range = copy.deepcopy(env_config["obs_r_range"])
        self.clear_r = env_config["clear_r"]
        self.min_pursuer_evader_init_dis = env_config["min_pursuer_evader_init_dis"]
        self.timestep_penalty = env_config["timestep_penalty"]
        self.collision_penalty = env_config["collision_penalty"]
        self.goal_reward = env_config["goal_reward"]

        # Initialize vortex cores
        self.cores.clear()
        centers = None
        for i in range(len(env_config["cores"]["positions"])):
            center = env_config["cores"]["positions"][i]
            clockwise = env_config["cores"]["clockwise"][i]
            gamma = env_config["cores"]["Gamma"][i]
            core = Core(center[0], center[1], clockwise, gamma)
            self.cores.append(core)
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))

        if centers is not None:
            self.core_centers = KDTree(centers)

        # Initialize obstacles
        self.obstacles.clear()
        for i in range(len(env_config["obstacles"]["positions"])):
            center = env_config["obstacles"]["positions"][i]
            r = env_config["obstacles"]["r"][i]
            obs = Obstacle(center[0], center[1], r)
            self.obstacles.append(obs)

        # Configure pursuers
        self.pursuers.clear()
        for i in eval_config["pursuers"]["index"]:
            pursuer = Pursuer(i)
            # Set configuration parameters
            pursuer.robot_type = 'pursuer'
            pursuer.dt = eval_config["pursuers"]["dt"][i]
            pursuer.N = eval_config["pursuers"]["N"][i]
            pursuer.length = eval_config["pursuers"]["length"][i]
            pursuer.width = eval_config["pursuers"]["width"][i]
            pursuer.obs_dis = eval_config["pursuers"]["obs_dis"][i]
            pursuer.r = eval_config["pursuers"]["r"][i]
            pursuer.detect_r = eval_config["pursuers"]["detect_r"][i]
            pursuer.a = np.array(eval_config["pursuers"]["a"][i])
            pursuer.w = np.array(eval_config["pursuers"]["w"][i])
            pursuer.perception.range = eval_config["pursuers"]["perception"]["range"][i]
            pursuer.perception.angle = eval_config["pursuers"]["perception"]["angle"][i]
            pursuer.max_speed = eval_config["pursuers"]["max_speed"][i]
            pursuer.start = tuple(eval_config["pursuers"]["start"][i])
            pursuer.init_theta = eval_config["pursuers"]["init_theta"][i]
            pursuer.init_speed = eval_config["pursuers"]["init_speed"][i]
            pursuer.distance_capture = eval_config["pursuers"]["distance_capture"][i]
            pursuer.angle_capture = eval_config["pursuers"]["angle_capture"][i]
            # Initialize dynamics
            pursuer.compute_k()
            pursuer.compute_actions()
            current_v = self.get_current_velocity(float(pursuer.start[0]), float(pursuer.start[1]))
            pursuer.reset_state(current_velocity=current_v)
            self.pursuers.append(pursuer)

        # Configure evaders
        self.evaders.clear()
        for i in eval_config["evaders"]["index"]:
            evader = Evader(i)
            # Set configuration parameters
            evader.robot_type = 'evader'
            evader.dt = eval_config["evaders"]["dt"][i]
            evader.N = eval_config["evaders"]["N"][i]
            evader.length = eval_config["evaders"]["length"][i]
            evader.width = eval_config["evaders"]["width"][i]
            evader.obs_dis = eval_config["evaders"]["obs_dis"][i]
            evader.r = eval_config["evaders"]["r"][i]
            evader.detect_r = eval_config["evaders"]["detect_r"][i]
            evader.a = np.array(eval_config["evaders"]["a"][i])
            evader.w = np.array(eval_config["evaders"]["w"][i])
            evader.perception.range = eval_config["evaders"]["perception"]["range"][i]
            evader.perception.angle = eval_config["evaders"]["perception"]["angle"][i]
            evader.max_speed = eval_config["evaders"]["max_speed"][i]
            evader.start = tuple(eval_config["evaders"]["start"][i])
            evader.init_theta = eval_config["evaders"]["init_theta"][i]
            evader.init_speed = eval_config["evaders"]["init_speed"][i]
            # Initialize dynamics
            evader.compute_k()
            evader.compute_actions()
            current_v = self.get_current_velocity(float(evader.start[0]), float(evader.start[1]))
            evader.reset_state(current_velocity=current_v)
            self.evaders.append(evader)

        return self.get_pursuers_observations()

    def episode_data(self) -> dict:
        """
        Serialize environment state for persistence.

        Captures complete environment and robot state including:
        - Environment parameters
        - Vortex core configurations
        - Obstacle positions
        - Robot parameters and trajectories

        Returns:
            dict: Structured dictionary containing:
                - env: Environment parameters
                - cores: Vortex core data
                - obstacles: Obstacle data
                - pursuers/evaders: Robot states
        """
        episode = dict()

        # Environment parameters
        episode["env"] = {}
        episode["env"]["seed"] = self.sd
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["vortex_core_radius"] = self.vortex_core_radius
        episode["env"]["v_rel_max"] = self.v_rel_max
        episode["env"]["p"] = self.p
        episode["env"]["v_range"] = copy.deepcopy(self.v_range)
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["min_pursuer_evader_init_dis"] = self.min_pursuer_evader_init_dis
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward

        # Vortex core data
        episode["env"]["cores"] = {}
        episode["env"]["cores"]["positions"] = []
        episode["env"]["cores"]["clockwise"] = []
        episode["env"]["cores"]["Gamma"] = []
        for core in self.cores:
            episode["env"]["cores"]["positions"].append([core.x, core.y])
            episode["env"]["cores"]["clockwise"].append(core.clockwise)
            episode["env"]["cores"]["Gamma"].append(core.Gamma)

        # Obstacle data
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x, obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # Pursuer configurations
        episode["pursuers"] = {}
        episode["pursuers"]["index"] = []
        episode["pursuers"]["robot_type"] = []
        episode["pursuers"]["dt"] = []
        episode["pursuers"]["N"] = []
        episode["pursuers"]["length"] = []
        episode["pursuers"]["width"] = []
        episode["pursuers"]["obs_dis"] = []
        episode["pursuers"]["r"] = []
        episode["pursuers"]["detect_r"] = []
        episode["pursuers"]["a"] = []
        episode["pursuers"]["w"] = []
        episode["pursuers"]["perception"] = {}
        episode["pursuers"]["perception"]["range"] = []
        episode["pursuers"]["perception"]["angle"] = []
        episode["pursuers"]["perception"]["communication_range"] = []
        episode["pursuers"]["perception"]["communication_angle"] = []
        episode["pursuers"]["max_speed"] = []
        episode["pursuers"]["start"] = []
        episode["pursuers"]["init_theta"] = []
        episode["pursuers"]["init_speed"] = []
        episode["pursuers"]["action_history"] = []
        episode["pursuers"]["trajectory"] = []
        episode["pursuers"]["target_id"] = []
        episode["pursuers"]["target_position"] = []
        episode["pursuers"]["distance_capture"] = []
        episode["pursuers"]["angle_capture"] = []

        for pursuer in self.pursuers:
            episode["pursuers"]["index"].append(pursuer.id)
            episode["pursuers"]["robot_type"].append(pursuer.robot_type)
            episode["pursuers"]["dt"].append(pursuer.dt)
            episode["pursuers"]["N"].append(pursuer.N)
            episode["pursuers"]["length"].append(pursuer.length)
            episode["pursuers"]["width"].append(pursuer.width)
            episode["pursuers"]["obs_dis"].append(pursuer.obs_dis)
            episode["pursuers"]["r"].append(pursuer.r)
            episode["pursuers"]["detect_r"].append(pursuer.detect_r)
            episode["pursuers"]["a"].append(list(pursuer.a))
            episode["pursuers"]["w"].append(list(pursuer.w))
            episode["pursuers"]["perception"]["range"].append(pursuer.perception.range)
            episode["pursuers"]["perception"]["angle"].append(pursuer.perception.angle)
            episode["pursuers"]["max_speed"].append(pursuer.max_speed)
            episode["pursuers"]["start"].append(list(pursuer.start))
            episode["pursuers"]["init_theta"].append(pursuer.init_theta)
            episode["pursuers"]["init_speed"].append(pursuer.init_speed)
            episode["pursuers"]["action_history"].append(copy.deepcopy(pursuer.action_history))
            episode["pursuers"]["trajectory"].append(copy.deepcopy(pursuer.trajectory))
            episode["pursuers"]["distance_capture"].append(pursuer.distance_capture)
            episode["pursuers"]["angle_capture"].append(pursuer.angle_capture)

        # Evader configurations
        episode["evaders"] = {}
        episode["evaders"]["index"] = []
        episode["evaders"]["robot_type"] = []
        episode["evaders"]["dt"] = []
        episode["evaders"]["N"] = []
        episode["evaders"]["length"] = []
        episode["evaders"]["width"] = []
        episode["evaders"]["obs_dis"] = []
        episode["evaders"]["r"] = []
        episode["evaders"]["detect_r"] = []
        episode["evaders"]["a"] = []
        episode["evaders"]["w"] = []
        episode["evaders"]["perception"] = {}
        episode["evaders"]["perception"]["range"] = []
        episode["evaders"]["perception"]["angle"] = []
        episode["evaders"]["max_speed"] = []
        episode["evaders"]["start"] = []
        episode["evaders"]["init_theta"] = []
        episode["evaders"]["init_speed"] = []
        episode["evaders"]["action_history"] = []
        episode["evaders"]["trajectory"] = []

        for evader in self.evaders:
            episode["evaders"]["index"].append(evader.id)
            episode["evaders"]["robot_type"].append(evader.robot_type)
            episode["evaders"]["dt"].append(evader.dt)
            episode["evaders"]["N"].append(evader.N)
            episode["evaders"]["length"].append(evader.length)
            episode["evaders"]["width"].append(evader.width)
            episode["evaders"]["obs_dis"].append(evader.obs_dis)
            episode["evaders"]["r"].append(evader.r)
            episode["evaders"]["detect_r"].append(evader.detect_r)
            episode["evaders"]["a"].append(list(evader.a))
            episode["evaders"]["w"].append(list(evader.w))
            episode["evaders"]["perception"]["range"].append(evader.perception.range)
            episode["evaders"]["perception"]["angle"].append(evader.perception.angle)
            episode["evaders"]["max_speed"].append(evader.max_speed)
            episode["evaders"]["start"].append(list(evader.start))
            episode["evaders"]["init_theta"].append(evader.init_theta)
            episode["evaders"]["init_speed"].append(evader.init_speed)
            episode["evaders"]["action_history"].append(copy.deepcopy(evader.action_history))
            episode["evaders"]["trajectory"].append(copy.deepcopy(evader.trajectory))

        return episode

    def save_episode(self, filename: str):
        """
        Save current episode data to file.

        Serializes and persists the complete environment state including:
        - Environment configuration
        - Vortex core parameters
        - Obstacle positions
        - Robot states and trajectories

        Args:
            filename (str): Output file path for saving JSON data
        """
        episode = self.episode_data()
        with open(filename, "w") as file:
            json.dump(episode, file)
    # =========================================================================

    # =========================================================================

    def clear_env(self):
        """Clear all entities and cached spatial structures from the environment."""
        self.cores = []
        self.obstacles = []
        self.evaders = []
        self.pursuers = []
        self.num_cores = 0
        self.num_obs = 0
        self.num_evaders = 0
        self.num_pursuers = 0


    def add_obstacle(self, x: float, y: float, radius: float):
        """Insert one obstacle into the current environment."""

        obs = Obstacle(float(x), float(y), radius)
        self.obstacles.append(obs)
        self.num_obs += 1

    def add_core(self, pos: ndarray, direction: bool, v_edge: float):
        """Add a current-field core to the environment."""
        center = pos
        # Random rotation direction
        direction = direction
        # Reduced intensity parameters
        v_edge = v_edge
        gamma = 2 * np.pi * (self.vortex_core_radius * 0.5) * v_edge
        core = Core(center[0], center[1], bool(direction), gamma)

        self.cores.append(core)


    def generate_obstacles(self, num: int):
        """Regenerate the obstacle field with a fixed obstacle count."""
        self.obstacles = []
        self.num_obs = num

        if num <= 0: return

        iteration = 10000
        count = 0
        while count < num:
            center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
            r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])

            # Patrol Collision Check
            if self._is_obstacle_too_close_to_patrol(center[0], center[1], r):
                iteration -= 1
                if iteration <= 0: break
                continue

            obs = Obstacle(center[0], center[1], r)
            if self.check_obstacle(obs):
                self.obstacles.append(obs)
                count += 1

            iteration -= 1
            if iteration <= 0:
                logger.warning(f"Failed to generate all {num} obstacles, generated {count}")
                break

    def get_observation(self) -> Tuple[List[Any], List[Any]]:
        """Return pursuer and evader observations with shared perception applied."""

        raw_obs, _ = self.get_pursuers_observations()


        global_seen_evaders = set()
        for p in self.pursuers:
            if not p.deactivated:
                global_seen_evaders.update(p.perception.observed_evaders)


        final_pursuer_obs = self.update_pursuers_observation_with_aggregated_evaders_ids(
            raw_obs, list(global_seen_evaders)
        )


        has_target = (len(global_seen_evaders) > 0)
        for obs in final_pursuer_obs:
            if obs is not None:
                obs['has_active_target'] = np.array([1.0 if has_target else 0.0], dtype=np.float32)


        evader_obs, _ = self.get_evaders_observation()

        return final_pursuer_obs, evader_obs

    def _is_obstacle_too_close_to_patrol(self, x: float, y: float, r: float, min_safe_dist: float = 3.0) -> bool:
        """Check whether an obstacle conflicts with the patrol-point layout."""
        obs_pos = np.array([x, y])
        for pp in self.patrol_points:
            pp_pos = np.array(pp.position)
            dist = np.linalg.norm(obs_pos - pp_pos)

            if dist < (r + min_safe_dist):
                return True
        return False
