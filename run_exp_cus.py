import argparse
import copy
import json
import os
import random
import sys
import time as t_module
from datetime import datetime
from collections import defaultdict
import glob
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import KDTree
import torch
import wandb
from scipy.stats import pearsonr
import pandas as pd


from config_manager import ConfigManager
from utils import logger as logger
from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent
from robots.pursuer import Pursuer
from robots.evader import Evader
from policy.decentralized_heuristic import HeuristicAgent


plt.rcParams['svg.fonttype'] = 'none'


sys.path.insert(0, "./thirdparty")


def convert_to_tensor(data, device):
    """Recursively convert np.array to torch.Tensor."""
    if isinstance(data, dict):
        return {key: convert_to_tensor(value, device) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        return data

def calculate_encirclement_variance_using_pursuer_logic(pursuers, evaders, detection_range=10.0):
    """Compute the variance of encirclement angles for active pursuers."""
    active_pursuers = [p for p in pursuers if not p.deactivated]
    active_evaders = [e for e in evaders if not e.deactivated]

    if not active_evaders or len(active_pursuers) < 2:
        return None

    target_evader = None
    engaging_pursuers = []
    range_threshold = detection_range

    for ev in active_evaders:
        current_engagers = []
        for p in active_pursuers:
            dist = np.linalg.norm([p.x - ev.x, p.y - ev.y])
            if dist < range_threshold:
                current_engagers.append(p)

        if len(current_engagers) > len(engaging_pursuers):
            engaging_pursuers = current_engagers
            target_evader = ev

    if len(engaging_pursuers) < 2:
        return None

    try:


        adjacent_angles, _ = active_pursuers[0].calculate_encirclement_geometry(
            engaging_pursuers, np.array([target_evader.x, target_evader.y])
        )
    except AttributeError:
        return None

    if not adjacent_angles:
        return None

    return np.var(adjacent_angles)


class ScenarioLoader:
    """Build experiment scenarios for custom evaluations."""
    def __init__(self, env: MarineEnv):
        """Initialize the instance."""
        self.env = env

    def load(self, config: dict):
        """Load one custom scenario and prepare its runtime state."""
        # Custom scenarios rebuild the environment from scratch before applying overrides.
        self.env.num_pursuers = config.get("num_pursuers", len(self.env.pursuers))
        self.env.reset()
        self.env.clear_env()
        self.env.enable_logging = True

        mode = config.get("mode", "random")
        if mode == "custom":
            self._setup_custom(config)
        else:
            self._setup_random(config)

        # Group delayed events by rollout step so they can be triggered in O(1) time.
        event_schedule = defaultdict(list)
        if "delayed_events" in config:
            for event in config["delayed_events"]:
                step = event["step"]
                event_schedule[step].append(event)

        if hasattr(self.env, "get_observation"):
            obs = self.env.get_observation()
        else:
            raise NotImplementedError("Env must implement get_observation method.")

        ep_data = getattr(self.env, "episode_data", {})
        if callable(ep_data): ep_data = ep_data()

        return obs, event_schedule, ep_data

    def _setup_random(self, config):
        """Initialize a randomly generated scenario from the config."""
        # Random mode still uses explicit counts so experiments remain reproducible.
        self.env.num_obs =  config.get("num_obs", None)
        self.env.num_pursuers = config.get("num_pursuers", None)
        self.env.num_evaders = config.get(
            "initial_evaders", None)
        assert self.env.num_pursuers is not None and self.env.num_evaders is not None and self.env.num_obs is not None, "Random mode requires num_pursuers, num_evaders, and num_obs to be specified in config."
        self.env.min_pursuer_evader_init_dis = config.get("min_pursuer_evader_init_dis", None)
        self.env.num_cores = config.get("num_cores", 0)

        self.env.reset()


    def _setup_custom(self, config):
        """Initialize a hand-crafted scenario from the config."""
        # Custom layouts bypass procedural placement and instantiate entities directly.
        if "obstacles" in config:
            self.env.obstacles = []
            self.env.num_obs = 0
            for obs in config["obstacles"]:
                if obs.get("type") == "wall":
                    self._create_wall(obs)
                else:
                    if hasattr(self.env, "add_obstacle"):
                        self.env.add_obstacle(float(obs["x"]), float(obs["y"]), float(obs["r"]))

        if "pursuer_pos" in config:
            for i, pos in enumerate(config["pursuer_pos"]):
                p = Pursuer(i, enable_logging=True)
                p.start = np.array(pos)
                p.x, p.y = pos[0], pos[1]
                p.init_theta = np.random.uniform(low=0.0, high=2 * np.pi)
                p.init_speed = np.random.uniform(low=0.0, high=p.max_speed)
                current_v = self.env.get_current_velocity(p.start[0], p.start[1])

                p.reset_state(current_velocity=current_v)
                self.env.pursuers.append(p)
                self.env.num_pursuers += 1

        if "evader_pos" in config:
            for i, pos in enumerate(config["evader_pos"]):
                e = Evader(i)
                e.start = np.array(pos)
                e.x, e.y = pos[0], pos[1]
                e.init_theta = np.random.uniform(low=0.0, high=2 * np.pi)
                e.init_speed = np.random.uniform(low=0.0, high=e.max_speed)
                current_v = self.env.get_current_velocity(e.start[0], e.start[1])

                e.reset_state(current_velocity=current_v)
                self.env.evaders.append(e)
                self.env.num_evaders += 1

        if "core_config" in config:
            for i, pos in enumerate(config["core_config"]):
                pos = np.array(pos)
                direction = pos[2]
                v_edge = pos[3]
                self.env.add_core(pos, direction, v_edge)

            centers = None
            for core in self.env.cores:
                c = np.array([[core.x, core.y]])
                centers = c if centers is None else np.vstack((centers, c))
            if centers is not None:
                self.core_centers = KDTree(centers)


    def _create_wall(self, wall_def):
        """Expand a wall description into obstacle primitives."""
        start = np.array(wall_def["start"])
        end = np.array(wall_def["end"])
        r = wall_def["r"]
        gap = wall_def.get("gap", r * 0.5)
        vec = end - start
        length = np.linalg.norm(vec)
        direction = vec / (length + 1e-6)
        step = max(r * 0.1, r * 2 - gap)
        num_balls = int(length / step) + 1
        for i in range(num_balls):
            pos = start + direction * (i * step)
            if hasattr(self.env, "add_obstacle"):
                self.env.add_obstacle(float(pos[0]), float(pos[1]), float(r))


class EpisodeDataCollector:
    """Collect per-step data for one evaluation episode."""
    def __init__(self, total_patrol_points):
        """Initialize the instance."""
        self.time_series = defaultdict(list)
        self.total_patrol_points = max(1, total_patrol_points)

    def log_step(self, step, pursuers, evaders, gate_weights, visited_set_len):
        """Record one timestep of episode statistics."""
        self.time_series["step"].append(step)


        coverage_percentage = visited_set_len / self.total_patrol_points
        self.time_series["coverage_rate"].append(coverage_percentage)


        for i, p in enumerate(pursuers):
            if p.deactivated:
                self.time_series[f"p{i}_gate"].append(None)
                self.time_series[f"p{i}_dist"].append(None)
                continue

            w = gate_weights[i] if i < len(gate_weights) else None
            self.time_series[f"p{i}_gate"].append(w)

            min_d = float("inf")
            active_evaders = [e for e in evaders if not e.deactivated]
            for e in active_evaders:
                d = np.linalg.norm([p.x - e.x, p.y - e.y])
                if d < min_d:
                    min_d = d
            if min_d == float("inf"):
                min_d = -1.0

            self.time_series[f"p{i}_dist"].append(min_d)

    def summarize(self):
        """Summarize the collected episode statistics."""
        return {"time_series": dict(self.time_series)}


def spawn_dynamic_evader(env, pos, current_step):
    """Spawn a new evader during a delayed-event scenario."""
    from environment.env import Evader


    new_id = len(env.evaders)
    new_evader = Evader(new_id)
    new_evader.id = new_id
    new_evader.start = np.array(pos)
    new_evader.x = pos[0]
    new_evader.y = pos[1]
    new_evader.init_theta = np.random.uniform(low=0.0, high=2 * np.pi)
    new_evader.init_speed = np.random.uniform(low=0.0, high=float(new_evader.max_speed))
    current_v = env.get_current_velocity(new_evader.start[0], new_evader.start[1])


    new_evader.reset_state(current_velocity=current_v)


    new_evader.spawn_step = current_step


    env.evaders.append(new_evader)
    env.num_evaders += 1

    logger.info(f"⚡ Event: Spawned Evader {new_id} at {pos} (Step {current_step})")


def evaluation(
    scenario_config: dict,
    agent,
    evader_agent: ApfAgent,
    eval_env: MarineEnv,
    use_rl=True,
    use_iqn=True,
    act_adaptive=False,
    save_episode=False,
    ep_idx=0,
):
    """Evaluate one agent on a custom SSC scenario.

    Args:
        scenario_config: Scenario definition loaded from the config file.
        agent: Pursuer policy being evaluated.
        evader_agent: Evader controller used during evaluation.
        eval_env: Environment instance for the rollout.

    Returns:
        dict: Aggregated metrics and saved trajectory data for the episode.
    """
    # Fixed per-episode evaluation RNG: every model × training-seed evaluates on
    # the exact same layout sequence, enabling paired comparisons across seeds.
    _eval_rng_seed = 100000 + ep_idx
    eval_env.rd = np.random.RandomState(_eval_rng_seed)
    np.random.seed(_eval_rng_seed)

    # Load the initial map state and any delayed events defined by the scenario.
    loader = ScenarioLoader(eval_env)
    states, event_schedule, episode_env_init_config = loader.load(scenario_config)
    pursuer_states, evader_states = states

    # Track when each entity first becomes active so later analysis can separate phases.
    for p in eval_env.pursuers:
        p.spawn_step = 0
    for e in eval_env.evaders:
        e.spawn_step = 0


    initial_evaders_count = scenario_config.get(
        "initial_evaders", eval_env.num_evaders
    )
    phase_milestones = {
        "patrol_start": 0,
        "first_spawn": None,
        "first_contact": None,
        "all_captured": None,
        "patrol_resume_start": None,
        "patrol_complete": None,
    }

    # Keep time-series data separate from scalar episode summaries.
    collector = EpisodeDataCollector(total_patrol_points=len(eval_env.patrol_points))

    pursuer_gate_logs = [[] for _ in range(len(eval_env.pursuers))]
    pursuer_deac_logs = [[] for _ in range(len(eval_env.pursuers))]

    evader_deac_logs = defaultdict(list)

    # Runtime Vars
    rob_num = len(eval_env.pursuers)
    rewards = [0.0] * rob_num
    times = [0.0] * rob_num
    energies = [0.0] * rob_num
    computation_times = []
    collision_id_set = set()
    experiment_min_distance = float("inf")
    episode_angle_variances = []

    end_episode = False
    length = 0
    dt = eval_env.pursuers[0].dt * eval_env.pursuers[0].N

    # These counters distinguish "all evaders gone" from "all scheduled events consumed."
    total_scheduled_events = sum([len(v) for v in event_schedule.values()])
    events_triggered_count = 0

    while not end_episode:
        # Trigger delayed scenario events before selecting the next actions.
        if length in event_schedule:
            for event in event_schedule[length]:
                if event["type"] == "spawn_evader":

                    if phase_milestones["first_spawn"] is None:
                        phase_milestones["first_spawn"] = length

                    spawn_pos_config = event["pos"]


                    # Avoid spawning directly on top of currently active evaders.
                    avoid_positions = []
                    for e in eval_env.evaders:
                        if not e.deactivated:
                            avoid_positions.append(np.array([e.x, e.y]))


                    MIN_SPAWN_DIST = 4.0
                    MIN_OBS_MARGIN = 3.0
                    MAX_RETRIES = 50

                    final_spawn_pos = [eval_env.width/2.0, eval_env.height/2.0]
                    success_gen = False


                    # Retry a few nearby candidates before falling back to a noisy placement.
                    for attempt in range(MAX_RETRIES):
                        candidate_pos = None


                        if isinstance(spawn_pos_config, str) and spawn_pos_config == "ambush_in_sight":
                            # Bias the spawn toward a pursuer that is far from the arena center.
                            active_pursuers = [p for p in eval_env.pursuers if not p.deactivated]

                            if active_pursuers:
                                center_pos = np.array([eval_env.width/2.0, eval_env.height/2.0])


                                p_data = []
                                for p in active_pursuers:
                                    d = np.linalg.norm(np.array([p.x, p.y]) - center_pos)
                                    p_data.append((d, p))

                                p_data.sort(key=lambda x: x[0], reverse=True)
                                candidate_count = max(1, int(len(active_pursuers) * 0.5))
                                if len(active_pursuers) <= 3: candidate_count = len(active_pursuers)
                                top_candidates = [item[1] for item in p_data[:candidate_count]]


                                target_p = np.random.choice(top_candidates)
                                p_pos = np.array([target_p.x, target_p.y])


                                vec_to_center = center_pos - p_pos
                                dist_to_center = np.linalg.norm(vec_to_center)
                                perception_r = getattr(target_p, 'perception_range', 20.0)
                                spawn_dist = perception_r * 0.6


                                angle_noise = np.random.uniform(-0.52, 0.52)

                                if dist_to_center > 1e-3:
                                    cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)
                                    base_dir = vec_to_center / dist_to_center
                                    rotated_dir = np.array([
                                        base_dir[0]*cos_a - base_dir[1]*sin_a,
                                        base_dir[0]*sin_a + base_dir[1]*cos_a
                                    ])
                                    candidate_pos = p_pos + rotated_dir * spawn_dist
                                else:
                                    candidate_pos = p_pos + np.random.randn(2) * 5.0
                            else:
                                candidate_pos = np.array([56.0, 56.0])


                        elif isinstance(spawn_pos_config, (list, tuple)):
                            # Fixed coordinates can still be jittered when they collide with live entities.
                            candidate_pos = np.array(spawn_pos_config)
                            if attempt > 0 or len(avoid_positions) > 0:
                                candidate_pos += np.random.randn(2) * 3.0


                        if candidate_pos is not None:

                            candidate_pos[0] = np.clip(candidate_pos[0], 1.0, eval_env.width - 1.0)
                            candidate_pos[1] = np.clip(candidate_pos[1], 1.0, eval_env.height - 1.0)

                            is_collision = False


                            for exist_p in avoid_positions:
                                if np.linalg.norm(candidate_pos - exist_p) < MIN_SPAWN_DIST:
                                    is_collision = True
                                    break


                            if not is_collision:
                                for obs in eval_env.obstacles:

                                    obs_pos = np.array([obs.x, obs.y])
                                    dist_to_obs = np.linalg.norm(candidate_pos - obs_pos)

                                    if dist_to_obs < (obs.r + MIN_OBS_MARGIN):
                                        is_collision = True
                                        break

                            if not is_collision:
                                final_spawn_pos = candidate_pos.tolist()
                                success_gen = True
                                break


                    if not success_gen and candidate_pos is not None:


                        final_spawn_pos = (candidate_pos + np.random.randn(2) * 4.0).tolist()


                    spawn_dynamic_evader(eval_env, final_spawn_pos, current_step=length)
                    avoid_positions.append(np.array(final_spawn_pos))

                    pursuer_states, evader_states = eval_env.get_observation()
                    events_triggered_count += 1


        # Convert observations and gather one action from every active policy.
        pursuer_states = [convert_to_tensor(s, agent.device) for s in pursuer_states]
        action = []
        current_step_weights = [None] * rob_num

        # Pursuers may be RL-driven or heuristic, but the rollout interface is identical.
        for i, rob in enumerate(eval_env.pursuers):
            if rob.deactivated:
                action.append(None)
                pursuer_gate_logs[i].append(None)
                pursuer_deac_logs[i].append(True)
                continue

            start = t_module.perf_counter()
            if use_rl:
                if use_iqn:
                    if act_adaptive:
                        a, _, _, _ = agent.act_adaptive(pursuer_states[i])
                    else:
                        a, _, _ = agent.act(pursuer_states[i])
                else:
                    a, _ = agent.act_dqn(pursuer_states[i])
            else:
                a = agent.act(pursuer_states[i])
            end = t_module.perf_counter()
            computation_times.append(end - start)

            # Save gating outputs for later visualization and correlation analysis.
            raw_w = None
            try:
                policy = getattr(agent, "policy_local", None)
                if (
                    policy
                    and hasattr(policy, "attention_weights")
                    and policy.attention_weights is not None
                ):
                    fused = policy.attention_weights.fused
                    raw_w = (
                        fused.detach().cpu().numpy().tolist()
                        if isinstance(fused, torch.Tensor)
                        else fused.tolist()
                    )
            except Exception:
                raise RuntimeError("Error retrieving gating weights from agent policy.")

            is_gated_net = hasattr(agent, "gating_type") and agent.gating_type != "no_gating"
            current_step_weights[i] = raw_w[0] if policy is not None and is_gated_net else None
            pursuer_gate_logs[i].append(raw_w)
            pursuer_deac_logs[i].append(False)

            action.append(a)

        action = [a[0] if a is not None else None for a in action]

        # Delayed events can grow the evader list, so iterate over the live objects directly.
        evaders_action = []
        for j, evader_obj in enumerate(eval_env.evaders):
            is_dead = bool(evader_obj.deactivated)
            evader_deac_logs[evader_obj.id].append(is_dead)

            if is_dead:
                evaders_action.append(None)
                continue

            a = evader_agent.act(evader_states[j])
            evaders_action.append(a)

        # Advance the environment once both action lists have been assembled.
        pursuer_states, reward, done, info = eval_env.step((action, evaders_action))
        evader_states, _ = eval_env.get_evaders_observation()

        # Record both patrol progress and per-agent efficiency metrics for this step.
        collector.log_step(
            length,
            eval_env.pursuers,
            eval_env.evaders,
            current_step_weights,
            len(eval_env.visited_set),
        )

        for i, rob in enumerate(eval_env.pursuers):
            if rob.collision and rob.id not in collision_id_set:
                collision_id_set.add(rob.id)
            if rob.deactivated:
                continue
            rewards[i] += agent.GAMMA**length * reward[i]
            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])

        # Phase milestones let the post-processing code separate patrol and pursuit behavior.
        active_pursuers = [r for r in eval_env.pursuers if not r.deactivated]
        all_captured_now = eval_env.check_all_evader_is_captured()
        pending_events = total_scheduled_events > events_triggered_count


        current_threat_exists = eval_env.num_evaders > 0 and not all_captured_now
        has_active_threat = current_threat_exists or pending_events


        if phase_milestones["first_contact"] is None:
            for p in active_pursuers:
                if len(p.perception.observed_evaders) > 0:
                    phase_milestones["first_contact"] = length
                    break


        has_ever_spawned = (phase_milestones["first_spawn"] is not None) or (
            initial_evaders_count > 0
        )

        if has_ever_spawned and not has_active_threat:
            if phase_milestones["all_captured"] is None:
                phase_milestones["all_captured"] = length
                phase_milestones["patrol_resume_start"] = length


        is_full_coverage = len(eval_env.visited_set) == len(eval_env.patrol_points)
        if is_full_coverage and phase_milestones["patrol_complete"] is None:
            phase_milestones["patrol_complete"] = length


        is_fail = len(active_pursuers) < 3
        is_task_complete = False


        if initial_evaders_count == 0 and total_scheduled_events == 0:
            if is_full_coverage:
                is_task_complete = True


        elif initial_evaders_count > 0 and total_scheduled_events == 0:

            if all_captured_now:
                is_task_complete = True


        else:


            if not has_active_threat and is_full_coverage:

                if has_ever_spawned:
                    is_task_complete = True

        if is_task_complete or is_fail or length >= 1000:
            end_episode = True
            logger.info(
                f"End Ep: L={length}, Succ={is_task_complete}, num_active_pursuers={len(active_pursuers)} Milestones={phase_milestones}"
            )

        # These metrics are only meaningful while at least one threat is still active.
        capture_dist_threshold = (
            eval_env.pursuers[0].distance_capture * 1.5 if eval_env.pursuers else 10.0
        )

        if current_threat_exists:

            is_observing = False
            for p in active_pursuers:
                if len(p.perception.observed_evaders) > 0:
                    is_observing = True
                    break

            if is_observing:
                ang_var = calculate_encirclement_variance_using_pursuer_logic(
                    eval_env.pursuers, eval_env.evaders, capture_dist_threshold
                )
                if ang_var is not None:
                    episode_angle_variances.append(ang_var)

            # Keep the closest pursuer-evader distance seen during the threat phase.
            curr_min_dist = float("inf")
            active_e_objs = [e for e in eval_env.evaders if not e.deactivated]
            if active_e_objs:
                for r1 in active_pursuers:
                    for ev in active_e_objs:
                        d = np.linalg.norm([r1.x - ev.x, r1.y - ev.y])
                        if d < curr_min_dist:
                            curr_min_dist = d
                if curr_min_dist != float("inf"):
                    experiment_min_distance = min(
                        experiment_min_distance, curr_min_dist
                    )

        length += 1

    # Package trajectories and aggregate scalar metrics after the scenario terminates.
    # ==============================================================================


    final_task_complete = is_task_complete


    is_collision_free = (len(collision_id_set) == 0)
    final_strict_success = final_task_complete and is_collision_free

    avg_reward = np.mean(rewards)
    collision_ratio = len(collision_id_set) / rob_num
    max_time = max(times) if times else 0
    avg_energy = np.mean(energies)

    visited_count = len(eval_env.visited_set)
    total_time_seconds = length * dt
    coverage_rate = visited_count / (total_time_seconds + 1e-6)

    avg_angle_variance = (
        np.mean(episode_angle_variances) if episode_angle_variances else float("nan")
    )
    if experiment_min_distance == float("inf"):
        experiment_min_distance = float('nan')
    # Time per evader (Safe Calc)
    total_evaders_seen = max(1, eval_env.num_evaders)
    avg_time_per_evader = max_time / total_evaders_seen

    # Correlation Calc
    all_weights, all_urgencies = [], []
    ts_data = collector.time_series
    for i in range(rob_num):
        weights = ts_data[f"p{i}_gate"]
        dists = ts_data[f"p{i}_dist"]
        for w, d in zip(weights, dists):
            if w is not None and d is not None and d > 0:
                all_weights.append(w)
                all_urgencies.append(1.0 / (d + 0.1))

    if len(all_weights) > 10 and np.var(all_weights) > 1e-6:
        avg_correlation, _ = pearsonr(np.array(all_weights)[:, 0], all_urgencies)
        if np.isnan(avg_correlation):
            avg_correlation = 0.0
    else:

        avg_correlation = float('nan')

    # --- Trajectory Construction (New Metadata-Rich Format) ---
    trajectories = None
    episode_log = None

    if save_episode:
        trajectories = []


        for i, p in enumerate(eval_env.pursuers):
            traj_entry = {
                "id": p.id,
                "type": "pursuer",
                "spawn_step": getattr(p, "spawn_step", 0),
                "trajectory": copy.deepcopy(p.trajectory),
                "deactivated": pursuer_deac_logs[i],
                "gating_weights": pursuer_gate_logs[i],
            }
            trajectories.append(traj_entry)


        for j, e in enumerate(eval_env.evaders):


            traj_entry = {
                "id": e.id,
                "type": "evader",
                "spawn_step": getattr(e, "spawn_step", 0),
                "trajectory": copy.deepcopy(e.trajectory),
                "deactivated": evader_deac_logs[e.id],
            }
            trajectories.append(traj_entry)

        episode_log = collector.summarize()

        episode_log["milestones"] = phase_milestones

    return {
        "strict_success": final_strict_success,       # [New] Real Success Rate
        "task_completion": final_task_complete,       # [Renamed] Old Success Rate
        "reward": avg_reward,
        "time": max_time,
        "time_per_evader": avg_time_per_evader,
        "energy": avg_energy,
        "collision_ratio": collision_ratio,
        "min_distance": experiment_min_distance,
        "coverage_rate": coverage_rate,
        "angle_variance": avg_angle_variance,
        "correlation": avg_correlation,
        "trajectories": trajectories,
        "episode_log": episode_log,
        "episode_env_init_config": episode_env_init_config,
    }


# ==============================================================================
# 6. Main Driver & Plotting
# ==============================================================================


def natural_sort_key(s):
    """Build a natural-sort key for file names."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def plot_results_from_config(plot_config_path):
    """Render grouped plots from a saved plot config."""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    import json


    with open(plot_config_path, "r") as f:
        plot_config = json.load(f)

    csv_path = plot_config["paths"]["episodes_csv"]
    save_dir = plot_config["paths"]["plots_save_dir"]
    model_groups = plot_config["model_groups"]
    colors = plot_config["colors"]


    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}. Did you move the folder?")

    df = pd.read_csv(csv_path)
    all_algos = df['algo_name'].unique()
    scenarios = df['scenario'].unique()


    global_models = ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"]
    for m in all_algos:
        if m not in global_models: global_models.append(m)
    model_color_map = {m: colors[i % len(colors)] for i, m in enumerate(global_models)}


    METRICS_CONFIG = {
        "strict_success": {"title": "Success Rate (Task + Safe)", "ylim": (0, 1.05), "plot_type": "bar", "normalize": False, "exclude_models": [], "show_error": False},
        "task_completion": {"title": "Task Completion Rate", "ylim": (0, 1.05), "plot_type": "bar", "normalize": False, "exclude_models": [], "show_error": False},
        "time": {"title": "Episode Time (s)", "ylim": None, "plot_type": "box", "normalize": False, "exclude_models": ["random"]},
        "normalized_time": {"title": "Normalized Episode Time", "ylim": (0, 1.05), "plot_type": "box", "normalize": True, "exclude_models": []},
        "energy": {"title": "Energy Consumption", "ylim": None, "plot_type": "box", "normalize": False, "exclude_models": []},
        "time_per_evader": {"title": "Time per Evader (s)", "ylim": None, "plot_type": "box", "normalize": False, "exclude_models": ["random"]},
        "collision_ratio": {"title": "Collision Ratio", "ylim": (1e-3, 2.0), "plot_type": "bar", "normalize": False, "exclude_models": [], "log_scale": True, "min_val": 1e-3, "show_error": True},
        "min_distance": {"title": "Min Inter-Robot Distance", "ylim": None, "plot_type": "box", "normalize": False, "exclude_models": []},
        "coverage_rate": {"title": "Coverage Efficiency", "ylim": None, "plot_type": "bar", "normalize": False, "exclude_models": [], "show_error": True},
        "angle_variance": {"title": "Encirclement Angle Variance", "ylim": None, "plot_type": "box", "normalize": False, "exclude_models": []},
        "correlation": {"title": "Gating-Distance Correlation", "ylim": (-0.2, 1.1), "plot_type": "bar", "normalize": False, "exclude_models": [], "show_error": True},
    }


    for group_name, active_models in model_groups.items():
        valid_models = [m for m in active_models if m in all_algos]
        if not valid_models: continue

        group_dir = os.path.join(save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for metric_key, config in METRICS_CONFIG.items():

            actual_col = "time" if metric_key == "normalized_time" else metric_key
            if actual_col not in df.columns: continue

            exclude_list = config.get("exclude_models", [])
            plot_models = [m for m in valid_models if not any(ex.lower() in m.lower() for ex in exclude_list)]
            if not plot_models: continue

            is_log = config.get("log_scale", False)
            min_val = config.get("min_val", 1e-3)
            show_error = config.get("show_error", True)


            curr_num_models = len(plot_models)
            group_width = 0.75
            intra_gap_ratio = 0.10
            current_bar_width = group_width / (curr_num_models + (curr_num_models - 1) * intra_gap_ratio)
            step = current_bar_width * (1 + intra_gap_ratio)
            offsets = np.linspace(-step * (curr_num_models - 1) / 2, step * (curr_num_models - 1) / 2, curr_num_models)

            fig, ax = plt.subplots(figsize=(6, 5))
            x_pos = np.arange(len(scenarios))
            legend_handles = []


            global_max = 1.0
            if config["normalize"]:
                all_vals = df[df['algo_name'].isin(plot_models)][actual_col].dropna()
                if len(all_vals) > 0 and all_vals.max() > 0:
                    global_max = all_vals.max()

            for j, model in enumerate(plot_models):
                means, std_devs, raw_data_list = [], [], []

                for scen in scenarios:

                    values = df[(df['scenario'] == scen) & (df['algo_name'] == model)][actual_col].dropna().tolist()

                    if config["normalize"] and values:
                        values = [v / global_max for v in values]

                    if not values:
                        means.append(0)
                        std_devs.append(0)
                        raw_data_list.append([])
                    else:
                        means.append(np.mean(values))
                        std_devs.append(np.std(values))
                        raw_data_list.append(values)


                if is_log:
                    means = [max(m, min_val) for m in means]
                    if config["plot_type"] == "box":
                        raw_data_list = [[max(v, min_val) for v in d] for d in raw_data_list]

                color = model_color_map[model]
                positions = x_pos + offsets[j]


                if config["plot_type"] == "bar":
                    bottom_val = min_val if is_log else None
                    heights = [m - bottom_val for m in means] if is_log else means
                    current_yerr = std_devs if show_error else None

                    ax.bar(positions, heights, yerr=current_yerr, capsize=3, width=current_bar_width,
                           bottom=bottom_val, label=model, color=color, alpha=0.85, edgecolor="white", linewidth=0.7)

                elif config["plot_type"] == "box":
                    valid_data = [d for d in raw_data_list if len(d) > 0]
                    valid_pos = [positions[idx] for idx, d in enumerate(raw_data_list) if len(d) > 0]

                    if valid_data:
                        bp = ax.boxplot(valid_data, positions=valid_pos, widths=current_bar_width * 0.8,
                                        patch_artist=True, manage_ticks=False, showfliers=False)
                        for patch in bp["boxes"]:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                            patch.set_edgecolor(color)
                        for element in ['medians', 'whiskers', 'caps']:
                            plt.setp(bp[element], color='k' if element == 'medians' else color)

                    legend_handles.append(mpatches.Patch(color=color, label=model, alpha=0.7))


            ax.set_xticks(x_pos)
            clean_labels = [s.replace("_", "\n") if len(s) > 12 else s for s in scenarios]
            ax.set_xticklabels(clean_labels, rotation=0, fontsize=9)
            ax.set_title(f"[{group_name}] {config['title']}", fontsize=12, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            if is_log: ax.set_yscale("log")
            if config["ylim"]: ax.set_ylim(config["ylim"])
            elif config["normalize"]: ax.set_ylim(0, 1.1)

            if config["plot_type"] == "bar":
                ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9, ncol=2)
            else:
                ax.legend(handles=legend_handles, fontsize=8, loc="best", frameon=True, framealpha=0.9, ncol=2)

            plt.tight_layout()


            filename = metric_key + ("_norm" if config["normalize"] else "") + ("_log" if is_log else "")
            base_path = os.path.join(group_dir, filename)
            # fig.savefig(f"{base_path}_ori.png", dpi=300)
            fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
            plt.close(fig)


class NumpyEncoder(json.JSONEncoder):
    """Implement the NumpyEncoder helper."""

    def default(self, obj):
        """Convert unsupported objects into JSON-safe values."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def run_experiment(
    eval_schedules,
    args,
    agents,
    evader_agents,
    envs,
    names,
    evaluations,
    config_type,
):
    """Run the configured custom experiment suite.

    Args:
        eval_schedules: Scenario groups to evaluate.
        args: Parsed command-line arguments.
        agents: Evaluated pursuer agents.
        evader_agents: Evader controllers paired with each environment.
        envs: Environment instances used for evaluation.
        names: Model names aligned with the agents list.
        evaluations: Evaluation callables.
        config_type: Label for the current experiment family.

    Returns:
        str: Path to the generated experiment directory.
    """
    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")

    single_model = getattr(args, "model", None)
    if single_model:
        # Single-model mode: raw data lands on the data disk under
        # <out_root>/raw/<Model>/seed<N>/ so cross-seed summaries can be
        # recomputed from immutable raw outputs at any time.
        out_root = getattr(args, "out_root", "DSGN_temporal_eval")
        seed_tag = args.seed if args.seed is not None else "NA"
        exp_dir = os.path.join(out_root, "raw", single_model, f"seed{seed_tag}")
        plots_dir = os.path.join(exp_dir, "Statistics_Plots")
    else:
        root_dir = os.path.join(os.getcwd(), "All_Experiments")
        models_str = "-".join(names)
        folder_name = f"{timestamp}_{config_type}_Models-{models_str}"
        exp_dir = os.path.join(root_dir, folder_name)
        plots_dir = os.path.join(exp_dir, "Statistics_Plots")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)


    seed = eval_schedules.get("random_seed", 42)
    np.random.seed(seed)
    random.seed(seed)


    config_dump = {
        "args": vars(args) if hasattr(args, '__dict__') else str(args),
        "eval_schedules": eval_schedules,
        "config_type": config_type
    }
    with open(os.path.join(exp_dir, "experiment_config.json"), "w") as f:
        json.dump(config_dump, f, indent=4)


    test_cases_input = eval_schedules.get("test_cases", {})
    if not test_cases_input:
        logger.warning("⚠️ No 'test_cases' found in configuration!")
        return exp_dir

    if isinstance(test_cases_input, list):
        grouped_scenarios = {"Default_Group": test_cases_input}
    else:
        grouped_scenarios = test_cases_input

    scenario_groups = getattr(args, "scenario_groups", None)
    if scenario_groups:
        requested = set(scenario_groups)
        grouped_scenarios = {g: v for g, v in grouped_scenarios.items() if g in requested}
        if not grouped_scenarios:
            raise ValueError(
                f"None of the requested scenario groups {sorted(requested)} "
                f"exist in the config test_cases."
            )
        logger.info(f"🎯 Scenario group filter active: {sorted(grouped_scenarios.keys())}")


    colors = [
        "#E76F51",
        "#457B9D",
        "#2A9D8F",  # 3. Teal
        "#6D597A",  # 4. Purple
        "#E9C46A",  # 5. Yellow
        "#B5838D",  # 6. Rose
        "#5F6F94",  # 7. Slate
        "#8AB17D",  # 8. Sage
        "#F4A261",  # 9. Sandy
        "#6B705C"   # 10. Olive
    ]

    global_final_output = {}


    for group_name, scenarios in grouped_scenarios.items():
        logger.info(f"\n" + "="*50)
        logger.info(f"🚀 Starting Scenario Group: {group_name}")
        logger.info("="*50)

        group_exp_dir = os.path.join(exp_dir, group_name)
        group_plots_dir = os.path.join(plots_dir, group_name)
        os.makedirs(group_exp_dir, exist_ok=True)
        os.makedirs(group_plots_dir, exist_ok=True)

        group_aggregated_results = defaultdict(lambda: defaultdict(list))

        for scen in scenarios:
            logger.info(f" Running Scenario: {scen['name']}")
            num_episodes = scen.get("num_episodes", 1)

            for ep_i in range(num_episodes):
                for j, agent_name in enumerate(names):
                    logger.info(f"   Episode {ep_i+1}/{num_episodes} - Model: {agent_name}")

                    use_iqn, use_rl = True, True
                    if agent_name in ["DQN", "Heuristic"]: use_iqn = False
                    if agent_name == "Heuristic": use_rl = False


                    metrics = evaluations[j](
                        scen, agents[j], evader_agents[j], envs[j],
                        use_rl=use_rl, use_iqn=use_iqn, act_adaptive=False,
                        save_episode=args.save_trajectory, ep_idx=ep_i,
                    )


                    for k, v in metrics.items():
                        if isinstance(v, float) and np.isnan(v):
                            metrics[k] = None


                    key = f"{scen['name']}/{agent_name}"
                    for m_key, m_val in metrics.items():
                        if m_key not in ["trajectories", "episode_log", "episode_env_init_config"]:
                            group_aggregated_results[key][m_key].append(m_val)


                    if args.save_trajectory and metrics["trajectories"]:
                        fpath = os.path.join(group_exp_dir, f"{scen['name']}_{agent_name}_ep{ep_i}.json")
                        dump_data = {
                            "metadata": {
                                "group": group_name,
                                "scenario": scen["name"],
                                "model": agent_name,
                                "episode_idx": ep_i,
                            },
                            "metrics": {k: v for k, v in metrics.items() if k not in ["trajectories", "episode_log", "episode_env_init_config"]},
                            "trajectories": metrics["trajectories"],
                            "episode_log": metrics.get("episode_log", {}),
                            "episode_env_init_config": metrics.get("episode_env_init_config", {}),
                        }
                        with open(fpath, "w") as f:
                            json.dump(dump_data, f, cls=NumpyEncoder)


        global_final_output[group_name] = {k: dict(v) for k, v in group_aggregated_results.items()}


        logger.info(f"Process {os.getpid()} - Constructing Tidy DataFrames for CSV export...")
        episode_records = []
        for key, metrics_dict in group_aggregated_results.items():
            scen_name, agent_name = key.split("/")
            num_eps = len(metrics_dict.get('task_completion', []))

            for ep_idx in range(num_eps):
                record = {
                    "scenario": scen_name,
                    "algo_name": agent_name,
                    "episode_id": ep_idx
                }

                for m_key, m_list in metrics_dict.items():
                    record[m_key] = m_list[ep_idx] if ep_idx < len(m_list) else None
                episode_records.append(record)

        df_ep = pd.DataFrame(episode_records)
        episodes_csv_path = os.path.join(group_exp_dir, "metrics_episodes_raw.csv")
        df_ep.to_csv(episodes_csv_path, index=False)


        numeric_cols = [col for col in df_ep.columns if col not in ['scenario', 'algo_name', 'episode_id']]
        summary_df = df_ep.groupby(['scenario', 'algo_name'])[numeric_cols].agg(['mean', 'std']).reset_index()

        summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
        summary_csv_path = os.path.join(group_exp_dir, "metrics_summary_stats.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved Tidy CSVs & Stats to {group_exp_dir}")


        plot_config = {
            "model_groups": {
                "Group1_Main": ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic"],
                "Group2_Ablation": ["DSGN", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"],
                "Group3_All": ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"]
            },
            "colors": colors,
            "paths": {

                "episodes_csv": os.path.abspath(episodes_csv_path),
                # "agents_csv": os.path.abspath(agents_csv_path),
                "plots_save_dir": os.path.abspath(group_plots_dir),
                "group_exp_dir": os.path.abspath(group_exp_dir)
            }
        }

        plot_config_path = os.path.join(group_exp_dir, "plot_config.json")
        with open(plot_config_path, "w") as f:
            json.dump(plot_config, f, indent=4)

        logger.info(f"Saved Master Plot Configuration to {plot_config_path}")


        logger.info(f"📊 Generating statistical plots for group [{group_name}]...")
        try:
            # Single-model mode skips cross-model statistical plots; the
            # temporal episode-level tables are still exported as raw data.
            if not getattr(args, "model", None):
                plot_results_from_config(plot_config_path)
        except Exception as e:
            logger.error(f"Plotting failed for group {group_name}: {e}")


        logger.info(f"📈 Generating temporal analysis plots for group [{group_name}]...")
        try:
            analyzer = TemporalAnalyzer(plot_config_path)
            analyzer.parse_and_plot()
        except Exception as e:
            logger.error(f"Temporal analysis failed for group {group_name}: {e}")


    final_results_path = os.path.join(exp_dir, "final_results_all_groups.json")
    with open(final_results_path, "w") as f:
        json.dump(global_final_output, f, indent=4, cls=NumpyEncoder)

    logger.info(f"🎉 All experiments finished. Data saved to {exp_dir}")
    return exp_dir


# ==============================================================================
# 7. Entry Point
# ==============================================================================


def main_experiment_process(args):
    """Initialize agents and execute the experiment process."""
    seed = 6
    config_type = "custom"

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(project_root, "config", "exp_config_cus.yaml")

    config_manager = initialize_config_manager(config_file)
    device = args.device
    exp_config = config_manager.get_instance()
    exp_schedule = exp_config.get("exp_schedule")

    initialize_wandb(exp_schedule, args)

    agents = []
    evader_agents = []
    envs = []
    names = []
    evaluations = []

    model_names = [args.model] if getattr(args, "model", None) else args.models

    for model_name in model_names:
        logger.info(f"Initializing Model: {model_name}")
        env_inst = MarineEnv(seed)
        envs.append(env_inst)

        agent = load_agent_factory(
            model_name, device, args,
            model_path=getattr(args, "model_path", None) or None,
        )
        agents.append(agent)

        dummy_evader = env_inst.evaders[0] if env_inst.evaders else None
        apf = ApfAgent(dummy_evader.a, dummy_evader.w)

        evader_agents.append(apf)
        names.append(model_name)
        evaluations.append(evaluation)

    generated_dir = run_experiment(
        exp_schedule,
        args,
        agents,
        evader_agents,
        envs,
        names,
        evaluations,
        config_type,
    )

    if not getattr(args, "no_wandb", False):
        wandb.finish()
    return generated_dir


def load_agent_factory(model_name, device, args, model_path=None):
    """Factory to create agents and load weights.

    Args:
        model_name: Name of the model/policy to instantiate.
        device: Torch device to run the agent on.
        args: Parsed command-line arguments.
        model_path: Optional override for the weight directory (used by the
            single-model multi-seed evaluation pipeline).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = project_root


    model_paths = {
        "DSGN": os.path.join(
            base_model_dir,
            "TrainedModels/DSGN/training_2026-01-04-23-18-22_vector"
        ),
        "AVERAGE": os.path.join(
            base_model_dir,
            "TrainedModels/AVERAGE/training_2026-01-08-19-51-26_average"
        ),
        "RULE_BASED": os.path.join(
            base_model_dir,
            "TrainedModels/RULE_BASED/training_2026-01-08-19-51-23_rule_based"
        ),
        "RANDOM": os.path.join(
            base_model_dir,
            "TrainedModels/RANDOM/training_2026-01-08-19-51-25_random"
        ),
        "DQN": os.path.join(
            base_model_dir,
            "TrainedModels/DQN/training_2026-01-30-18-10-49_no_gating"
            ),
        "IQN": os.path.join(
            base_model_dir,
            "TrainedModels/IQN/training_2026-01-30-17-51-46_no_gating"
            ),
        "TransIQN": os.path.join(
            base_model_dir,
            "TrainedModels/TransIQN/training_2026-01-30-18-40-04_no_gating"
            ),
        "DSGN-WO-BETA": os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-WO-BETA/training_2026-01-31-05-55-34_vector"
            ),
        "DSGN-SCALAR": os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-SCALAR/training_2026-09-01-00-00-00_scalar"
            ),
        "DSGN-SCALAR-WO-BETA": os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-SCALAR-WO-BETA/training_2026-09-01-00-00-00_scalar"
            ),
        "Heuristic": None,
    }

    if model_path:
        model_paths[model_name] = model_path

    # Evaluation must never silently substitute randomly initialized weights.
    if model_name in model_paths:
        checkpoint = os.path.join(model_paths[model_name], "latest_model.pth")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Missing checkpoint for {model_name}: {checkpoint}. "
                ""
                "Provide trained weights using the model path override."
            )

    if model_name == "DSGN":
        agent = Agent(device=device, model_name="DSGN", gating_type="vector")
        if os.path.exists(model_paths["DSGN"]):
            agent.load_model(model_paths["DSGN"], device)
        else:
            logger.warning(
                f"Model path not found for DSGN: {model_paths['DSGN']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "AVERAGE":
        agent = Agent(device=device, model_name="AVERAGE", gating_type="average")
        if os.path.exists(model_paths["AVERAGE"]):
            agent.load_model(model_paths["AVERAGE"], device)
        else:
            logger.warning(
                f"Model path not found for AVERAGE: {model_paths['AVERAGE']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "RULE_BASED":
        agent = Agent(device=device, model_name="RULE_BASED", gating_type="rule_based")
        if os.path.exists(model_paths["RULE_BASED"]):
            agent.load_model(model_paths["RULE_BASED"], device)
        else:
            logger.warning(
                f"Model path not found for RULE_BASED: {model_paths['RULE_BASED']}, please check the path and try again."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "RANDOM":
        agent = Agent(device=device, model_name="RANDOM", gating_type="random")
        if os.path.exists(model_paths["RANDOM"]):
            agent.load_model(model_paths["RANDOM"], device)
        else:
            logger.warning(
                f"Model path not found for RANDOM: {model_paths['RANDOM']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "DQN":
        agent = Agent(device=device, model_name="DQN", gating_type="no_gating", use_iqn=False)
        if os.path.exists(model_paths["DQN"]):
            agent.load_model(model_paths["DQN"], device)
        else:
            logger.warning(
                f"Model path not found for DQN: {model_paths['DQN']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "IQN":
        agent = Agent(device=device, model_name="IQN", gating_type="no_gating")
        if os.path.exists(model_paths["IQN"]):
            agent.load_model(model_paths["IQN"], device)
        else:
            logger.warning(
                f"Model path not found for IQN: {model_paths['IQN']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "TransIQN":
        agent = Agent(device=device, model_name="TransIQN", gating_type="no_gating")
        if os.path.exists(model_paths["TransIQN"]):
            agent.load_model(model_paths["TransIQN"], device)
        else:
            logger.warning(
                f"Model path not found for TransIQN: {model_paths['TransIQN']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "DSGN-WO-BETA":
        agent = Agent(device=device, model_name="DSGN-WO-BETA", gating_type="vector")
        if os.path.exists(model_paths["DSGN-WO-BETA"]):
            agent.load_model(model_paths["DSGN-WO-BETA"], device)
        else:
            logger.warning(
                f"Model path not found for DSGN-WO-BETA: {model_paths['DSGN-WO-BETA']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "DSGN-SCALAR":
        agent = Agent(device=device, model_name="DSGN-SCALAR", gating_type="scalar")
        if os.path.exists(model_paths["DSGN-SCALAR"]):
            agent.load_model(model_paths["DSGN-SCALAR"], device)
        else:
            logger.warning(
                f"Model path not found for DSGN-SCALAR: {model_paths['DSGN-SCALAR']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "DSGN-SCALAR-WO-BETA":
        agent = Agent(device=device, model_name="DSGN-SCALAR-WO-BETA", gating_type="scalar")
        if os.path.exists(model_paths["DSGN-SCALAR-WO-BETA"]):
            agent.load_model(model_paths["DSGN-SCALAR-WO-BETA"], device)
        else:
            logger.warning(
                f"Model path not found for DSGN-SCALAR-WO-BETA: {model_paths['DSGN-SCALAR-WO-BETA']}, using random weights."
            )
            raise FileNotFoundError
        return agent

    elif model_name == "Heuristic":
        agent = HeuristicAgent(device=device, model_name="Heuristic")
        return agent

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_config_manager(config_file):
    """Load and return a configuration manager instance."""
    config_manager = ConfigManager()
    config_manager.load_config(config_file)
    return config_manager


def initialize_wandb(config, args):
    """Initialize Weights and Biases logging."""
    if getattr(args, "no_wandb", False):
        return
    try:
        wandb.init(
            project=args.project,
            group=args.group,
            name=f"Exp_{args.models}",
            config=config,
        )
    except Exception:
        pass


class TemporalAnalyzer:
    """Analyze temporal metrics from saved experiment outputs."""

    def __init__(self, plot_config_path: str):
        """Initialize the instance."""
        with open(plot_config_path, "r", encoding="utf-8") as f:
            plot_config = json.load(f)

        self.plot_config_path = plot_config_path
        self.exp_dir = plot_config["paths"]["group_exp_dir"]
        self.save_dir = os.path.join(self.exp_dir, "Temporal_Plots")
        self.analysis_dir = os.path.join(self.exp_dir, "Temporal_Analysis")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        self.model_groups = plot_config["model_groups"]
        colors = plot_config["colors"]
        self.seconds_per_step = 0.5

        global_models = [
            "DSGN",
            "TransIQN",
            "DQN",
            "IQN",
            "Heuristic",
            "DSGN-WO-BETA",
            "DSGN-SCALAR",
            "DSGN-SCALAR-WO-BETA",
            "AVERAGE",
            "RULE_BASED",
            "RANDOM",
        ]

        for _, active_models in self.model_groups.items():
            for m in active_models:
                if m not in global_models:
                    global_models.append(m)

        self.model_color_map = {
            m: colors[i % len(colors)] for i, m in enumerate(global_models)
        }


        self.known_models = sorted(global_models, key=len, reverse=True)

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def get_model_color(self, model_name: str) -> str:
        """Return the configured color for a model name."""
        return self.model_color_map.get(model_name, "#000000")

    @staticmethod
    def _is_nan(x: Any) -> bool:
        """Check whether a value should be treated as NaN."""
        return isinstance(x, float) and np.isnan(x)

    @staticmethod
    def _to_step(x: Any) -> Optional[int]:
        """Convert a value into an integer step index when possible."""
        if x is None:
            return None
        try:
            if isinstance(x, float) and np.isnan(x):
                return None
            return int(round(float(x)))
        except Exception:
            return None

    @staticmethod
    def _safe_series_value(series: List[float], step: Optional[int]) -> float:
        """Read a series value with basic safety checks."""
        if not series:
            return np.nan
        if step is None:
            return np.nan
        if step < 0:
            return np.nan
        if step >= len(series):
            return float(series[-1])
        return float(series[step])

    @staticmethod
    def _safe_duration(a: Optional[int], b: Optional[int]) -> float:
        """Compute a safe duration value."""
        if a is None or b is None:
            return np.nan
        if b < a:
            return np.nan
        return float(b - a)

    @staticmethod
    def _safe_slope(c1: float, c2: float, dt: float) -> float:
        """Compute a slope while handling degenerate cases."""
        if np.isnan(c1) or np.isnan(c2) or np.isnan(dt) or dt <= 0:
            return np.nan
        return float((c2 - c1) / dt)

    @staticmethod
    def _mean_std_n(values: List[Any]) -> Tuple[float, float, int]:
        """Compute mean, standard deviation, and sample count."""
        cleaned = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
                if np.isnan(fv):
                    continue
                cleaned.append(fv)
            except Exception:
                continue

        n = len(cleaned)
        if n == 0:
            return np.nan, np.nan, 0
        if n == 1:
            return float(cleaned[0]), 0.0, 1
        arr = np.asarray(cleaned, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)), n

    @staticmethod
    def _fmt_mean_std(mean_v: float, std_v: float, n: int, decimals: int = 2) -> str:
        """Format summary statistics for reporting."""
        if n == 0 or np.isnan(mean_v):
            return "NA"
        return f"{mean_v:.{decimals}f}±{std_v:.{decimals}f}"

    def _fallback_parse_from_filename(self, fpath: str) -> Tuple[str, str, Optional[int]]:
        """Fallback to filename parsing when metadata is missing."""
        fname = os.path.basename(fpath)
        ep_match = re.search(r"_ep(\d+)\.json$", fname)
        episode_idx = int(ep_match.group(1)) if ep_match else None

        base_name = fname
        if ep_match:
            base_name = fname[:ep_match.start()]


        for model_name in self.known_models:
            suffix = "_" + model_name
            if base_name.endswith(suffix):
                scenario_name = base_name[: -len(suffix)]
                return scenario_name, model_name, episode_idx


        parts = base_name.split("_")
        if len(parts) >= 2:
            model_name = parts[-1]
            scenario_name = "_".join(parts[:-1])
            return scenario_name, model_name, episode_idx

        return "Unknown_Scenario", "Unknown_Model", episode_idx

    def _extract_episode_payload(self, fpath: str) -> Optional[Dict[str, Any]]:
        """Extract the payload for a saved episode file."""
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read JSON: {fpath} | {e}")
            return None

        meta = content.get("metadata", {}) or {}
        if meta:
            scenario_name = meta.get("scenario", "Unknown_Scenario")
            model_name = meta.get("model", "Unknown_Model")
            episode_idx = meta.get("episode_idx", None)
        else:
            scenario_name, model_name, episode_idx = self._fallback_parse_from_filename(fpath)

        episode_log = content.get("episode_log", {}) or {}
        time_series = episode_log.get("time_series", {}) or {}
        coverage_series = time_series.get("coverage_rate", []) or []

        milestones = (
            episode_log.get("milestones", None)
            or content.get("milestones", None)
            or {}
        )

        metrics = content.get("metrics", {}) or {}

        return {
            "file_path": fpath,
            "scenario": scenario_name,
            "model": model_name,
            "episode_idx": episode_idx,
            "coverage_series": coverage_series,
            "milestones": milestones,
            "metrics": metrics,
            "raw_content": content,
        }

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def _compute_episode_temporal_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute temporal metrics for one episode."""
        series = payload["coverage_series"]
        ms = payload["milestones"] or {}
        metrics = payload["metrics"] or {}

        first_spawn = self._to_step(ms.get("first_spawn"))
        all_captured = self._to_step(ms.get("all_captured"))
        patrol_resume_start = self._to_step(ms.get("patrol_resume_start"))
        patrol_complete = self._to_step(ms.get("patrol_complete"))

        total_steps = len(series) - 1 if len(series) > 0 else np.nan

        coverage_at_switch = self._safe_series_value(series, first_spawn)
        coverage_at_capture = self._safe_series_value(series, all_captured)
        coverage_at_resume = self._safe_series_value(series, patrol_resume_start)
        coverage_at_complete = self._safe_series_value(series, patrol_complete)

        patrol_complete_steps = self._safe_duration(0, patrol_complete)
        switch_to_capture_steps = self._safe_duration(first_spawn, all_captured)
        capture_to_resume_steps = self._safe_duration(all_captured, patrol_resume_start)
        resume_to_patrol_complete_steps = self._safe_duration(
            patrol_resume_start, patrol_complete
        )
        switch_to_patrol_complete_steps = self._safe_duration(
            first_spawn, patrol_complete
        )

        pre_switch_slope = self._safe_slope(
            0.0,
            coverage_at_switch,
            float(first_spawn) if first_spawn is not None and first_spawn > 0 else np.nan,
        )

        capture_phase_slope = self._safe_slope(
            coverage_at_switch,
            coverage_at_capture,
            switch_to_capture_steps,
        )

        resume_phase_slope = self._safe_slope(
            coverage_at_resume,
            coverage_at_complete,
            resume_to_patrol_complete_steps,
        )

        overall_post_switch_slope = self._safe_slope(
            coverage_at_switch,
            coverage_at_complete,
            switch_to_patrol_complete_steps,
        )

        coverage_gain_during_capture = (
            coverage_at_capture - coverage_at_switch
            if not np.isnan(coverage_at_capture) and not np.isnan(coverage_at_switch)
            else np.nan
        )

        coverage_gain_after_capture = (
            coverage_at_complete - coverage_at_capture
            if not np.isnan(coverage_at_complete) and not np.isnan(coverage_at_capture)
            else np.nan
        )

        record = {
            "scenario": payload["scenario"],
            "model": payload["model"],
            "episode_idx": payload["episode_idx"],
            "source_file": os.path.basename(payload["file_path"]),
            "strict_success": metrics.get("strict_success", None),
            "task_completion": metrics.get("task_completion", None),
            "final_time_sec": metrics.get("time", np.nan),
            "final_collision_ratio": metrics.get("collision_ratio", np.nan),
            "final_angle_variance": metrics.get("angle_variance", np.nan),
            "final_min_distance": metrics.get("min_distance", np.nan),
            "total_steps": total_steps,
            "first_spawn": first_spawn,
            "all_captured": all_captured,
            "patrol_resume_start": patrol_resume_start,
            "patrol_complete": patrol_complete,
            "coverage_at_switch": coverage_at_switch,
            "coverage_at_capture": coverage_at_capture,
            "coverage_at_resume": coverage_at_resume,
            "coverage_at_complete": coverage_at_complete,
            "patrol_complete_steps": patrol_complete_steps,
            "switch_to_capture_steps": switch_to_capture_steps,
            "capture_to_resume_steps": capture_to_resume_steps,
            "resume_to_patrol_complete_steps": resume_to_patrol_complete_steps,
            "switch_to_patrol_complete_steps": switch_to_patrol_complete_steps,
            "pre_switch_slope": pre_switch_slope,
            "capture_phase_slope": capture_phase_slope,
            "resume_phase_slope": resume_phase_slope,
            "overall_post_switch_slope": overall_post_switch_slope,
            "coverage_gain_during_capture": coverage_gain_during_capture,
            "coverage_gain_after_capture": coverage_gain_after_capture,
        }
        return record

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def _collect_all_data(self):
        """Collect temporal data from all saved episodes."""
        json_files = glob.glob(os.path.join(self.exp_dir, "*_ep*.json"))
        if not json_files:
            logger.warning(f"No trajectory JSON files found in {self.exp_dir}")
            return {}, pd.DataFrame()

        data_pool = defaultdict(
            lambda: defaultdict(lambda: {"coverage": [], "milestones": [], "episode_rows": []})
        )
        episode_rows = []

        for fpath in sorted(json_files):
            payload = self._extract_episode_payload(fpath)
            if payload is None:
                continue

            scenario_name = payload["scenario"]
            model_name = payload["model"]


            if payload["coverage_series"]:
                data_pool[scenario_name][model_name]["coverage"].append(payload["coverage_series"])
                data_pool[scenario_name][model_name]["milestones"].append(payload["milestones"])


            row = self._compute_episode_temporal_metrics(payload)
            episode_rows.append(row)
            data_pool[scenario_name][model_name]["episode_rows"].append(row)

        df_episode = pd.DataFrame(episode_rows)
        return data_pool, df_episode

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def _export_temporal_tables(self, df_episode: pd.DataFrame):
        """Export temporal statistics tables."""
        if df_episode.empty:
            logger.warning("Temporal episode dataframe is empty; skip exporting tables.")
            return


        episode_csv = os.path.join(self.analysis_dir, "temporal_episode_metrics.csv")
        df_episode.to_csv(episode_csv, index=False)


        grouped_records = []
        metric_cols = [
            "coverage_at_switch",
            "coverage_at_capture",
            "coverage_at_resume",
            "coverage_at_complete",
            "patrol_complete_steps",
            "switch_to_capture_steps",
            "capture_to_resume_steps",
            "resume_to_patrol_complete_steps",
            "switch_to_patrol_complete_steps",
            "pre_switch_slope",
            "capture_phase_slope",
            "resume_phase_slope",
            "overall_post_switch_slope",
            "coverage_gain_during_capture",
            "coverage_gain_after_capture",
            "final_time_sec",
            "final_collision_ratio",
            "final_angle_variance",
            "final_min_distance",
        ]

        for (scenario, model), g in df_episode.groupby(["scenario", "model"], dropna=False):
            record = {
                "scenario": scenario,
                "model": model,
                "episodes": len(g),
            }

            strict_mean, strict_std, strict_n = self._mean_std_n(g["strict_success"].tolist())
            task_mean, task_std, task_n = self._mean_std_n(g["task_completion"].tolist())
            record["strict_success_rate_mean"] = strict_mean
            record["strict_success_rate_std"] = strict_std
            record["strict_success_rate_n"] = strict_n
            record["task_completion_rate_mean"] = task_mean
            record["task_completion_rate_std"] = task_std
            record["task_completion_rate_n"] = task_n

            for col in metric_cols:
                mean_v, std_v, n_v = self._mean_std_n(g[col].tolist())
                record[f"{col}_mean"] = mean_v
                record[f"{col}_std"] = std_v
                record[f"{col}_n"] = n_v

            grouped_records.append(record)

        df_summary = pd.DataFrame(grouped_records)
        df_summary = df_summary.sort_values(["scenario", "model"]).reset_index(drop=True)

        summary_csv = os.path.join(self.analysis_dir, "temporal_summary_stats.csv")
        df_summary.to_csv(summary_csv, index=False)


        paper_rows = []
        for _, row in df_summary.iterrows():
            paper_rows.append(
                {
                    "scenario": row["scenario"],
                    "model": row["model"],
                    "strict_success_rate": self._fmt_mean_std(
                        row["strict_success_rate_mean"],
                        row["strict_success_rate_std"],
                        int(row["strict_success_rate_n"]),
                        decimals=2,
                    ),
                    "task_completion_rate": self._fmt_mean_std(
                        row["task_completion_rate_mean"],
                        row["task_completion_rate_std"],
                        int(row["task_completion_rate_n"]),
                        decimals=2,
                    ),
                    "coverage_at_switch": self._fmt_mean_std(
                        row["coverage_at_switch_mean"],
                        row["coverage_at_switch_std"],
                        int(row["coverage_at_switch_n"]),
                        decimals=3,
                    ),
                    "switch_to_capture_steps": self._fmt_mean_std(
                        row["switch_to_capture_steps_mean"],
                        row["switch_to_capture_steps_std"],
                        int(row["switch_to_capture_steps_n"]),
                        decimals=1,
                    ),
                    "resume_to_patrol_complete_steps": self._fmt_mean_std(
                        row["resume_to_patrol_complete_steps_mean"],
                        row["resume_to_patrol_complete_steps_std"],
                        int(row["resume_to_patrol_complete_steps_n"]),
                        decimals=1,
                    ),
                    "switch_to_patrol_complete_steps": self._fmt_mean_std(
                        row["switch_to_patrol_complete_steps_mean"],
                        row["switch_to_patrol_complete_steps_std"],
                        int(row["switch_to_patrol_complete_steps_n"]),
                        decimals=1,
                    ),
                    "pre_switch_slope": self._fmt_mean_std(
                        row["pre_switch_slope_mean"],
                        row["pre_switch_slope_std"],
                        int(row["pre_switch_slope_n"]),
                        decimals=4,
                    ),
                    "capture_phase_slope": self._fmt_mean_std(
                        row["capture_phase_slope_mean"],
                        row["capture_phase_slope_std"],
                        int(row["capture_phase_slope_n"]),
                        decimals=4,
                    ),
                    "resume_phase_slope": self._fmt_mean_std(
                        row["resume_phase_slope_mean"],
                        row["resume_phase_slope_std"],
                        int(row["resume_phase_slope_n"]),
                        decimals=4,
                    ),
                }
            )

        df_paper_long = pd.DataFrame(paper_rows)
        paper_long_csv = os.path.join(self.analysis_dir, "temporal_paper_table_long.csv")
        df_paper_long.to_csv(paper_long_csv, index=False)


        wide_records = []
        for scenario, g in df_paper_long.groupby("scenario", dropna=False):
            record = {"scenario": scenario}
            for _, r in g.iterrows():
                model = r["model"]
                record[f"{model}__C_switch"] = r["coverage_at_switch"]
                record[f"{model}__T_capture"] = r["switch_to_capture_steps"]
                record[f"{model}__T_resume"] = r["resume_to_patrol_complete_steps"]
                record[f"{model}__T_postswitch"] = r["switch_to_patrol_complete_steps"]
                record[f"{model}__S_pre"] = r["pre_switch_slope"]
                record[f"{model}__S_cap"] = r["capture_phase_slope"]
                record[f"{model}__S_resume"] = r["resume_phase_slope"]
            wide_records.append(record)

        df_paper_wide = pd.DataFrame(wide_records)
        paper_wide_csv = os.path.join(self.analysis_dir, "temporal_paper_table_wide.csv")
        df_paper_wide.to_csv(paper_wide_csv, index=False)

        logger.info(f"Saved temporal episode metrics  -> {episode_csv}")
        logger.info(f"Saved temporal summary stats    -> {summary_csv}")
        logger.info(f"Saved temporal paper table     -> {paper_long_csv}")
        logger.info(f"Saved temporal paper wide      -> {paper_wide_csv}")

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def _plot_single_scenario(
        self,
        scen_name: str,
        models_data: Dict[str, Dict[str, List[Any]]],
        save_dir: str,
        group_name: str,
        zoom_xlim=(10, 90),
    ):
        """Plot temporal metrics for one scenario."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax_ins = ax.inset_axes([0.52, 0.35, 0.30, 0.50])

        zoom_y_min, zoom_y_max = float("inf"), float("-inf")
        has_zoom_data = False
        z_start, z_end = zoom_xlim
        max_len = 0
        sec_per_step = self.seconds_per_step

        for m_data in models_data.values():
            for seq in m_data["coverage"]:
                if seq:
                    max_len = max(max_len, len(seq))

        if max_len == 0:
            plt.close(fig)
            return

        sorted_models = [m for m in self.model_color_map.keys() if m in models_data]
        for m in models_data.keys():
            if m not in sorted_models:
                sorted_models.append(m)

        for idx, model_name in enumerate(sorted_models):
            m_data = models_data[model_name]
            color = self.get_model_color(model_name)
            raw_curves = m_data["coverage"]

            aligned_curves = []
            for seq in raw_curves:
                if not seq:
                    continue
                if len(seq) < max_len:
                    aligned_curves.append(seq + [seq[-1]] * (max_len - len(seq)))
                else:
                    aligned_curves.append(seq[:max_len])

            if not aligned_curves:
                continue

            arr = np.array(aligned_curves, dtype=float)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)


            x_axis = np.arange(len(mean_curve)) * sec_per_step


            ax.plot(
                x_axis,
                mean_curve,
                label=model_name,
                color=color,
                linewidth=2,
                linestyle="-",
                alpha=0.9,
            )
            ax.fill_between(
                x_axis,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.15,
            )


            ax_ins.plot(
                x_axis,
                mean_curve,
                color=color,
                linewidth=2.5,
                linestyle="-",
                alpha=0.9,
            )
            ax_ins.fill_between(
                x_axis,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.10,
            )


            zoom_mask = (x_axis >= z_start) & (x_axis <= z_end)
            if np.any(zoom_mask):
                has_zoom_data = True
                zoom_segment_mean = mean_curve[zoom_mask]
                local_max_mean = np.max(zoom_segment_mean)
                local_min_mean = np.min(zoom_segment_mean)
                zoom_y_max = max(zoom_y_max, local_max_mean)
                zoom_y_min = min(zoom_y_min, max(0, local_min_mean))


            milestones_list = m_data["milestones"]
            avg_milestones = {}
            for key in ["first_spawn", "all_captured"]:
                steps = [
                    self._to_step(m.get(key))
                    for m in milestones_list
                    if m and self._to_step(m.get(key)) is not None
                ]
                if steps:
                    avg_milestones[key] = float(np.mean(steps))

            y_mark = 0.02 + idx * 0.025


            if "first_spawn" in avg_milestones:
                x = avg_milestones["first_spawn"] * sec_per_step
                ax.plot(x, y_mark, marker="^", color=color, markersize=8, alpha=0.8)

            if "all_captured" in avg_milestones:
                x = avg_milestones["all_captured"] * sec_per_step
                ax.plot(x, y_mark, marker="v", color=color, markersize=8, alpha=0.8)
                ax.axvline(x=x, color=color, linestyle=":", alpha=0.4)

        if has_zoom_data:
            ax_ins.set_xlim(z_start, z_end)
            y_margin = (zoom_y_max - zoom_y_min) * 0.02 if zoom_y_max > zoom_y_min else 0.02
            ax_ins.set_ylim(max(0, zoom_y_min - y_margin), min(1.05, zoom_y_max + y_margin))
            ax_ins.tick_params(axis="x", labelsize=10)
            ax_ins.tick_params(axis="y", labelsize=10)
            ax_ins.grid(True, linestyle="--", alpha=0.5)
            ax.indicate_inset_zoom(ax_ins, edgecolor="gray", alpha=0.5, linewidth=1.5)
        else:
            ax_ins.remove()

        ax.set_title(
            f"[{group_name}] Coverage Rate over Time: {scen_name}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )


        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Coverage Rate (Normalized)", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9, ncol=2)

        plt.tight_layout()
        base_filename = f"Coverage_Time_{scen_name}"
        fig.savefig(os.path.join(save_dir, f"{base_filename}.png"), dpi=900, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{base_filename}.svg"), format="svg", bbox_inches="tight")
        plt.close(fig)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def parse_and_plot(self):
        """Parse saved experiment files and generate temporal plots."""
        data_pool, df_episode = self._collect_all_data()

        if df_episode.empty:
            logger.warning("No valid episode data found for temporal analysis.")
            return


        self._export_temporal_tables(df_episode)


        for group_name, active_models in self.model_groups.items():
            group_save_dir = os.path.join(self.save_dir, group_name)
            os.makedirs(group_save_dir, exist_ok=True)

            for scen, models_data in data_pool.items():
                filtered_models_data = {
                    m: d for m, d in models_data.items() if m in active_models
                }
                if filtered_models_data:
                    self._plot_single_scenario(
                        scen,
                        filtered_models_data,
                        group_save_dir,
                        group_name,
                    )

        logger.info(f"Temporal plots saved to   -> {self.save_dir}")
        logger.info(f"Temporal tables saved to  -> {self.analysis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--device", type=str, default="cuda:0")
    parser.add_argument("-S", "--save_trajectory", action="store_true") # Default True if needed
    parser.add_argument("--project", type=str, default="baseline_experiment")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
                "DSGN", "TransIQN", "DQN", "IQN",
                "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA",
                "AVERAGE", "RULE_BASED", "RANDOM",
                "Heuristic"
                ],
        help="Models: DSGN TransIQN DQN IQN DSGN-WO-BETA DSGN-SCALAR DSGN-SCALAR-WO-BETA AVERAGE RULE_BASED RANDOM",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single-model mode: evaluate one model × one training seed only.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed number for single-model mode (used in the output path).",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Override the weight directory for single-model mode.",
    )
    parser.add_argument(
        "--out_root", type=str, default="DSGN_temporal_eval",
        help="Output root directory for single-model raw data (default: DSGN_temporal_eval).",
    )
    parser.add_argument(
        "--scenario_groups", nargs="+", default=None,
        help="Only run these scenario groups, e.g. Task_Switching.",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable wandb logging (recommended for parallel evaluation).",
    )

    args = parser.parse_args()
    main_experiment_process(args)
