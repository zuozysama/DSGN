import argparse
import copy
import json
import os
import sys
import time as t_module
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import pandas as pd


from config_manager import ConfigManager
from utils import logger as logger
from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent
from policy.decentralized_heuristic import HeuristicAgent

sys.path.insert(0, "./thirdparty")


plt.rcParams['svg.fonttype'] = 'none'


def convert_to_tensor(data, device):
    """Recursively convert np.array to torch.Tensor."""
    if isinstance(data, dict):
        return {key: convert_to_tensor(value, device) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        return data

def calculate_grid_coverage(visited_grids, grid_resolution=1.0):
    """Convert visited-grid count into covered area."""
    return len(visited_grids) * (grid_resolution ** 2)

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
            engaging_pursuers,
            np.array([target_evader.x, target_evader.y])
        )
    except AttributeError:
        # Fallback if static method not implemented yet
        return None

    if not adjacent_angles:
        return None


    return np.var(adjacent_angles)


# 接续式评估开关: 围捕完成后回合不终止, 清零巡逻覆盖后继续跑到全覆盖或总回合上限
# (由 main_experiment_process 根据 --post-capture-patrol / --episode-budget 参数设置)
POST_CAPTURE_PATROL = False
EPISODE_BUDGET = 1000


def reset_patrol_coverage(eval_env):
    """Clear patrol coverage so the post-capture patrol phase restarts from zero.

    Resets the environment-level visited set and every patrol point's visit
    bookkeeping (visit_count / last_visited_step) together, keeping the three
    consistent; otherwise check_reached would refuse to re-trigger arrival
    events for points visited during the capture phase.
    """
    eval_env.visited_set.clear()
    for patrol_point in eval_env.patrol_points:
        patrol_point.reset()


def evaluation(
    states,
    agent,
    evader_agent,
    eval_env: MarineEnv,
    use_rl=True,
    use_iqn=True,
    act_adaptive=False,
    save_episode=False,
):
    """Run one standard evaluation episode and collect summary metrics."""
    rob_num = len(eval_env.pursuers)
    rewards = [0.0] * rob_num
    times = [0.0] * rob_num
    energies = [0.0] * rob_num
    computation_times = []
    gating_weights = [[] for _ in range(rob_num)]
    deactivated_timelines = [[] for _ in range(rob_num + len(eval_env.evaders))]

    # Keep episode-level metric buffers separate from the discounted reward totals.
    episode_angle_variances = []
    weight_urgency_pairs = [[] for _ in range(rob_num)]

    end_episode = False
    length = 0
    collision_id_set = set()
    experiment_min_distance = float("inf")

    # 接续式阶段标记: 围捕完成步 / 巡逻完成步 (未达成则为 None)
    capture_step = None
    patrol_complete_step = None

    pursuer_states, evader_states = states
    dt = eval_env.pursuers[0].dt * eval_env.pursuers[0].N

    while not end_episode:
        # Convert the latest observations before policy inference.
        action = []
        pursuer_states = [convert_to_tensor(s, agent.device) for s in pursuer_states]
        current_step_weights = [None] * rob_num

        for _i in range(rob_num):
            gating_weights[_i].append(None)

        for i, rob in enumerate(eval_env.pursuers):
            if rob.deactivated:
                action.append(None)
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

            # Save fused attention weights so they can be correlated with target urgency.
            try:
                policy = getattr(agent, "policy_local", None)
                # if policy and hasattr(policy, "gate_outputs_for_loss"):
                #     raw_w_tensor = policy.gate_outputs_for_loss
                #     if raw_w_tensor is not None:
                #         raw_w = raw_w_tensor.detach().cpu().numpy().flatten()
                #         gating_weights[i][-1] = raw_w.tolist()

                #         if hasattr(policy, 'gating_type'):
                #             if policy.gating_type == 'scalar':
                #                 w_encircle = raw_w[0]
                #             elif policy.gating_type == 'vector':
                #                 w_encircle = np.mean(raw_w)
                #             else:
                #                 w_encircle = 0.5
                #         else:
                #              w_encircle = np.mean(raw_w)

                #         current_step_weights[i] = w_encircle

                if policy and hasattr(policy, "attention_weights") and policy.attention_weights is not None:
                     fused = policy.attention_weights.fused
                     raw_w = fused.detach().cpu().numpy().tolist() if isinstance(fused, torch.Tensor) else fused.tolist()
                     gating_weights[i][-1] = raw_w
                     current_step_weights[i] = np.mean(raw_w)
            except Exception:
                raise RuntimeError("Error retrieving gating weights from agent policy.")

            action.append(a)

        action = [a[0] if a is not None else None for a in action]

        evaders_action = []
        for j, evader in enumerate(eval_env.evaders):
            if evader.deactivated:
                evaders_action.append(None)
                continue
            a = evader_agent.act(evader_states[j])
            evaders_action.append(a)

        # Step the shared environment once both sides have selected actions.
        pursuer_states, reward, done, info = eval_env.step((action, evaders_action))
        evader_states, _ = eval_env.get_evaders_observation()

        for i, rob in enumerate(eval_env.pursuers):
            if rob.collision and rob.id not in collision_id_set:
                collision_id_set.add(rob.id)
            if rob.deactivated: continue
            rewards[i] += agent.GAMMA**length * reward[i]
            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])

        for pi, p_rob in enumerate(eval_env.pursuers):
            deactivated_timelines[pi].append(bool(p_rob.deactivated))
        for ej, e_rob in enumerate(eval_env.evaders):
            deactivated_timelines[rob_num + ej].append(bool(e_rob.deactivated))

        # =========================================================

        # =========================================================
        active_pursuers = [r for r in eval_env.pursuers if not r.deactivated]
        active_evaders = [r for r in eval_env.evaders if not r.deactivated]

        # Coverage is aggregated at the end from the environment-level visited set.
        capture_dist_threshold = eval_env.pursuers[0].distance_capture * 1.5 if eval_env.pursuers else 10.0
        is_observing = False
        for p in active_pursuers:
            if len(p.perception.observed_evaders) > 0:
                is_observing = True
                break

        if is_observing:
            current_angle_var = calculate_encirclement_variance_using_pursuer_logic(
                eval_env.pursuers,
                eval_env.evaders,
                detection_range=capture_dist_threshold
            )
            if current_angle_var is not None:
                episode_angle_variances.append(current_angle_var)

        # Compare gating intensity with the urgency of currently observed evaders only.
        for i, p in enumerate(eval_env.pursuers):
            if p.deactivated or current_step_weights[i] is None:
                continue

            observed_ids = p.perception.observed_evaders

            if not observed_ids:
                # Preserve zero-target samples so the correlation still reflects idle behavior.
                weight_urgency_pairs[i].append((current_step_weights[i], 0.0))
            else:
                min_e_dist = float('inf')
                target_found = False

                for e in active_evaders:
                    if e.id in observed_ids:
                        d = np.linalg.norm([p.x - e.x, p.y - e.y])
                        if d < min_e_dist:
                            min_e_dist = d
                            target_found = True

                if target_found:
                    urgency = 1.0 / (min_e_dist + 0.1)
                    weight_urgency_pairs[i].append((current_step_weights[i], urgency))

        # =========================================================

        # Track the closest interaction seen in this rollout for safety analysis.
        min_distance = float("inf")
        if len(active_pursuers) > 1:
            for i, r1 in enumerate(active_pursuers):
                for j, r2 in enumerate(active_pursuers):
                    if r1.id != r2.id:
                        d = np.linalg.norm(np.array([r1.x, r1.y]) - np.array([r2.x, r2.y]))
                        min_distance = min(min_distance, d)
        for r1 in active_pursuers:
            for ev in active_evaders:
                d = np.linalg.norm(np.array([r1.x, r1.y]) - np.array([ev.x, ev.y]))
                min_distance = min(min_distance, d)

        if min_distance != float("inf"):
            experiment_min_distance = min(experiment_min_distance, min_distance)

        # -----------------------------------------------------------
        # 接续式两阶段逻辑: 围捕完成 -> 清零巡逻覆盖 -> 继续巡逻
        # -----------------------------------------------------------
        if capture_step is None and eval_env.check_all_evader_is_captured():
            capture_step = length
            if POST_CAPTURE_PATROL:
                reset_patrol_coverage(eval_env)

        if (
            POST_CAPTURE_PATROL
            and capture_step is not None
            and patrol_complete_step is None
            and eval_env.check_full_coverage()
        ):
            patrol_complete_step = length

        if POST_CAPTURE_PATROL:
            # 接续式: 围捕完成后继续跑, 直到巡逻全覆盖 / 总回合上限 / 追捕者减员
            end_episode = (
                (length >= EPISODE_BUDGET)
                or (len(active_pursuers) < 3)
                or (patrol_complete_step is not None)
            )
        else:
            # 原口径: 围捕完成即结束
            end_episode = (
                (length >= EPISODE_BUDGET)
                or (len(active_pursuers) < 3)
                or (capture_step is not None)
            )
        if end_episode:
            logger.info(
                f"Episode ended at length {length}, len(pursuers) = {len([pursuer for pursuer in eval_env.pursuers if not pursuer.deactivated])}, "
                f"check_all_evader_is_captured = {eval_env.check_all_evader_is_captured()}, "
                f"capture_step = {capture_step}, patrol_complete_step = {patrol_complete_step}"
            )
        length += 1

    # Summarize the rollout after the termination condition has been reached.


    is_pursuit_success = eval_env.check_all_evader_is_captured()


    has_collision = (len(collision_id_set) > 0)
    collision_ratio = len(collision_id_set) / rob_num


    is_true_success = is_pursuit_success and (not has_collision)

    # 接续式新增指标: 最终覆盖率 (回合结束时刻已访问巡逻点占比)
    final_coverage_ratio = (
        len(eval_env.visited_set) / len(eval_env.patrol_points)
        if eval_env.patrol_points else 0.0
    )

    max_pursuit_time = max(times) if times else 0
    avg_pursuit_time = max_pursuit_time / eval_env.num_evaders if eval_env.num_evaders > 0 else 0
    success_energies = [energies[i] for i, rob in enumerate(eval_env.pursuers) if not rob.deactivated]


    if experiment_min_distance == float("inf"):
        experiment_min_distance = float('nan')

    # Coverage is reported as visited patrol points per simulated second.
    visited_count = len(eval_env.visited_set)

    total_time_seconds = length * dt

    coverage_area = visited_count
    coverage_rate = coverage_area / (total_time_seconds + 1e-6)

    avg_angle_variance = np.mean(episode_angle_variances) if episode_angle_variances else float('nan')

    # Aggregate per-agent correlations only when enough samples were collected.
    correlations = []
    for pairs in weight_urgency_pairs:
        if len(pairs) > 10:
            weights = [p[0] for p in pairs]
            urgencies = [p[1] for p in pairs]
            if np.var(weights) < 1e-6:
                corr = 0.0
            else:
                corr, _ = pearsonr(weights, urgencies)
                if np.isnan(corr): corr = 0.0
            correlations.append(corr)
    avg_correlation = np.mean(correlations) if correlations else float('nan')

    trajectories = None
    if save_episode:
        trajectories = [
            copy.deepcopy(rob.trajectory)
            for rob in eval_env.pursuers + eval_env.evaders
        ]
        total_robots = len(eval_env.pursuers) + len(eval_env.evaders)

        if len(trajectories) < total_robots:
            pass

        for ri in range(len(trajectories)):
            traj_entry = trajectories[ri]
            if not isinstance(traj_entry, dict):
                traj_entry = {"trajectory": traj_entry}

            if ri < len(deactivated_timelines):
                traj_entry["deactivated"] = deactivated_timelines[ri]

            if ri < len(gating_weights):
                traj_entry["gating_weights"] = gating_weights[ri]
            trajectories[ri] = traj_entry

    return (
        is_pursuit_success, # 0
        rewards,            # 1
        computation_times,  # 2
        max_pursuit_time,   # 3
        avg_pursuit_time,   # 4
        success_energies,   # 5
        trajectories,       # 6
        collision_ratio,    # 7
        experiment_min_distance, # 8
        coverage_rate,      # 9
        avg_angle_variance, # 10
        avg_correlation,    # 11
        is_true_success,    # 12
        capture_step,          # 13 围捕完成步 (接续式两阶段标记)
        patrol_complete_step,  # 14 巡逻完成步 (未达成 None)
        final_coverage_ratio,  # 15 回合结束覆盖率
    )


def exp_setup(envs, eval_schedule, i):
    """Set up the experiment environment for evaluation."""
    observations = []
    for test_env in envs:
        test_env.num_pursuers = eval_schedule["num_pursuers"][i]
        test_env.num_evaders = eval_schedule["num_evaders"][i]
        test_env.num_cores = eval_schedule["num_cores"][i]
        test_env.num_obs = eval_schedule["num_obstacles"][i]
        test_env.min_pursuer_evader_init_dis = eval_schedule[
            "min_pursuer_evader_init_dis"
        ][i]

        state, collision, _ = test_env.reset()
        evader_state, _ = test_env.get_evaders_observation()
        states = [state, evader_state]
        observations.append(states)
    return observations


def dashboard(eval_schedule, indent=0):
    """Render the experiment summary dashboard."""
    for key, value in eval_schedule.items():
        if isinstance(value, dict):
            logger.info(f"Process {os.getpid()} -" + "  " * indent + f"{key}:")
        else:
            logger.info(f"Process {os.getpid()} -" + "  " * indent + f"{key}: {value}")


def run_experiment(eval_schedules, args, agents, evader_agents, envs, names, evaluations, config_type):
    """Evaluate all requested models on the selected experiment schedule.

    Args:
        eval_schedules: Experiment schedules loaded from the selected config file.
        args: Parsed command-line arguments.
        agents: Pursuer agents to evaluate.
        evader_agents: Evader controllers aligned with the environments.
        envs: Environment instances used for evaluation.
        names: Model names aligned with the agents list.
        evaluations: List of evaluation functions, strictly matching the length of agents.
        config_type: Label used for naming the experiment output directory.
    """
    color_palette = [
        "#E76F51",
        "#457B9D",
    "#2A9D8F",  # 3. Teal
    "#6D597A",  # 4. Purple
    "#E9C46A",  # 5. Yellow
    "#B5838D",  # 6. Rose (New)
    "#5F6F94",  # 7. Slate
    "#8AB17D",  # 8. Sage
    "#F4A261",  # 9. Sandy
    "#6B705C"   # 10. Olive (New)
]
    colors = color_palette

    assert len(agents) == len(evader_agents) == len(envs) == len(names) == len(evaluations),\
        f"Length Mismatch: Agents({len(agents)}), Envs({len(envs)}), Names({len(names)}), Evals({len(evaluations)})"

    save_trajectory = args.save_trajectory
    save_eval_config = getattr(args, "save_eval_config", False)
    save_configs = save_trajectory or save_eval_config
    logger.info(f"Process {os.getpid()} - save_trajectory: {save_trajectory}, save_eval_config: {save_eval_config}")

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")


    # 实验产物统一保存到数据盘 (优先 DSGN_EXP_DIR), 方便后续复现
    root_dir = os.environ.get("DSGN_EXP_DIR") or os.path.join(os.getcwd(), "All_Experiments")
    os.makedirs(root_dir, exist_ok=True)

    models_str = "-".join(names)
    folder_name = f"{timestamp}_{config_type}_Models-{models_str}"

    exp_dir = os.path.join(root_dir, folder_name)
    os.makedirs(exp_dir, exist_ok=True)


    plots_dir = os.path.join(exp_dir, "Statistics_Plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"Saving experiment data to {exp_dir}")

    robot_nums = []
    summary_stats = {}

    # Raw Data for nested list structure: [scenario][model] -> list
    all_successes_exp = []
    all_true_successes_exp = []
    all_rewards_exp = []
    all_success_times_exp = []
    per_evader_success_times_exp = []
    all_success_energies_exp = []
    all_collision_ratios_exp = []
    all_experiment_min_distance_exp = []
    all_trajectories_exp = [] if save_trajectory else None
    all_eval_configs_exp = [] if save_configs else None


    all_coverage_rates_exp = []
    all_angle_variances_exp = []
    all_correlations_exp = []

    # 接续式两阶段指标
    all_capture_steps_exp = []
    all_patrol_complete_steps_exp = []
    all_final_coverage_ratios_exp = []

    # Each schedule entry represents one scenario family with its own episode count.
    for idx, count in enumerate(eval_schedules["num_episodes"]):
        dashboard(eval_schedules, idx)
        curr_n_pursuers = eval_schedules["num_pursuers"][idx]
        curr_n_evaders = eval_schedules["num_evaders"][idx]

        scenario_key = (
            f"Scenario_{idx}_Pursuers{curr_n_pursuers}_Evaders{curr_n_evaders}"
        )
        summary_stats[scenario_key] = {}

        robot_nums.append(curr_n_pursuers)

        all_successes = [[] for _ in agents]
        all_true_successes = [[] for _ in agents]
        all_rewards = [[] for _ in agents]
        all_computation_times = [[] for _ in agents]
        all_max_success_times_with_success = [[] for _ in agents]
        all_avg_success_times_with_success = [[] for _ in agents]
        all_success_energies = [[] for _ in agents]
        all_trajectories = [[] for _ in agents] if save_trajectory else None
        all_eval_configs = [[] for _ in agents] if save_configs else None
        all_collision_ratios = [[] for _ in agents]
        all_experiment_min_distance = [[] for _ in agents]


        all_coverage_rates = [[] for _ in agents]
        all_angle_variances = [[] for _ in agents]
        all_correlations = [[] for _ in agents]

        all_capture_steps = [[] for _ in agents]
        all_patrol_complete_steps = [[] for _ in agents]
        all_final_coverage_ratios = [[] for _ in agents]

        # Roll out every model on a fresh copy of the current scenario.
        for i in range(count):

            observations = exp_setup(envs, eval_schedules, idx)


            for j in range(len(agents)):
                logger.info(f"Evaluating all agents on schedule_{idx}_agent_{j}_ep_{i}")
                agent = agents[j]
                evader_agent = evader_agents[j]
                env = envs[j]
                name = names[j]
                eval_func = evaluations[j]

                if save_configs:
                    all_eval_configs[j].append(env.episode_data())

                obs = observations[j]

                use_rl = True
                use_iqn = True
                if name == "Heuristic":
                    use_rl = False
                    use_iqn = False
                if name == "DQN":
                    use_iqn = False
                # Keep heuristic and DQN agents on their legacy inference paths.

                (pur_success, rewards_ep, comp_times, max_t, avg_t, energies_ep,
                 trajs, coll_ratio, min_dist,
                 # Unpack new metrics
                 cov_rate, ang_var, corr, true_success,
                 capture_step, patrol_complete_step, final_coverage_ratio) = eval_func(
                    obs, agent, evader_agent, env,
                    use_rl=use_rl, use_iqn=use_iqn, act_adaptive=False,
                    save_episode=save_trajectory
                )

                all_successes[j].append(pur_success)
                all_true_successes[j].append(true_success)
                all_rewards[j] += rewards_ep
                all_computation_times[j] += comp_times
                all_max_success_times_with_success[j].append(max_t)
                all_avg_success_times_with_success[j].append(avg_t)
                all_success_energies[j] += energies_ep
                all_collision_ratios[j].append(coll_ratio)
                # all_experiment_min_distance[j].append(min_dist)


                all_experiment_min_distance[j].append(min_dist if not np.isnan(min_dist) else None)


                all_coverage_rates[j].append(cov_rate)
                all_angle_variances[j].append(ang_var if not np.isnan(ang_var) else None)
                all_correlations[j].append(corr if not np.isnan(corr) else None)

                all_capture_steps[j].append(capture_step)
                all_patrol_complete_steps[j].append(patrol_complete_step)
                all_final_coverage_ratios[j].append(final_coverage_ratio)

                if save_trajectory:
                    all_trajectories[j].append(copy.deepcopy(trajs))

        # Convert raw episode outputs into per-scenario summary statistics.
        for k, name in enumerate(names):
            pur_s_rate = np.sum(all_successes[k]) / len(all_successes[k]) if all_successes[k] else 0
            true_s_rate = np.sum(all_true_successes[k]) / len(all_true_successes[k]) if all_true_successes[k] else 0
            avg_r = np.mean(all_rewards[k]) if all_rewards[k] else 0
            avg_t = np.mean(all_max_success_times_with_success[k]) if all_max_success_times_with_success[k] else 0
            avg_e = np.mean(all_success_energies[k]) if all_success_energies[k] else 0
            avg_coll = np.mean(all_collision_ratios[k]) if all_collision_ratios[k] else 0

            # New stats
            avg_cov = np.mean(all_coverage_rates[k]) if all_coverage_rates[k] else 0


            valid_angs = [v for v in all_angle_variances[k] if v is not None]
            avg_ang = np.mean(valid_angs) if valid_angs else 0.0

            valid_corrs = [v for v in all_correlations[k] if v is not None]
            avg_corr = np.mean(valid_corrs) if valid_corrs else 0.0
            logger.info(
                f"[{name}] Pur_Succ: {pur_s_rate:.2f} | True_Succ: {true_s_rate:.2f} | Rew: {avg_r:.1f} | Coll: {avg_coll:.2f} | "
                f"Cov: {avg_cov:.1f} | AngVar: {avg_ang:.2f} | Corr: {avg_corr:.2f}"
            )

            # 接续式两阶段指标汇总
            n_captured = sum(1 for s in all_capture_steps[k] if s is not None)
            n_patrol_done = sum(1 for s in all_patrol_complete_steps[k] if s is not None)
            patrol_s_rate = (n_patrol_done / n_captured) if n_captured > 0 else 0.0
            patrol_steps_valid = [
                p - c for c, p in zip(all_capture_steps[k], all_patrol_complete_steps[k])
                if c is not None and p is not None
            ]
            avg_patrol_steps = np.mean(patrol_steps_valid) if patrol_steps_valid else 0.0
            avg_capture_steps = np.mean([s for s in all_capture_steps[k] if s is not None]) if n_captured else 0.0
            avg_final_cov = np.mean(all_final_coverage_ratios[k]) if all_final_coverage_ratios[k] else 0.0
            logger.info(
                f"[{name}] Patrol_Succ: {patrol_s_rate:.2f} (n={n_patrol_done}/{n_captured}) | "
                f"AvgPatrolSteps: {avg_patrol_steps:.1f} | FinalCov: {avg_final_cov:.3f}"
            )

            summary_stats[scenario_key][name] = {
                "pur_success_rate": float(pur_s_rate),
                "true_success_rate": float(true_s_rate),
                "avg_reward": float(avg_r),
                "avg_time": float(avg_t),
                "avg_energy": float(avg_e),
                "avg_collision": float(avg_coll),
                # New stats
                "coverage_rate": float(avg_cov),
                "angle_variance": float(avg_ang),
                "weight_correlation": float(avg_corr),
                # 接续式两阶段指标
                "patrol_success_rate": float(patrol_s_rate),
                "avg_capture_steps": float(avg_capture_steps),
                "avg_patrol_steps": float(avg_patrol_steps),
                "final_coverage_ratio": float(avg_final_cov),
            }
        logger.info("\n")

        all_successes_exp.append(all_successes)
        all_true_successes_exp.append(all_true_successes)
        all_rewards_exp.append(all_rewards)
        all_success_times_exp.append(all_max_success_times_with_success)
        per_evader_success_times_exp.append(all_avg_success_times_with_success)
        all_success_energies_exp.append(all_success_energies)
        all_collision_ratios_exp.append(all_collision_ratios)
        all_experiment_min_distance_exp.append(all_experiment_min_distance)

        all_coverage_rates_exp.append(all_coverage_rates)
        all_angle_variances_exp.append(all_angle_variances)
        all_correlations_exp.append(all_correlations)

        all_capture_steps_exp.append(all_capture_steps)
        all_patrol_complete_steps_exp.append(all_patrol_complete_steps)
        all_final_coverage_ratios_exp.append(all_final_coverage_ratios)

        if save_trajectory:
            all_trajectories_exp.append(copy.deepcopy(all_trajectories))
        if save_configs:
            all_eval_configs_exp.append(copy.deepcopy(all_eval_configs))

    # Save Data
    exp_data = {

        "eval_metadata": {
            "post_capture_patrol": POST_CAPTURE_PATROL,
            "episode_budget": EPISODE_BUDGET,
        },

        "robot_nums": robot_nums,

        "eval_schedules": eval_schedules,
        "names": names,
        "all_successes_exp": all_successes_exp,
        "all_true_successes_exp": all_true_successes_exp,
        "all_rewards_exp": all_rewards_exp,
        "all_success_times_exp": all_success_times_exp,


        "per_evader_success_times_exp": per_evader_success_times_exp,

        "all_success_energies_exp": all_success_energies_exp,
        "all_collision_ratios_exp": all_collision_ratios_exp,
        "all_experiment_min_distance_exp": all_experiment_min_distance_exp,
        "all_coverage_rates_exp": all_coverage_rates_exp,
        "all_angle_variances_exp": all_angle_variances_exp,
        "all_correlations_exp": all_correlations_exp,
        "all_capture_steps_exp": all_capture_steps_exp,
        "all_patrol_complete_steps_exp": all_patrol_complete_steps_exp,
        "all_final_coverage_ratios_exp": all_final_coverage_ratios_exp,
    }

    if save_trajectory:
        exp_data["all_trajectories_exp"] = all_trajectories_exp
    if save_configs:
        exp_data["all_eval_configs_exp"] = all_eval_configs_exp


    # Plotting

    with open(os.path.join(exp_dir, "exp_results.json"), "w") as file:
        json.dump(exp_data, file)

    with open(os.path.join(exp_dir, "summary_stats.json"), "w") as file:
        json.dump(summary_stats, file, indent=4)


    logger.info(f"Process {os.getpid()} - Constructing Tidy DataFrames for CSV export...")

    episode_records = []
    agent_records = []


    for i, count in enumerate(eval_schedules["num_episodes"]):
        scenario_name = f"{eval_schedules['num_pursuers'][i]}v{eval_schedules['num_evaders'][i]}"

        for j, name in enumerate(names):

            for ep in range(len(all_successes_exp[i][j])):
                episode_records.append({
                    "scenario": scenario_name,
                    "robot_num": eval_schedules['num_pursuers'][i],
                    "evader_num": eval_schedules['num_evaders'][i],
                    "algo_name": name,
                    "episode_id": ep,
                    "pursuit_success": int(all_successes_exp[i][j][ep]),
                    "true_success": int(all_true_successes_exp[i][j][ep]),
                    "max_success_time": all_success_times_exp[i][j][ep],
                    "avg_success_time": per_evader_success_times_exp[i][j][ep],
                    "collision_ratio": all_collision_ratios_exp[i][j][ep],
                    "min_distance": all_experiment_min_distance_exp[i][j][ep],
                    "coverage_rate": all_coverage_rates_exp[i][j][ep],
                    "angle_variance": all_angle_variances_exp[i][j][ep],
                    "gating_correlation": all_correlations_exp[i][j][ep],
                    "capture_step": all_capture_steps_exp[i][j][ep],
                    "patrol_complete_step": all_patrol_complete_steps_exp[i][j][ep],
                    "patrol_success": int(all_patrol_complete_steps_exp[i][j][ep] is not None),
                    "final_coverage_ratio": all_final_coverage_ratios_exp[i][j][ep]
                })


            for k in range(len(all_rewards_exp[i][j])):
                agent_records.append({
                    "scenario": scenario_name,
                    "algo_name": name,
                    "reward": all_rewards_exp[i][j][k],

                    "energy": all_success_energies_exp[i][j][k] if k < len(all_success_energies_exp[i][j]) else None
                })


    df_ep = pd.DataFrame(episode_records)
    df_ag = pd.DataFrame(agent_records)

    episodes_csv_path = os.path.join(exp_dir, "metrics_episodes_raw.csv")
    agents_csv_path = os.path.join(exp_dir, "metrics_agents_raw.csv")

    df_ep.to_csv(episodes_csv_path, index=False)
    df_ag.to_csv(agents_csv_path, index=False)


    logger.info(f"Process {os.getpid()} - Generating summary statistics CSV...")


    summary_df = df_ep.groupby(['scenario', 'robot_num', 'evader_num', 'algo_name']).agg(

        pur_success_mean=('pursuit_success', 'mean'),
        pur_success_std=('pursuit_success', 'std'),
        true_success_mean=('true_success', 'mean'),
        true_success_std=('true_success', 'std'),


        max_time_mean=('max_success_time', 'mean'),
        max_time_std=('max_success_time', 'std'),
        avg_time_mean=('avg_success_time', 'mean'),
        avg_time_std=('avg_success_time', 'std'),


        coll_ratio_mean=('collision_ratio', 'mean'),
        coll_ratio_std=('collision_ratio', 'std'),
        min_dist_mean=('min_distance', 'mean'),
        min_dist_std=('min_distance', 'std'),


        cov_rate_mean=('coverage_rate', 'mean'),
        cov_rate_std=('coverage_rate', 'std'),
        angle_var_mean=('angle_variance', 'mean'),
        angle_var_std=('angle_variance', 'std'),
        gating_corr_mean=('gating_correlation', 'mean'),
        gating_corr_std=('gating_correlation', 'std')
    ).reset_index()

    summary_csv_path = os.path.join(exp_dir, "metrics_summary_stats.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved Statistical Summary to: {summary_csv_path}")


    scenarios_list = [f"{r}v{e}" for r, e in zip(eval_schedules["num_pursuers"], eval_schedules["num_evaders"])]

    plot_config = {
        "model_groups": {
        "Group1_Main": ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic"],
        "Group2_Ablation": ["DSGN", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"],
        "Group3_All": ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"]
        },
        "colors": colors,
        "scenarios": scenarios_list,
        "paths": {
            "episodes_csv": os.path.abspath(episodes_csv_path),
            "agents_csv": os.path.abspath(agents_csv_path),
            "plots_save_dir": os.path.abspath(plots_dir)
        }
    }

    plot_config_path = os.path.join(exp_dir, "plot_config.json")
    with open(plot_config_path, "w") as f:
        json.dump(plot_config, f, indent=4)

    # logger.info(f"Saved Master Plot Configuration to {plot_config_path}")


    try:
        plot_results_from_config(plot_config_path)
    except Exception as e:
        # logger.error(f"Plotting failed: {e}")
        pass

    return exp_dir


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

    episodes_csv = plot_config["paths"]["episodes_csv"]
    agents_csv = plot_config["paths"]["agents_csv"]
    save_dir = plot_config["paths"]["plots_save_dir"]
    model_groups = plot_config["model_groups"]
    colors = plot_config["colors"]
    scenarios = plot_config.get("scenarios")


    if not os.path.exists(episodes_csv):
        raise FileNotFoundError(f"Data not found at {episodes_csv}")

    df_ep = pd.read_csv(episodes_csv)
    df_ag = pd.read_csv(agents_csv)


    if not scenarios:
        scenarios = df_ep['scenario'].unique().tolist()


    all_algos = df_ep['algo_name'].unique()
    global_models = ["DSGN", "TransIQN", "DQN", "IQN", "Heuristic", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA", "AVERAGE", "RULE_BASED", "RANDOM"]
    for m in all_algos:
        if m not in global_models: global_models.append(m)
    model_color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(global_models)}


    def get_data_matrix(df, metric_col, target_scenarios, target_models):
        """Collect metric values for the requested scenarios and models."""
        data = []
        for s in target_scenarios:
            s_data = []
            for m in target_models:

                vals = df[(df['scenario'] == s) & (df['algo_name'] == m)][metric_col].dropna().tolist()
                s_data.append(vals)
            data.append(s_data)
        return data


    for group_name, active_models in model_groups.items():

        active_models = [m for m in active_models if m in all_algos]
        if not active_models:
            continue


        group_dir = os.path.join(save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)


        metrics_map = [
            {
                "key": "pursuit_success",
                "data": get_data_matrix(df_ep, "pursuit_success", scenarios, active_models),
                "title": "Task Completion Rate", "type": "bar", "ylabel": "Rate", "ylim": (0, 1.05), "show_error": False
            },
            {
                "key": "success_w_o_collision",
                "data": get_data_matrix(df_ep, "true_success", scenarios, active_models),
                "title": "Success Rate (Task + Safe)", "type": "bar", "ylabel": "Rate", "ylim": (0, 1.05), "show_error": False
            },
            {
                "key": "collision_ratio",
                "data": get_data_matrix(df_ep, "collision_ratio", scenarios, active_models),
                "title": "Collision Ratio", "type": "bar", "ylabel": "Ratio", "ylim": (1e-3, 2.0), "log_scale": True, "min_val": 1e-3
            },
            {
                "key": "normalized_time",
                "data": get_data_matrix(df_ep, "max_success_time", scenarios, active_models),
                "title": "Normalized Episode Time", "type": "box", "ylabel": "Norm. Time", "normalize": True
            },
            {
                "key": "absolute_time_no_random",
                "data": get_data_matrix(df_ep, "max_success_time", scenarios, active_models),
                "title": "Episode Time (s)", "type": "box", "ylabel": "Time Steps", "normalize": False, "exclude_models": ["RANDOM", "Random"]
            },
            {
                "key": "absolute_time_evader_no_random",
                "data": get_data_matrix(df_ep, "avg_success_time", scenarios, active_models),
                "title": "Time per Evader (s)", "type": "box", "ylabel": "Time Steps", "normalize": False, "exclude_models": ["RANDOM", "Random"]
            },
            {
                "key": "min_distance",
                "data": get_data_matrix(df_ep, "min_distance", scenarios, active_models),
                "title": "Min Inter-Robot Distance", "type": "box", "ylabel": "Distance (m)"
            },
            {
                "key": "angle_variance",
                "data": get_data_matrix(df_ep, "angle_variance", scenarios, active_models),
                "title": "Encirclement Angle Variance", "type": "box", "ylabel": "Variance"
            },
            {
                "key": "coverage_rate",
                "data": get_data_matrix(df_ep, "coverage_rate", scenarios, active_models),
                "title": "Coverage Efficiency", "type": "box", "ylabel": "Rate", "ylim": (0, 1.05)
            },
            {
                "key": "gating_correlation",
                "data": get_data_matrix(df_ep, "gating_correlation", scenarios, active_models),
                "title": "Gating-Distance Correlation", "type": "bar", "ylabel": "Pearson Coeff", "ylim": (-0.2, 1.1)
            },
            {
                "key": "energy",
                "data": get_data_matrix(df_ag, "energy", scenarios, active_models),
                "title": "Energy Consumption", "type": "box", "ylabel": "Energy"
            }
        ]


        num_scenarios = len(scenarios)
        x_indices = np.arange(num_scenarios)

        for m_cfg in metrics_map:
            excluded_names = m_cfg.get("exclude_models", [])


            plot_model_indices = [
                i for i, name in enumerate(active_models)
                if not any(ex in name for ex in excluded_names)
            ]

            curr_num_models = len(plot_model_indices)
            if curr_num_models == 0:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            is_log = m_cfg.get("log_scale", False)
            min_val = m_cfg.get("min_val", 1e-3)
            show_error = m_cfg.get("show_error", True)


            group_width = 0.75
            intra_gap_ratio = 0.10
            current_bar_width = group_width / (curr_num_models + (curr_num_models - 1) * intra_gap_ratio)
            step = current_bar_width * (1 + intra_gap_ratio)
            offsets = np.linspace(-step * (curr_num_models - 1) / 2, step * (curr_num_models - 1) / 2, curr_num_models)


            legend_patches = []
            for idx in plot_model_indices:
                name = active_models[idx]
                color = model_color_map[name]
                legend_patches.append(mpatches.Patch(color=color, label=name, alpha=0.8))

            for i in range(num_scenarios):
                scale_factor = 1.0
                if m_cfg.get("normalize", False):
                    scenario_means = []
                    for j in plot_model_indices:
                        raw_vals = m_cfg["data"][i][j]
                        if raw_vals: scenario_means.append(np.mean(raw_vals))
                    if scenario_means and max(scenario_means) > 1e-6:
                        scale_factor = 1.0 / max(scenario_means)

                for k, model_idx in enumerate(plot_model_indices):
                    raw_data = m_cfg["data"][i][model_idx]
                    model_name = active_models[model_idx]

                    if not raw_data: raw_data = [0]
                    plot_data = [d * scale_factor for d in raw_data]
                    if is_log: plot_data = [max(d, min_val) for d in plot_data]

                    pos = x_indices[i] + offsets[k]
                    color = model_color_map[model_name]

                    if m_cfg["type"] == "bar":
                        mean_val = np.mean(plot_data)
                        std_val = np.std(plot_data)
                        bottom_val = min_val if is_log else None
                        height = mean_val - bottom_val if is_log else mean_val
                        current_yerr = std_val if show_error else None

                        ax.bar(pos, height, yerr=current_yerr, width=current_bar_width,
                               bottom=bottom_val, color=color, capsize=3, edgecolor='white', alpha=0.9)

                    elif m_cfg["type"] == "box":
                        bp = ax.boxplot(plot_data, positions=[pos], widths=current_bar_width * 0.8,
                                        patch_artist=True, manage_ticks=False, showfliers=False)
                        for patch in bp['boxes']:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        for element in ['whiskers', 'caps', 'medians']:
                            plt.setp(bp[element], color='black')
                        plt.setp(bp['medians'], linewidth=1.5, color='black')


            ax.set_xticks(x_indices)
            ax.set_xticklabels(scenarios, fontsize=10, fontweight='bold')
            ax.set_xlabel("Scenario Scale (Pursuers vs Evaders)", fontsize=10)
            ax.set_ylabel(m_cfg["ylabel"], fontsize=10)
            ax.set_title(f"[{group_name}] {m_cfg['title']}", fontsize=12, fontweight='bold', pad=12)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            if is_log: ax.set_yscale("log")
            if "ylim" in m_cfg: ax.set_ylim(m_cfg["ylim"])
            ax.legend(handles=legend_patches, loc='best', fontsize=8, frameon=True, framealpha=0.9)
            plt.tight_layout()


            filename = f"{m_cfg['key']}"
            if is_log: filename += "_log"
            base_path = os.path.join(group_dir, filename)

            # fig.savefig(f"{base_path}_ori.png", dpi=300)
            fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{base_path}.svg", format="svg", bbox_inches='tight')
            plt.close(fig)

        logger.info(f"Successfully generated plots for {group_name} in {group_dir}")


def _box_plot(ax, data, pos, color):
    """Draw a styled box plot for one metric."""
    if not data:
        return
    box = ax.boxplot(
        data,
        positions=[pos],
        flierprops={"marker": ".", "markersize": 1},
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
    for line in box["medians"]:
        line.set_color("black")


def initialize_wandb(config, args):
    """Initialize Weights and Biases logging."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    # wandb 产物统一保存到数据盘 (优先 DSGN_SAVE_DIR), 避免系统盘被占满
    save_root = os.environ.get("DSGN_SAVE_DIR") or project_root
    wandb_dir = os.path.join(save_root, "wandb_logs")
    os.makedirs(wandb_dir, exist_ok=True)
    try:
        wandb.init(
            project=args.project,
            group=args.group,
            name=f"Exp_{args.models}",
            config=config,
            dir=wandb_dir,
        )
    except Exception:
        pass


def initialize_config_manager(config_file):
    """Load and return a configuration manager instance."""
    config_manager = ConfigManager()
    config_manager.load_config(config_file)
    return config_manager


def load_agent_factory(model_name, device, args):
    """Factory to create agents and load weights."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = project_root


    model_paths = {
        "DSGN": os.environ.get("DSGN_MODEL_PATH_DSGN") or os.path.join(
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
        "DSGN-WO-BETA": os.environ.get("DSGN_MODEL_PATH_DSGN_WO_BETA") or os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-WO-BETA/training_2026-01-31-05-55-34_vector"
            ),
        "DSGN-SCALAR": os.environ.get("DSGN_MODEL_PATH_DSGN_SCALAR") or os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-SCALAR/training_2026-09-01-00-00-00_scalar"
            ),
        "DSGN-SCALAR-WO-BETA": os.environ.get("DSGN_MODEL_PATH_DSGN_SCALAR_WO_BETA") or os.path.join(
            base_model_dir,
            "TrainedModels/DSGN-SCALAR-WO-BETA/training_2026-09-01-00-00-00_scalar"
            ),
        "Heuristic": None,
    }

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


def main_experiment_process(args):
    """Initialize agents and execute the standard experiment process.

    Args:
        args: Parsed command-line arguments.

    Returns:
        str: Path to the generated experiment directory.
    """
    global POST_CAPTURE_PATROL, EPISODE_BUDGET
    POST_CAPTURE_PATROL = args.post_capture_patrol
    EPISODE_BUDGET = args.episode_budget
    logger.info(
        f"Process {os.getpid()} - post_capture_patrol={POST_CAPTURE_PATROL}, "
        f"episode_budget={EPISODE_BUDGET}"
    )

    seed = 6
    exp_config_list = [
        "exp_config_small_scale.yaml",
        "exp_config_medium_scale.yaml",
        "exp_config_large_scale.yaml",
    ]

    if args.config == 0:
        config_type = "small"
    elif args.config == 1:
        config_type = "medium"
    elif args.config in {2, -1}:
        config_type = "large"
    else:
        raise ValueError(
            f"Unsupported config index: {args.config}. "
            "Use 0 (small), 1 (medium), 2 (large)."
        )

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(
        project_root, "config", f"{exp_config_list[args.config]}"
    )

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


    for model_name in args.models:
        logger.info(f"Initializing Model: {model_name}")
        env_inst = MarineEnv(seed)
        envs.append(env_inst)
        agent = load_agent_factory(model_name, device, args)
        agents.append(agent)
        evader_agents.append(ApfAgent(env_inst.evaders[0].a, env_inst.evaders[0].w))
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
        config_type
    )

    wandb.finish()
    return generated_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--device", type=str, default="cuda:0")
    parser.add_argument("-S", "--save_trajectory", action="store_true")
    parser.add_argument(
        "--save-eval-config", action="store_true",
        help="保存每回合环境配置快照 (eval_configs) 到 exp_results.json, 与轨迹保存解耦"
    )
    parser.add_argument(
        "--post-capture-patrol", action="store_true",
        help="接续式评估: 围捕完成后清零巡逻覆盖并继续运行到全覆盖或总回合上限"
    )
    parser.add_argument(
        "--episode-budget", type=int, default=1000,
        help="总回合步数上限 (接续式建议 1500, 为巡逻阶段保留观测窗口)"
    )
    parser.add_argument("-C", "--config", type=int, required=True)
    # parser.add_argument("-G", "--gating_type", type=str, default="unspecified")
    parser.add_argument("--project", type=str, default="baseline_experiment")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument(
        "--models", nargs="+", default=[
            "DSGN", "TransIQN", "DQN", "IQN",
            "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA",
            "AVERAGE", "RULE_BASED", "RANDOM",
            "Heuristic"
            ], help="Models: TERL IQN DQN MEAN"
    )

    args = parser.parse_args()
    main_experiment_process(args)
