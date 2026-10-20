from collections import defaultdict
import copy
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.animation as animation

from utils import logger
from config_manager import ConfigManager

# mpl.use('TkAgg')
mpl.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import environment.env as marine_env


class EnvVisualizer:
    """
    Environment visualization class for drawing and displaying robot trajectories and perception data in the environment.

    Args:
        seed (int): Random seed, defaults to 0.
        draw_envs (bool): Whether to draw only the environment, defaults to False.
        draw_traj (bool): Whether to draw only the final trajectories, defaults to False.
        video_plots (bool): Whether to generate video plots, defaults to False.
        plot_dist (bool): Whether to plot the return distribution for IQN agents, defaults to False.
        plot_q_values (bool): Whether to plot Q-values, defaults to False.
        dpi (int): Display DPI, defaults to 96.

    Attributes:
        env (MarineEnv): Environment object.
        fig (matplotlib.figure.Figure): Figure object.
        axis_graph (matplotlib.axes.Axes): Axes object for drawing the map.
        robots_plot (list): List of robot plot objects.
        robots_last_pos (list): List of robots' last positions.
        robots_traj_plot (list): List of robot trajectory plot objects.
        LiDAR_beams_plot (list): List of LiDAR beam plot objects.
        axis_title (matplotlib.axes.Axes): Axes object for the title.
        axis_action (matplotlib.axes.Axes): Axes object for action and steering state.
        axis_goal (matplotlib.axes.Axes): Axes object for goal measurements.
        axis_perception (matplotlib.axes.Axes): Axes object for perception output.
        axis_dvl (matplotlib.axes.Axes): Axes object for DVL measurements.
        axis_dist (list): List of axes objects for return distribution.
        axis_q_values (matplotlib.axes.Axes): Axes object for Q-values.
        episode_actions (list): Action sequence loaded from episode data.
        episode_actions_quantiles (None or list): Quantile data for actions.
        episode_actions_taus (None or list): Tau data for actions.
        plot_dist (bool): Whether to plot return distribution.
        plot_q_values (bool): Whether to plot Q-values.
        draw_envs (bool): Whether to draw the environment.
        draw_traj (bool): Whether to draw final trajectories.
        video_plots (bool): Whether to generate video plots.
        plots_save_dir (None or str): Directory to save video plots.
        dpi (int): Display DPI.
        agent_name (None or str): Name of the agent.
        agent (None or object): Agent object (IQN or DQN).
        configs (None or dict): Evaluation configurations.
        episodes (None or dict): Evaluation episodes to visualize.

    Methods:
        init_visualize(self, env_configs=None):
            Initialize visualization settings.
        plot_graph(self, axis):
            Draw the map of the current environment.
        plot_robots(self, axis, traj_color=None):
            Draw robot positions and trajectories.
        plot_action_and_steer_state(self, action):
            Draw action and steering state.
        plot_measurements(self, robot_idx, R_matrix=None):
            Draw perception data.
        plot_return_dist(self, action):
            Draw return distribution.
        plot_action_values(self, action):
            Draw Q-values.
        one_step(self, actions, robot_idx=0):
            Execute one step and update visualization.
        init_animation(self):
            Initialize animation.
        visualize_control(self, action_sequence, start_idx=0):
            Update robot states and display animation of the action sequence.
        load_env_config(self, episode_dict):
            Load environment configuration from a dictionary.
        load_env_config_from_eval_files(self, config_f, eval_f, eval_id, env_id):
            Load environment configuration from evaluation files.
        load_env_config_from_json_file(self, filename):
            Load environment configuration from a JSON file.
        play_eval_episode(self, eval_id, episode_id, colors, robot_ids=None):
            Play an evaluation episode.
        save_figure(self, filename):
            Save the figure to a file.
        play_episode(self, trajectories, colors, robot_ids=None, max_steps=None, start_step=0):
            Play an episode and display robot trajectories.
        draw_dist_plot(self, trajectories, robot_id, step_id, colors):
            Draw the action distribution plot for a specific step.
        action_data(self, robot_id):
            Get action data for a specific robot ID.
        draw_trajectory(self, trajectories, colors, name=None):
            Draw a trajectory plot.
        draw_video_plots(self, episode, save_dir, start_idx, agent):
            Generate video plots.
    """

    def __init__(
        self,
        seed: int = 0,
        draw_envs: bool = False,  # Mode 2: plot the environment
        draw_traj: bool = False,  # Mode 3: plot final trajectories given action sequences
        video_plots: bool = False,  # Mode 4: Generate plots for a video
        plot_dist: bool = False,  # If return distributions are needed (for IQN agent) in the video
        plot_q_values: bool = False,  # If Q values are needed in the video
        dpi: int = 150,  # Monitor DPI
        trajectory_min_alpha: float = 0.32,
        trajectory_max_alpha: float = 0.9,
        video_trajectory_tail_steps=None,
    ):
        """Initialize the instance."""
        self.env = marine_env.MarineEnv(seed)
        self.env.reset()
        self.fig = None  # figure for visualization
        self.axis_graph = None  # sub figure for the map
        self.robots_plot = []
        self.robots_last_pos = []
        self.robots_traj_plot = []
        self.current_visualization_step = 0
        self.current_visualization_max_steps = 1
        self.trajectory_min_alpha = trajectory_min_alpha
        self.trajectory_max_alpha = trajectory_max_alpha
        self.LiDAR_beams_plot = []
        self.axis_title = None  # sub figure for title
        self.axis_action = None  # sub figure for action command and steer data
        self.axis_goal = None  # sub figure for relative goal measurement
        self.axis_perception = None  # sub figure for perception output
        self.axis_dvl = None  # sub figure for DVL measurement
        self.axis_dist = []  # sub figure(s) for return distribution of actions
        self.axis_q_values = None  # sub figure for Q values of actions
        self.axis_gating_weights = None  # sub figure for gating weights visualization

        self.episode_actions = []  # action sequence load from episode data
        self.episode_actions_quantiles = None
        self.episode_actions_taus = None

        self.plot_dist = plot_dist  # draw return distribution of actions
        self.plot_q_values = plot_q_values  # draw Q values of actions
        self.draw_envs = draw_envs  # draw only the envs
        self.draw_traj = draw_traj  # draw only final trajectories
        self.video_plots = video_plots  # draw video plots
        self.plots_save_dir = None  # video plots save directory
        self.dpi = dpi  # monitor DPI
        self.agent_name = None  # agent name
        self.agent = None  # agent (IQN or DQN for plot data)

        self.configs = None  # evaluation configs
        self.episodes = None  # evaluation episodes to visualize

        self.config_vis = ConfigManager().get_instance()

    def init_visualize(self, env_configs=None):
        """
        Initialize visualization settings.
        """
        # initialize subplot for the map, robot state and sensor measurements
        if self.draw_envs:
            # Mode 2: plot the environment
            if env_configs is None:
                self.fig, self.axis_graph = plt.subplots(1, 1, figsize=(8, 8))
            else:
                num = len(env_configs)
                if num % 3 == 0:
                    self.fig, self.axis_graphs = plt.subplots(
                        int(num / 3), 3, figsize=(8 * 3, 8 * int(num / 3))
                    )
                else:
                    self.fig, self.axis_graphs = plt.subplots(
                        1, num, figsize=(8 * num, 8)
                    )
        elif self.draw_traj:
            if self.plot_dist:
                self.fig = plt.figure(figsize=(24, 16))
                spec = self.fig.add_gridspec(5, 6)

                self.axis_graph = self.fig.add_subplot(spec[:, :4])
                self.axis_perception = self.fig.add_subplot(spec[:2, 4:])
                self.axis_dist.append(self.fig.add_subplot(spec[2:, 4]))
                self.axis_dist.append(self.fig.add_subplot(spec[2:, 5]))
            else:
                # Mode 3: plot final trajectories given action sequences
                self.fig, self.axis_graph = plt.subplots(figsize=(16, 16))
        elif self.video_plots:
            # Mode 4: Generate 1080p video plots
            w = 1920
            h = 1080
            self.fig = plt.figure(figsize=(w / self.dpi, h / self.dpi), dpi=self.dpi)
            if self.agent_name in ["DSGN", "AVERAGE", "RULE_BASED", "RANDOM", "DSGN-WO-BETA", "DSGN-SCALAR", "DSGN-SCALAR-WO-BETA"]:
                spec = self.fig.add_gridspec(6, 12, hspace=0.85, wspace=0.45)
                self.axis_graph = self.fig.add_subplot(spec[:, :8])
                title_name = self.agent_name if self.agent_name != "DSGN-WO-BETA" else "DSGN (w/o beta)"
                if title_name == "DSGN-SCALAR": title_name = "DSGN (scalar gate)"
                if title_name == "DSGN-SCALAR-WO-BETA": title_name = "DSGN-scalar (w/o beta)"
                if title_name == "RULE_BASED": title_name = "Rule-based-gate"
                self.axis_graph.set_title(
                    f"{title_name} performance", fontweight="bold", fontsize=26, pad=10
                )
                self.axis_perception = self.fig.add_subplot(spec[0:3, 8:])
                self.axis_gating_weights = self.fig.add_subplot(spec[3:6, 8:])
            elif self.agent_name == "IQN":
                spec = self.fig.add_gridspec(6, 12, hspace=0.85, wspace=0.45)
                self.axis_graph = self.fig.add_subplot(spec[:, :8])
                self.axis_graph.set_title(
                    "IQN performance", fontweight="bold", fontsize=26, pad=10
                )
                self.axis_perception = self.fig.add_subplot(spec[0:3, 8:])
            elif self.agent_name == "TransIQN":
                spec = self.fig.add_gridspec(6, 12, hspace=0.85, wspace=0.45)
                self.axis_graph = self.fig.add_subplot(spec[:, :8])
                self.axis_graph.set_title(
                    "TransIQN performance", fontweight="bold", fontsize=26, pad=10
                )
                self.axis_perception = self.fig.add_subplot(spec[0:3, 8:])
            elif self.agent_name == "DQN":
                spec = self.fig.add_gridspec(6, 12, hspace=0.85, wspace=0.45)
                self.axis_graph = self.fig.add_subplot(spec[:, :8])
                self.axis_graph.set_title(
                    "DQN performance", fontweight="bold", fontsize=26, pad=10
                )
                self.axis_perception = self.fig.add_subplot(spec[0:3, 8:])
            elif self.agent_name == "Heuristic":
                spec = self.fig.add_gridspec(6, 12, hspace=0.85, wspace=0.45)
                self.axis_graph = self.fig.add_subplot(spec[:, :8])
                self.axis_graph.set_title(
                    "Heuristic performance", fontweight="bold", fontsize=26, pad=10
                )
                self.axis_perception = self.fig.add_subplot(spec[0:3, 8:])
        else:
            # Mode 1 (default): Display an episode
            # -----------------------------------------------------------

            # -----------------------------------------------------------
            self.fig = plt.figure(figsize=(25.6, 14.4))


            spec = self.fig.add_gridspec(6, 6, wspace=0.3, hspace=0.75)


            self.axis_graph = self.fig.add_subplot(spec[:, :4])


            self.axis_perception = self.fig.add_subplot(spec[0:2, 4:])


            self.axis_gating_weights = self.fig.add_subplot(spec[3:6, 4:])


        if self.draw_envs and env_configs is not None:
            for i, env_config in enumerate(env_configs):
                self.load_env_config(env_config)
                if len(env_configs) % 3 == 0:
                    self.plot_graph(self.axis_graphs[int(i / 3), i % 3])
                else:
                    self.plot_graph(self.axis_graphs[i])
        else:
            self.plot_graph(self.axis_graph)

    def plot_graph(self, axis):
        """
        Draw the environment map and initial robot positions.

        Args:
            axis (matplotlib.axes.Axes): Axes object for plotting.
        """
        # x_pos = list(np.linspace(0.0, self.env.width, 100))
        # y_pos = list(np.linspace(0.0, self.env.height, 100))
        expand_range = 0.0
        axis_range = np.array([-expand_range, 1.0 + expand_range])
        x_range = int(self.env.width) * axis_range
        y_range = int(self.env.height) * axis_range
        num_point = int(100 * (2 * expand_range + 1))
        x_pos = list(np.linspace(x_range[0], x_range[1], num_point))
        y_pos = list(np.linspace(y_range[0], y_range[1], num_point))
        # x_pos = list(np.linspace(-1*self.env.width , 2*self.env.width, 3*100))
        # y_pos = list(np.linspace(-1*self.env.height , 2*self.env.height, 3*100))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        speeds = np.zeros((len(x_pos), len(y_pos)))
        for m, x in enumerate(x_pos):
            for n, y in enumerate(y_pos):
                v = self.env.get_current_velocity(x, y)
                speed = np.clip(np.linalg.norm(v), 0.1, 10)
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
                speeds[n, m] = np.log(speed)

        cmap = cm.get_cmap("Blues")
        cmap = cmap(np.linspace(0, 1, 20))
        cmap = mpl.colors.ListedColormap(cmap[10:, :-1])

        axis.contourf(x_pos, y_pos, speeds, cmap=cmap)  # Draw filled contour plot
        # axis.quiver(pos_x, pos_y, arrow_x, arrow_y, width=0.001, scale_units='xy', scale=2.0)  # Draw quiver plot (vector field)

        for obs in self.env.obstacles:
            axis.add_patch(
                mpl.patches.Circle(
                    (float(obs.x), float(obs.y)), radius=obs.r, color="m"
                )
            )

        axis.set_aspect("equal")
        # axis.set_xlim([0.0, self.env.width])
        # axis.set_ylim([0.0, self.env.height])
        # Set axis range to 3 times the environment size
        # axis.set_xlim(-self.env.width, 2 * self.env.width)  # x-axis from -100m to 200m
        # axis.set_ylim(-self.env.height, 2 * self.env.height)  # y-axis from -100m to 200m
        axis.set_xlim(x_range[0], x_range[1])
        axis.set_ylim(y_range[0], y_range[1])
        axis.set_xticks([])
        axis.set_yticks([])

        for patrol_point in self.env.patrol_points:
            axis.scatter(
                patrol_point.position[0],
                patrol_point.position[1],
                color="r",
                s=5,
                marker="D",
            )

        # Plot start state of each robot
        for idx, robot in enumerate(self.env.pursuers + self.env.evaders):
            if not self.draw_envs:
                # Select color based on robot type
                if robot.robot_type == "pursuer":
                    color = (0.6, 0.7, 1.0)
                elif robot.robot_type == "evader":
                    color = (0.75, 0.6, 1.0)
                else:
                    color = "yellow"  # Fallback color

                # Draw the robot's starting position
                axis.scatter(
                    robot.start[0],
                    robot.start[1],
                    marker="o",
                    color=color,
                    s=3,
                    zorder=6,
                )
                # axis.text(robot.start[0] - 1, robot.start[1] + 1, str(robot.id), color=color, fontsize=15, zorder=8)

            # Record robot positions and trajectories
            self.robots_last_pos.append([])
            self.robots_traj_plot.append([])

        self.plot_robots(axis)

    def plot_robots(self, axis, traj_color=None, is_last=False, observation_id=-1):
        """
        Draw robot positions and trajectories.

        Args:
            axis (matplotlib.axes.Axes): Axes object for plotting.
            traj_color (list, optional): List of trajectory colors, defaults to None.
            is_last (bool, optional): Whether this is the last frame, defaults to False.
            observation_id (int, optional): ID of the robot to highlight, defaults to -1.
        """
        if not self.draw_envs:
            for robot_plot in self.robots_plot:
                robot_plot.remove()
            self.robots_plot.clear()

        robot_scale = 1.2
        for i, robot in enumerate(self.env.pursuers + self.env.evaders):
            # if robot.deactivated:
            #     continue

            d = np.array(
                [[0.5 * robot_scale * robot.length], [0.5 * robot_scale * robot.width]]
            )
            rot = np.array(
                [
                    [np.cos(robot.theta), -np.sin(robot.theta)],
                    [np.sin(robot.theta), np.cos(robot.theta)],
                ]
            )
            d_r = rot @ d
            xy = (robot.x - d_r[0, 0], robot.y - d_r[1, 0])

            angle_d = robot.theta / np.pi * 180

            # Select color based on robot type
            if robot.robot_type == "pursuer":
                robot_color = "lime"
            elif robot.robot_type == "evader":
                robot_color = "purple"
            else:
                robot_color = "yellow"  # Fallback color

            if self.draw_traj and robot.robot_type == "pursuer":
                pass
                # robot.check_capture_current_target(self.env.pursuers, self.env.evaders)
                # if robot.robot_type == "pursuer":
                #     robot_color = "lime" if robot.is_current_target_captured else 'r'
            else:
                # Draw robot velocity (add initial length to avoid being hidden by the robot plot)
                velocity_len = np.linalg.norm(robot.velocity)
                if velocity_len > 1e-6:
                    robot_r = 0.5 * np.linalg.norm(
                        np.array([robot.length, robot.width])
                    )
                    init_len = robot_scale * robot_r + 0.1
                    scaled_len = (velocity_len + init_len) / velocity_len
                    vx, vy = (
                        scaled_len * robot.velocity[0],
                        scaled_len * robot.velocity[1],
                    )
                else:
                    vx, vy = 0.0, 0.0

                self.robots_plot.append(
                    axis.quiver(
                        robot.x,
                        robot.y,
                        vx,
                        vy,
                        color="r",
                        width=0.005,
                        headlength=5,
                        headwidth=3,
                        scale_units="xy",
                        scale=1,
                    )
                )  # Draw a vector with an arrow on the axes

            # Draw detection range for pursuers
            if robot.robot_type == "pursuer":
                detection_circle = mpl.patches.Circle(
                    (robot.x, robot.y),
                    radius=20,
                    color="lightgreen",
                    alpha=0.055,
                    zorder=1,
                )
                self.robots_plot.append(axis.add_patch(detection_circle))

            # else:
            #     # Draw robot velocity (add initial length to avoid being hidden by the robot plot)
            #     robot_r = 0.5 * np.linalg.norm(np.array([robot.length, robot.width]))
            #     init_len = robot_scale * robot_r + 0.1
            #     velocity_len = np.linalg.norm(robot.velocity)
            #     scaled_len = (velocity_len + init_len) / velocity_len
            #     self.robots_plot.append(
            #         axis.quiver(robot.x, robot.y, scaled_len * robot.velocity[0], scaled_len * robot.velocity[1],
            #                     color="r", width=0.005, headlength=5, headwidth=3, scale_units='xy',
            #                     scale=1))  # Draw a vector with an arrow on the axes

            # Draw robot
            self.robots_plot.append(
                axis.add_patch(
                    mpl.patches.Rectangle(
                        xy,
                        robot_scale * robot.length,
                        robot_scale * robot.width,
                        color=robot_color,
                        angle=angle_d,
                        zorder=7,
                    )
                )
            )
            if robot.id == observation_id:

                text_obj = axis.text(
                    robot.x,
                    robot.y + robot.length * 1.5,
                    f"ID: {robot.id}",
                    color="white",
                    fontsize=16,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=20,
                )


                text_obj.set_path_effects(
                    [pe.withStroke(linewidth=3, foreground="black")]
                )

                self.robots_plot.append(text_obj)
            if robot.deactivated and robot.robot_type == "evader":
                self.robots_plot.append(
                    axis.scatter(
                        robot.x, robot.y, color="gold", marker="*", s=66, zorder=9
                    )
                )  # Draw a star
            if robot.deactivated and robot.robot_type == "pursuer" and not is_last:
                self.robots_plot.append(
                    axis.scatter(robot.x, robot.y, color="red", marker="x", s=66, zorder=9)
                )
            if not self.draw_envs:
                if self.robots_last_pos[i]:
                    progress = self.current_visualization_step / max(
                        1, self.current_visualization_max_steps - 1
                    )
                    traj_alpha = self.trajectory_min_alpha + progress * (
                        self.trajectory_max_alpha - self.trajectory_min_alpha
                    )
                    h = axis.plot(
                        (self.robots_last_pos[i][0], robot.x),
                        (self.robots_last_pos[i][1], robot.y),
                        color="tab:orange" if traj_color is None else traj_color[i],
                        linewidth=1.35,
                        alpha=traj_alpha,
                    )
                    self.robots_traj_plot[i].append(h)

                self.robots_last_pos[i] = [robot.x, robot.y]

    def plot_action_and_steer_state(self, action):
        """
        Draw action and steering state.

        Args:
            action (int): Action ID.
        """
        self.axis_action.clear()

        a, w = self.env.pursuers[0].action_list[action]

        if self.video_plots:
            self.axis_action.text(1, 3, "action", fontsize=25)
            self.axis_action.text(1, 2, f"a: {a:.2f}", fontsize=20)
            self.axis_action.text(1, 1, f"w: {w:.2f}", fontsize=20)
            self.axis_action.set_xlim([0, 2.5])
            self.axis_action.set_ylim([0, 4])
        else:
            x_pos = 0.15
            self.axis_action.text(
                x_pos, 6, "Steer actions", fontweight="bold", fontsize=15
            )
            self.axis_action.text(
                x_pos, 5, f"Acceleration (m/s^2): {a:.2f}", fontsize=15
            )
            self.axis_action.text(
                x_pos, 4, f"Angular velocity (rad/s): {w:.2f}", fontsize=15
            )

            # Robot steer state
            self.axis_action.text(
                x_pos, 2, "Steer states", fontweight="bold", fontsize=15
            )
            self.axis_action.text(
                x_pos,
                1,
                f"Forward speed (m/s): {self.env.pursuers[0].speed:.2f}",
                fontsize=15,
            )
            self.axis_action.text(
                x_pos,
                0,
                f"Orientation (rad): {self.env.pursuers[0].theta:.2f}",
                fontsize=15,
            )
            self.axis_action.set_ylim([-1, 7])

        self.remove_axis_ticks_and_spines(self.axis_action)

    def plot_gating_weights(self, gating_weights_history, current_step, window_size=None, episode_log=None):
        """Run plot gating weights."""
        ax = self.axis_gating_weights
        ax.clear()

        # -----------------------------

        # -----------------------------
        if window_size is not None:
            start_step = max(0, current_step - window_size)
            x_limit_max = max(start_step + window_size, current_step)
        else:
            start_step = 0

            x_limit_max = len(gating_weights_history) if len(gating_weights_history) > 0 else 100

        # -----------------------------

        # -----------------------------
        data_limit = min(current_step + 1, len(gating_weights_history))

        steps = []
        encircle_vals = []
        patrol_vals = []

        for t in range(start_step, data_limit):
            gw = gating_weights_history[t]
            if gw is not None:
                gw_flat = np.array(gw).flatten()
                if len(gw_flat) == 2:
                    steps.append(t)
                    encircle_vals.append(gw_flat[0])
                    patrol_vals.append(gw_flat[1])

        if not steps:
            ax.text(0.5, 0.5, "Waiting for data...", ha="center")
            ax.set_xlim([start_step, x_limit_max])
            return

        # -----------------------------

        # -----------------------------
        ax.plot(steps, encircle_vals, color="#FF6B6B", linewidth=2, label="Encircle")
        ax.plot(steps, patrol_vals, color="#4ECDC4", linewidth=2, label="Patrol")

        ax.scatter(steps[-1], encircle_vals[-1], color="#FF6B6B", s=40, zorder=5)
        ax.scatter(steps[-1], patrol_vals[-1], color="#4ECDC4", s=40, zorder=5)

        # -----------------------------

        # -----------------------------
        if episode_log and "milestones" in episode_log:
            milestones = episode_log["milestones"]


            valid_milestones = [
                (k, v) for k, v in milestones.items()
                if v is not None and isinstance(v, (int, float))
            ]


            sorted_events = sorted(valid_milestones, key=lambda item: item[1])


            label_map = {
                "patrol_start": "Patrol",
                "first_spawn": "Spawn",
                "first_contact": "Contact",
                "all_captured": "Captured",
                "patrol_resume_start": "Resume",
                "patrol_complete": "Complete",
            }
            grouped_events = defaultdict(list)
            for event_name, event_step in sorted_events:
                grouped_events[event_step].append(
                    label_map.get(event_name, event_name.replace("_", " ").title())
                )

            visible_events = [
                (event_step, labels)
                for event_step, labels in sorted(grouped_events.items())
                if start_step <= event_step <= x_limit_max and event_step <= current_step
            ]
            if len(visible_events) <= 1:
                y_offsets = [0.92]
            else:
                y_offsets = [
                    0.92 - 0.20 * i / (len(visible_events) - 1)
                    for i in range(len(visible_events))
                ]

            for i, (event_step, labels) in enumerate(visible_events):
                ax.axvline(x=event_step, color="#999999", linestyle=":", alpha=0.5, linewidth=1)

                label_text = "/".join(labels)
                if len(visible_events) == 1:
                    label_ha = "center"
                elif i == 0:
                    label_ha = "left"
                elif i == len(visible_events) - 1:
                    label_ha = "right"
                else:
                    label_ha = "center"
                y_pos = y_offsets[i]

                ax.text(
                    x=event_step,
                    y=y_pos,
                    s=label_text,
                    transform=ax.get_xaxis_transform(),
                    rotation=0,
                    va="center",
                    ha=label_ha,
                    fontsize=9,
                    color="#555555",
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
                    clip_on=False
                )

                ax.plot(event_step, 1.0, marker="v", color="#999999",
                        transform=ax.get_xaxis_transform(),
                        markersize=7, clip_on=True, alpha=0.78, zorder=6)
        # -----------------------------

        # -----------------------------
        if encircle_vals[-1] > patrol_vals[-1]:
            ax.set_facecolor("#FFF0F0")
            status_text = "MODE: ENCIRCLE"
            status_color = "#D32F2F"
        else:
            ax.set_facecolor("#F0FFFF")
            status_text = "MODE: PATROL"
            status_color = "#0097A7"

        ax.text(
            0.5, 1.08, status_text,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=status_color, va="bottom", ha="center",
            clip_on=False,
        )

        # -----------------------------

        # -----------------------------
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([start_step, x_limit_max])

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    def remove_axis_ticks_and_spines(self, axis):
        """
        Remove ticks and spines from the given axes.

        Args:
            axis (matplotlib.axes.Axes): Axes object from which to remove ticks and spines.
        """
        axis.set_xticks([])  # Remove ticks
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)  # Remove spines
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)

    def plot_measurements(self, robot_idx, R_matrix=None):
        """
        Draw perception data.

        Args:
            robot_idx (int): Index of the robot.
            R_matrix (np.ndarray, optional): Relationship matrix, defaults to None.
        """
        self.axis_perception.clear()

        rob = self.env.pursuers[robot_idx]

        # if rob.reach_goal:
        #     logger.info(f"robot {robot_idx} reached goal, no measurements are available!")
        #     return

        legend_size = 12
        font_size = 15

        rob.perception_output(
            obstacles=self.env.obstacles,
            pursuers=self.env.pursuers,
            evaders=self.env.evaders,
            patrol_points=self.env.patrol_points,
        )

        # Plot detected objects in the robot frame (rotate x-axis by 90 degrees upward in the plot)
        # range_c = 'g' if rob.cooperative else 'r'
        range_c = "g"
        self.axis_perception.add_patch(
            mpl.patches.Circle(
                (0, 0),
                rob.perception.range,
                facecolor=range_c,
                edgecolor=range_c,
                alpha=0.22,
                linewidth=1.2,
                zorder=0,
            )
        )
        # Add a circular patch to the axes to represent the robot's perception range
        # Plot self velocity (add initial length to avoid being hidden by the robot plot)
        robot_scale = 1.5
        robot_r = 0.5 * np.linalg.norm(np.array([rob.length, rob.width]))
        init_len = robot_scale * robot_r
        abs_velocity_r = rob.perception.observation["self"][0:2]
        velocity_len = np.linalg.norm(abs_velocity_r)

        if velocity_len > 1e-6:
            scaled_len = (velocity_len + init_len) / velocity_len
            vx = -scaled_len * abs_velocity_r[1]
            vy = scaled_len * abs_velocity_r[0]
        else:
            vx, vy = 0.0, 0.0

        self.axis_perception.quiver(
            0.0,
            0.0,
            vx,
            vy,
            color="r",
            width=0.008,
            headlength=5,
            headwidth=3,
            scale_units="xy",
            scale=1,
        )
        # Draw a vector with an arrow on the axes to represent the robot's velocity vector

        robot_c = "g"
        self.axis_perception.add_patch(
            mpl.patches.Rectangle(
                (-0.5 * robot_scale * rob.width, -0.5 * robot_scale * rob.length),
                robot_scale * rob.width,
                robot_scale * rob.length,
                color=robot_c,
            )
        )  # Draw a rectangular patch to represent the robot's position and size

        x_pos = 0
        y_pos = 0
        relation_pos = [[0.0, 0.0]]

        for i, obs in enumerate(rob.perception.observation["statics"]):
            # Rotate by 90 degrees
            self.axis_perception.add_patch(
                mpl.patches.Circle(
                    (-obs[1], obs[0]),
                    obs[2],
                    color="m",
                    zorder=1,
                )
            )
            relation_pos.append([-obs[1], obs[0]])
            # Include in observation info
            # self.axis_observation.text(x_pos,y_pos,f"position: ({obs[0]:.2f},{obs[1]:.2f}), radius: {obs[2]:.2f}")
            # y_pos += 1

        # self.axis_observation.text(x_pos,y_pos,"Static obstacles",fontweight="bold",fontsize=15)
        # y_pos += 2

        # if rob.cooperative:
        # for i,obj_history in enumerate(rob.perception.observation["dynamic"].values()):
        # Get the number of pursuers and evaders separately
        num_pursuers = len(rob.perception.observation["pursuers"])
        num_evaders = len(rob.perception.observation["evaders"])
        for i, obj in enumerate(
            rob.perception.observation["pursuers"]
            + rob.perception.observation["evaders"]
        ):
            if i < num_pursuers:
                color_velocity = "r"  # Arrow color for pursuers
                color_circle = "g"  # Circle color for pursuers
            else:
                color_velocity = "b"  # Arrow color for evaders (blue)
                color_circle = "orange"  # Circle color for evaders (yellow)
            # Plot the current position
            # pos = obj_history[-1][:2]

            # Plot velocity (rotate by 90 degrees)
            # velocity_len = np.linalg.norm(rob.velocity)
            # scaled_len = (velocity_len + init_len) / velocity_len
            # self.axis_perception.quiver(-obj[1], obj[0], -scaled_len * obj[3], scaled_len * obj[2],
            #                             color=color_velocity,
            #                             width=0.008, headlength=5, headwidth=3, scale_units='xy',
            #                             scale=1)  # Draw a vector with an arrow on the axes to represent the velocity or direction of an object (e.g., target or robot)
            velocity_len = np.linalg.norm(rob.velocity)
            if velocity_len > 1e-6:
                scaled_len = (velocity_len + init_len) / velocity_len
                vx = -scaled_len * obj[3]
                vy = scaled_len * obj[2]
            else:
                vx, vy = 0.0, 0.0

            self.axis_perception.quiver(
                -obj[1],
                obj[0],
                vx,
                vy,
                color=color_velocity,
                width=0.008,
                headlength=5,
                headwidth=3,
                scale_units="xy",
                scale=1,
            )
            # Draw a vector with an arrow on the axes to represent the velocity or direction of an object (e.g., target or robot)

            # Plot position (rotate by 90 degrees)
            self.axis_perception.add_patch(
                mpl.patches.Circle(
                    (-obj[1], obj[0]),
                    rob.detect_r,
                    color=color_circle,
                    zorder=1,
                )
            )  # Add a circular patch to the axes to represent the detection range of an object
            relation_pos.append([-obj[1], obj[0]])

            # Include history in observation info
            # self.axis_observation.text(x_pos,y_pos,f"position: ({obj[0]:.2f},{obj[1]:.2f}), velocity: ({obj[2]:.2f},{obj[3]:.2f})")
            # y_pos += 1

        # self.axis_observation.text(x_pos,y_pos,"Other Robots",fontweight="bold",fontsize=15)
        # y_pos += 2

        if R_matrix is not None:
            # Plot relation matrix
            length = len(R_matrix)
            assert len(relation_pos) == length, (
                "The number of objects does not match the size of the relation matrix"
            )
            for i in range(length):
                for j in range(length):
                    self.axis_perception.plot(
                        [relation_pos[i][0], relation_pos[j][0]],
                        [relation_pos[i][1], relation_pos[j][1]],
                        linewidth=2 * R_matrix[i][j],
                        color="k",
                        zorder=0,
                    )

        self.axis_perception.set_xlim(
            [-rob.perception.range - 1, rob.perception.range + 1]
        )
        self.axis_perception.set_ylim(
            [-rob.perception.range - 1, rob.perception.range + 1]
        )
        self.axis_perception.set_aspect("equal")  # Set aspect ratio to equal
        self.axis_perception.set_title(f"Pursuer {robot_idx}", fontsize=20, pad=8)  # Set title

        self.remove_axis_ticks_and_spines(self.axis_perception)

    def plot_return_dist(self, action):
        """
        Draw the return distribution.

        Args:
            action (dict): Action dictionary containing quantiles and CVaR values.
        """
        for axis in self.axis_dist:
            axis.clear()

        dist_interval = 1
        mean_bar = 0.35
        idx = 0

        xlim = [np.inf, -np.inf]
        for idx, cvar in enumerate(action["cvars"]):
            ylabelright = []

            quantiles = np.array(action["quantiles"][idx])

            q_means = np.mean(quantiles, axis=0)
            max_a = np.argmax(q_means)
            i = None
            for i, a in enumerate(self.env.pursuers[0].action_list):
                q_mean = q_means[i]

                ylabelright.append("\n".join([f"a: {a[0]:.2f}", f"w: {a[1]:.2f}"]))

                # ylabelright.append(f"mean: {q_mean:.2f}")

                self.axis_dist[idx].axhline(
                    i * dist_interval, color="black", linewidth=2.0, zorder=0
                )
                self.axis_dist[idx].scatter(
                    quantiles[:, i],
                    i * np.ones(len(quantiles[:, i])) * dist_interval,
                    color="g",
                    marker="x",
                    s=80,
                    linewidth=3,
                )
                self.axis_dist[idx].hlines(
                    y=i * dist_interval,
                    xmin=np.min(quantiles[:, i]),
                    xmax=np.max(quantiles[:, i]),
                    zorder=0,
                )
                if i == max_a:
                    self.axis_dist[idx].vlines(
                        q_mean,
                        ymin=i * dist_interval - mean_bar,
                        ymax=i * dist_interval + mean_bar,
                        color="red",
                        linewidth=5,
                    )
                else:
                    self.axis_dist[idx].vlines(
                        q_mean,
                        ymin=i * dist_interval - mean_bar,
                        ymax=i * dist_interval + mean_bar,
                        color="blue",
                        linewidth=3,
                    )

            self.axis_dist[idx].tick_params(axis="x", labelsize=15)
            self.axis_dist[idx].set_ylim([-1.0, i + 1])
            self.axis_dist[idx].set_yticks([])
            if idx == len(action["cvars"]) - 1:
                self.axis_dist[idx].set_yticks(range(0, i + 1))
                self.axis_dist[idx].yaxis.tick_right()
                self.axis_dist[idx].set_yticklabels(labels=ylabelright, fontsize=15)

            if len(action["cvars"]) > 1:
                if idx == 0:
                    self.axis_dist[idx].set_title(
                        "adaptive: " + r"$\phi$" + f" = {cvar:.2f}", fontsize=20
                    )
                else:
                    self.axis_dist[idx].set_title(
                        r"$\phi$" + f" = {cvar:.2f}", fontsize=20
                    )
            else:
                self.axis_dist[idx].set_title(r"$\phi$" + f" = {cvar:.2f}", fontsize=20)

            xlim[0] = min(xlim[0], np.min(quantiles) - 5)
            xlim[1] = max(xlim[1], np.max(quantiles) + 5)

        for idx, cvar in enumerate(action["cvars"]):
            self.axis_dist[idx].set_xlim(xlim)

    def plot_action_values(self, action):
        """
        Draw Q-values.

        Args:
            action (dict): Action dictionary containing Q-values.
        """
        self.axis_q_values.clear()

        dist_interval = 1
        mean_bar = 0.35
        ylabelright = []

        q_values = np.array(action["qvalues"])
        max_a = np.argmax(q_values)
        i = None
        for i, a in enumerate(self.env.pursuers[0].action_list):
            ylabelright.append("\n".join([f"a: {a[0]:.2f}", f"w: {a[1]:.2f}"]))
            self.axis_q_values.axhline(
                i * dist_interval, color="black", linewidth=2.0, zorder=0
            )
            if i == max_a:
                self.axis_q_values.vlines(
                    q_values[i],
                    ymin=i * dist_interval - mean_bar,
                    ymax=i * dist_interval + mean_bar,
                    color="red",
                    linewidth=8,
                )
            else:
                self.axis_q_values.vlines(
                    q_values[i],
                    ymin=i * dist_interval - mean_bar,
                    ymax=i * dist_interval + mean_bar,
                    color="blue",
                    linewidth=5,
                )

        self.axis_q_values.set_title("action values", fontsize=20)
        self.axis_q_values.tick_params(axis="x", labelsize=15)
        self.axis_q_values.set_ylim([-1.0, i + 1])
        self.axis_q_values.set_yticks(range(0, i + 1))
        self.axis_q_values.yaxis.tick_right()
        self.axis_q_values.set_yticklabels(labels=ylabelright, fontsize=15)
        self.axis_q_values.set_xlim([np.min(q_values) - 5, np.max(q_values) + 5])

    def one_step(self, actions, robot_idx=0):
        """
        Execute one step and update visualization.

        Args:
            actions (Tuple(list, list)): List of actions.
            robot_idx (int): Index of the robot, defaults to 0.
        """
        assert len(actions) == len(self.env.pursuers), (
            "Number of actions not equal number of robots!"
        )
        pursuer_actions, evader_actions = actions
        action = None
        for i, action in enumerate(pursuer_actions):
            rob = self.env.pursuers[i]
            current_velocity = self.env.get_current_velocity(rob.x, rob.y)
            rob.update_state(action, current_velocity)

        self.plot_robots(self.axis_graph)
        self.plot_measurements(robot_idx)
        # if not self.plot_dist and not self.plot_q_values:
        #     self.plot_action_and_steer_state(action["action"])

        if self.step % self.env.pursuers[0].N == 0:
            if self.plot_dist:
                self.plot_return_dist(action)
            elif self.plot_q_values:
                self.plot_action_values(action)

        self.step += 1

    def init_animation(self):
        """
        Initialize animation.
        """
        self.plot_robots(self.axis_graph)

    def visualize_control(self, action_sequence, start_idx=0):
        """
        Update robot states and display animation of the action sequence.

        Args:
            action_sequence (list): Sequence of actions.
            start_idx (int): Starting index, defaults to 0.
        """
        actions = []

        # Counter for updating distributions plot
        self.step = start_idx

        for i, a in enumerate(action_sequence):
            for _ in range(self.env.pursuers[0].N):
                actions.append(a)

        if self.video_plots:
            for i, action in enumerate(actions):
                self.one_step(action)
                self.fig.savefig(
                    f"{self.plots_save_dir}/step_{self.step}.png",
                    pad_inches=0.2,
                    dpi=self.dpi,
                )
        else:
            for i, action in enumerate(actions):
                self.one_step(action)
            plt.show()

    def load_env_config(self, episode_dict):
        """
        Load environment configuration from a dictionary.

        Args:
            episode_dict (dict): Dictionary containing environment configuration.
        """
        episode = copy.deepcopy(episode_dict)

        self.env.reset_with_eval_config(episode)

        if self.plot_dist:
            # Load action cvars, quantiles, and taus
            self.episode_actions_cvars = episode["robot"]["actions_cvars"]
            self.episode_actions_quantiles = episode["robot"]["actions_quantiles"]
            self.episode_actions_taus = episode["robot"]["actions_taus"]
        elif self.plot_q_values:
            # Load action values
            self.episode_actions_values = episode["robot"]["actions_values"]

    def load_env_config_from_eval_files(self, config_f, eval_f, eval_id, env_id):
        """
        Load environment configuration from evaluation files.

        Args:
            config_f (str): Path to the configuration file.
            eval_f (str): Path to the evaluation file.
            eval_id (int): Evaluation ID.
            env_id (int): Environment ID.
        """
        with open(config_f, "r") as f:
            episodes = json.load(f)
        episode = episodes[f"env_{env_id}"]
        eval_file = np.load(eval_f, allow_pickle=True)
        episode["robot"]["action_history"] = copy.deepcopy(
            eval_file["actions"][eval_id][env_id]
        )
        self.load_env_config(episode)

    def load_env_config_from_json_file(self, filename):
        """
        Load environment configuration from a JSON file.

        Args:
            filename (str): Path to the JSON file.
        """
        with open(filename, "r") as f:
            episode = json.load(f)
        self.load_env_config(episode)

    def load_eval_config_and_episode(self, config_file, eval_file):
        """
        Load evaluation configurations and episode data.

        Args:
            config_file (str): Path to the configuration file.
            eval_file (str): Path to the evaluation file.
        """
        with open(config_file, "r") as f:
            self.configs = json.load(f)

        self.episodes = np.load(eval_file, mmap_mode="r", allow_pickle=True)

    def play_eval_episode(self, eval_id, episode_id, colors, robot_ids=None):
        """
        Play an evaluation episode.

        Args:
            eval_id (int): Evaluation ID.
            episode_id (int): Episode ID.
            colors (list): List of colors.
            robot_ids (list, optional): List of robot IDs, defaults to None.
        """
        self.env.reset_with_eval_config(self.configs[episode_id])
        self.init_visualize()

        trajectories = self.episodes["trajectories"][eval_id][episode_id]

        self.play_episode(trajectories, colors, robot_ids)

    def save_figure(self, filename, save_path):
        """
        Save the figure to a file.

        Args:
            filename (str): Filename.
            save_path (str): Save path.
        """
        full_path = os.path.join(save_path, filename)
        if self.fig:
            _, ext = os.path.splitext(full_path)
            output_format = ext.lstrip(".").lower() or "pdf"
            self.fig.savefig(full_path, format=output_format)
            logger.info(f"Figure saved to {full_path}")
            if output_format == "pdf":
                svg_path = os.path.splitext(full_path)[0] + ".svg"
                with mpl.rc_context({"svg.fonttype": "none"}):
                    self.fig.savefig(svg_path, format="svg")
                logger.info(f"Figure saved to {svg_path}")
        else:
            logger.info("No figure to save.")
        plt.close()

    def play_episode(
        self,
        trajectories,
        colors,
        robot_ids=None,
        max_steps=None,
        start_step=0,
        save_video_path=None,
        save_image_path=None,
        episode_log=None
    ):
        """
        Play an episode, display robot trajectories, and optionally save video/image.
        """

        all_robots_meta = []

        for i, traj in enumerate(trajectories):

            plot_observation = False if robot_ids is None else i in robot_ids


            if isinstance(traj, dict):
                traj_data = traj.get("trajectory", traj)
                gating_weights = traj.get("gating_weights", None)
                deactivated_status = traj.get("deactivated", False)
            else:
                traj_data = traj
                gating_weights = None
                deactivated_status = False


            robot_meta = {
                "id": i,
                "type": traj['type'],
                "spawn_step": traj["spawn_step"],
                "traj_len": len(traj_data),
                "gating_weights": gating_weights,
                "deactivated": deactivated_status,
                "plot_observation": plot_observation,
                "init_state": traj_data[0],
            }
            all_robots_meta.append(robot_meta)


        event_schedule = defaultdict(list)
        for rob_meta in all_robots_meta:
            if rob_meta["spawn_step"] > 0:
                event_schedule[rob_meta["spawn_step"]].append({
                    "type": "spawn_robot",
                    "robot_id": rob_meta["id"],
                    "rob_type": rob_meta["type"],
                    "init_state": rob_meta["init_state"],
                })

        def spawn_dynamic_evader(env, rob_config, current_step):
            """Run spawn dynamic evader."""
            from environment.env import Evader
            init_state = rob_config["init_state"]
            id = rob_config["robot_id"]
            type = rob_config["type"]

            new_robot = Evader(id)

            new_robot.x = init_state[0]
            new_robot.y = init_state[1]
            new_robot.velocity = np.array([init_state[2], init_state[3]])
            new_robot.theta = init_state[4]
            new_robot.deactivated = False


            new_robot.spawn_step = current_step


            env.evaders.append(new_robot)
            env.num_evaders += 1

            logger.info(f"⚡ Event: Spawned Evader {id} at {init_state} (Step {current_step})")


        robot_info_map = {r["id"]: r for r in all_robots_meta}


        num_pursuers = len(self.env.pursuers)
        num_evaders = len(self.env.evaders)

        for _ in range(num_evaders):
             if len(all_robots_meta) > num_pursuers:
                all_robots_meta.pop(-1)


        sorted_robots_for_cam = sorted(all_robots_meta, key=lambda x: x["traj_len"])


        if sorted_robots_for_cam:
            sorted_robots_for_cam[-1]["plot_observation"] = True


        if max_steps is None:
            max_steps = sorted_robots_for_cam[-1]["traj_len"] if sorted_robots_for_cam else 0
        self.current_visualization_max_steps = max(1, max_steps)


        robots_to_follow = [r for r in sorted_robots_for_cam if r["plot_observation"]]


        writer = None
        if save_video_path:
            fps = 1.0 / 0.3
            metadata = dict(title='Exp Video', artist='Matplotlib')

            try:

                writer = animation.FFMpegWriter(
                    fps=fps, metadata=metadata, codec='libx264',
                    extra_args=['-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-crf', '18', '-preset', 'slow']
                )
            except Exception:

                logger.warning("libx264 not found, using mpeg4...")
                writer = animation.FFMpegWriter(
                    fps=fps, metadata=metadata, codec='mpeg4',
                    extra_args=['-q:v', '1', '-b:v', '8000k']
                )

            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            writer.setup(self.fig, save_video_path, dpi=self.dpi)
            logger.info(f"Recording video to {save_video_path}...")


        idx = 0
        current_robot_step = 0

        try:
            for i in range(max_steps):
                self.current_visualization_step = i


                if i in event_schedule:
                    for event in event_schedule[i]:
                        if event["type"] == "spawn_robot":
                            spawn_dynamic_evader(self.env, event, i)


                            self.robots_last_pos.append([])
                            self.robots_traj_plot.append([])
                            assert len(self.robots_last_pos) == len(self.robots_traj_plot) == len(self.env.pursuers) + len(self.env.evaders), "Robots count mismatch after spawning evader!"

                if not robots_to_follow: break


                if i >= robots_to_follow[idx]["traj_len"]:
                    current_robot_step = 0
                    idx += 1
                    if idx >= len(robots_to_follow): break


                current_obs_id = robots_to_follow[idx]["id"]


                self.plot_robots(self.axis_graph, colors, observation_id=current_obs_id)
                self.plot_measurements(current_obs_id)


                if (self.axis_gating_weights is not None
                    and robots_to_follow[idx].get("gating_weights") is not None):
                    self.plot_gating_weights(robots_to_follow[idx]["gating_weights"], i, episode_log=episode_log)


                for j, rob in enumerate(self.env.pursuers + self.env.evaders):
                    if rob.deactivated:
                        continue


                    if j >= len(trajectories):
                        continue


                    traj_source = trajectories[j]
                    traj_data = traj_source.get("trajectory", traj_source) if isinstance(traj_source, dict) else traj_source


                    cur_step = i
                    if rob.spawn_step > 0:
                        cur_step -= rob.spawn_step


                    if cur_step < len(traj_data):
                        rob.x = traj_data[cur_step][0]
                        rob.y = traj_data[cur_step][1]
                        rob.theta = traj_data[cur_step][2]
                        rob.speed = traj_data[cur_step][3]
                        rob.velocity = np.array(traj_data[cur_step][4:])


                    if j in robot_info_map:
                        deactivated_info = robot_info_map[j]["deactivated"]


                        if isinstance(deactivated_info, list):
                            if cur_step < len(deactivated_info):
                                rob.deactivated = deactivated_info[cur_step]


                all_dead = all([rob.deactivated for rob in self.env.pursuers + self.env.evaders])
                if all_dead and i < 1000:
                    self.plot_robots(self.axis_graph, colors, is_last=True)
                    if writer: writer.grab_frame()
                    break

                if writer:
                    writer.grab_frame()
                    if i % 50 == 0:
                        print(f"Recording frame {i}/{max_steps}", end='\r')

        except Exception as e:
            logger.error(f"Error during animation: {e}")
            import traceback
            traceback.print_exc()
        finally:

            if writer:
                writer.finish()
                logger.info(f"\nVideo saved: {save_video_path}")

            if save_image_path:
                self.save_figure(os.path.basename(save_image_path), os.path.dirname(save_image_path))


    def draw_dist_plot(self, trajectories, robot_id, step_id, colors):
        """
        Draw the action distribution plot for a specific step.

        Args:
            trajectories (list): List of trajectories.
            robot_id (int): Robot ID.
            step_id (int): Step ID.
            colors (list): List of colors.
        """
        self.init_visualize()

        for i in range(step_id + 1):
            self.plot_robots(self.axis_graph, traj_color=colors)
            for j, rob in enumerate(self.env.pursuers):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i + 1][0]
                rob.y = trajectories[j][i + 1][1]
                rob.theta = trajectories[j][i + 1][2]
                rob.speed = trajectories[j][i + 1][3]
                rob.velocity = np.array(trajectories[j][i + 1][4:])
                if i + 1 == len(trajectories[j]) - 1:
                    rob.deactivated = True

        # Plot observation
        self.plot_measurements(robot_id)

        action = self.action_data(robot_id)
        self.plot_return_dist(action)

        self.fig.savefig("IQN_dist_plot.png", bbox_inches="tight")

    def action_data(self, robot_id):
        """
        Get action data for a specific robot ID.

        Args:
            robot_id (int): Robot ID.

        Returns:
            dict | int: Action data dictionary.
        """
        rob = self.env.pursuers[robot_id]
        observation, _ = rob.perception_output(
            self.env.obstacles, self.env.pursuers, self.env.evaders
        )

        action = None
        if self.agent_name == "adaptive_IQN":
            # Compute total distribution and adaptive CVaR distribution
            a_cvar, quantiles_cvar, _, cvar = self.agent.act_adaptive(observation)
            a_greedy, quantiles_greedy, _ = self.agent.act(observation)

            action = dict(
                action=[a_cvar, a_greedy],
                cvars=[cvar, 1.0],
                quantiles=[quantiles_cvar[0], quantiles_greedy[0]],
            )
        elif self.agent_name == "IQN":
            a_greedy, quantiles_greedy, _ = self.agent.act(observation)

            action = dict(
                action=[a_greedy], cvars=[1.0], quantiles=[quantiles_greedy[0]]
            )
        elif self.agent_name == "DQN":
            a, qvalues = self.agent.act_dqn(observation)

            action = dict(action=a, qvalues=qvalues[0])
        elif self.agent_name == "APF" or self.agent_name == "RVO":
            action = self.agent.act(observation)
        if action is None:
            raise ValueError("Unknown agent name!")
        return action

    def draw_trajectory(self, trajectories, colors, name=None):
        """
        Draw a trajectory plot.

        Args:
            trajectories (list): List of trajectories.
            colors (list): List of colors.
            name (str): Filename.
        """
        # Used in Mode 3
        self.init_visualize()

        # Select a robot that is active until the end of the episode
        robot_id = 0
        max_length = 0
        for i, traj in enumerate(trajectories):
            logger.info("rob: ", i, " len: ", len(traj))
            if len(traj) > max_length:
                robot_id = i
                max_length = len(traj)

        logger.info("\n")

        for i in range(len(trajectories[robot_id]) - 1):
            self.plot_robots(self.axis_graph, traj_color=colors)
            for j, rob in enumerate(self.env.pursuers):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i + 1][0]
                rob.y = trajectories[j][i + 1][1]
                rob.theta = trajectories[j][i + 1][2]
                rob.speed = trajectories[j][i + 1][3]
                rob.velocity = np.array(trajectories[j][i + 1][4:])
                if i + 1 == len(trajectories[j]) - 1:
                    rob.deactivated = True

        # for robot_plot in self.robots_plot:
        #     robot_plot.remove()
        # self.robots_plot.clear()

        fig_name = "trajectory_test.png" if name is None else f"trajectory_{name}.png"
        self.fig.savefig(fig_name, bbox_inches="tight")


    def draw_video_plots(self, trajectories, colors, save_dir, start_idx=0):

        """Draw the auxiliary plots used in video rendering."""
        pass
