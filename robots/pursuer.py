import copy
from typing import List, Tuple, Optional
import time

import numpy as np
import wandb

import utils
from config_manager import ConfigManager
from robots.evader import Evader
from robots.perception import Perception
from robots.robot import Robot
from utils import logger as logger


class Pursuer(Robot):
    """
    Pursuer class, inheriting from the Robot class, representing a pursuer in a pursuit-evasion scenario.

    Attributes:
        robot_type (str): Robot type, set to 'pursuer'.
        max_speed (float): Maximum speed.
        perception (Perception): Perception object, set as a pursuer type.
        captured_evaderId_list (List[int]): List of evader IDs captured by this pursuer.
        is_current_target_captured (bool): Whether the current tracking target has been captured.
        distance_capture (float): Maximum distance allowed for a successful capture.
        angle_capture (float): Minimum required encirclement angle for capture.
    """

    def __init__(self, index, *, enable_logging: bool = False):
        """
        Initializes a pursuer instance.

        Args:
            index (int): Pursuer index.
        """
        super().__init__(index)
        # Pursuer-specific attributes
        # Overriding base class attributes
        self.pur_config = ConfigManager.get_instance()
        self.robot_type = "pursuer"
        self.capture_event_count = 0

        self.max_speed = self.pur_config.get("pursuer.max_speed")
        self.a = self.pur_config.get("pursuer.a")
        self.w = self.pur_config.get("pursuer.w")
        self.perception = Perception(is_evader=False)

        self.captured_evaderId_list: List[int] = []  # List of captured evaders
        self.is_current_target_captured = (
            False  # Whether the current target has been captured
        )
        self.is_pursuing = False  # Whether the pursuer is actively pursuing

        self.distance_capture = self.pur_config.get(
            "env.capture_distance", default=8.0
        )  # Capture range threshold
        self.angle_capture = np.pi  # Minimum encirclement angle required for capture

        # Precomputed values
        self.compute_k()  # Compute and update water resistance coefficient
        self.compute_actions()  # Compute and update action list

        # State dimensions
        self.num_self_state = 4
        self.num_static_state = 25
        self.num_pursuer_state = 35
        self.num_evader_state = 7 * self.perception.max_evader_num
        self.num_patrol_point_state = 96  # 16 points * 6 features

        # Feature dimensions
        self.num_self_feature = 4
        self.num_static_feature = 5
        self.num_pursuer_feature = 7
        self.num_evader_feature = 7
        self.num_patrol_point_feature = 6

        self.enable_logging = enable_logging  # Logging flag

    def find_related_evader(self, evaders: List[Evader]):
        """
        Find evaders within the encirclement distance.

        Args:
            evaders (List[Evader]): List of all evaders.

        Returns:
            List[Evader]: List of evaders within the capture range, sorted by distance.
        """

        self_position = np.array([self.x, self.y])

        related_evaders = sorted(
            [
                e
                for e in evaders
                if np.linalg.norm(np.array([e.x, e.y]) - self_position)
                < self.distance_capture
                and not e.deactivated
            ],
            key=lambda e: np.linalg.norm(np.array([e.x, e.y]) - self_position),
        )
        return related_evaders

    def check_capture_current_target(
        self, pursuers, evaders: List[Evader]
    ) -> Tuple[bool, List[float], int, Optional[int], Optional[dict]]:
        """Check whether the current target has been captured."""
        # Configuration parameters
        MAX_ANGLE_GAP = np.pi  # Maximum allowable angle gap (180 degrees)
        MAX_ANGLE_RATIO = (
            3.0  # Maximum angle should not exceed 3 times the minimum angle
        )
        MIN_PURSUERS = 2  # Minimum number of pursuers required besides itself (at least 3 in total)

        self_position = np.array([self.x, self.y])

        # Check each evader
        for evader in evaders:
            if evader.deactivated:
                continue

            evader_position = np.array([evader.x, evader.y])

            # Check distance from the evader
            if np.linalg.norm(evader_position - self_position) > self.distance_capture:
                continue


            active_pursuers = [
                pursuers[p_id]
                for p_id in self.perception.observed_pursuers
                if not pursuers[p_id].deactivated
            ]
            if len(active_pursuers) < MIN_PURSUERS:
                continue


            pursuer_positions = np.array([[p.x, p.y] for p in active_pursuers])
            distances_to_evader = np.linalg.norm(
                pursuer_positions - evader_position, axis=1
            )

            # Get all pursuers within capture distance
            nearby_pursuers = [
                p
                for i, p in enumerate(active_pursuers)
                if distances_to_evader[i] < self.distance_capture
            ]

            # Check if the number of pursuers meets the minimum requirement
            if len(nearby_pursuers) < MIN_PURSUERS:
                continue


            involved_pursuers = nearby_pursuers + [self]


            adjacent_angles, max_angle = self.calculate_encirclement_geometry(involved_pursuers, evader_position)
            # ========================================

            if not adjacent_angles:
                continue

            min_angle = min(adjacent_angles)

            # Validate encirclement:


            if max_angle > MAX_ANGLE_GAP or max_angle > min_angle * MAX_ANGLE_RATIO:

                continue

            # ==========================================

            # ==========================================
            self.is_current_target_captured = True
            self.captured_evaderId_list.append(evader.id)


            # logger.info(

            # )

            capture_info = {
                "event": "capture",
                "leader_id": self.id,
                "evader_id": evader.id,
                "team_size": len(nearby_pursuers) + 1,
                "max_angle_gap": max_angle,
                "timestamp": time.time(),
            }

            self.capture_event_count += 1
            return True, adjacent_angles, len(nearby_pursuers), evader.id, capture_info

        # No capturable evader found
        return False, [], 0, None, None

    @staticmethod
    def calculate_encirclement_geometry(active_pursuers, evader_pos):
        """Compute the encirclement geometry around a target."""
        if not active_pursuers:
            return [], 0.0

        # Compute angles relative to evader
        pursuer_angles = [
            np.arctan2(p.y - evader_pos[1], p.x - evader_pos[0]) % (2 * np.pi)
            for p in active_pursuers
        ]
        pursuer_angles.sort()

        # Compute angles between adjacent pursuers
        adjacent_angles = [
            (pursuer_angles[(i + 1) % len(pursuer_angles)] - pursuer_angles[i])
            % (2 * np.pi)
            for i in range(len(pursuer_angles))
        ]

        max_angle_gap = max(adjacent_angles) if adjacent_angles else 0.0

        return adjacent_angles, max_angle_gap

    # -------------------------------------------------------------------------
    # Perception Logic (Core)
    # -------------------------------------------------------------------------

    def perception_output(
        self,
        obstacles,
        pursuers,
        evaders,
        in_robot_frame=True,
        patrol_points=None,
        current_step: int = None,
    ):
        """Build the perception output for the pursuer."""
        if self.deactivated:
            return None, self.collision

        # 1. Clear previous observations
        self.perception.observation["self"].clear()
        self.perception.observation["statics"].clear()
        self.perception.observation["evaders"].clear()
        self.perception.observation["pursuers"].clear()
        self.perception.observation["patrol_points"].clear()
        self.perception.observation["masks"].clear()
        self.perception.observation["types"].clear()
        self.perception.observed_pursuers.clear()
        self.perception.observed_evaders.clear()
        self.perception.observed_obstacles.clear()

        # ----------------------------------------------------------
        # A. Self State
        # ----------------------------------------------------------
        if in_robot_frame:
            # Velocity relative to seabed in robot's frame
            abs_velocity_r = self.project_to_robot_frame(self.velocity, is_vector=True)
            # Obstacle distance logic...
            if obstacles:
                # Extract obstacle coordinates as numpy array
                obstacle_positions = np.array(
                    [[obs.x, obs.y] for obs in obstacles]
                )  # Shape: (num_obstacles, 2)
                self_position = np.array([self.x, self.y])  # Shape: (2,)

                # Compute Euclidean distances between all obstacles and robot
                distances = np.linalg.norm(
                    obstacle_positions - self_position, axis=1
                )  # Shape: (num_obstacles,)

                # Find nearest obstacle distance, return perception range if beyond detection
                min_distance = (
                    np.min(distances)
                    if np.any(distances < self.perception.range)
                    else self.perception.range
                )
            else:
                # Return max perception range when no obstacles
                min_distance = self.perception.range

            self.perception.observation["self"] = (
                list(abs_velocity_r) + [min_distance] + [self.is_pursuing]
            )
            self.perception.observation["masks"].append(True)
            assert len(self.perception.observation["self"]) == self.num_self_state

        # ----------------------------------------------------------
        # B. Pursuers Perception
        # ----------------------------------------------------------
        valid_teammate_positions = []
        for k, pursuer in enumerate(pursuers):
            if pursuer is self or pursuer.deactivated:
                continue

            valid_teammate_positions.append(np.array([pursuer.x, pursuer.y]))

            if not self.check_detection(pursuer.x, pursuer.y, pursuer.detect_r):
                continue

            self.perception.observed_pursuers.append(pursuer.id)
            if not self.collision:
                self.check_collision(pursuer.x, pursuer.y, pursuer.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(
                    np.array([pursuer.x, pursuer.y]), False
                )
                v_r = self.project_to_robot_frame(pursuer.velocity)
                dis = np.linalg.norm(pos_r)
                angle = np.arctan2(pos_r[1], pos_r[0])

                pursuing_signal = pursuer.is_pursuing
                new_obs = list(
                    np.concatenate((pos_r, v_r, [dis, angle, pursuing_signal]))
                )
                self.perception.observation["pursuers"].append(new_obs)

        # Pursuer Masks
        p_count = len(self.perception.observation["pursuers"])
        p_masks = [True] * min(p_count, self.perception.max_pursuer_num)
        if p_count < self.perception.max_pursuer_num:
            p_masks.extend([False] * (self.perception.max_pursuer_num - p_count))
        self.perception.observation["masks"].extend(
            p_masks[: self.perception.max_pursuer_num]
        )

        # ----------------------------------------------------------
        # C. Evaders Perception (Local Detection)
        # ----------------------------------------------------------
        for j, evader in enumerate(evaders):
            if evader.deactivated:
                continue

            # Local FOV detection
            if not self.check_detection(evader.x, evader.y, evader.detect_r):
                continue

            self.perception.observed_evaders.append(evader.id)
            if not self.collision:
                self.check_collision(evader.x, evader.y, evader.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(
                    np.array([evader.x, evader.y]), False
                )
                v_r = self.project_to_robot_frame(evader.velocity)
                dis = np.linalg.norm(pos_r)
                pos_angle = np.arctan2(pos_r[1], pos_r[0])
                heading_err = utils.calculate_angle_from_components(
                    float(v_r[0]), float(v_r[1])
                )
                new_obs = (
                    list(
                        np.concatenate(
                            (
                                pos_r,
                                v_r,
                            )
                        )
                    )
                    + [dis, pos_angle]
                    + heading_err
                )
                self.perception.observation["evaders"].append(new_obs)

        # Evader Masks

        e_count = len(self.perception.observation["evaders"])
        e_masks = [True] * min(e_count, self.perception.max_evader_num)
        if e_count < self.perception.max_evader_num:
            e_masks.extend([False] * (self.perception.max_evader_num - e_count))
        self.perception.observation["masks"].extend(
            e_masks[: self.perception.max_evader_num]
        )

        # ----------------------------------------------------------
        # D. Patrol Points Perception
        # ----------------------------------------------------------
        if patrol_points is not None:
            for i, patrol_point in enumerate(patrol_points):
                pp_world_pos = np.array(
                    [patrol_point.position[0], patrol_point.position[1]]
                )

                # Calculate min_dist to any teammate (for coordination features)
                if valid_teammate_positions:
                    dists = [
                        np.linalg.norm(t_pos - pp_world_pos)
                        for t_pos in valid_teammate_positions
                    ]
                    min_dist = min(dists)
                else:
                    min_dist = 999.0

                if in_robot_frame:
                    pp_x, pp_y = float(patrol_point.position[0]), float(
                        patrol_point.position[1]
                    )


                    pos_r = self.project_to_robot_frame(np.array([pp_x, pp_y]), False)
                    dis = np.linalg.norm(pos_r)
                    pos_angle = np.arctan2(pos_r[1], pos_r[0])


                    norm_idleness = patrol_point.get_norm_idleness(current_step)


                    self.perception.observation["patrol_points"].append(
                        [pos_r[0], pos_r[1], dis, pos_angle, norm_idleness, min_dist]
                    )

        # Patrol Masks
        pp_count = len(self.perception.observation["patrol_points"])
        pp_masks = [True] * min(pp_count, self.perception.max_patrol_point_num)
        if pp_count < self.perception.max_patrol_point_num:
            pp_masks.extend([False] * (self.perception.max_patrol_point_num - pp_count))
        self.perception.observation["masks"].extend(
            pp_masks[: self.perception.max_patrol_point_num]
        )

        # ----------------------------------------------------------
        # E. Static Obstacles Perception
        # ----------------------------------------------------------
        for i, obs in enumerate(obstacles):
            if not self.check_detection(obs.x, obs.y, obs.r):
                continue
            self.perception.observed_obstacles.append(i)
            if not self.collision:
                self.check_collision(obs.x, obs.y, obs.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([obs.x, obs.y]), False)
                dis = np.linalg.norm(pos_r)
                pos_angle = np.arctan2(pos_r[1], pos_r[0])
                self.perception.observation["statics"].append(
                    [pos_r[0], pos_r[1], obs.r, dis, pos_angle]
                )

        # Static Masks
        s_count = len(self.perception.observation["statics"])
        s_masks = [True] * min(s_count, self.perception.max_obstacle_num)
        if s_count < self.perception.max_obstacle_num:
            s_masks.extend([False] * (self.perception.max_obstacle_num - s_count))
        self.perception.observation["masks"].extend(
            s_masks[: self.perception.max_obstacle_num]
        )

        # ----------------------------------------------------------
        # F. Types & Masks Construction
        # ----------------------------------------------------------

        # Masks Order: Self(1) -> Pursuers(N) -> Evaders(M) -> Patrol(K) -> Statics(L)

        self.perception.observation["types"].extend(
            [0]  # Self
            + [1] * self.perception.max_pursuer_num  # Pursuers
            + [2] * self.perception.max_evader_num  # Evaders
            + [3] * self.perception.max_patrol_point_num  # Patrol points
            + [4] * self.perception.max_obstacle_num  # Statics
        )


        total_len = (
            1
            + self.perception.max_pursuer_num
            + self.perception.max_evader_num
            + self.perception.max_patrol_point_num
            + self.perception.max_obstacle_num
        )

        assert (
            len(self.perception.observation["masks"]) == total_len
        ), f"Masks length mismatch! Expected {total_len}, got {len(self.perception.observation['masks'])}"

        assert len(self.perception.observation["types"]) == len(
            self.perception.observation["masks"]
        ), f"Types vs Masks mismatch! Types: {len(self.perception.observation['types'])}, Masks: {len(self.perception.observation['masks'])}"

        # ----------------------------------------------------------
        # G. Sorting and Formatting
        # ----------------------------------------------------------
        self_state = copy.deepcopy(self.perception.observation["self"])
        evader_obs = self.copy_sort(
            self.perception.max_evader_num, "evaders", in_robot_frame
        )
        pursuer_obs = self.copy_sort(
            self.perception.max_pursuer_num, "pursuers", in_robot_frame
        )
        patrol_obs = self.copy_sort(
            self.perception.max_patrol_point_num, "patrol_points", in_robot_frame
        )
        static_obs = self.copy_sort(
            self.perception.max_obstacle_num, "statics", in_robot_frame
        )

        observation_dict = {
            "self": np.array(self_state).reshape(self.num_self_feature),
            "pursuers": np.array(pursuer_obs).reshape(-1, self.num_pursuer_feature),
            "evaders": np.array(evader_obs).reshape(-1, self.num_evader_feature),
            "obstacles": np.array(static_obs).reshape(-1, self.num_static_feature),
            "patrol_points": np.array(patrol_obs).reshape(
                -1, self.num_patrol_point_feature
            ),
            "masks": np.array(self.perception.observation["masks"]),
            "types": np.array(self.perception.observation["types"]),
        }

        return observation_dict, self.collision

    def update_aggregated_evader_perception(
        self,
        aggregated_evader_ids: List[int],
        evaders: List[Evader],
        original_observation_dict=None,
        in_robot_frame=True,
    ):
        """
        Update the aggregated evader perception based on the list of observed evader IDs.
        Args:
            in_robot_frame: Whether the perception is in the robot's coordinate frame.
            aggregated_evader_ids (List[int]): List of evader IDs observed by the pursuer.
            evaders (List[Evader]): List of all evaders in the environment.
        """
        original_observed_evader_ids = self.perception.observed_evaders.copy()


        self.perception.observation["evaders"].clear()
        self.perception.observed_evaders.clear()


        for evader in evaders:
            if evader.id in aggregated_evader_ids:

                self.perception.observed_evaders.append(evader.id)

                if in_robot_frame:
                    pos_r = self.project_to_robot_frame(
                        np.array([evader.x, evader.y]), False
                    )
                    v_r = self.project_to_robot_frame(evader.velocity)
                    dis = np.linalg.norm(pos_r)
                    pos_angle = np.arctan2(pos_r[1], pos_r[0])
                    heading_angle_error = utils.calculate_angle_from_components(
                        float(v_r[0]), float(v_r[1])
                    )
                    new_obs = (
                        list(
                            np.concatenate(
                                (
                                    pos_r,
                                    v_r,
                                )
                            )
                        )
                        + [dis, pos_angle]
                        + heading_angle_error
                    )
                    self.perception.observation["evaders"].append(new_obs)


        e_count = len(self.perception.observation["evaders"])
        e_masks = [True] * min(e_count, self.perception.max_evader_num)
        if e_count < self.perception.max_evader_num:
            e_masks.extend([False] * (self.perception.max_evader_num - e_count))


        evader_obs = self.copy_sort(
            self.perception.max_evader_num, "evaders", in_robot_frame
        )


        original_observation_dict["evaders"] = np.array(evader_obs).reshape(
            -1, self.num_evader_feature
        )


        start_idx = 1 + self.perception.max_pursuer_num
        end_idx = start_idx + self.perception.max_evader_num

        original_observation_dict["masks"][start_idx:end_idx] = e_masks[
            : self.perception.max_evader_num
        ]

        return original_observation_dict

    def log_collisions(self, collision_entities: Tuple[str, int]):
        """
        Log collision events for the pursuer.

        Args:
            collision_entities (Tuple[str, int]): Tuple containing the type of entity collided with and its ID.
        """
        entity_type, entity_id = collision_entities
        if self.enable_logging:
            logger.warning(
                f"💥 Collision detected! Pursuer {self.id} collided with {entity_type} {entity_id}.")
