import numpy as np
import copy
from config_manager import ConfigManager


class HeuristicAgent:
    """
    Decentralized Heuristic Baseline (FSM-APF) for Simultaneous Patrol and Encirclement.

    Key Mechanisms:
    1. Task Allocation: Local Inhibition based on distance ranking (Decentralized & Reactive).
    2. Exploration (SEARCH): Utility-based heuristic maximizing spatiotemporal coverage (Idleness).
    3. Exploitation (ENCIRCLE): Centripetal attraction + Tangential (Vortex) repulsion to
       ensure implicit structural convergence and uniform encirclement.
    4. Collision Avoidance & Kinematics: Mixed APF with Position Repulsion and Velocity Damping,
       including piecewise target soft-landing and non-target dynamic avoidance.
    """

    def __init__(self, device="cpu", model_name="Heuristic"):
        """
        Args:
            a_space (list): Available linear accelerations.
            w_space (list): Available angular velocities.
        """

        self.config = ConfigManager.get_instance()

        self.a = self.config.get("pursuer.a")
        self.w = self.config.get("pursuer.w")

        self.device = device  # Not used, but kept for consistency with other agents
        self.model_name = model_name
        self.GAMMA = 0.99

        # --- APF Physical Parameters ---
        self.m = 500.0  # Virtual mass for kinematic mapping

        # 1. Search/Patrol Parameters
        self.k_att_patrol = 3.0  # Attraction to patrol nodes
        self.k_rep_agent_patrol = 2000.0  # Teammate repulsion (spatial dispersion)
        self.d0_agent_patrol = 12.0  # Keep teammates far away during exploration

        # 2. Encirclement Parameters (Target locked)
        self.k_att_encircle = 8.0  # Centripetal attraction to target evader
        self.k_tan_encircle = 1500.0  # Tangential/Vortex force for uniform distribution
        self.k_rep_agent_encircle = 800.0  # Collision avoidance during encirclement
        self.d0_agent_encircle = 4.0  # Safe distance between teammates


        self.k_v_agent = 3.0
        self.k_v_obs = 5.0
        self.k_v_target_brake = 4.0


        self.k_rep_obs = 4000.0
        self.d0_obs = 6.0
        self.k_rep_evader = 1500.0
        self.k_v_evader = 3.0
        self.d0_evader_avoid = 4.0

        # Mission constraints
        self.capture_radius = self.config.get("env.capture_distance", default=8.0)
        self.target_surround_radius = self.capture_radius * 0.75

    def _extract_valid_observations(self, obs_dict):
        """
        Extracts valid entities from zero-padded arrays using the Boolean masks.
        Handles both batched ([1, 5, 7]) and unbatched inputs, and cleanly converts PyTorch Tensors to NumPy.
        """

        def to_numpy_and_squeeze(data):

            """Run to numpy and squeeze."""
            if hasattr(data, "cpu"):
                data = data.cpu().detach().numpy()
            elif hasattr(data, "numpy"):
                data = data.numpy()


            if len(data.shape) > 1 and data.shape[0] == 1:
                return data[0]
            return data


        pursuers_arr = to_numpy_and_squeeze(obs_dict["pursuers"])
        evaders_arr = to_numpy_and_squeeze(obs_dict["evaders"])
        patrols_arr = to_numpy_and_squeeze(obs_dict["patrol_points"])
        obstacles_arr = to_numpy_and_squeeze(obs_dict["obstacles"])
        masks_arr = to_numpy_and_squeeze(obs_dict["masks"])


        masks_bool = masks_arr.astype(bool)


        N = pursuers_arr.shape[0]
        M = evaders_arr.shape[0]
        K = patrols_arr.shape[0]
        L = obstacles_arr.shape[0]


        idx_self = 1
        idx_pur = idx_self + N
        idx_eva = idx_pur + M
        idx_pat = idx_eva + K
        idx_obs = idx_pat + L


        valid_pursuers = pursuers_arr[masks_bool[idx_self:idx_pur]]
        valid_evaders = evaders_arr[masks_bool[idx_pur:idx_eva]]
        valid_patrols = patrols_arr[masks_bool[idx_eva:idx_pat]]
        valid_obstacles = obstacles_arr[masks_bool[idx_pat:idx_obs]]

        return valid_pursuers, valid_evaders, valid_patrols, valid_obstacles

    def calculate_mixed_repulsion(self, pos_r, v_rel_r, radius, k_rep, k_v, d0, safety_margin=1.0):
        """Compute the mixed repulsion force field."""
        dist = np.linalg.norm(pos_r)
        clearance = dist - radius - safety_margin

        if clearance >= d0 or clearance <= 0.01:
            return np.zeros(2)


        mag_pos = k_rep * ((1.0 / clearance) - (1.0 / d0)) / (clearance**2)
        dir_pos = -1.0 * pos_r / dist
        f_pos = mag_pos * dir_pos


        v_ao = np.dot(v_rel_r, pos_r / dist)
        f_vel = np.zeros(2)
        if v_ao > 0:
            mag_vel = k_v * v_ao * ((1.0 / clearance) - (1.0 / d0))
            f_vel = -1.0 * mag_vel * (pos_r / dist)

        return f_pos + f_vel

    def act(self, obs_dict):
        """Main decision loop. Outputs the discrete action index."""
        # --- 1. State Parsing ---

        self_state = obs_dict["self"]
        if hasattr(self_state, "cpu"):
            self_state = self_state.cpu().detach().numpy()
        elif hasattr(self_state, "numpy"):
            self_state = self_state.numpy()

        if len(self_state.shape) == 2 and self_state.shape[0] == 1:
            self_state = self_state[0]

        v_current_r = self_state[0:2]  # Robot's velocity in egocentric frame

        valid_pursuers, valid_evaders, valid_patrols, valid_obstacles = (
            self._extract_valid_observations(obs_dict)
        )

        total_force = np.zeros(2)

        # ==========================================
        # STAGE 1: LOCAL INHIBITION TASK ALLOCATION
        # ==========================================
        best_target_evader = None
        min_distance = float("inf")

        if len(valid_evaders) > 0:
            for evader in valid_evaders:
                evader_pos_r = evader[0:2]
                dist_to_evader = np.linalg.norm(evader_pos_r)

                # Count how many visible teammates are closer to THIS evader than me
                closer_teammates_count = 0
                for pur in valid_pursuers:
                    pur_pos_r = pur[0:2]
                    # Vector from teammate to evader in my egocentric frame
                    dist_pur_to_evader = np.linalg.norm(pur_pos_r - evader_pos_r)

                    if dist_pur_to_evader < dist_to_evader:
                        closer_teammates_count += 1

                # Local Inhibition Trigger: if 3 or more teammates are closer, ignore this target!
                # This naturally resolves conflicts without communication.
                if closer_teammates_count < 3:
                    if dist_to_evader < min_distance:
                        min_distance = dist_to_evader
                        best_target_evader = evader

        # ==========================================
        # STAGE 2: FORCE FIELD GENERATION (FSM)
        # ==========================================
        if best_target_evader is not None:
            # ------------------------------------------------
            # MODE: ENCIRCLE (Target locked)
            # ------------------------------------------------
            evader_pos_r = best_target_evader[0:2]
            target_v_r = best_target_evader[2:4]
            evader_dist = np.linalg.norm(evader_pos_r)


            if evader_dist > self.target_surround_radius:
                magnitude = self.k_att_encircle * (
                    evader_dist - self.target_surround_radius
                )
                direction = evader_pos_r / evader_dist
                total_force += magnitude * direction
            else:
                # Slight repulsion if too close
                magnitude = self.k_att_encircle * (
                    self.target_surround_radius - evader_dist
                )
                direction = -1.0 * evader_pos_r / max(evader_dist, 0.01)
                total_force += magnitude * direction


            v_rel_to_target = v_current_r - target_v_r
            v_ao_target = np.dot(v_rel_to_target, evader_pos_r / max(evader_dist, 0.01))
            if v_ao_target > 0 and evader_dist < self.target_surround_radius + 2.0:
                brake_mag = self.k_v_target_brake * v_ao_target * (1.0 / max(evader_dist, 0.5))
                total_force += -1.0 * brake_mag * (evader_pos_r / max(evader_dist, 0.01))


            if evader_dist > 0.1:
                normal_vec = evader_pos_r / evader_dist
                tangent_vec = np.array([-normal_vec[1], normal_vec[0]])  # Rotated 90 degrees

                for pur in valid_pursuers:
                    pur_pos_r = pur[0:2]
                    pur_v_r = pur[2:4]
                    v_rel_r = v_current_r - pur_v_r

                    dist_pur_to_evader = np.linalg.norm(pur_pos_r - evader_pos_r)

                    # Only apply tangential force for teammates also close to the target
                    if dist_pur_to_evader < self.capture_radius * 1.5:
                        # Standard repulsion vector from teammate
                        teammate_dist = np.linalg.norm(pur_pos_r)
                        if teammate_dist < 0.01:
                            continue
                        rep_vec = -1.0 * pur_pos_r / teammate_dist

                        # Project teammate repulsion onto the tangent vector!
                        # This slides the agent along the encirclement perimeter.
                        tangent_proj_mag = np.dot(rep_vec, tangent_vec)

                        # Apply force
                        force_mag = self.k_tan_encircle * (
                            1.0 / max(teammate_dist, 0.5)
                        )
                        total_force += force_mag * tangent_proj_mag * tangent_vec

                        # Mixed APF collision avoidance with teammate
                        total_force += self.calculate_mixed_repulsion(
                            pur_pos_r, v_rel_r, 1.0,
                            self.k_rep_agent_encircle, self.k_v_agent, self.d0_agent_encircle
                        )

        else:
            # ------------------------------------------------
            # MODE: SEARCH (No valid targets)
            # ------------------------------------------------
            if len(valid_patrols) > 0:
                utilities = []
                for p in valid_patrols:
                    pos_r_x, pos_r_y, dis, angle, norm_idleness, min_dist_teammate = p
                    # Utility strictly promotes spatial dispersion & temporal coverage
                    u = (15.0 * norm_idleness) - (1.0 * dis) + (0.5 * min_dist_teammate)
                    utilities.append(u)

                best_patrol_idx = np.argmax(utilities)
                best_patrol_pos_r = valid_patrols[best_patrol_idx][0:2]
                best_patrol_dist = valid_patrols[best_patrol_idx][2]

                if best_patrol_dist > 0.1:
                    total_force += self.k_att_patrol * (
                        best_patrol_pos_r / best_patrol_dist
                    )

            # Repel teammates strongly during search to maximize map coverage
            for pur in valid_pursuers:
                pur_pos_r = pur[0:2]
                pur_v_r = pur[2:4]
                v_rel_r = v_current_r - pur_v_r

                total_force += self.calculate_mixed_repulsion(
                    pur_pos_r, v_rel_r, 1.0,
                    self.k_rep_agent_patrol, self.k_v_agent, self.d0_agent_patrol
                )

        # ==========================================
        # STAGE 3: GENERAL OBSTACLE AVOIDANCE
        # ==========================================

        for obs in valid_obstacles:
            obs_pos_r = obs[0:2]
            obs_radius = obs[2]
            v_rel_r = v_current_r - np.zeros(2)
            total_force += self.calculate_mixed_repulsion(
                obs_pos_r, v_rel_r, obs_radius,
                self.k_rep_obs, self.k_v_obs, self.d0_obs, safety_margin=1.5
            )


        for evader in valid_evaders:
            if best_target_evader is not None and np.array_equal(evader, best_target_evader):
                continue

            ev_pos_r = evader[0:2]
            ev_v_r = evader[2:4]
            v_rel_r = v_current_r - ev_v_r
            total_force += self.calculate_mixed_repulsion(
                ev_pos_r, v_rel_r, 1.0,
                self.k_rep_evader, self.k_v_evader, self.d0_evader_avoid
            )

        # ==========================================
        # STAGE 4: KINEMATIC MAPPING (Force -> Action)
        # ==========================================
        force_magnitude = np.linalg.norm(total_force)

        if force_magnitude < 1e-3:
            target_angle = 0.0
            acc_proj = 0.0
        else:
            # Robot coordinates: heading is +X (angle 0)
            target_angle = np.arctan2(total_force[1], total_force[0])
            acc_total = total_force / self.m
            acc_proj = acc_total[0]  # Acceleration along current heading


        a_arr = np.array(self.a, dtype=float)
        w_arr = np.array(self.w, dtype=float)

        # Map to angular velocity (w_space)
        w_idx = np.argmin(np.abs(w_arr - target_angle))

        # Map to linear acceleration (a_space)
        if abs(target_angle) > np.pi / 2:
            # Turn is too sharp; brake/reverse
            a_idx = np.argmin(a_arr)
        else:
            # Accelerate forward/backward based on X-axis force projection
            if acc_proj > 0:
                pos_indices = np.where(a_arr >= 0)[0]
                best_idx = np.argmin(np.abs(a_arr[pos_indices] - acc_proj))
                a_idx = pos_indices[best_idx]
            else:
                neg_indices = np.where(a_arr <= 0)[0]
                best_idx = np.argmin(np.abs(a_arr[neg_indices] - acc_proj))
                a_idx = neg_indices[best_idx]

        action_idx = int(a_idx * len(w_arr) + w_idx)
        return [action_idx]
