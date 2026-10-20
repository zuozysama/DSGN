import numpy as np
import copy
from config_manager import ConfigManager
from utils import logger as logger


class ApfAgent:
    """
    Agent class based on the Artificial Potential Field (APF) method.

    The APF method uses virtual force fields to guide the agent's movement, including:
    1. Repulsive forces from obstacles
    2. Velocity-dependent forces from dynamic obstacles
    3. Repulsive forces from boundaries


    Attributes:
        k_rep (float): Repulsive force constant, controlling the strength of repulsion.
        k_v (float): Velocity force constant, controlling the strength of velocity-dependent forces.
        m (float): Robot mass (kilograms), used to convert force to acceleration.
        d0 (float): Obstacle influence distance threshold (meters), beyond which no force is generated.
        n (int): Exponent of repulsive force, affecting the rate of force decay with distance.
        min_vel (float): Minimum velocity threshold, below which acceleration is triggered.
        boundary_k (float): Additional coefficient for boundary forces.
        acc_threshold (float): Threshold for acceleration selection.
        a (np.array): Available linear acceleration options.
        w (np.array): Available angular velocity options.
        GAMMA (float): Discount factor, used for reward computation in reinforcement learning.
    """

    def __init__(self, a, w):
        """
        Initialize an APF agent instance.

        Args:
            a (list of float): List of available linear acceleration options.
            w (list of float): List of available angular velocity options.
        """
        # Basic force field parameters
        self.k_rep = 5000.0  # Repulsive force constant (increased to generate sufficient force)
        self.k_v = 1.0  # Velocity force constant
        self.m = 500  # Robot mass (kg)
        self.d0 = 10.0  # Obstacle distance threshold (m)
        self.n = 2  # Exponent of repulsive force
        self.min_vel = 1.0  # Minimum velocity threshold

        # Boundary force related parameters
        self.boundary_k = 10.0  # Boundary force coefficient
        self.out_bound_k = 20.0  # Force enhancement coefficient when out of bounds

        # Acceleration selection parameters
        self.acc_threshold = 0.1  # Acceleration selection threshold

        # Action space
        self.a = np.array(a)  # Available linear accelerations
        self.w = np.array(w)  # Available angular velocities

        self.GAMMA = 0.99  # Discount factor

        # Environment configuration
        self.apf_config = ConfigManager.get_instance()
        self.x_min = 0.0
        self.x_max = self.apf_config.get("env.width")
        self.y_min = 0.0
        self.y_max = self.apf_config.get("env.height")

    def position_force(self, position, radius):
        """
        Calculate repulsive force from an obstacle.

        Uses the standard APF repulsive force formula, considering a safety distance.
        F = k_rep * ((1/d - 1/d0) / d^n) * dir

        Args:
            position (np.array): Position vector of the obstacle in the robot's coordinate system.
            radius (float): Radius of the obstacle.

        Returns:
            np.array: Repulsive force vector. Returns zero vector if distance exceeds threshold.
        """
        d_obs = np.linalg.norm(position) - radius - 0.8  # Consider safety distance
        if d_obs >= self.d0:
            return np.zeros(2)

        magnitude = self.k_rep * ((1 / d_obs) - (1 / self.d0)) / (d_obs ** self.n)
        direction = -1.0 * position / np.linalg.norm(position)
        return magnitude * direction

    def velocity_force(self, v_ao, position, radius):
        """
        Calculate velocity-dependent force from a dynamic obstacle.

        Generates additional repulsive force for approaching dynamic obstacles, with magnitude related to relative velocity and distance.

        Args:
            v_ao (float): Projection of relative velocity along the line of connection.
            position (np.array): Position vector of the dynamic obstacle.
            radius (float): Radius of the dynamic obstacle.

        Returns:
            np.array: Velocity force vector. Returns zero vector if distance exceeds threshold or relative velocity is negative.
        """
        d_obs = np.linalg.norm(position) - radius - 0.8

        if d_obs >= self.d0:
            return np.zeros(2)

        if v_ao > 0:  # Generate force only for approaching cases
            magnitude = -self.k_v * v_ao * ((1 / d_obs) - (1 / self.d0)) / (d_obs ** 1)
            direction = position / np.linalg.norm(position)
            return magnitude * direction
        return np.zeros(2)

    def boundary_force(self, position):
        """
        Calculate boundary repulsive force using a unified force field function.

        Considers environmental scale, calculates repulsive force for each boundary separately, and significantly enhances force when out of bounds.
        Out-of-bounds cases use stronger nonlinear decay to ensure quick return to within boundaries.

        Args:
            position (np.array): Current position vector of the agent.

        Returns:
            np.array: Boundary repulsive force vector.
        """
        x, y = position
        force = np.zeros(2)

        def calculate_force(dist, is_positive):
            """
            Calculate boundary force in a single direction.

            Args:
                dist (float): Distance to the boundary.
                is_positive (bool): Whether the force is in the positive direction.

            Returns:
                float: Calculated force (considering direction).
            """
            if dist < self.d0:
                k = self.k_rep * (self.out_bound_k if dist < 0 else 1.0)
                magnitude = k * ((1 / abs(dist)) - (1 / self.d0)) / (abs(dist) ** (self.n - 0.5))
                return magnitude if is_positive else -magnitude
            return 0.0

        # X-direction boundary force
        x_min_dist = x - self.x_min
        x_max_dist = self.x_max - x
        force[0] = calculate_force(x_min_dist, True) + calculate_force(x_max_dist, False)

        # Y-direction boundary force
        y_min_dist = y - self.y_min
        y_max_dist = self.y_max - y
        force[1] = calculate_force(y_min_dist, True) + calculate_force(y_max_dist, False)

        # Consider environmental scale
        env_scale = np.sqrt(self.x_max * self.y_max) / 100.0
        return force * self.boundary_k * env_scale

    def act(self, observation):
        """
        Select an action based on the observed state.

        Chooses an appropriate combination of acceleration and angular velocity based on force field calculations.
        Special attention is given to handling out-of-bounds situations and low velocity cases.

        Args:
            observation (list): Observation vector of length 59, containing:
                - [0:4]: Self state (x, y, theta, speed)
                - [4:19]: Static obstacle states (x, y, radius) * 5
                - [19:]: Dynamic obstacle states (x, y, vx, vy) * 10

        Returns:
            int: Selected action index, representing the (acceleration, angular_velocity) combination.
        """
        assert len(observation) == 59, "The state size does not equal 59"

        obs_array = np.array(observation)
        static_start_idx = 4
        dynamic_start_idx = 19

        # Extract observation data
        ego = obs_array[:static_start_idx]
        static = obs_array[static_start_idx:dynamic_start_idx]
        dynamic = obs_array[dynamic_start_idx:]

        # Initialize state variables
        total_force_repulsion = np.zeros(2)
        position_global_frame = ego[:2]
        velocity = ego[2:4]
        velocity_magnitude = np.linalg.norm(velocity)

        # Handle static obstacles
        static_obstacles = static.reshape(-1, 3)
        for obs in static_obstacles:
            if np.linalg.norm(obs[:2]) > 1e-3:
                total_force_repulsion += self.position_force(obs[:2], float(obs[2]))

        # Handle dynamic obstacles
        dynamic_obstacles = dynamic.reshape(-1, 4)
        for obs in dynamic_obstacles:
            if np.linalg.norm(obs[:2]) > 1e-3:
                pos = obs[:2]
                vel = obs[2:4]

                e_ao = pos / np.linalg.norm(pos)
                v_ao = np.dot(velocity - vel, e_ao)

                if v_ao >= 0.0:
                    total_force_repulsion += self.position_force(pos, 0.8)
                    total_force_repulsion += self.velocity_force(v_ao, pos, 0.8)

        # Add boundary force
        boundary_repulsion = self.boundary_force(position_global_frame)
        total_force_repulsion += boundary_repulsion

        # Check if out of bounds
        is_out_bounds = (position_global_frame[0] < self.x_min or
                         position_global_frame[0] > self.x_max or
                         position_global_frame[1] < self.y_min or
                         position_global_frame[1] > self.y_max)

        # Calculate motion direction
        force_total = total_force_repulsion
        force_magnitude = np.linalg.norm(force_total)

        velocity_angle = 0.0 if velocity_magnitude <= 1e-3 else np.arctan2(velocity[1], velocity[0])
        force_angle = np.arctan2(force_total[1], force_total[0])


        diff_angle = force_angle - velocity_angle
        diff_angle = np.mod(diff_angle + np.pi, 2 * np.pi) - np.pi

        # Select angular velocity (considering boundary conditions)
        if is_out_bounds or position_global_frame[0] < self.x_min + self.d0 * 0.5 or\
                position_global_frame[0] > self.x_max - self.d0 * 0.5 or\
                position_global_frame[1] < self.y_min + self.d0 * 0.5 or\
                position_global_frame[1] > self.y_max - self.d0 * 0.5:
            # Near or beyond boundaries, select non-zero angular velocity
            # 1. Collect indices and values of all non-zero angular velocities
            available_w = [(i, w) for i, w in enumerate(self.w) if np.abs(w) > 0.1]

            # 2. If there are available non-zero angular velocities, select the one closest to the target angle
            if available_w:
                w_idx = min(available_w, key=lambda x: abs(x[1] - diff_angle))[0]
            else:
                # If no suitable angular velocity is found, use the standard minimum distance selection
                w_idx = np.argmin(np.abs(self.w - diff_angle))
        else:
            w_idx = np.argmin(np.abs(self.w - diff_angle))

        # Calculate acceleration
        a_total = force_total / self.m
        velocity_direction = np.array([1.0, 0.0])
        if velocity_magnitude > 1e-3:
            velocity_direction = velocity / velocity_magnitude

        a_proj = np.dot(a_total, velocity_direction)

        # Select acceleration
        a = copy.deepcopy(self.a)

        if is_out_bounds:
            # Out-of-bounds case: Forcefully select corrective acceleration
            if np.dot(force_total, velocity) > 0:
                a_idx = 0  # Need to decelerate
            else:
                a_idx = len(a) - 1  # Need to accelerate
        else:
            # Normal case: Select acceleration based on projection
            if abs(a_proj) > self.acc_threshold:
                # Significant acceleration demand
                if a_proj > 0:
                    a_idx = np.argmin(np.abs(a[a > 0] - a_proj))
                else:
                    a_idx = np.argmin(np.abs(a[a < 0] - a_proj))
            else:
                a_idx = len(a) // 2

        return int(a_idx * len(self.w) + w_idx)
