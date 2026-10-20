import numpy as np
from config_manager import ConfigManager


class Perception:
    """
    Perception class used to define the perception range of pursuers and evaders, as well as observed objects.

    Args:
        is_evader (bool): Whether it is an evader, default is False.

    Attributes:
        range (float): Perception range (meters).
        angle (float): Perception angle range (radians).
        max_obstacle_num (int): Maximum number of obstacles considered.
        max_pursuer_num (int): Maximum number of pursuers considered.
        max_evader_num (int): Maximum number of evaders considered.
        observation (dict): Perception format, containing information about itself, static objects, pursuers, and evaders.
        observed_obstacles (list): List of indices of observed obstacles.
        observed_pursuers (list): List of indices of observed pursuers.
        observed_evaders (list): List of indices of observed evaders.

    Methods:
        __init__(self, is_evader=False): Initialize an instance of the Perception class.
        observation_format(self, is_evader): Return the corresponding perception format based on whether it is an evader.
    """

    def __init__(self, is_evader=False):
        """
        Initialize an instance of the Perception class.

        Args:
            is_evader (bool): Whether it is an evader. If it is an evader, set a specific perception format.
        """
        self.perception_config = ConfigManager.get_instance()
        self.range = 20.0  # Perception range (meters)
        self.angle = 2 * np.pi  # Perception angle range (radians)
        self.max_obstacle_num = 5  # Maximum number of obstacles considered
        self.max_pursuer_num = 5  # Maximum number of pursuers considered
        self.max_evader_num = self.perception_config.get("perception.max_evader_num") if not is_evader else 5 # Maximum number of evaders considered
        self.observation = self.observation_format(is_evader)
        self.observed_obstacles: list[int] = []  # List of indices of observed obstacles
        self.observed_pursuers: list[int] = []  # List of indices of observed pursuers
        self.observed_evaders: list[int] = []  # List of indices of observed evaders

    def observation_format(self, is_evader):
        """
        Return the corresponding perception format based on whether it is an evader.

        Args:
            is_evader (bool): Whether it is an evader.

        Returns:
            dict: Perception format dictionary containing information about itself, static objects, pursuers, and evaders.
        """
        if is_evader:
            return dict(self=[], statics=[], pursuers=[], evaders=[])
        else:
            self.max_patrol_point_num = 16
            return dict(self=[], pursuers=[], evaders=[], statics=[], patrol_points=[], masks=[], types=[])
