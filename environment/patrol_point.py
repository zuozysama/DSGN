"""Patrol point utilities for surveillance-style environment logic."""
import math
from typing import Tuple

import numpy as np


class PatrolPoint:
    """Implement the PatrolPoint helper."""

    def __init__(
            self,
            position: Tuple[float, float],
            point_id: int = 0,
            threshold: float = 2.0,
            normalization_factor: float = 250.0
    ):
        """Initialize the instance."""
        self.position = position
        self.id = point_id
        self.threshold = threshold
        self.normalization_factor = normalization_factor


        self.visit_count = 0


        self.last_visited_step = -int(self.normalization_factor)

    def reset(self):
        """Run reset."""
        self.visit_count = 0
        self.last_visited_step = -int(self.normalization_factor)

    def update_visit(self, timestep: int):
        """Run update visit."""
        self.visit_count += 1
        self.last_visited_step = timestep


    def get_idleness(self, current_timestep: int) -> int:
        """Run get idleness."""

        if current_timestep is None:
            # print("[PatrolPoint] Warning: current_timestep is None, should only happen in the visualization mode.")
            return 0
        return max(0, current_timestep - self.last_visited_step)

    def check_reached(self, agent_pos: Tuple[float, float], current_timestep: int) -> bool:
        """Run check reached."""

        dx = self.position[0] - agent_pos[0]
        dy = self.position[1] - agent_pos[1]
        distance = math.hypot(dx, dy)

        if distance <= self.threshold:
            # ==============================================================

            # ==============================================================


            time_since_last = current_timestep - self.last_visited_step
            if self.visit_count > 0 and time_since_last < 50:
                return False


            current_norm_idleness = self.get_idleness(current_timestep) / self.normalization_factor
            if self.visit_count > 0 and current_norm_idleness < 0.05:
                return False

            # ==============================================================

            # ==============================================================

            return True

        return False

    def get_norm_idleness(self, current_timestep: int) -> float:
        """Run get norm idleness."""

        raw_idleness = self.get_idleness(current_timestep)


        norm_idleness = min(raw_idleness / self.normalization_factor, 2.0)

        return norm_idleness

    def __repr__(self):
        """Return a readable string representation."""
        return (f"PatrolPoint(id={self.id}, pos={self.position}, "
                f"visits={self.visit_count}, idle={self.get_idleness(self.last_visited_step)})")
