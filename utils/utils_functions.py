import math
import numpy as np
import wandb

from utils.logger_setter import logger as logger


def find_robot_by_id(robot_id, robot_list):
    """
    Find a robot object in a list based on its ID.

    This function iterates through the given list of robots to find an object matching the specified ID.
    If a matching robot object is found, it is returned; otherwise, a runtime error is raised.

    Args:
        robot_id (int | signedinteger): The ID of the robot to find.
        robot_list (list): A list of robot objects, each expected to have an `id` attribute.

    Returns:
        Pursuer | Evader: The robot object matching the specified ID.

    Raises:
        RuntimeError: If no matching robot object is found in the list.

    """
    try:
        if robot_id is None:
            raise ValueError("robot_id cannot be None!")
        if not isinstance(robot_id, int | np.signedinteger):
            raise ValueError("robot_id must be an integer!")

        for robot in robot_list:
            if robot_id == robot.id:
                # logger.info(f"Finding robot with ID: {robot_id}!!!!!!!!=================")
                # print(f"Finding robot with ID: {robot_id}!!!!!!!!=================")
                return robot
        raise AttributeError(f"Wrong robot_id: {robot_id} input!")
    except Exception as e:
        logger.error(f"An error occurred in find_robot_by_id: {e}")
        raise RuntimeError("Robot not found in robot_list!")


def calculate_angle_from_components(v1: float, v2: float) -> list:
    """Compute an angle from vector components."""
    # Calculate the angle corresponding to the velocity components
    theta = np.arctan2(v2, v1)


    if theta < 0:
        theta += 2 * np.pi
    return [theta]


def to_list(input_data):
    """
    Convert any input data into a list.

    Args:
        input_data: Can be a scalar, iterable, or NumPy array.

    Returns:
        list: The converted list.
    """
    if isinstance(input_data, np.ndarray):
        # If it's a NumPy array, convert using tolist()
        return input_data.tolist()
    elif isinstance(input_data, (list, tuple, set)):
        # If it's already a list, tuple, or set, convert to a list
        return list(input_data)
    elif isinstance(input_data, (np.integer, np.floating, int, float, str)):
        # If it's a single scalar value (including NumPy scalars and Python basic types), wrap in a list
        return [input_data]
    else:
        # For other iterable objects (e.g., dictionaries), convert their keys to a list
        try:
            return list(input_data)
        except TypeError:
            # If it cannot be iterated, treat it as a single element in a list
            return [input_data]


def generate_centers(width: float, height: float, fov_radius: float = 20.0, safety_margin: float = 0.95):
    """Generate center coordinates for the current field layout."""


    max_grid_side = fov_radius * np.sqrt(2) * safety_margin


    num_x = int(math.ceil(width / max_grid_side))
    num_y = int(math.ceil(height / max_grid_side))


    delta_x = width / num_x
    delta_y = height / num_y


    # linspace(start, stop, num)


    x_centers = np.linspace(delta_x / 2, width - delta_x / 2, num_x)
    y_centers = np.linspace(delta_y / 2, height - delta_y / 2, num_y)


    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing='xy')


    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    return centers


def log_gating_weights(fused_weight, step):
    """Log gating weights distribution"""

    gated_weights_tensor = fused_weight
    gated_weights_array = gated_weights_tensor.detach().cpu().numpy()


    wandb.log({
        "Gating/Fused_Weights_Distribution": wandb.Histogram(gated_weights_array)
    })


    wandb.log({
        "Gating/Fused_Weights_Mean": np.mean(gated_weights_array),
        "Gating/Fused_Weights_Std": np.std(gated_weights_array),
        "Gating/Fused_Weights_Min": np.min(gated_weights_array),
        "Gating/Fused_Weights_Max": np.max(gated_weights_array),
        "Global_timestep": step
    })


import torch

def batch_obs_dict(obs_list, device):
    """Batch observation dictionaries into array form."""
    if len(obs_list) == 0:
        return None


    keys = obs_list[0].keys()

    batched_dict = {}

    for key in keys:


        val_list = [obs[key] for obs in obs_list]


        stacked_val = np.stack(val_list)


        if key in ['masks', 'types']:


            tensor_val = torch.tensor(stacked_val, dtype=torch.long, device=device)

            if key == 'masks':
                tensor_val = torch.tensor(stacked_val, dtype=torch.bool, device=device)
        else:

            tensor_val = torch.tensor(stacked_val, dtype=torch.float32, device=device)

        batched_dict[key] = tensor_val

    return batched_dict
