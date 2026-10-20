import torch
import numpy as np


class ReplayBuffer:
    """Store and sample replay transitions."""
    def __init__(self, buffer_size, batch_size, device, max_pursuers, max_evaders, max_obstacles, max_patrol_points = 16):
        """
        Initialize the replay buffer
        Args:
            buffer_size (int): Maximum capacity of the buffer
            batch_size (int): Batch size for each sampling
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.size = 0
        self.ptr = 0

        # Define feature dimensions for various entity types
        self.self_dim = 4
        self.pursuer_dim = 7
        self.evader_dim = 7
        self.obstacle_dim = 5

        # Initialize storage with NumPy arrays (float32 to save memory)
        self.self_state = np.zeros((buffer_size, self.self_dim), dtype=np.float32)
        self.next_self_state = np.zeros_like(self.self_state)

        self.pursuers = np.zeros((buffer_size, max_pursuers, self.pursuer_dim), dtype=np.float32)
        self.next_pursuers = np.zeros_like(self.pursuers)

        self.evaders = np.zeros((buffer_size, max_evaders, self.evader_dim), dtype=np.float32)
        self.next_evaders = np.zeros_like(self.evaders)

        self.obstacles = np.zeros((buffer_size, max_obstacles, self.obstacle_dim), dtype=np.float32)
        self.next_obstacles = np.zeros_like(self.obstacles)

        self.masks = np.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles + max_patrol_points), dtype=bool)
        self.next_masks = np.zeros_like(self.masks)

        self.types = np.zeros((buffer_size, 1 + max_pursuers + max_evaders + max_obstacles + max_patrol_points), dtype=np.int64)
        self.next_types = np.zeros_like(self.types)

        self.patrol_points = np.zeros((buffer_size, 16, 6), dtype=np.float32)
        self.next_local_patch = np.zeros_like(self.patrol_points)


        self.has_active_target = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_has_active_target = np.zeros_like(self.has_active_target)

        self.actions = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)

    def add(self, obs, action, reward, next_obs, done):
        """Add data to the buffer"""
        idx = self.ptr

        # Store current observations
        self.self_state[idx] = obs['self']
        self.pursuers[idx] = obs['pursuers']
        self.evaders[idx] = obs['evaders']
        self.obstacles[idx] = obs['obstacles']
        self.masks[idx] = obs['masks']
        self.types[idx] = obs['types']
        self.patrol_points[idx] = obs['patrol_points']


        self.has_active_target[idx] = obs['has_active_target']

        # Store next observations
        self.next_self_state[idx] = next_obs['self']
        self.next_pursuers[idx] = next_obs['pursuers']
        self.next_evaders[idx] = next_obs['evaders']
        self.next_obstacles[idx] = next_obs['obstacles']
        self.next_masks[idx] = next_obs['masks']
        self.next_types[idx] = next_obs['types']
        self.next_local_patch[idx] = next_obs['patrol_points']


        self.next_has_active_target[idx] = next_obs['has_active_target']

        # Store actions, rewards, and done flags
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        # Update pointer and buffer size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Sample a batch of data from the buffer"""
        if self.size < self.batch_size:
            raise ValueError(f"Insufficient data in buffer: {self.size} available, {self.batch_size} required.")

        indices = np.random.randint(0, self.size, size=self.batch_size)

        def to_tensor(data, dtype=torch.float32):
            """Run to tensor."""
            return torch.tensor(data, dtype=dtype, device=self.device)

        batch = {
            'observations': {
                'self': to_tensor(self.self_state[indices]),
                'pursuers': to_tensor(self.pursuers[indices]),
                'evaders': to_tensor(self.evaders[indices]),
                'obstacles': to_tensor(self.obstacles[indices]),
                'masks': torch.tensor(self.masks[indices], dtype=torch.bool, device=self.device),
                'types': torch.tensor(self.types[indices], dtype=torch.long, device=self.device),
                'patrol_points': to_tensor(self.patrol_points[indices]),


                'has_active_target': to_tensor(self.has_active_target[indices])
            },
            'actions': to_tensor(self.actions[indices]),
            'rewards': to_tensor(self.rewards[indices]),
            'next_observations': {
                'self': to_tensor(self.next_self_state[indices]),
                'pursuers': to_tensor(self.next_pursuers[indices]),
                'evaders': to_tensor(self.next_evaders[indices]),
                'obstacles': to_tensor(self.next_obstacles[indices]),
                'masks': torch.tensor(self.next_masks[indices], dtype=torch.bool, device=self.device),
                'types': torch.tensor(self.next_types[indices], dtype=torch.long, device=self.device),
                'patrol_points': to_tensor(self.next_local_patch[indices])
            },
            'dones': torch.tensor(self.dones[indices], dtype=torch.bool, device=self.device)
        }
        return batch

    def __len__(self):
        """Run   len  ."""
        return self.size
