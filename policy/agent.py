import json
import random
import os
from collections import deque
from typing import Literal

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
import wandb

from policy.MLPDQN import MLPDQNPolicy
from policy.MLPIQN import MLPIQNPolicy
from policy.TransIQN import TransIQNPolicy
from policy.DSGN import DSGNPolicy, DSGNConfig
from policy.ablation_mlp_with_target_selection import (
    MlpWithTargetSelectPolicy,
    MlpWithTargetSelectConfig,
)
from policy.ablation_transformer_without_target_selection import (
    TransformerWithoutTargetSelectPolicy,
    TransformerWithoutTargetSelectConfig,
)
from policy.replay_buffer import ReplayBuffer
from utils import logger


class Agent:
    """
    Reinforcement learning agent class, integrated with wandb monitoring.

    Attributes:
        device (str): The device to use ('cpu' or 'cuda')
        LR (float): Learning rate
        TAU (float): Soft update parameter
        GAMMA (float): Discount factor
        BUFFER_SIZE (int): Experience replay buffer size
        BATCH_SIZE (int): Experience replay batch size
        training (bool): Whether in training mode
        action_size (int): Size of the action space
        use_iqn (bool): Whether to use IQN policy network
        policy_local (nn.Module): Local policy network
        policy_target (nn.Module): Target policy network
        optimizer (optim.Optimizer): Optimizer
        memory (ReplayBuffer): Experience replay buffer
        wandb_project (str): wandb project name
        rewards_window (deque): Sliding window for rewards
        losses_window (deque): Sliding window for losses
        training_step (int): Training step counter
    """

    def __init__(
        self,
        hidden_dimension=256,
        num_heads=8,
        num_layers=4,
        action_size=9,
        BATCH_SIZE=256,
        BUFFER_SIZE=1_000_000,
        LR=1e-4,
        TAU=1.0,
        GAMMA=0.99,
        device="cpu",
        seed=0,
        training=True,
        use_iqn=True,
        model_name=None,
        wandb_project="multi-agent-rl-iqn",
        gating_type: Literal[
            "scalar", "vector", "average", "rule_based", "random", "no_gating"
        ] = "unspecified",
        gating_reg_coeff: float = None,
    ):
        """
        Initialize Agent instance.

        Args:
            hidden_dimension (int): Dimension of the hidden layer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers in the network.
            action_size (int): Size of the action space.
            BATCH_SIZE (int): Batch size.
            BUFFER_SIZE (int): Buffer size.
            LR (float): Learning rate.
            TAU (float): Soft update parameter.
            GAMMA (float): Discount factor.
            device (str): The device to use ('cpu' or 'cuda').
            seed (int): Random seed.
            training (bool): Whether in training mode.
            use_iqn (bool): Whether to use IQN policy network.
            model_name (str): Name of the model to use.
            wandb_project (str): wandb project name.
            gating_type (Literal): gating type
        """
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.training = training
        self.action_size = action_size
        self.use_iqn = use_iqn
        self.model_name = model_name
        self.gating_reg_coeff = gating_reg_coeff
        self.gating_type = gating_type

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if training:
            # wandb related initialization
            self.wandb_project = wandb_project
            self.rewards_window = deque(maxlen=100)
            self.losses_window = deque(maxlen=100)
            self.training_step = 0

            assert self.model_name is not None, "model_name must be specified!"
            if use_iqn:
                # Policy network initialization
                if self.model_name in [
                    "TERL-S",
                    "DSGN",
                    "AVERAGE",
                    "RULE_BASED",
                    "RANDOM",
                    "DSGN-WO-BETA",
                    "DSGN-SCALAR",
                    "DSGN-SCALAR-WO-BETA",
                ]:
                    config = DSGNConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed,
                        num_quantiles=32,
                        num_cosine_features=64,
                        learning_rate=LR,
                        gradient_clip=1.0,
                        gating_type=gating_type,
                    )
                    self.policy_local = DSGNPolicy(config=config)
                    self.policy_target = DSGNPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                # elif self.model_name == "MlpWithTargetSelect":
                #     config = MlpWithTargetSelectConfig(
                #         hidden_dim=hidden_dimension,
                #         num_heads=num_heads,
                #         num_layers=num_layers,
                #         action_size=action_size,
                #         device=device,
                #         seed=seed
                #     )
                #     self.policy_local = MlpWithTargetSelectPolicy(config=config)
                #     self.policy_target = MlpWithTargetSelectPolicy(config=config)
                #     self.policy_target.load_state_dict(self.policy_local.state_dict())
                # elif self.model_name == "TransformerWithoutTargetSelect":
                #     config = TransformerWithoutTargetSelectConfig(
                #         hidden_dim=hidden_dimension,
                #         num_heads=num_heads,
                #         num_layers=num_layers,
                #         action_size=action_size,
                #         device=device,
                #         seed=seed
                #     )
                #     self.policy_local = TransformerWithoutTargetSelectPolicy(config=config)
                #     self.policy_target = TransformerWithoutTargetSelectPolicy(config=config)
                #     self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "IQN":
                    config = DSGNConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed,
                        num_quantiles=32,
                        num_cosine_features=64,
                        learning_rate=LR,
                        gradient_clip=1.0,
                        gating_type=gating_type,
                    )
                    self.policy_local = MLPIQNPolicy(config=config)
                    self.policy_target = MLPIQNPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
                elif self.model_name == "TransIQN":
                    config = DSGNConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed,
                        num_quantiles=32,
                        num_cosine_features=64,
                        learning_rate=LR,
                        gradient_clip=1.0,
                        gating_type=gating_type,
                    )
                    self.policy_local = TransIQNPolicy(config=config)
                    self.policy_target = TransIQNPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())
            else:
                # DQN policy network initialization
                if self.model_name == "DQN":
                    config = DSGNConfig(
                        hidden_dim=hidden_dimension,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        action_size=action_size,
                        device=device,
                        seed=seed,
                        num_quantiles=32,
                        num_cosine_features=64,
                        learning_rate=LR,
                        gradient_clip=1.0,
                        gating_type=gating_type,
                    )
                    self.policy_local = MLPDQNPolicy(config=config)
                    self.policy_target = MLPDQNPolicy(config=config)
                    self.policy_target.load_state_dict(self.policy_local.state_dict())

            self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, 5, 8, 5)

    def act_dqn(self, state, eps=0.0, use_eval=True):
        """
        Select action based on DQN policy (Supports Batch Inference).
        """
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()

        with torch.no_grad():
            # action_values shape: [Batch_Size, Action_Dim]
            action_values = self.policy_local(state)
        self.policy_local.train()


        batch_size = action_values.shape[0]

        greedy_actions = torch.argmax(action_values, dim=1)
        random_actions = torch.randint(
            0, self.action_size, (batch_size,), device=self.device
        )

        rand_vals = torch.rand(batch_size, device=self.device)
        actions_tensor = torch.where(rand_vals > eps, greedy_actions, random_actions)

        return actions_tensor.cpu().numpy(), action_values.cpu().data.numpy()

    def convert_to_tensor(self, data, device):
        """Recursively convert np.array in dictionary to torch.Tensor"""
        if isinstance(data, dict):
            # If it's a dictionary, recursively process each key-value pair
            return {
                key: self.convert_to_tensor(value, device)
                for key, value in data.items()
            }
        elif isinstance(data, np.ndarray):
            # If it's an np.array, convert to torch.Tensor and move to device
            return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # Other types remain unchanged
            return data

    def act(self, state, eps=0.0, cvar=1.0, use_eval=True):
        """
        Select action based on IQN policy (Supports Batch Inference).
        """

        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()


        with torch.no_grad():

            quantiles, taus = self.policy_local(state, self.policy_local.K, cvar)
            # action_values shape: [Batch_Size, Action_Dim]
            action_values = quantiles.mean(dim=1)

        self.policy_local.train()


        batch_size = action_values.shape[0]


        greedy_actions = torch.argmax(action_values, dim=1)  # Shape: [Batch]


        random_actions = torch.randint(
            0, self.action_size, (batch_size,), device=self.device
        )


        rand_vals = torch.rand(batch_size, device=self.device)


        actions_tensor = torch.where(rand_vals > eps, greedy_actions, random_actions)


        return (
            actions_tensor.cpu().numpy(),
            quantiles.cpu().data.numpy(),
            taus.cpu().data.numpy(),
        )

    def act_adaptive(self, state, eps=0.0):
        """
        Select action with adaptive CVaR value.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon value for epsilon-greedy strategy.

        Returns:
            int: Selected action.
            np.ndarray: Quantiles.
            np.ndarray: Tau values.
            float: CVaR value.
        """
        cvar = self.adjust_cvar(state)
        action, quantiles, taus = self.act(state, eps, cvar)
        return action, quantiles, taus, cvar

    def adjust_cvar(self, state):
        """Adjust CVaR value based on state"""
        obstacles = state["obstacles"]
        pursuers = state["pursuers"]
        evaders = state["evaders"]

        closest_d = np.inf

        for entity in obstacles:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - entity[2] - 0.8
            closest_d = min(closest_d, dist)

        for entity in pursuers:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - 1.6
            closest_d = min(closest_d, dist)

        for entity in evaders:
            if any(np.abs(entity[:2])) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(entity[:2]) - 1.6
            closest_d = min(closest_d, dist)

        cvar = 1.0
        if closest_d < 10.0:
            cvar = closest_d / 10.0

        return cvar

    def train(self, global_step: int):
        """
        Select training method based on policy.

        Returns:
            float: Training loss.
        """
        if self.use_iqn:
            return self.train_IQN(global_step)
        else:
            return self.train_DQN()

    def train_IQN(self, global_step: int):
        """
        Train IQN model with Annealed Beta-Prior Regularization.
        Configured for Stability: Long Warm-up & Low Coefficient.
        """
        # --- 1. Sampling & Main RL Logic ---
        states, actions, rewards, next_states, dones = self.memory.sample().values()

        actions = actions.unsqueeze(-1).long()
        rewards = rewards.unsqueeze(-1).float()
        dones = dones.unsqueeze(-1).float()

        self.optimizer.zero_grad()

        # Compute Q Targets
        Q_targets_next, _ = self.policy_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)
        Q_targets = rewards.unsqueeze(-1) + (
            self.GAMMA * Q_targets_next * (1.0 - dones.unsqueeze(-1))
        )

        # Compute Q Expected
        Q_expected, taus = self.policy_local(states)
        Q_expected = Q_expected.gather(
            2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1)
        )

        # Quantile Huber Loss
        td_error = Q_targets - Q_expected
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
        main_loss = quantil_l.sum(dim=1).mean(dim=1).mean()

        # =================================================================

        # =================================================================

        weighted_aux_loss = torch.tensor(0.0, device=self.device)
        curr_coeff = 0.0

        if self.model_name in ["DSGN", "DSGN-SCALAR", "AVERAGE", "RULE_BASED", "RANDOM"]:
            gate_outputs = self.policy_local.gate_outputs_for_loss
            gating_type = self.policy_local.gating_type

            if gating_type in ["vector", "scalar"]:


                start_step = 1_000_000
                end_step = 3_000_000


                if gating_type == "vector":
                    target_coeff = 0.001
                else:
                    target_coeff = (
                        0.0006
                    )

                if global_step > start_step:
                    progress = min(
                        1.0, (global_step - start_step) / (end_step - start_step)
                    )
                    curr_coeff = target_coeff * progress

                if curr_coeff > 0:
                    # 1. Get Mask [B, 1]
                    has_evader_mask = states["has_active_target"].to(self.device)

                    # 2. Extract Gate
                    if gating_type == "scalar":

                        target_gate = gate_outputs[:, 0].unsqueeze(1)
                    else:
                        target_gate = gate_outputs

                    # 3. Compute Loss
                    aux_loss = compute_beta_regularization_loss(
                        target_gate, has_evader_mask, gating_type=gating_type
                    )

                    weighted_aux_loss = curr_coeff * aux_loss

        assert weighted_aux_loss==0.0 if self.model_name in ["DSGN-WO-BETA", "DSGN-SCALAR-WO-BETA", "MLPIQN", "TransIQN"] else True, "Aux loss should be zero for models without Beta regularization"
        # =================================================================
        # Final Optimization
        # =================================================================

        total_loss = main_loss + weighted_aux_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        # --- Logging ---
        if hasattr(self, "use_wandb") and self.use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "Loss/Total": total_loss.item(),
                    "Loss/Main_RL": main_loss.item(),
                    "Loss/Beta_Reg_Weighted": weighted_aux_loss.item(),
                    "Coeff/Beta_Reg": curr_coeff,
                    "Gate/Mean": gate_outputs.mean().item()
                    if gate_outputs is not None
                    else 0.0,
                    "Step": global_step,
                }
            )

        return total_loss.detach().cpu().numpy()

    def train_DQN(self):
        """
        Train DQN model.

        Returns:
            float: Training loss.
        """
        states, actions, rewards, next_states, dones = self.memory.sample().values()
        actions = actions.unsqueeze(-1).long()
        rewards = rewards.unsqueeze(-1).float()
        dones = dones.unsqueeze(-1).float()

        self.optimizer.zero_grad()

        # Compute target values
        Q_targets_next = self.policy_target(next_states)
        Q_targets_next, _ = Q_targets_next.max(dim=1, keepdim=True)
        Q_targets = rewards + (1 - dones) * self.GAMMA * Q_targets_next

        # Compute expected values
        Q_expected = self.policy_local(states)
        Q_expected = Q_expected.gather(1, actions)

        # Compute Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def soft_update(self):
        """Soft-update the target network with online-network weights."""
        for target_param, local_param in zip(
            self.policy_target.parameters(), self.policy_local.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def save_latest_model(self, directory, filename=None):
        """
        Save the model.
        - If filename is provided (Latest): Save Model + Optimizer (Agent handles this).
        - If filename is None (Checkpoint): Save Model Weights Only (Delegate to Policy).
        """
        if filename:


            try:
                os.makedirs(directory, exist_ok=True)
                save_path = os.path.join(directory, filename)


                save_dict = {
                    "model_state_dict": self.policy_local.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                    if hasattr(self, "optimizer")
                    else None,
                }

                torch.save(save_dict, save_path)


                config_name = (
                    filename.replace(".pth", "").replace(".pt", "") + "_config.json"
                )
                if config_name == filename:
                    config_name = "config.json"
                config_path = os.path.join(directory, config_name)
                with open(config_path, "w") as f:
                    json.dump(vars(self.policy_local.config), f)

                logger.info(
                    f"Process {os.getpid()} |Latest Model (w/ Optimizer) saved to {save_path}"
                )
            except Exception as e:
                logger.error(f"Process {os.getpid()} |Error saving latest model: {e}")
                raise
        else:


            self.policy_local.save(directory, filename=None)

    def load_model(self, path, device="cpu", version=None, filename=None):
        """
        Load model.
        Args:
            filename: If set (e.g., 'latest_model.pth'), loads that specific file.
        """
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        try:
            target_path = None
            candidate_paths = []

            # Priority 1: explicit filename or directory.
            if filename:
                direct_path = (
                    filename if os.path.isabs(filename) else os.path.join(path, filename)
                )

                if os.path.isfile(direct_path):
                    candidate_paths.append(direct_path)
                else:
                    candidate_paths.append(
                        os.path.join(direct_path, "network_params_v1.pth")
                    )

            # Priority 2: explicit checkpoint version.
            elif version is not None:
                version_str = str(version)
                checkpoint_dir = (
                    version_str
                    if version_str.startswith("checkpoint_")
                    else f"checkpoint_{version_str}"
                )
                candidate_paths.extend(
                    [
                        os.path.join(path, checkpoint_dir, "network_params_v1.pth"),
                        os.path.join(path, f"network_params_v{version_str}.pth"),
                    ]
                )

            # Priority 3: default to the root latest checkpoint.
            else:
                candidate_paths.append(os.path.join(path, "latest_model.pth"))


            for candidate_path in candidate_paths:

                if os.path.isfile(candidate_path):
                    target_path = candidate_path
                    break

            if target_path is None:
                searched_paths = ", ".join(candidate_paths)
                raise FileNotFoundError(
                    f"Unable to find model file. Searched: {searched_paths}"
                )
        except Exception as e:
            logger.error(
                f"Process {os.getpid()} |Failed to determine model file path: {e}"
            )
            raise
        # -------------------------------------------------------------

        # -------------------------------------------------------------

        if self.use_iqn:
            if self.model_name == "TERL-S":

                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "DSGN":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "AVERAGE":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "RULE_BASED":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "RANDOM":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "IQN":
                self.policy_local = MLPIQNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "TransIQN":
                self.policy_local = TransIQNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "DSGN-WO-BETA":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "DSGN-SCALAR":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            elif self.model_name == "DSGN-SCALAR-WO-BETA":
                self.policy_local = DSGNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
        else:
            if self.model_name == "DQN":
                self.policy_local = MLPDQNPolicy.load(
                    path, device, version=None, exact_file_path=target_path
                )
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")


        if self.policy_target is not None:
            self.policy_target.load_state_dict(self.policy_local.state_dict())
            self.policy_target.eval()

        # -------------------------------------------------------------

        # -------------------------------------------------------------
        if hasattr(self, "optimizer") and self.optimizer is not None:
            try:
                checkpoint = torch.load(target_path, map_location=device)

                if (
                    isinstance(checkpoint, dict)
                    and "optimizer_state_dict" in checkpoint
                ):
                    opt_state = checkpoint["optimizer_state_dict"]
                    if opt_state is not None:
                        self.optimizer.load_state_dict(opt_state)
                        logger.info(
                            f"Process {os.getpid()} |✅ Optimizer state restored."
                        )
                    else:

                        logger.info(
                            f"Process {os.getpid()} |Loaded weights only (no optimizer state in file)."
                        )
                else:
                    logger.info(
                        f"Process {os.getpid()} |Loaded weights only (legacy format)."
                    )

            except Exception as e:
                logger.warning(
                    f"Process {os.getpid()} |❌ Failed to restore optimizer: {e}. Continue with fresh optimizer."
                )

        logger.info(
            f"Process {os.getpid()} |Agent loaded successfully from {target_path}"
        )


@torch.jit.script
def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0):
    """
    Calculate Huber loss.

    Args:
        td_errors (torch.Tensor): Temporal difference errors.
        k (float): Threshold parameter in Huber loss.

    Returns:
        torch.Tensor: Huber loss.
    """
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss


def compute_beta_regularization_loss(
    gate_outputs, has_evader_mask, gating_type="vector"
):
    """
    Computes Beta-Prior NLL Loss with Differential Configuration.

    Args:
        gate_outputs: Tensor range (0, 1).
                      Scalar: [B, 1] (Probability of Encircle mode)
                      Vector: [B, H] (Probability of Encircle feature)
        has_evader_mask: Tensor [B, 1], 1.0 if target visible, 0.0 otherwise.
        gating_type: 'scalar' or 'vector'.
    """
    # 1. Numerical Stability
    eps = 1e-6
    g = torch.clamp(gate_outputs, eps, 1.0 - eps)

    # 2. Broadcast Mask
    mask = has_evader_mask.expand_as(g)

    # ==========================================

    # ==========================================

    if gating_type == "scalar":


        no_target_beta = 1.5


        target_alpha_val = 1.0
        target_beta_val = 1.0

    else:


        no_target_beta = 4.0


        target_alpha_val = 0.2
        target_beta_val = 0.2

    # ==========================================

    # ==========================================

    target_alpha = target_alpha_val * mask + 1.0 * (1.0 - mask)
    target_beta = target_beta_val * mask + no_target_beta * (1.0 - mask)

    # ==========================================

    # ==========================================

    log_term_1 = (target_alpha - 1.0) * torch.log(g)
    log_term_2 = (target_beta - 1.0) * torch.log(1.0 - g)

    nll = -(log_term_1 + log_term_2)

    return torch.mean(nll)
