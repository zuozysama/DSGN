import json
import os
from typing import Dict, Tuple, Optional
from utils import logger as logger
from policy.common.vector_state_extractor import vector_encoder
from policy.DSGN import DSGNConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """Implement a residual MLP block."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize the instance."""
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class MLPDQNPolicy(nn.Module):
    """
    [Baseline Model] MLP-DQN with DeepSets Architecture

    Architecture:
    1. Encoders: Shared MLPs (same as TERL/IQN).
    2. Aggregation: Masked Mean Pooling (DeepSets).
    3. Reasoning: Stacked ResMLPBlocks.
    4. Head: Standard DQN Head (Outputs Q-values directly).

    Difference from IQN:
    - No Cosine Embedding.
    - No Quantile dimensions.
    - Output shape is [Batch, Action_Size] instead of [Batch, N, Action_Size].
    """

    def __init__(self, config: Optional[DSGNConfig] = None, **kwargs):
        """Initialize the instance."""
        super().__init__()

        # --- 1. Config Init ---
        assert config is not None, "TERL Config must be provided"
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.num_hiddens = config.hidden_dim
        self.device = config.device

        torch.manual_seed(config.seed)

        # Feature specs
        self.feature_dims = {
            "self": 4,
            "pursuers": 7,
            "evaders": 7,
            "patrol_points": 6,
            "obstacles": 5,
        }
        self.entity_order = [
            "self",
            "pursuers",
            "evaders",
            "patrol_points",
            "obstacles",
        ]

        # --- 2. Encoders ---
        self.entity_encoders = nn.ModuleDict(
            {
                name: vector_encoder(dim, self.num_hiddens)
                for name, dim in self.feature_dims.items()
            }
        )

        # --- 3. Fusion ---
        fusion_input_dim = len(self.feature_dims) * self.num_hiddens
        self.fusion_compression = nn.Sequential(
            nn.Linear(fusion_input_dim, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.GELU(),
        )

        # --- 4. Deep Reasoning ---
        reasoning_layers = []
        for _ in range(config.num_layers):
            reasoning_layers.append(ResMLPBlock(self.num_hiddens, dropout=0.1))
        self.reasoning_net = nn.Sequential(*reasoning_layers)

        # --- 5. DQN Head (Simplified) ---

        self.hidden_layer = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.output_layer = nn.Linear(self.num_hiddens, config.action_size)
        self.layer_norm = nn.LayerNorm(self.num_hiddens)

        self.to(self.device)

    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the input tensors before a forward pass."""
        if not isinstance(obs, dict):
            raise ValueError(f"obs must be a dictionary, got {type(obs)}")
        for key in self.entity_order:
            if key not in obs:
                raise KeyError(
                    f"Missing required key '{key}' in observation dictionary."
                )

    def _masked_mean_pooling(
        self, features: torch.Tensor, raw_inputs: torch.Tensor
    ) -> torch.Tensor:
        """DeepSets Aggregation"""
        is_active = torch.abs(raw_inputs).sum(dim=-1) > 1e-6
        mask = is_active.unsqueeze(-1).float()
        masked_features = features * mask
        sum_features = masked_features.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        return sum_features / valid_counts

    def encode_and_pool(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Same DeepSets logic as MLP-IQN"""
        pooled_outputs = []
        for entity_type in self.entity_order:
            encoder = self.entity_encoders[entity_type]
            raw_input = obs[entity_type]
            encoded = encoder(raw_input)

            if entity_type == "self":
                if encoded.dim() == 3 and encoded.shape[1] == 1:
                    pooled_outputs.append(encoded.squeeze(1))
                elif encoded.dim() == 2:
                    pooled_outputs.append(encoded)
                else:
                    raise ValueError(f"'self' encoding shape error")
            else:
                pooled = self._masked_mean_pooling(encoded, raw_input)
                pooled_outputs.append(pooled)

        concatenated = torch.cat(pooled_outputs, dim=-1)
        compressed = self.fusion_compression(concatenated)
        context = self.reasoning_net(compressed)
        return context

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DQN Forward Pass
        Returns: [Batch, Action_Size] (Q-values)
        """
        self._validate_input(obs)

        # 1. Backbone: [B, H]
        features = self.encode_and_pool(obs)

        # 2. Output Head (No Cosine Integration)
        features = F.gelu(self.hidden_layer(features))
        features = self.layer_norm(features)
        q_values = self.output_layer(features)  # [B, Action_Size]

        return q_values

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str, filename: str = None) -> None:

        """Save model parameters to disk."""
        os.makedirs(directory, exist_ok=True)
        if filename:
            params_path = os.path.join(directory, filename)
            config_name = (
                filename.replace(".pth", "").replace(".pt", "") + "_config.json"
            )
            if config_name == filename:
                config_name = "config.json"
            config_path = os.path.join(directory, config_name)
        else:
            existing_versions = []
            for fname in os.listdir(directory):
                if fname.startswith("network_params_v") and fname.endswith(".pth"):
                    try:
                        version = int(fname[len("network_params_v") : -4])
                        existing_versions.append(version)
                    except ValueError:
                        continue
            new_version = max(existing_versions, default=0) + 1
            params_path = os.path.join(directory, f"network_params_v{new_version}.pth")
            config_path = os.path.join(directory, f"config_v{new_version}.json")

        save_dict = {"model_state_dict": self.state_dict()}
        torch.save(save_dict, params_path)
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f)
        logger.info(f"MLP-DQN Model saved to {params_path}")

    @classmethod
    def load(
        cls,
        directory: str,
        device: str = "cpu",
        version: Optional[int] = None,
        exact_file_path=None,
        **kwargs,
    ) -> "MLPDQNPolicy":

        """Load model parameters from disk."""
        if exact_file_path:
            params_path = exact_file_path
            config_path = exact_file_path.replace(".pth", "_config.json")
            if not os.path.exists(config_path):
                config_path = (
                    exact_file_path.replace(".pth", "").replace(".pt", "")
                    + "_config.json"
                )
                if config_path == exact_file_path:
                    config_path = os.path.join(directory, "config.json")
        else:
            versions = []
            if os.path.exists(directory):
                for fname in os.listdir(directory):
                    if fname.startswith("network_params_v") and fname.endswith(".pth"):
                        try:
                            ver = int(fname[len("network_params_v") : -4])
                            versions.append(ver)
                        except ValueError:
                            continue
            if not versions:
                raise ValueError(f"No valid model files found in {directory}")
            if version is None:
                version = max(versions)
            params_path = os.path.join(directory, f"network_params_v{version}.pth")
            config_path = os.path.join(directory, f"config_v{version}.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DSGNConfig(**config_dict)
        config.device = device
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        model = cls(config)
        checkpoint = torch.load(params_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        logger.info(f"Loaded MLP-DQN model from {params_path}")
        return model
