import json
import os
from typing import Dict, Tuple, Optional, List
from utils import logger as logger
from policy.common.vector_state_extractor import vector_encoder


from policy.DSGN import DSGNConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """
    [Parameter Matching Component]
    A Residual MLP Block designed to mimic the FeedForward Network (FFN)
    parameters inside a Transformer Layer.

    Structure: Input -> LayerNorm -> Linear(H->4H) -> GELU -> Linear(4H->H) -> Add -> Output
    Parameter count approx: 8 * H^2 (Similar to Transformer FFN)
    """

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
        # Residual connection structure similar to Transformer block
        """Run a forward pass."""
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class MLPIQNPolicy(nn.Module):
    """
    [Baseline Model] MLP-IQN with DeepSets Architecture & Parameter Matching

    Architecture:
    1. Independent Encoders (Shared MLPs) -> Same as TERL.
    2. Masked Mean Pooling (DeepSets) -> Handles variable length.
    3. Deep Reasoning MLP -> Replaces Transformer, matches parameter count.
    4. IQN Head -> Same as TERL.
    """

    def __init__(self, config: Optional[DSGNConfig] = None, **kwargs):
        """Initialize the instance."""
        super().__init__()

        # --- 1. Initialization & Config ---
        assert config is not None, "TERL Config must be provided for Baseline"
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.num_hiddens = config.hidden_dim
        self.device = config.device

        torch.manual_seed(config.seed)

        # Feature dimensions (Must match TERL exactly)
        self.feature_dims = {
            "self": 4,
            "pursuers": 7,
            "evaders": 7,
            "patrol_points": 6,
            "obstacles": 5,
        }
        # Order matters for concatenation
        self.entity_order = [
            "self",
            "pursuers",
            "evaders",
            "patrol_points",
            "obstacles",
        ]

        # IQN parameters
        self.K = config.num_quantiles
        self.n = config.num_cosine_features
        self.register_buffer(
            "pis",
            torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n),
        )

        # --- 2. Encoders (phi function) ---
        # Same structure as TERL
        self.entity_encoders = nn.ModuleDict(
            {
                name: vector_encoder(dim, self.num_hiddens)
                for name, dim in self.feature_dims.items()
            }
        )

        # We also keep Type Embedding to be fair, though simple concatenation makes it implicit.
        # To strictly match parameters, we can include it or skip it.
        # Here we skip it because MLP explicit concatenation already distinguishes types.

        # --- 3. Fusion & Reasoning (rho function) ---
        # Input: Self(H) + Pursuers(H) + Evaders(H) + Patrols(H) + Obstacles(H) = 5 * H
        fusion_input_dim = len(self.feature_dims) * self.num_hiddens

        # 3.1 Compression Layer: 5H -> H
        self.fusion_compression = nn.Sequential(
            nn.Linear(fusion_input_dim, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.GELU(),
        )

        # 3.2 Deep Reasoning Network (Parameter Matching)
        # Replaces the Transformer Encoder.
        # If TERL has N transformer layers, we use N ResMLPBlocks.
        reasoning_layers = []
        for _ in range(config.num_layers):
            reasoning_layers.append(ResMLPBlock(self.num_hiddens, dropout=0.1))

        self.reasoning_net = nn.Sequential(*reasoning_layers)

        # --- 4. IQN Head (Identical to TERL) ---
        self.cos_embedding = nn.Linear(self.n, self.num_hiddens)
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
        """
        Performs Mean Pooling while ignoring padding (zeros).

        Args:
            features: Encoded features [Batch, N, Hidden]
            raw_inputs: Original inputs [Batch, N, Raw_Dim] (used to detect padding)

        Returns:
            pooled: [Batch, Hidden]
        """
        # 1. Generate Mask: Check if raw input row is all zeros (Padding)
        # Assuming padding is strictly all-zeros.
        # Sum absolute values across feature dim. If close to 0, it's padding.
        mask = torch.abs(raw_inputs).sum(dim=-1) > 1e-6
        mask = mask.unsqueeze(-1).float()  # [B, N, 1]

        # 2. Mask Features (Zero out padding positions)
        masked_features = features * mask

        # 3. Sum Pooling
        sum_features = masked_features.sum(dim=1)  # [B, H]

        # 4. Count Valid Elements (Clamp to 1 to avoid divide-by-zero)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

        # 5. Mean Pooling
        mean_features = sum_features / valid_counts

        return mean_features

    def encode_and_pool(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DeepSets Logic: Encode -> Pool -> Concat -> Compress -> Reason
        """
        pooled_outputs = []

        # Iterate in fixed order: Self, Pursuers, Evaders, Patrol, Obstacles
        for entity_type in self.entity_order:

            encoder = self.entity_encoders[entity_type]
            raw_input = obs[entity_type]  # [B, N, D] or [B, D]

            # Encode
            encoded = encoder(raw_input)  # [B, N, H] or [B, H]

            if entity_type == "self":
                # Self is always single entity [B, 4]. Encoder returns [B, H].
                # No pooling needed.
                pooled_outputs.append(encoded)
            else:
                # Variable length entities [B, N, D] -> [B, N, H] -> [B, H]
                # If raw_input is empty (some edge cases), handle gracefully?
                # Usually standard collate fn pads them, so dimensions are safe.
                pooled = self._masked_mean_pooling(encoded, raw_input)
                pooled_outputs.append(pooled)

        # Concat: [B, 5 * H]
        concatenated = torch.cat(pooled_outputs, dim=-1)

        # Compress: [B, H]
        compressed = self.fusion_compression(concatenated)

        # Deep Reasoning: [B, H] -> [B, H] (through N layers)
        reasoning_context = self.reasoning_net(compressed)

        return reasoning_context

    def calc_cos(
        self, batch_size: int, num_tau: int = 8, cvar: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cosine values for IQN
        """
        if batch_size <= 0 or num_tau <= 0 or cvar <= 0:
            raise ValueError("batch_size, num_tau and cvar must be positive")

        taus = torch.rand(batch_size, num_tau).to(self.device).unsqueeze(-1)
        taus = torch.pow(taus, cvar).clamp(0, 1)
        cos = torch.cos(taus * self.pis)
        return cos, taus

    def forward(
        self, obs: Dict[str, torch.Tensor], num_tau: int = 8, cvar: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""
        self._validate_input(obs)
        batch_size = obs["self"].shape[0]

        # 1. DeepSets Aggregation & Reasoning -> [B, H]
        features = self.encode_and_pool(obs)

        # 2. IQN Processing (Cosine Embedding)
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos_features = F.gelu(self.cos_embedding(cos))  # [B, num_tau, H]

        # 3. Combine Policy Features with Quantile Features
        # features: [B, H] -> [B, 1, H]
        # mult: [B, 1, H] * [B, num_tau, H] -> [B, num_tau, H]
        # reshape: [B * num_tau, H]
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, -1)

        # 4. Output MLP
        features = F.gelu(self.hidden_layer(features))
        features = self.layer_norm(features)
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, -1), taus

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str, filename: str = None) -> None:
        """
        Save the model parameters and configuration.
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # --- Determine Paths ---
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
                for filename in os.listdir(directory):
                    if filename.startswith("network_params_v") and filename.endswith(
                        ".pth"
                    ):
                        try:
                            version = int(filename[len("network_params_v") : -4])
                            existing_versions.append(version)
                        except ValueError:
                            continue

                new_version = max(existing_versions, default=0) + 1
                params_path = os.path.join(
                    directory, f"network_params_v{new_version}.pth"
                )
                config_path = os.path.join(directory, f"config_v{new_version}.json")

            # --- Save ---
            save_dict = {"model_state_dict": self.state_dict()}
            torch.save(save_dict, params_path)

            with open(config_path, "w") as f:
                json.dump(vars(self.config), f)

            logger.info(f"Model saved to {params_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(
        cls,
        directory: str,
        device: str = "cpu",
        version: Optional[int] = None,
        exact_file_path=None,
        **kwargs,
    ) -> "MLPIQNPolicy":
        """
        Load the MLP-IQN model.
        """
        try:
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
                    for filename in os.listdir(directory):
                        if filename.startswith(
                            "network_params_v"
                        ) and filename.endswith(".pth"):
                            try:
                                ver = int(filename[len("network_params_v") : -4])
                                versions.append(ver)
                            except ValueError:
                                continue

                if not versions:
                    raise ValueError(f"No valid model files found in {directory}")

                if version is None:
                    version = max(versions)
                elif version not in versions:
                    raise FileNotFoundError(
                        f"Version {version} not found in {directory}"
                    )

                params_path = os.path.join(directory, f"network_params_v{version}.pth")
                config_path = os.path.join(directory, f"config_v{version}.json")

            # --- Load Config ---
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = DSGNConfig(**config_dict)
            else:
                raise RuntimeError(f"Config json not found for Baseline model.")

            # Update device and kwargs
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # --- Build Model ---
            model = cls(config)

            # --- Load Weights ---
            checkpoint = torch.load(params_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded MLP-IQN model weights (from full checkpoint).")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded MLP-IQN model weights (legacy format).")

            model.to(device)
            return model

        except Exception as e:
            logger.error(f"Error loading MLP-IQN model: {e}")
            raise
