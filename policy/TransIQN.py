import json
import os
from typing import Dict, Tuple, Optional
from utils import logger as logger
from policy.common.vector_state_extractor import vector_encoder


try:
    from policy.DSGN import DSGNConfig
except ImportError:

    from dataclasses import dataclass

    @dataclass
    class DSGNConfig:
        """Implement the DSGNConfig helper."""
        hidden_dim: int = 128
        num_heads: int = 4
        num_layers: int = 2
        action_size: int = 5
        num_quantiles: int = 32
        num_cosine_features: int = 64
        device: str = "cpu"
        seed: int = 42
        learning_rate: float = 1e-4
        gradient_clip: float = 10.0
        gating_type: str = "vector"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """Implement a residual MLP block."""

    def __init__(
        self, hidden_dim: int, expansion_factor: int = 8, dropout: float = 0.1
    ):
        """Initialize the instance."""
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)


        self.linear1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
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


class TransIQNPolicy(nn.Module):
    """
    [Baseline Model] Trans-IQN (Monolithic Architecture)

    Design Philosophy:
    - This serves as the "Strongest Baseline" to isolate the contribution of Task Disentanglement.
    - Components:
        1. Identical Encoders as TERL.
        2. Identical Transformer Encoder as TERL (Same depth/heads).
        3. NO Dual-Stream (Target/Patrol Selection).
        4. NO Gating Mechanism.
    - Aggregation:
        It uses the Transformer's output at Index 0 (Self-Token) as the monolithic context vector,
        forcing the network to compress both Patrol and Encirclement intents into a single representation.
    """

    def __init__(self, config: Optional[DSGNConfig] = None, **kwargs):
        """
        Initialize Trans-IQN policy network
        """
        super().__init__()

        # --- 1. Init & Config ---
        assert config is not None, "TERL Config must be provided"
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.num_hiddens = config.hidden_dim
        self.device = config.device

        torch.manual_seed(config.seed)

        # Feature dimensions (Same as TERL)
        self.feature_dims = {
            "self": 4,
            "pursuers": 7,
            "evaders": 7,
            "patrol_points": 6,
            "obstacles": 5,
        }
        # Order: Self(0) -> Pursuers(1) -> Evaders(2) -> Patrol(3) -> Statics(4)
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

        # --- 2. Encoders (Identical to TERL) ---
        self.entity_encoders = nn.ModuleDict(
            {
                name: vector_encoder(dim, self.num_hiddens)
                for name, dim in self.feature_dims.items()
            }
        )
        self.type_embedding = nn.Embedding(
            len(self.feature_dims.keys()), self.num_hiddens
        )

        # --- 3. Transformer Encoder (Identical to TERL) ---

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_hiddens,
            nhead=config.num_heads,
            dim_feedforward=self.num_hiddens * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )

        # LayerNorm for the Monolithic Context
        self.ln_context = nn.LayerNorm(self.num_hiddens)

        # --- 4. Parameter Matching Block (Updated!) ---


        self.param_matching_block = ResMLPBlock(
            hidden_dim=self.num_hiddens, expansion_factor=8, dropout=0.1
        )

        # --- 5. IQN Head (Identical) ---
        self.cos_embedding = nn.Linear(self.n, self.num_hiddens)
        self.hidden_layer = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.output_layer = nn.Linear(self.num_hiddens, config.action_size)
        self.layer_norm = nn.LayerNorm(self.num_hiddens)

        self.to(self.device)

    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the input tensors before a forward pass."""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")
        if obs["self"].dim() != 2:
            raise ValueError("self features must be 2D")
        for key in self.entity_order:
            if key not in obs:
                raise KeyError(f"Missing required key '{key}' in observation.")

    def encode_monolithic(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes entities and aggregates them via Transformer into a single Monolithic Context.

        Returns:
            context_feature: [Batch, Hidden]
        """
        B = obs["self"].shape[0]
        encoded_features = []

        # 1. Feature Encoding
        for entity_type in self.entity_order:
            encoder = self.entity_encoders[entity_type]
            features = obs[entity_type]
            if entity_type == "self":
                encoded = encoder(features).unsqueeze(1)
            else:
                encoded = encoder(features)
            encoded_features.append(encoded)

        entity_embed = torch.cat(encoded_features, dim=1)
        type_embed = self.type_embedding(obs["types"].long())
        tokens = entity_embed + type_embed

        # 2. Transformer Processing
        # mask: 1=Valid, 0=Padding. Transformer needs ~mask (True=Padding)
        attention_mask = ~obs["masks"].bool()
        transformed = self.transformer_encoder(
            tokens, src_key_padding_mask=attention_mask
        )

        # 3. Monolithic Aggregation
        # Unlike TERL which splits this into streams, Trans-IQN assumes
        # the Agent Token (Index 0) contains all necessary global information.
        context_feature = transformed[:, 0, :]  # [B, H]

        # Normalize
        context_feature = self.ln_context(context_feature)

        # 4. Parameter Matching / Reasoning Depth
        context_feature = self.param_matching_block(context_feature)

        return context_feature

    def calc_cos(
        self, batch_size: int, num_tau: int = 8, cvar: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cosine values
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

        # 1. Get Monolithic Context [B, H]
        features = self.encode_monolithic(obs)

        # 2. IQN Processing
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos_features = F.gelu(self.cos_embedding(cos))  # [B, num_tau, H]

        # 3. Combine Policy Features with Quantile Features
        # features: [B, H] -> [B, 1, H] * [B, num_tau, H] -> [B, num_tau, H]
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, -1)

        # 4. Output
        features = F.gelu(self.hidden_layer(features))
        features = self.layer_norm(features)
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, -1), taus

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --- Save & Load Implementation (Strictly Matching TERL) ---
    def save(self, directory: str, filename: str = None) -> None:
        """
        Save the model parameters and configuration.
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # --- 1. Determine Paths ---
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
                params_path = os.path.join(
                    directory, f"network_params_v{new_version}.pth"
                )
                config_path = os.path.join(directory, f"config_v{new_version}.json")

            # --- 2. Construct Dictionary ---
            save_dict = {"model_state_dict": self.state_dict()}

            # --- 3. Save ---
            torch.save(save_dict, params_path)

            with open(config_path, "w") as f:
                json.dump(vars(self.config), f)

            logger.info(f"Trans-IQN Model saved to {params_path}")

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
    ) -> "TransIQNPolicy":
        """
        Load the Trans-IQN model (Compatible with both weights-only and full-checkpoint formats).
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
                # --- 1. Find Version ---
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

            # --- 2. Load Config ---
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = DSGNConfig(**config_dict)
            else:
                raise RuntimeError(f"Config json not found. Cannot reconstruct model.")

            # Update device parameters
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # --- 3. Build Model ---
            model = cls(config)

            # --- 4. Load Weights ---
            checkpoint = torch.load(params_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded Trans-IQN model weights (from full checkpoint).")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded Trans-IQN model weights (legacy format).")

            model.to(device)
            return model

        except Exception as e:
            logger.error(f"Error loading Trans-IQN model: {e}")
            raise
