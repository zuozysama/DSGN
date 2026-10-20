import json
import math
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
from utils import logger as logger
from policy.common.vector_state_extractor import vector_encoder
from policy.common.TargetSelectionModule import TargetSelectionModule
from policy.common.patrol_selection_module import PatrolSelectionModule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSGNConfig:
    """Configuration class for DSGN policy network"""

    hidden_dim: int
    num_heads: int
    num_layers: int
    action_size: int
    num_quantiles: int
    num_cosine_features: int
    device: str
    seed: int
    learning_rate: float
    gradient_clip: float

    gating_type: Literal["scalar", "vector", "average", "random", "rule_based"]


class GatedFusionModule(nn.Module):
    """Fuse patrol and target features with a gating mechanism."""

    def __init__(self, feature_dim: int, gating_type: str):
        """Initialize the instance."""
        super().__init__()
        self.gating_type = gating_type
        self.feature_dim = feature_dim


        self.gating_input_dim = feature_dim * 3


        if self.gating_type in ["scalar", "vector"]:
            self.gating_net_base = nn.Sequential(
                nn.Linear(self.gating_input_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
            )

            if self.gating_type == "scalar":
                logger.info("Initializing with Informed Scalar Gating (2D-Softmax)")
                self.gating_head = nn.Linear(feature_dim, 2)
                self.gating_head.weight.data.uniform_(-1e-3, 1e-3)
                self.gating_head.bias.data.zero_()

            elif self.gating_type == "vector":
                logger.info("Initializing with Informed Vector Gating (D-dim Sigmoid)")
                self.gating_head = nn.Linear(feature_dim, feature_dim)
                self.gating_head.weight.data.uniform_(-1e-3, 1e-3)
                self.gating_head.bias.data.zero_()

        else:
            logger.info(
                f"⚠️ [Baseline] Initializing with {self.gating_type} Gating (No Parameters)"
            )


        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        context_feature: torch.Tensor,
        target_feature: torch.Tensor,
        patrol_feature: torch.Tensor,
        rule_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Run a forward pass."""
        batch_size = context_feature.shape[0]
        device = context_feature.device


        if self.gating_type == "average":
            # Fixed 0.5
            w_target = 0.5
            w_patrol = 0.5

            # Logs
            gate_outputs_for_loss = torch.zeros(1, device=device)  # Dummy
            fused_weights_log = torch.tensor([0.5, 0.5], device=device)

        elif self.gating_type == "random":
            # Random U(0, 1)
            # [B, 1]
            rand_val = torch.rand(batch_size, 1, device=device)
            w_target = rand_val
            w_patrol = 1.0 - rand_val

            gate_outputs_for_loss = rand_val
            fused_weights_log = torch.cat([rand_val, 1.0 - rand_val], dim=1)  # [B, 2]

        elif self.gating_type == "rule_based":
            # Rule-based (Distance Inversed)
            if rule_weight is None:
                raise ValueError("rule_weight must be provided for rule_based gating")


            w_target = rule_weight
            w_patrol = 1.0 - rule_weight

            gate_outputs_for_loss = rule_weight
            fused_weights_log = torch.cat([rule_weight, 1.0 - rule_weight], dim=1)

        else:
            # Learned Gating (Scalar/Vector)
            # Prepare inputs: Concat[Context, Target, Patrol]
            gating_inputs = torch.cat(
                [context_feature, target_feature, patrol_feature], dim=-1
            )

            gate_hidden = self.gating_net_base(gating_inputs)
            gate_logits = self.gating_head(gate_hidden)

            if self.gating_type == "scalar":
                # [B, 2]
                weights = F.softmax(gate_logits, dim=-1)
                w_target = weights[:, 0].unsqueeze(1)  # [B, 1]
                w_patrol = weights[:, 1].unsqueeze(1)  # [B, 1]

                gate_outputs_for_loss = weights
                fused_weights_log = weights

            elif self.gating_type == "vector":
                # [B, H]
                g = torch.sigmoid(gate_logits)
                w_target = g
                w_patrol = 1.0 - g

                gate_outputs_for_loss = g
                fused_weights_log = torch.stack(
                    [g.mean(dim=-1), (1 - g).mean(dim=-1)], dim=-1
                )


        weighted_target_feat = w_target * target_feature
        weighted_patrol_feat = w_patrol * patrol_feature


        # [B, 3H] -> [B, H] via Projection
        fusion_input = torch.cat(
            [context_feature, weighted_target_feat, weighted_patrol_feat], dim=-1
        )
        final_output = self.fusion_projection(fusion_input)

        return final_output, gate_outputs_for_loss, fused_weights_log


class DSGNPolicy(nn.Module):
    """
    DSGN policy network implementation (v5 Architecture Upgrade)
    With Baseline Support
    """

    def __init__(self, config: Optional[DSGNConfig] = None, **kwargs):
        """
        Initialize DSGN policy network

        Args:
            config: DSGN configuration object
            **kwargs: Optional configuration parameters, will override defaults in config
        """
        super().__init__()

        # Initialize parameters using config class or kwargs
        # if config is None:
        #     config = TERLConfig()
        assert config is not None, "DSGN Config must be provided"
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.num_hiddens = config.hidden_dim
        self.device = config.device
        self.gating_type = self.config.gating_type

        # Logs
        self.attention_weights = None
        # This attribute will store outputs needed for regularization loss
        self.gate_outputs_for_loss = None

        torch.manual_seed(config.seed)

        # Feature dimensions
        self.feature_dims = {
            "self": 4,  # [vx, vy, min_obs_dis, pursuing_signal]
            "pursuers": 7,  # [px, py, vx, vy, dist, angle, pursuing_signal]
            "evaders": 7,  # [px, py, vx, vy, dist, pos_angle, head_angle]
            "patrol_points": 6,  # [px, py, dist, angle, visited_count, last_visited_time]
            "obstacles": 5,  # [px, py, radius, dist, angle]
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

        # Encoders
        self.entity_encoders = nn.ModuleDict(
            {
                name: vector_encoder(dim, self.num_hiddens)
                for name, dim in self.feature_dims.items()
            }
        )
        self.type_embedding = nn.Embedding(
            len(self.feature_dims.keys()), self.num_hiddens
        )

        # Transformer Encoder
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

        # LayerNorms (Agent-Centric Context is 1*H)
        self.ln_att_context = nn.LayerNorm(self.num_hiddens)
        self.ln_att_evaders = nn.LayerNorm(self.num_hiddens)
        self.ln_att_patrols = nn.LayerNorm(self.num_hiddens)

        # Sub-Modules
        self.target_selection = TargetSelectionModule(self.num_hiddens)
        self.patrol_selection = PatrolSelectionModule(self.num_hiddens)

        # Gated Fusion (Feature dim is H)
        self.gated_fusion_module = GatedFusionModule(
            feature_dim=self.num_hiddens, gating_type=self.gating_type
        )

        # IQN & Output
        self.cos_embedding = nn.Linear(self.n, self.num_hiddens)
        self.hidden_layer = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.output_layer = nn.Linear(self.num_hiddens, config.action_size)
        self.layer_norm = nn.LayerNorm(self.num_hiddens)

        self.to(self.device)
        self._last_target_weights = None
        self._last_transformer_weights = None

    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the input tensors before a forward pass."""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")
        if obs["self"].dim() != 2:
            raise ValueError("self features must be 2D")

    def encode_entities(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Modified Encoder with Agent-Centric Pooling and Baseline Logic
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
        attention_mask = ~obs["masks"].bool()
        transformed = self.transformer_encoder(
            tokens, src_key_padding_mask=attention_mask
        )

        # --- Agent-Centric Pooling ---

        context_feature = transformed[:, 0, :]  # [B, H]

        # 3. Extract Entity Features
        type_evader = self.entity_order.index("evaders")
        type_patrol = self.entity_order.index("patrol_points")

        # Evaders
        evader_indices = obs["types"] == type_evader
        num_evaders_per_batch = evader_indices.sum(dim=1)
        max_evaders = num_evaders_per_batch.max().item()

        flat_mask_e = obs["masks"].masked_select(evader_indices)
        flat_features_e = transformed.masked_select(
            evader_indices.unsqueeze(-1).expand(-1, -1, transformed.size(-1))
        )
        evader_mask = flat_mask_e.reshape(B, max_evaders)
        evader_features = flat_features_e.reshape(B, max_evaders, -1)

        # Patrols
        patrol_indices = obs["types"] == type_patrol
        num_patrols_per_batch = patrol_indices.sum(dim=1)
        max_patrols = num_patrols_per_batch.max().item()

        flat_mask_p = obs["masks"].masked_select(patrol_indices)
        flat_features_p = transformed.masked_select(
            patrol_indices.unsqueeze(-1).expand(-1, -1, transformed.size(-1))
        )
        patrol_mask = flat_mask_p.reshape(B, max_patrols)
        patrol_features = flat_features_p.reshape(B, max_patrols, -1)


        rule_weight = None
        if self.gating_type == "rule_based":

            # obs['evaders'] shape: [B, max_evaders, 7]

            raw_evader_feats = obs["evaders"]


            # shape: [B, max_evaders]
            dists = torch.norm(raw_evader_feats[:, :, :2], dim=-1)


            # obs['masks']: 1=Valid, 0=Padding


            # evader_mask: [B, max_evaders], 1=Valid, 0=Padding
            valid_mask = evader_mask.bool()
            dists_masked = dists.clone()
            dists_masked[~valid_mask] = float("inf")


            # min_dist: [B, 1]
            min_dist, _ = dists_masked.min(dim=1, keepdim=True)


            reaction_dist = 10.0


            rule_weight = torch.clamp(reaction_dist / (min_dist + 0.1), 0.0, 1.0)


            rule_weight[min_dist == float("inf")] = 0.0

        # 4. Normalization
        context_norm = self.ln_att_context(context_feature)
        evaders_norm = self.ln_att_evaders(evader_features)
        patrols_norm = self.ln_att_patrols(patrol_features)

        # 5. Head Processing
        target_feat, att_w_t = self.target_selection(
            context_norm, evaders_norm, evader_mask
        )
        patrol_feat, att_w_p = self.patrol_selection(
            context_norm, patrols_norm, patrol_mask
        )

        # 6. Gated Fusion (With Baseline Support)
        fused_features, gate_outputs, fused_weights_log = self.gated_fusion_module(
            context_norm,
            target_feat,
            patrol_feat,
            rule_weight=rule_weight,  # Pass rule weight if computed
        )

        # 7. Store Logs
        self.gate_outputs_for_loss = gate_outputs
        AttentionWeights = namedtuple(
            "AttentionWeights", ["target_selection", "patrol_selection", "fused"]
        )
        self.attention_weights = AttentionWeights(
            target_selection=att_w_t,
            patrol_selection=att_w_p,
            fused=fused_weights_log.detach(),
        )

        return fused_features

    def calc_cos(
        self, batch_size: int, num_tau: int = 8, cvar: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cosine values

        Args:
            batch_size: Batch size
            num_tau: Number of tau samples
            cvar: CVaR parameter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine values and tau values
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

        # 1. Encode & Fuse
        features = self.encode_entities(obs)  # [B, H]

        # 2. IQN Processing
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos_features = F.gelu(self.cos_embedding(cos))  # [B, num_tau, H]

        # 3. Combine Policy Features with Quantile Features
        # features: [B, H] -> [B, 1, H]
        # cos: [B, num_tau, H]
        # Element-wise mult and reshape -> [B * num_tau, H]
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, -1)

        # 4. Output
        features = F.gelu(self.hidden_layer(features))
        features = self.layer_norm(features)
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, -1), taus

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return attention weights for inspection."""
        return {
            "target_selection": self.attention_weights.target_selection
            if self.attention_weights
            else None,
            "fused": self.attention_weights.fused if self.attention_weights else None,
        }

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str, filename: str = None) -> None:
        """
        Save the model parameters and configuration.

        Args:
            directory: Save directory.
            file_name: If provided, save to this specific filename (overwrite).
                       If None, use auto-increment versioning (network_params_v{N}.pth).
        """
        try:
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
    ) -> "DSGNPolicy":
        """
        Load the DSGN model (Compatible with both weights-only and full-checkpoint formats).
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


            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = DSGNConfig(**config_dict)
            else:

                raise RuntimeError(
                    f"DSGN config json not found for version {version}. Cannot reconstruct model."
                )


            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)


            model = cls(config)


            checkpoint = torch.load(params_path, map_location=device)


            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    f"Loaded model weights (from full checkpoint) version {version}"
                )
            else:

                model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights (legacy format) version {version}")

            model.to(device)
            return model

        except Exception as e:
            logger.error(f"Error loading DSGN model: {e}")
            raise
