import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from utils import logger as logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MlpWithTargetSelectConfig:
    """Configuration class for policy network"""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    action_size: int = 9
    num_quantiles: int = 32
    num_cosine_features: int = 64
    device: str = 'cpu'
    seed: int = 0
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0


def encoder(input_dimension: int, output_dimension: int) -> nn.Sequential:
    """Create encoder model"""
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.LayerNorm(output_dimension),
        nn.ReLU()
    )


class TargetSelectionModule(nn.Module):
    """Attention module specifically for target selection of evaders"""

    def __init__(self, hidden_dim: int):
        """Initialize the instance."""
        super().__init__()
        self.hidden_dim = hidden_dim

        self.self_transform = nn.Linear(2 * hidden_dim, hidden_dim)

        # Linear layer for transforming query
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

        # Linear layers for transforming key and value
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)

        # Layer Norm for feature fusion
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Scaling factor
        self.scale = math.sqrt(hidden_dim)

    def forward(self, self_feature: torch.Tensor, evader_features: torch.Tensor,
                evader_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_feature: Self features [batch_size, 2 * hidden_dim]
            evader_features: Evader features [batch_size, num_evaders, hidden_dim]
            evader_mask: Evader mask [batch_size, num_evaders]

        Returns:
            enhanced_feature: Enhanced features [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, num_evaders]
        """
        batch_size = self_feature.shape[0]  # [B, 2H]
        hidden_dim = self.hidden_dim
        num_evaders = evader_features.shape[1]

        # Add shape checking
        assert self_feature.shape == (
            batch_size, 2 * hidden_dim), f"Unexpected self_feature shape: {self_feature.shape}"
        assert evader_features.shape == (
            batch_size, num_evaders, hidden_dim), f"Unexpected evader_features shape: {evader_features.shape}"
        assert evader_mask.shape == (batch_size, num_evaders), f"Unexpected evader_mask shape: {evader_mask.shape}"

        self_feature = self.self_transform(self_feature)  # [B, H] # Transform self features

        # Transform query (self features)
        query = self.query_transform(self_feature).unsqueeze(1)  # [B, 1, H]

        # Transform key and value (evader features)
        keys = self.key_transform(evader_features)  # [B, N, H]
        values = self.value_transform(evader_features)  # [B, N, H]

        # Calculate attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # [B, 1, N]

        # Validate scores shape
        assert scores.shape == (batch_size, 1, num_evaders), f"Unexpected scores shape: {scores.shape}"

        # Apply mask
        if evader_mask is not None:
            scores = scores.masked_fill((~evader_mask.bool()).unsqueeze(1), float('-inf'))

        # Get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, N]

        # Get weighted features
        weighted_features = torch.matmul(attention_weights, values)  # [B, 1, H]
        weighted_features = weighted_features.squeeze(1)  # [B, H]

        # Feature fusion and normalization
        enhanced_feature = self.layer_norm(self_feature + weighted_features)

        # Final shape checking
        assert enhanced_feature.shape == (batch_size, hidden_dim)
        assert attention_weights.squeeze(1).shape == (batch_size, num_evaders)

        return enhanced_feature, attention_weights.squeeze(1)


class MlpWithTargetSelectPolicy(nn.Module):
    """
    Policy network implementation
    Uses Transformer architecture to process multi-entity inputs and IQN method to estimate action value distribution
    """

    def __init__(self, config: Optional[MlpWithTargetSelectConfig] = None, **kwargs):
        """
        Initialize policy network

        Args:
            config: configuration object
            **kwargs: Optional configuration parameters, will override defaults in config
        """
        super().__init__()

        # Initialize parameters using config class or kwargs
        if config is None:
            config = MlpWithTargetSelectConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        torch.manual_seed(config.seed)

        # Define feature dimensions for various entity types
        self.feature_dims = {
            'self': 4,  # [vx, vy, min_obs_dis, pursuing_signal]
            'pursuers': 7,  # [px, py, vx, vy, dist, angle, pursuing_signal]
            'evaders': 7,  # [px, py, vx, vy, dist, pos_angle, head_angle]
            'obstacles': 5  # [px, py, radius, dist, angle]
        }

        # IQN parameters
        self.K = config.num_quantiles
        self.n = config.num_cosine_features

        self.register_buffer('pis',
                             torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n))

        # Define feature encoders
        self.entity_encoders = nn.ModuleDict({
            name: encoder(dim, self.hidden_dim)
            for name, dim in self.feature_dims.items()
        })

        # Type embedding
        self.type_embedding = nn.Embedding(4, self.hidden_dim)

        # mlp embedding, ablation study
        self.global_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.feature_dims), self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU()
        )

        # Add target selection module
        self.target_selection = TargetSelectionModule(self.hidden_dim)

        # IQN related layers
        self.cos_embedding = nn.Linear(self.n, self.hidden_dim)

        # Output layer
        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, config.action_size)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # Initialize weights
        # self._init_weights()

        # Move model to specified device
        self.to(self.device)

        # For storing the last attention weights
        self._last_target_weights = None
        self._last_transformer_weights = None

    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    # Embedding layer initialized with standard normal distribution
                    nn.init.normal_(param.data, mean=0.0, std=0.02)
                elif 'layer_norm' in name:
                    # LayerNorm layer weights initialized to 1
                    nn.init.constant_(param.data, 1.0)
                elif 'output_layer' in name:
                    # Output layer uses a smaller initialization range to avoid overly large initial values
                    nn.init.uniform_(param.data, -0.003, 0.003)
                elif param.dim() > 1:
                    # For 2D and higher weights, use orthogonal initialization with gain=1/sqrt(2) for better initial scaling
                    nn.init.orthogonal_(param.data, gain=1 / math.sqrt(2))
                else:
                    # 1D weights use uniform distribution
                    bound = 1 / math.sqrt(param.size(0))
                    nn.init.uniform_(param.data, -bound, bound)
            elif 'bias' in name:
                # Bias terms initialized to 0
                nn.init.constant_(param.data, 0)

    def _validate_input(self, obs: Dict[str, torch.Tensor]) -> None:
        """Validate the format and dimensions of input data"""
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        required_keys = {'self', 'types', 'masks'}
        if not all(key in obs for key in required_keys):
            raise ValueError(f"obs must contain keys: {required_keys}")

        if obs['self'].dim() != 2:
            raise ValueError("self features must be 2-dimensional [batch_size, feature_dim]")

        if obs['self'].shape[1] != self.feature_dims['self']:
            raise ValueError(f"self features must have dimension {self.feature_dims['self']}")

    def encode_entities(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode entity features and process through Transformer

        Args:
            obs: Dictionary containing features of various entities

        Returns:
            torch.Tensor: Encoded features [batch_size, hidden_dim]
        """
        B = obs['self'].shape[0]
        encoded_features = dict()

        # Encode various entities
        for entity_type, encoder in self.entity_encoders.items():
            features = obs[entity_type]
            if entity_type == 'self':
                encoded = encoder(features).unsqueeze(1)
            else:
                encoded = encoder(features)
            encoded_features[entity_type] = encoded

        mean_feature_per_type = dict()

        for type_index, key in enumerate(encoded_features.keys()):
            type_indices = obs['types'] == type_index
            number_per_batch = type_indices.sum(dim=1)
            max_number = number_per_batch.max().item()

            flat_mask = obs['masks'].masked_select(type_indices)
            mask = flat_mask.reshape(B, max_number).long()

            valid_len = mask.sum(dim=1).clamp(min=1)
            mean_feature_per_type[key] = (encoded_features[key] * mask.unsqueeze(-1)).sum(dim=1) / valid_len.unsqueeze(-1)

        # ablation study, replace transformer with mlp
        mlp_embedding = torch.cat(list(mean_feature_per_type.values()), dim=-1)  # [B, H * 4]

        mlp_embedding = self.global_mlp(mlp_embedding)  # [B, 2H]

        # Extract evader features and mask - uniformly handle batch and single samples
        evader_indices = (obs['types'] == 2)  # [B, N]

        # Get the number of positions with type==2 per batch
        num_evaders_per_batch = evader_indices.sum(dim=1)  # [B]
        max_evaders = num_evaders_per_batch.max().item()

        # Use masked_select and reshape to handle irregular selections
        flat_mask = obs['masks'].masked_select(evader_indices)
        flat_features = encoded_features['evaders']

        # Reshape into regular shape
        evader_mask = flat_mask.reshape(B, max_evaders)  # [B, max_evaders]
        evader_features = flat_features.reshape(B, max_evaders, -1)  # [B, max_evaders, H]

        # Apply target selection module
        enhanced_features, attention_weights = self.target_selection(
            mlp_embedding,
            evader_features,
            evader_mask
        )

        # Store attention weights for visualization
        self._last_target_weights = attention_weights

        return enhanced_features

    def calc_cos(self, batch_size: int, num_tau: int = 8, cvar: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self,
                obs: Dict[str, torch.Tensor],
                num_tau: int = 8,
                cvar: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation

        Args:
            obs: Observation data dictionary
            num_tau: Number of tau samples
            cvar: CVaR parameter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quantiles: [batch_size, num_tau, action_size]
                - taus: [batch_size, num_tau, 1]
        """
        self._validate_input(obs)
        batch_size = obs['self'].shape[0]

        # Transformer encoding of observations
        features = self.encode_entities(obs)

        # IQN processing
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos_features = F.relu(self.cos_embedding(cos))

        # Feature combination
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, -1)

        # Output layer
        features = F.relu(self.hidden_layer(features))
        features = self.layer_norm(features)
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, -1), taus

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get the attention weights from the last forward pass"""
        return {
            'target_selection': self._last_target_weights,
            'transformer': self._last_transformer_weights
        }

    def count_parameters(self) -> int:
        """Count the number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: str) -> None:
        """
        Save the model, using version numbers to avoid overwriting existing files

        Args:
            directory: Save directory
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # Find existing version numbers in the directory
            existing_versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        version = int(filename[len("network_params_v"):-4])
                        existing_versions.append(version)
                    except ValueError:
                        continue

            # Determine the new version number
            new_version = max(existing_versions, default=0) + 1

            # Save model parameters with the new version number
            params_path = os.path.join(directory, f"network_params_v{new_version}.pth")
            torch.save(self.state_dict(), params_path)

            # Save the corresponding version configuration file
            config_path = os.path.join(directory, f"config_v{new_version}.json")
            with open(config_path, 'w') as f:
                json.dump(vars(self.config), f)

            logger.info(f"Model saved as version {new_version} in {directory}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls,
             directory: str,
             device: str = 'cpu',
             version: Optional[int] = None,
             **kwargs) -> 'MlpWithTargetSelectPolicy':
        """
        Load the model

        Args:
            directory: Model directory
            version: Specific version number to load, if None then load the latest version
            device: Device
            **kwargs: Other parameters

        Returns:
            MlpWithTargetSelectPolicy: Loaded model

        Raises:
            FileNotFoundError: When the specified version's model file does not exist
            ValueError: When there are no valid model files in the directory
        """
        try:
            # Get all version numbers
            versions = []
            for filename in os.listdir(directory):
                if filename.startswith("network_params_v") and filename.endswith(".pth"):
                    try:
                        ver = int(filename[len("network_params_v"):-4])
                        versions.append(ver)
                    except ValueError:
                        continue

            if not versions:
                raise ValueError(f"No valid model files found in {directory}")

            # Determine the version to load
            if version is None:
                version = max(versions)  # Load the latest version
            elif version not in versions:
                raise FileNotFoundError(f"Version {version} not found in {directory}")

            params_path = os.path.join(directory, f"network_params_v{version}.pth")
            config_path = os.path.join(directory, f"config_v{version}.json")

            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = MlpWithTargetSelectConfig(**config_dict)
            else:
                logger.warning(f"Config file for version {version} not found, using default configuration")
                config = MlpWithTargetSelectConfig()

            # Update device and other parameters
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Create model and load parameters
            model = cls(config)
            model.load_state_dict(torch.load(params_path, map_location=device))
            model.to(device)

            logger.info(f"Successfully loaded model version {version} from {directory}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
