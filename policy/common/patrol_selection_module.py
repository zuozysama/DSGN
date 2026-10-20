import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PatrolSelectionModule(nn.Module):
    """Attention module for selecting patrol targets"""

    def __init__(self, hidden_dim: int = 256):
        """Initialize the instance."""
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project the agent's self feature (Self + Global Context)
        self.self_transform = nn.Linear(hidden_dim, hidden_dim)

        # Query comes from self feature
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

        # Key/Value come from patrol target embeddings
        self.key_transform = nn.Linear(self.hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(self.hidden_dim, hidden_dim)


        # self.layer_norm = nn.LayerNorm(hidden_dim)

        # Attention scale factor
        self.scale = math.sqrt(hidden_dim)

    def forward(self, self_feature: torch.Tensor, patrol_features: torch.Tensor,
                patrol_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_feature: Tensor of shape [B, 2 * hidden_dim]
            patrol_features: Tensor of shape [B, N_patrol, hidden_dim]
            patrol_mask: Tensor of shape [B, N_patrol], True for valid, False for masked

        Returns:
            enhanced_feature: [B, hidden_dim]
            attention_weights: [B, N_patrol]
        """
        B, N, H = patrol_features.shape

        # Shape Checking
        assert self_feature.shape == (B, self.hidden_dim)
        assert patrol_features.shape == (B, N, self.hidden_dim)
        if patrol_mask is not None:
            assert patrol_mask.shape == (B, N)

        # Transform self feature
        self_feature = self.self_transform(self_feature)

        # Transform Query, Key, Value
        query = self.query_transform(self_feature).unsqueeze(1)  # [B, 1, H]
        key = self.key_transform(patrol_features)  # [B, N, H]
        value = self.value_transform(patrol_features)  # [B, N, H]

        # Calculate Scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [B, 1, N]
        assert scores.shape == (B, 1, N)

        # Apply Mask
        if patrol_mask is not None:
            # Mask out invalid patrol points with -inf
            scores = scores.masked_fill((~patrol_mask.bool()).unsqueeze(1), float('-inf'))

        # Get Attention Weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, N]


        # Use torch.isnan for readability and stability
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )

        # Weighted sum of patrol features
        weighted_features = torch.matmul(attention_weights, value).squeeze(1)  # [B, H]


        # Output is pure patrol intent vector.
        enhanced_feature = weighted_features

        assert enhanced_feature.shape == (B, self.hidden_dim)


        return enhanced_feature, attention_weights.squeeze(1)
