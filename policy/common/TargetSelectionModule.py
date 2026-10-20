import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class TargetSelectionModule(nn.Module):
    """Attention module specifically for target selection of evaders"""

    def __init__(self, hidden_dim: int = 256):
        """Initialize the instance."""
        super().__init__()
        self.hidden_dim = hidden_dim

        self.self_transform = nn.Linear(hidden_dim, hidden_dim)

        # Linear layer for transforming query
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

        # Linear layers for transforming key and value
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)


        # self.layer_norm = nn.LayerNorm(hidden_dim)

        # Scaling factor
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        context_feature: torch.Tensor,
        evader_features: torch.Tensor,
        evader_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""
        batch_size = context_feature.shape[0]
        hidden_dim = self.hidden_dim
        num_evaders = evader_features.shape[1]

        # ... (Shape checking omitted for brevity, keep them in your code) ...
        # Add shape checking
        assert context_feature.shape == (batch_size, hidden_dim), (
            f"Unexpected self_feature shape: {context_feature.shape}"
        )
        assert evader_features.shape == (batch_size, num_evaders, hidden_dim), (
            f"Unexpected evader_features shape: {evader_features.shape}"
        )
        assert evader_mask.shape == (batch_size, num_evaders), (
            f"Unexpected evader_mask shape: {evader_mask.shape}"
        )

        context_feature = self.self_transform(context_feature)

        # Transform query
        query = self.query_transform(context_feature).unsqueeze(1)  # [B, 1, H]

        # Transform key and value
        keys = self.key_transform(evader_features)  # [B, N, H]
        values = self.value_transform(evader_features)  # [B, N, H]

        # Calculate attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # [B, 1, N]

        # Apply mask
        if evader_mask is not None:
            # Mask out invalid evaders with -inf
            scores = scores.masked_fill(
                (~evader_mask.bool()).unsqueeze(1), float("-inf")
            )

        # Get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, N]


        # Use torch.isnan for readability
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights,
        )

        # Get weighted features
        weighted_features = torch.matmul(attention_weights, values)  # [B, 1, H]
        weighted_features = weighted_features.squeeze(1)  # [B, H]


        # Output is pure intent. If no targets, output is exactly 0 vector.
        enhanced_feature = weighted_features

        return enhanced_feature, attention_weights.squeeze(1)
