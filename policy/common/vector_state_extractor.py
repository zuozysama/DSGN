import torch.nn as nn
def vector_encoder(input_dimension: int, output_dimension: int) -> nn.Sequential:
    """Create encoder model"""
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.LayerNorm(output_dimension),
        nn.ReLU()
    )