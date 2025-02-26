from typing import NamedTuple, Tuple
import torch
from torch import nn
import torch.nn.functional as F

## copied from alpha_zero
class NetworkOutputs(NamedTuple):
    pi_prob: torch.Tensor
    value: torch.Tensor

## copied from alpha_zero, TODO: add as question how does it change the results?
def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)

## copied from alpha_zero
class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out

## copied from alpha_zero, simplified    
class AlphaZeroNet(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int = 19,
        num_filters: int = 64,
        num_fc_units: int = 64,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * h * w, num_actions),
            nn.Softmax(dim=1),  # Ensures valid probability distribution
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * h * w, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 4), # there are 4 players
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)

        return pi_logits, value

# Define loss function for AlphaZero with two heads
def policy_value_loss(policy_pred, policy_target, value_pred, value_target):
    """
    Computes the combined loss for AlphaZero network. The loss is a combination of
    policy cross-entropy and value mean squared error.
    
    Parameters:
        policy_pred: torch.Tensor - Output logits from the policy head (before softmax).
        policy_target: torch.Tensor - Target probability distribution.
        value_pred: torch.Tensor - Output from the value head (single scalar per sample).
        value_target: torch.Tensor - Target scalar value.

    Returns:
        Total loss combining policy cross-entropy and value mean squared error.
    """
    policy_loss = F.cross_entropy(policy_pred, policy_target)
    value_loss = F.mse_loss(value_pred, value_target)

    # You can weigh them differently if needed, but usually equal weighting works
    total_loss = policy_loss + value_loss
    return total_loss
