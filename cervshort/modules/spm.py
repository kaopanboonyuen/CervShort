"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  Shortcut-Perturbation Module (SPM)                         ║
║                                                                              ║
║  Generates a soft shortcut-sensitivity mask over the feature dimensions of  ║
║  backbone output Z. High-scoring dimensions are more likely to encode        ║
║  acquisition-dependent shortcuts (staining, illumination, debris).          ║
║                                                                              ║
║  The mask is used by TFPM to scale-down shortcut-prone features before      ║
║  they enter the morphology encoder.                                          ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Ref    : CervShort Figure 1 + Section 4                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutPerturbationModule(nn.Module):
    """
    SPM: learns a soft mask M_spur ∈ [0,1]^C over feature channels.

    Channels with high mask values are considered shortcut-prone and
    are down-weighted before being passed to morphology encoding.

    Args:
        feature_dim  : Backbone feature dimension C
        hidden_dim   : Intermediate projection dimension
        temperature  : Softmax temperature for mask sharpness
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        # Channel-wise shortcut importance estimator
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z : Backbone features [B, C]

        Returns:
            shortcut_mask : Soft mask [B, C] ∈ [0, 1]
                            High values → shortcut-prone dimensions
        """
        shortcut_mask = self.importance_net(Z.detach())  # [B, C]
        return shortcut_mask
