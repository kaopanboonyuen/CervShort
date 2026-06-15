"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Tri-Path Feature Purification Module (TFPM)                       ║
║                                                                              ║
║  Implements the three synergistic learning paths of CervShort:              ║
║    Path 1 — Morphology Extraction via Adaptive NC Segmentation              ║
║    Path 2 — Degradation-Invariant Shortcut Suppression                      ║
║    Path 3 — Domain-Stable Prototype Alignment                               ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Ref    : CervShort, Section 4.2 (Methodology)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from cervshort.modules.morphology_encoder import MorphologyEncoder
from cervshort.modules.degradation_projector import DegradationInvariantProjector
from cervshort.modules.prototype_alignment import PrototypeAlignmentModule


class TriPathFeaturePurificationModule(nn.Module):
    """
    TFPM: The three-branch purification module at the heart of CervShort.

    Given backbone features Z ∈ R^C and raw image x, TFPM produces:
      z_morph : pathology-grounded morphology embedding
      z_inv   : degradation-invariant (shortcut-suppressed) embedding
      z_proto : prototype-aligned cross-domain embedding

    The disentanglement objective is:
      Z = Z_morph ⊕ Z_spur

    Cross-branch orthogonality is enforced via:
      L_orth = ||Z_morph^T Z_spur||_2

    Args:
        feature_dim    : Backbone output dimension C
        num_classes    : Number of diagnostic categories
        num_domains    : Number of laboratory domains
        proto_dim      : Prototype / projected embedding dimension
        proto_momentum : EMA coefficient for prototype updates
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        num_classes: int = 5,
        num_domains: int = 5,
        proto_dim: int = 256,
        proto_momentum: float = 0.97,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.proto_dim   = proto_dim

        # ── Path 1: Morphology ──────────────────────────────────────────────
        self.morphology_encoder = MorphologyEncoder(
            feature_dim=feature_dim,
            out_dim=proto_dim,
        )

        # ── Path 2: Degradation invariance ──────────────────────────────────
        self.degradation_projector = DegradationInvariantProjector(
            feature_dim=feature_dim,
            out_dim=proto_dim,
        )

        # ── Path 3: Prototype alignment ──────────────────────────────────────
        self.prototype_module = PrototypeAlignmentModule(
            feature_dim=proto_dim,
            num_classes=num_classes,
            num_domains=num_domains,
            momentum=proto_momentum,
        )

        # ── Shared projection for Z_spur (shortcut subspace) ────────────────
        self.spur_projector = nn.Sequential(
            nn.Linear(feature_dim, proto_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proto_dim, proto_dim),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        Z: torch.Tensor,
        shortcut_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x            : Raw images          [B, 3, H, W]
            Z            : Backbone features   [B, C]
            shortcut_mask: Soft shortcut mask  [B, C]  (from SPM)
            labels       : Class labels        [B]
            domain_ids   : Domain labels       [B]

        Returns:
            dict with: z_morph, z_inv, z_spur, z_proto, proto_loss,
                       mask_consistency
        """
        # ── Path 1: Morphology encoding ──────────────────────────────────────
        morph_out = self.morphology_encoder(x=x, Z=Z)
        z_morph          = morph_out["z_morph"]          # [B, proto_dim]
        mask_consistency = morph_out["mask_consistency"] # scalar

        # ── Path 2: Degradation-invariant projection ─────────────────────────
        z_inv = self.degradation_projector(Z=Z)  # [B, proto_dim]

        # ── Shortcut subspace ────────────────────────────────────────────────
        if shortcut_mask is not None:
            Z_spur_raw = Z * shortcut_mask
        else:
            Z_spur_raw = Z
        z_spur = self.spur_projector(Z_spur_raw)  # [B, proto_dim]

        # ── Path 3: Prototype alignment ──────────────────────────────────────
        proto_out  = self.prototype_module(
            z=z_morph,
            labels=labels,
            domain_ids=domain_ids,
        )
        z_proto    = proto_out["z_proto"]    # [B, proto_dim]
        proto_loss = proto_out["proto_loss"] # scalar

        return {
            "z_morph"         : z_morph,
            "z_inv"           : z_inv,
            "z_spur"          : z_spur,
            "z_proto"         : z_proto,
            "proto_loss"      : proto_loss,
            "mask_consistency": mask_consistency,
        }
