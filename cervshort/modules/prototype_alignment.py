"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Path 3 — Domain-Stable Prototype Alignment                          ║
║                                                                              ║
║  Maintains per-class, per-domain prototype vectors and enforces cross-domain ║
║  consistency via:                                                            ║
║                                                                              ║
║    P_{k,d} = (1/|D_{k,d}|) Σ Z_morph^{(i)}   [EMA updated]                ║
║                                                                              ║
║    L_proto = Σ_k Σ_{d1,d2} ||P_{k,d1} − P_{k,d2}||²                       ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Ref    : CervShort Section 4.2.3                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PrototypeAlignmentModule(nn.Module):
    """
    Cross-domain prototype alignment (Path 3 of TFPM).

    Maintains an EMA prototype memory P[k, d] ∈ R^{proto_dim}
    for each (class k, domain d) pair. During forward:
      1. Update prototypes from current batch (EMA)
      2. Compute prototype-aligned embedding for each sample
      3. Return cross-domain alignment loss L_proto

    Args:
        feature_dim  : Input embedding dimension (proto_dim from Path 1)
        num_classes  : Number of Bethesda cytology classes
        num_domains  : Number of independent laboratories
        momentum     : EMA momentum for prototype updates (default 0.97)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 5,
        num_domains: int = 5,
        momentum: float = 0.97,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.momentum    = momentum

        # Prototype bank P[k, d] ∈ R^{feature_dim}; shape [K, D, F]
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, num_domains, feature_dim),
        )
        self.register_buffer(
            "proto_initialised",
            torch.zeros(num_classes, num_domains, dtype=torch.bool),
        )

        # Optional projection before prototype comparison
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_prototypes(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> None:
        """EMA update of prototype bank from current batch."""
        z_detach = z.detach()
        for k in range(self.num_classes):
            for d in range(self.num_domains):
                mask = (labels == k) & (domain_ids == d)
                if mask.sum() == 0:
                    continue
                batch_mean = z_detach[mask].mean(dim=0)
                if self.proto_initialised[k, d]:
                    self.prototypes[k, d] = (
                        self.momentum * self.prototypes[k, d]
                        + (1 - self.momentum) * batch_mean
                    )
                else:
                    self.prototypes[k, d] = batch_mean
                    self.proto_initialised[k, d] = True

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_proto_loss(self) -> torch.Tensor:
        """
        L_proto = Σ_k Σ_{d1 ≠ d2} ||P_{k,d1} − P_{k,d2}||²
        Only considers initialised prototype pairs.
        """
        total_loss = torch.tensor(0.0, device=self.prototypes.device)
        count = 0
        for k in range(self.num_classes):
            for d1 in range(self.num_domains):
                if not self.proto_initialised[k, d1]:
                    continue
                for d2 in range(d1 + 1, self.num_domains):
                    if not self.proto_initialised[k, d2]:
                        continue
                    diff = self.prototypes[k, d1] - self.prototypes[k, d2]
                    total_loss = total_loss + (diff ** 2).sum()
                    count += 1
        return total_loss / max(count, 1)

    # ──────────────────────────────────────────────────────────────────────────

    def _prototype_aligned_embedding(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor],
        domain_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Produce prototype-aligned embedding:
          z_proto = z + (P_k_mean − P_{k,d})
        This pulls the sample towards the class centroid across all domains.
        """
        z_proto = z.clone()
        if labels is None or domain_ids is None:
            return z_proto

        for i in range(z.size(0)):
            k = labels[i].item()
            d = domain_ids[i].item()
            if k >= self.num_classes or d >= self.num_domains:
                continue
            if not self.proto_initialised[k, d]:
                continue

            # Mean prototype across all domains for class k
            init_mask = self.proto_initialised[k]  # [D]
            if init_mask.sum() == 0:
                continue
            p_mean = self.prototypes[k][init_mask].mean(dim=0)  # [F]
            p_d    = self.prototypes[k, d]                       # [F]
            z_proto[i] = z[i] + (p_mean - p_d).detach()

        return z_proto

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z          : Morphology embeddings [B, feature_dim]
            labels     : Class labels          [B]
            domain_ids : Domain labels         [B]

        Returns:
            dict: z_proto [B, feature_dim], proto_loss scalar
        """
        z_proj = self.proj(z)  # [B, F]

        # Update prototypes (training mode only)
        if self.training and labels is not None and domain_ids is not None:
            self._update_prototypes(z_proj, labels, domain_ids)

        # Prototype alignment loss
        proto_loss = self._compute_proto_loss()

        # Prototype-aligned embedding
        z_proto = self._prototype_aligned_embedding(z_proj, labels, domain_ids)

        return {"z_proto": z_proto, "proto_loss": proto_loss}
