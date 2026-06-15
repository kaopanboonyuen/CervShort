"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Path 2 — Degradation-Invariant Shortcut Suppression                 ║
║                                                                              ║
║  Learns a projector h_ω that isolates shortcut-sensitive signals Z_spur     ║
║  and penalises them via:                                                     ║
║    L_orth = ||Z_morph^T Z_spur||₂      (orthogonality)                     ║
║    L_freq = ||F(Z_spur) ⊙ 1_{hi-freq}||₁  (frequency suppression)         ║
║    L_deg  = L_orth + β · L_freq                                             ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Ref    : CervShort Section 4.2.2                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DegradationInvariantProjector(nn.Module):
    """
    Degradation-invariant projection head (Path 2 of TFPM).

    Produces a shortcut-purified embedding z_inv from backbone features Z,
    by learning to be invariant to staining, illumination, and debris artifacts.

    Architecture:
      Z  →  MLP projector  →  z_inv (normalised)

    During training, the degradation loss (L_orth + β·L_freq) is used externally
    in CervShortLoss to penalise overlap with Z_morph and high-frequency content.

    Args:
        feature_dim : Backbone feature dimension
        out_dim     : Output embedding dimension
        beta        : Weight for frequency suppression term
        hi_freq_thr : Fraction of spectrum considered "high-frequency" [0–1]
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        out_dim: int = 256,
        beta: float = 0.1,
        hi_freq_thr: float = 0.5,
    ):
        super().__init__()
        self.beta        = beta
        self.hi_freq_thr = hi_freq_thr

        # Projector h_ω
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
            nn.Linear(feature_dim // 2, out_dim),
        )

        # Auxiliary head to isolate shortcut subspace Z_spur
        self.spur_head = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Project backbone features to degradation-invariant embedding.

        Args:
            Z : Backbone features [B, C]

        Returns:
            z_inv : Degradation-invariant embedding [B, out_dim]
        """
        z_inv = self.projector(Z)               # [B, out_dim]
        z_inv = F.normalize(z_inv, dim=-1)      # L2-normalise
        return z_inv

    # ──────────────────────────────────────────────────────────────────────────

    def compute_degradation_loss(
        self,
        Z: torch.Tensor,
        z_morph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L_deg = L_orth + β·L_freq.

        Args:
            Z       : Backbone features     [B, C]
            z_morph : Morphology embedding  [B, out_dim]

        Returns:
            dict: l_orth, l_freq, l_deg
        """
        z_spur = self.spur_head(Z)  # [B, out_dim]

        # ── Orthogonality constraint ──────────────────────────────────────────
        # L_orth = ||Z_morph^T Z_spur||_F
        # Both are [B, D]; compute batch-level dot product matrix
        dot = torch.bmm(
            z_morph.unsqueeze(2),   # [B, D, 1]
            z_spur.unsqueeze(1),    # [B, 1, D]
        )  # [B, D, D]
        l_orth = torch.norm(dot, p=2, dim=(-2, -1)).mean()

        # ── Frequency suppression ─────────────────────────────────────────────
        # Penalise high-frequency content in z_spur using 1D FFT
        fft_spur   = torch.fft.rfft(z_spur, norm="ortho")          # [B, out_dim//2+1]
        fft_mag    = fft_spur.abs()                                  # magnitudes
        n_freqs    = fft_mag.shape[-1]
        hi_idx     = int(self.hi_freq_thr * n_freqs)
        hi_freq_mask = torch.zeros(n_freqs, device=Z.device)
        hi_freq_mask[hi_idx:] = 1.0
        l_freq     = (fft_mag * hi_freq_mask).sum(dim=-1).mean()

        l_deg = l_orth + self.beta * l_freq

        return {"l_orth": l_orth, "l_freq": l_freq, "l_deg": l_deg,
                "z_spur": z_spur}
