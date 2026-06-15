"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CervShort — Loss Functions                                ║
║                                                                              ║
║  Implements the complete multi-term objective:                               ║
║                                                                              ║
║  L_CervShort = L_cls                                                        ║
║              + λ_m  · L_morph    (mask consistency)                         ║
║              + λ_deg · L_deg     (orthogonality + freq suppression)         ║
║              + λ_p  · L_proto   (cross-domain prototype alignment)          ║
║              + λ_adv · L_adv    (adversarial artifact robustness)           ║
║                                                                              ║
║  Reference: CervShort Equation (6) / Section 4.4                           ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class OrthogonalityLoss(nn.Module):
    """L_orth = ||Z_morph^T Z_spur||_F (batch-averaged)."""

    def forward(
        self, z_morph: torch.Tensor, z_spur: torch.Tensor
    ) -> torch.Tensor:
        # [B, D, 1] × [B, 1, D] → [B, D, D]
        dot = torch.bmm(z_morph.unsqueeze(2), z_spur.unsqueeze(1))
        return torch.norm(dot, p="fro", dim=(-2, -1)).mean()


class FrequencySuppressionLoss(nn.Module):
    """
    L_freq = ||F(Z_spur) ⊙ 1_{hi-freq}||_1
    Penalises high-frequency activations in the shortcut subspace.
    """

    def __init__(self, hi_freq_thr: float = 0.5):
        super().__init__()
        self.hi_freq_thr = hi_freq_thr

    def forward(self, z_spur: torch.Tensor) -> torch.Tensor:
        fft_mag  = torch.fft.rfft(z_spur, norm="ortho").abs()  # [B, D/2+1]
        n        = fft_mag.shape[-1]
        hi_idx   = int(self.hi_freq_thr * n)
        mask     = torch.zeros(n, device=z_spur.device)
        mask[hi_idx:] = 1.0
        return (fft_mag * mask).sum(-1).mean()


class DegradationInvarianceLoss(nn.Module):
    """
    L_deg = L_orth + β · L_freq
    Reference: CervShort Equation (4)
    """

    def __init__(self, beta: float = 0.1, hi_freq_thr: float = 0.5):
        super().__init__()
        self.beta   = beta
        self.l_orth = OrthogonalityLoss()
        self.l_freq = FrequencySuppressionLoss(hi_freq_thr)

    def forward(
        self,
        z_morph: torch.Tensor,
        z_spur:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_orth = self.l_orth(z_morph, z_spur)
        l_freq = self.l_freq(z_spur)
        l_deg  = l_orth + self.beta * l_freq
        return {"l_orth": l_orth, "l_freq": l_freq, "l_deg": l_deg}


class CervShortLoss(nn.Module):
    """
    Full CervShort multi-term training objective.

    L = L_cls + λ_m·L_morph + λ_deg·L_deg + λ_p·L_proto + λ_adv·L_adv

    Args:
        lambda_morph : Weight for mask consistency loss
        lambda_deg   : Weight for degradation invariance loss
        lambda_proto : Weight for prototype alignment loss
        lambda_adv   : Weight for adversarial augmentation loss
        beta_freq    : β weight for frequency suppression within L_deg
        label_smoothing: Label smoothing for cross-entropy
    """

    def __init__(
        self,
        lambda_morph: float = 0.5,
        lambda_deg:   float = 0.3,
        lambda_proto: float = 0.2,
        lambda_adv:   float = 0.1,
        beta_freq:    float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_morph = lambda_morph
        self.lambda_deg   = lambda_deg
        self.lambda_proto = lambda_proto
        self.lambda_adv   = lambda_adv

        self.ce_loss  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.deg_loss = DegradationInvarianceLoss(beta=beta_freq)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        logits:           torch.Tensor,
        labels:           torch.Tensor,
        z_morph:          torch.Tensor,
        z_spur:           torch.Tensor,
        proto_loss:       torch.Tensor,
        logits_aug:       torch.Tensor,
        mask_consistency: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits           : Clean predictions  [B, K]
            labels           : Ground truth        [B]
            z_morph          : Morphology embeddings [B, D]
            z_spur           : Shortcut embeddings   [B, D]
            proto_loss       : Pre-computed L_proto (scalar)
            logits_aug       : Predictions on artifact-augmented images [B, K]
            mask_consistency : L_morph from morphology encoder (scalar)

        Returns:
            dict with individual losses + total loss
        """
        # ── L_cls : Classification fidelity ──────────────────────────────────
        l_cls = self.ce_loss(logits, labels)

        # ── L_morph : Mask consistency (from Path 1) ──────────────────────────
        l_morph = mask_consistency

        # ── L_deg : Orthogonality + frequency suppression ─────────────────────
        deg_dict = self.deg_loss(z_morph, z_spur)
        l_deg    = deg_dict["l_deg"]

        # ── L_proto : Cross-domain prototype alignment (from Path 3) ──────────
        l_proto = proto_loss

        # ── L_adv : Adversarial artifact robustness ────────────────────────────
        l_adv = self.ce_loss(logits_aug, labels)

        # ── Total ──────────────────────────────────────────────────────────────
        loss = (
            l_cls
            + self.lambda_morph * l_morph
            + self.lambda_deg   * l_deg
            + self.lambda_proto * l_proto
            + self.lambda_adv   * l_adv
        )

        return {
            "loss"   : loss,
            "l_cls"  : l_cls.detach(),
            "l_morph": l_morph.detach(),
            "l_deg"  : l_deg.detach(),
            "l_orth" : deg_dict["l_orth"].detach(),
            "l_freq" : deg_dict["l_freq"].detach(),
            "l_proto": l_proto.detach(),
            "l_adv"  : l_adv.detach(),
        }
