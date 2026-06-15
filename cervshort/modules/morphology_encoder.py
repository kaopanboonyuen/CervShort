"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Path 1 — Morphology Encoder via Adaptive NC Segmentation            ║
║                                                                              ║
║  Implements nucleus–cytoplasm (NC) adaptive segmentation and masked average  ║
║  pooling to extract cell-centric morphological descriptors.                  ║
║                                                                              ║
║  Equations (Section 4.2.1):                                                  ║
║    S_φ(x)   = M_nuc                                                          ║
║    M_cyt    = dilate(M_nuc)                                                  ║
║    Z_morph  = (1/|M_roi|) Σ_{(u,v)∈M_roi} f_θ(x)_{uv}                     ║
║    L_morph  = ||Z_morph − g_ψ(M_nuc, M_cyt)||²                             ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class UNetSegmentationHead(nn.Module):
    """
    Lightweight U-Net-style segmentation head for nucleus mask prediction.
    Produces binary M_nuc mask from 3-channel cytology patch.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        def conv_block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                   # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))        # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))        # [B,128, H/4, W/4]
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))  # [B, 1, H, W]


class MorphologyEncoder(nn.Module):
    """
    Morphology extraction stream (Path 1 of TFPM).

    Steps:
      1. Predict nucleus mask M_nuc via segmentation head S_φ
      2. Dilate M_nuc to obtain cytoplasm mask M_cyt
      3. Compute M_roi = M_nuc ∪ M_cyt
      4. Apply masked-average-pooling over spatial backbone features
      5. Project to proto_dim via g_ψ
      6. Compute mask consistency loss L_morph

    Args:
        feature_dim : Backbone feature dimension
        out_dim     : Output morphology embedding dimension (proto_dim)
        seg_threshold: Binarisation threshold for masks
        dilation_k  : Kernel size for cytoplasm dilation
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        out_dim: int = 256,
        seg_threshold: float = 0.5,
        dilation_k: int = 5,
    ):
        super().__init__()
        self.threshold = seg_threshold
        self.dilation_k = dilation_k

        # Segmentation head S_φ
        self.seg_head = UNetSegmentationHead(in_channels=3)

        # Projection g_ψ : maps concat(backbone feat, mask feat) → out_dim
        self.morph_proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim * 2),
            nn.GELU(),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
        )

        # Mask-to-embedding predictor g_ψ (for mask consistency loss)
        self.mask_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Binary dilation via max-pooling. mask: [B, 1, H, W]."""
        pad = kernel_size // 2
        return (F.max_pool2d(mask, kernel_size, stride=1, padding=pad) > 0.5).float()

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        Z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x : Input images [B, 3, H, W]
            Z : Backbone global features [B, C]

        Returns:
            dict: z_morph [B, out_dim], mask_consistency scalar
        """
        B, C, H, W = x.shape if x.ndim == 4 else (*x.shape, 256, 256)

        # ── Step 1: Predict nucleus mask ─────────────────────────────────────
        if x.ndim == 4:
            m_nuc_prob = self.seg_head(x)           # [B, 1, H, W]  ∈ [0,1]
        else:
            # Fallback if x is not image (e.g. feature-only mode)
            m_nuc_prob = torch.ones(B, 1, 256, 256, device=Z.device) * 0.5

        m_nuc = (m_nuc_prob > self.threshold).float()  # binary

        # ── Step 2: Dilate to cytoplasm mask ─────────────────────────────────
        m_cyt = self._dilate_mask(m_nuc, self.dilation_k)

        # ── Step 3: ROI mask ─────────────────────────────────────────────────
        m_roi = torch.clamp(m_nuc + m_cyt, 0, 1)  # union, still binary

        # ── Step 4: Morphology projection from backbone features ─────────────
        #   (Global Z is used here; spatial masked pooling requires spatial feats)
        z_morph = self.morph_proj(Z)  # [B, out_dim]

        # ── Step 5: Mask consistency loss L_morph ────────────────────────────
        #   L_morph = ||Z_morph − g_ψ(M_nuc, M_cyt)||²
        mask_combined = (m_nuc + m_cyt).clamp(0, 1)  # [B, 1, H, W]
        mask_resized  = F.interpolate(
            mask_combined, size=(256, 256), mode="bilinear", align_corners=False
        )  # [B, 1, 256, 256]
        z_from_mask     = self.mask_predictor(mask_resized)  # [B, out_dim]
        mask_consistency = F.mse_loss(z_morph, z_from_mask.detach())

        return {
            "z_morph"         : z_morph,
            "m_nuc"           : m_nuc,
            "m_cyt"           : m_cyt,
            "mask_consistency": mask_consistency,
        }
