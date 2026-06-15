"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CervShort — Main Model                              ║
║                                                                              ║
║  Domain-Aware Shortcut Disruption for Robust Cervical Cancer Cytology        ║
║  Classification                                                              ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Affil. : Chulalongkorn University · Khon Kaen University, Thailand         ║
║  Fund   : C2F Postdoctoral Fellowship, Chulalongkorn University             ║
║                                                                              ║
║  Reference: "CervShort: Domain-Aware Shortcut Disruption for Robust          ║
║              Cervical Cancer Cytology Classification"                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from cervshort.modules.tfpm import TriPathFeaturePurificationModule
from cervshort.modules.spm import ShortcutPerturbationModule
from cervshort.utils.losses import CervShortLoss
from cervshort.augmentation.artifact_shift import ArtifactShiftAugmentation


class CervShort(nn.Module):
    """
    CervShort: Domain-Aware Shortcut Disruption Framework.

    Combines:
      1. Shortcut-Perturbation Module (SPM)
      2. Tri-Path Feature Purification Module (TFPM):
         - Path 1: Segmentation-guided morphology encoding
         - Path 2: Degradation-invariant projection
         - Path 3: Cross-domain prototype alignment
      3. Cross-Branch Contrastive Alignment Loss

    Args:
        backbone      : Backbone encoder ('resnet18', 'resnet50',
                        'densenet121', 'vit_l16')
        num_classes   : Number of cytology classes (default 5: NILM/ASC-US/LSIL/HSIL/SCC)
        num_domains   : Number of laboratory domains
        feature_dim   : Backbone output feature dimension
        proto_dim     : Prototype embedding dimension
        proto_momentum: EMA momentum for prototype updates
        lambda_morph  : Weight for morphology consistency loss
        lambda_deg    : Weight for degradation-invariance loss
        lambda_proto  : Weight for prototype alignment loss
        lambda_adv    : Weight for adversarial augmentation loss
    """

    def __init__(
        self,
        backbone: str = "vit_l16",
        num_classes: int = 5,
        num_domains: int = 5,
        feature_dim: int = 1024,
        proto_dim: int = 256,
        proto_momentum: float = 0.97,
        lambda_morph: float = 0.5,
        lambda_deg: float = 0.3,
        lambda_proto: float = 0.2,
        lambda_adv: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feature_dim = feature_dim

        # ── Backbone encoder ───────────────────────────────────────────────
        self.encoder = self._build_backbone(backbone, feature_dim)

        # ── Shortcut Perturbation Module ────────────────────────────────────
        self.spm = ShortcutPerturbationModule(feature_dim=feature_dim)

        # ── Tri-Path Feature Purification Module ────────────────────────────
        self.tfpm = TriPathFeaturePurificationModule(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_domains=num_domains,
            proto_dim=proto_dim,
            proto_momentum=proto_momentum,
        )

        # ── Final classifier (operates on concatenated tri-path features) ───
        self.classifier = nn.Sequential(
            nn.LayerNorm(proto_dim * 3),
            nn.Linear(proto_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # ── Artifact augmentation ───────────────────────────────────────────
        self.artifact_aug = ArtifactShiftAugmentation()

        # ── Loss ────────────────────────────────────────────────────────────
        self.criterion = CervShortLoss(
            lambda_morph=lambda_morph,
            lambda_deg=lambda_deg,
            lambda_proto=lambda_proto,
            lambda_adv=lambda_adv,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _build_backbone(self, name: str, feature_dim: int) -> nn.Module:
        """Load a pretrained backbone and strip its classification head."""
        import torchvision.models as tvm

        if name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, feature_dim)
        elif name == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, feature_dim)
        elif name == "densenet121":
            m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            in_f = m.classifier.in_features
            m.classifier = nn.Linear(in_f, feature_dim)
        elif name == "vit_l16":
            try:
                import timm
                m = timm.create_model("vit_large_patch16_224", pretrained=True,
                                      num_classes=feature_dim)
            except ImportError:
                m = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_V1)
                in_f = m.heads.head.in_features
                m.heads.head = nn.Linear(in_f, feature_dim)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        return m

    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features Z ∈ R^C."""
        return self.encoder(x)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x          : Input images  [B, 3, H, W]
            labels     : Class labels  [B]
            domain_ids : Lab domain IDs [B]
            return_loss: Whether to compute and return the full loss dict

        Returns:
            dict with keys: logits, loss (optional), z_morph, z_inv, z_proto
        """
        B = x.size(0)
        device = x.device

        # ── 1. Backbone encoding ────────────────────────────────────────────
        Z = self.encode(x)  # [B, C]

        # ── 2. Shortcut Perturbation Module ─────────────────────────────────
        shortcut_mask = self.spm(Z)  # [B, C] soft mask identifying shortcut dims

        # ── 3. Tri-Path Feature Purification ────────────────────────────────
        tfpm_out = self.tfpm(
            x=x,
            Z=Z,
            shortcut_mask=shortcut_mask,
            labels=labels,
            domain_ids=domain_ids,
        )
        z_morph  = tfpm_out["z_morph"]   # [B, proto_dim]
        z_inv    = tfpm_out["z_inv"]      # [B, proto_dim]
        z_proto  = tfpm_out["z_proto"]    # [B, proto_dim]
        z_spur   = tfpm_out["z_spur"]     # [B, proto_dim]

        # ── 4. Classification ────────────────────────────────────────────────
        z_final = torch.cat([z_morph, z_inv, z_proto], dim=-1)  # [B, proto_dim*3]
        logits  = self.classifier(z_final)  # [B, num_classes]

        out = {"logits": logits, "z_morph": z_morph, "z_inv": z_inv,
               "z_proto": z_proto, "z_spur": z_spur}

        # ── 5. Loss computation (training only) ──────────────────────────────
        if return_loss and labels is not None:
            # Adversarial artifact-shifted samples
            with torch.no_grad():
                x_aug = self.artifact_aug(x)
            Z_aug   = self.encode(x_aug)
            logits_aug = self.classifier(
                torch.cat(
                    list(self.tfpm(x=x_aug, Z=Z_aug,
                                   shortcut_mask=self.spm(Z_aug),
                                   labels=labels, domain_ids=domain_ids
                                   ).values())[:3],
                    dim=-1,
                )
            )

            losses = self.criterion(
                logits=logits,
                labels=labels,
                z_morph=z_morph,
                z_spur=z_spur,
                proto_loss=tfpm_out.get("proto_loss", torch.tensor(0.0, device=device)),
                logits_aug=logits_aug,
                mask_consistency=tfpm_out.get("mask_consistency",
                                              torch.tensor(0.0, device=device)),
            )
            out.update(losses)

        return out
