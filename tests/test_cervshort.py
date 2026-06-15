"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CervShort — Unit Tests                                          ║
║  Author: Teerapong Panboonyuen (Kao Panboonyuen)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from cervshort.utils.losses import CervShortLoss, DegradationInvarianceLoss
from cervshort.utils.metrics import (
    compute_auc, compute_fpr95,
    compute_causal_consistency,
    compute_shortcut_correlation,
)
from cervshort.augmentation.artifact_shift import ArtifactShiftAugmentation
from cervshort.modules.prototype_alignment import PrototypeAlignmentModule
from cervshort.modules.degradation_projector import DegradationInvariantProjector


# ─────────────────────────────────────────────────────────────────────────────
# Loss tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCervShortLoss:
    def _make_inputs(self, B=8, K=5, D=256):
        logits   = torch.randn(B, K)
        labels   = torch.randint(0, K, (B,))
        z_morph  = torch.randn(B, D)
        z_spur   = torch.randn(B, D)
        proto_l  = torch.tensor(0.05)
        logits_a = torch.randn(B, K)
        mask_c   = torch.tensor(0.02)
        return logits, labels, z_morph, z_spur, proto_l, logits_a, mask_c

    def test_loss_is_scalar(self):
        loss_fn = CervShortLoss()
        inputs  = self._make_inputs()
        out     = loss_fn(*inputs)
        assert out["loss"].shape == torch.Size([])

    def test_loss_is_positive(self):
        loss_fn = CervShortLoss()
        out = loss_fn(*self._make_inputs())
        assert out["loss"].item() > 0

    def test_all_components_present(self):
        loss_fn = CervShortLoss()
        out = loss_fn(*self._make_inputs())
        for key in ["loss","l_cls","l_morph","l_deg","l_proto","l_adv"]:
            assert key in out, f"Missing key: {key}"


class TestDegradationLoss:
    def test_orthogonality(self):
        loss_fn = DegradationInvarianceLoss()
        z_morph = torch.randn(16, 256)
        z_spur  = torch.randn(16, 256)
        out = loss_fn(z_morph, z_spur)
        assert out["l_orth"].item() >= 0
        assert out["l_freq"].item() >= 0
        assert out["l_deg"].item()  >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactAugmentation:
    def test_output_shape(self):
        aug = ArtifactShiftAugmentation(p_each=1.0)
        x = torch.rand(4, 3, 256, 256)
        x_aug = aug(x, intensity=1.0)
        assert x_aug.shape == x.shape

    def test_values_in_range(self):
        aug = ArtifactShiftAugmentation(p_each=1.0)
        x = torch.rand(2, 3, 64, 64)
        x_aug = aug(x, intensity=1.0)
        assert x_aug.min() >= 0.0
        assert x_aug.max() <= 1.0

    def test_zero_intensity_minimal_change(self):
        aug = ArtifactShiftAugmentation(p_each=0.0)
        x = torch.rand(2, 3, 64, 64)
        x_aug = aug(x, intensity=0.0)
        assert torch.allclose(x, x_aug, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_auc_range(self):
        import numpy as np
        y_true  = np.array([0,1,2,0,1,2,0,1,2,0])
        y_score = np.eye(3)[y_true] + np.random.rand(10, 3)*0.1
        y_score /= y_score.sum(axis=1, keepdims=True)
        auc = compute_auc(y_true, y_score)
        assert 50.0 <= auc <= 100.0

    def test_fpr95_range(self):
        import numpy as np
        y_true  = np.array([0,1,2]*10)
        y_score = np.random.dirichlet([1,1,1], 30)
        fpr = compute_fpr95(y_true, y_score)
        assert 0.0 <= fpr <= 100.0

    def test_causal_consistency_perfect(self):
        logits = torch.randn(16, 5)
        cc = compute_causal_consistency(logits, logits)
        assert abs(cc - 1.0) < 1e-5

    def test_shortcut_corr_range(self):
        feats = torch.randn(50, 256)
        art   = torch.randint(0, 2, (50,))
        corr  = compute_shortcut_correlation(feats, art)
        assert 0.0 <= corr <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Module tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrototypeAlignment:
    def test_forward_shape(self):
        mod = PrototypeAlignmentModule(feature_dim=256, num_classes=5, num_domains=5)
        z          = torch.randn(8, 256)
        labels     = torch.randint(0, 5, (8,))
        domain_ids = torch.randint(0, 5, (8,))
        out = mod(z, labels, domain_ids)
        assert out["z_proto"].shape == (8, 256)
        assert out["proto_loss"].shape == torch.Size([])

class TestDegradationProjector:
    def test_forward_shape(self):
        proj = DegradationInvariantProjector(feature_dim=512, out_dim=256)
        Z = torch.randn(8, 512)
        z_inv = proj(Z)
        assert z_inv.shape == (8, 256)

    def test_l2_normalised(self):
        proj = DegradationInvariantProjector(feature_dim=512, out_dim=256)
        Z = torch.randn(8, 512)
        z_inv = proj(Z)
        norms = z_inv.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
