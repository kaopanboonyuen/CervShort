"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Artifact-Shift Adversarial Augmentation (ASAA)                      ║
║                                                                              ║
║  Synthesises realistic laboratory-dependent perturbations used as            ║
║  adversarial negatives during CervShort training.                           ║
║                                                                              ║
║  Perturbation family Δ_artifact includes:                                   ║
║    • Illumination drift  (global brightness/contrast jitter)                ║
║    • Staining variation  (Reinhard colour shift)                             ║
║    • Focus blur          (Gaussian blur)                                     ║
║    • Debris occlusion    (random rectangular patches)                        ║
║    • Microscopic noise   (Gaussian / Poisson noise)                          ║
║    • Chromatic aberration (per-channel scale/shift)                         ║
║                                                                              ║
║  Adversarial sampling objective (Section 4.3):                               ║
║    δ* = argmax_{δ ∈ Δ} L_adv(x + δ)                                        ║
║    L_adv = CE(f_θ(x + δ), y)                                               ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import List, Optional


class ArtifactShiftAugmentation(nn.Module):
    """
    Differentiable and non-differentiable augmentation pool simulating
    real cytology lab-dependent perturbations.

    Args:
        p_each          : Probability of applying each individual augmentation
        intensity_range : Global intensity scale (0 = clean, 1 = max artifact)
        debris_count    : Max number of debris blobs per image
        debris_size_ratio: Max fraction of image covered by each debris patch
    """

    def __init__(
        self,
        p_each: float = 0.5,
        intensity_range: float = 1.0,
        debris_count: int = 5,
        debris_size_ratio: float = 0.08,
    ):
        super().__init__()
        self.p_each           = p_each
        self.intensity_range  = intensity_range
        self.debris_count     = debris_count
        self.debris_size_ratio = debris_size_ratio

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _illumination_drift(x: torch.Tensor, factor: float = 0.3) -> torch.Tensor:
        """Simulate non-uniform illumination drift."""
        B, C, H, W = x.shape
        # Smooth gradient field
        grad = torch.linspace(1 - factor, 1 + factor, W, device=x.device)
        grad = grad.view(1, 1, 1, W).expand(B, C, H, W)
        return (x * grad).clamp(0, 1)

    @staticmethod
    def _stain_shift(x: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
        """Reinhard-inspired per-channel colour shift (staining variation)."""
        B, C, H, W = x.shape
        shift = (torch.rand(B, C, 1, 1, device=x.device) - 0.5) * 2 * scale
        return (x + shift).clamp(0, 1)

    @staticmethod
    def _focus_blur(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
        """Gaussian blur to simulate out-of-focus microscopy."""
        k = max(3, int(4 * sigma + 1) | 1)   # odd kernel
        # 1D Gaussian
        t = torch.arange(k, device=x.device).float() - k // 2
        g = torch.exp(-t ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        # 2D kernel via outer product
        kernel = (g.unsqueeze(1) * g.unsqueeze(0)).view(1, 1, k, k)
        kernel = kernel.expand(x.size(1), 1, k, k)
        pad    = k // 2
        return F.conv2d(x, kernel, padding=pad, groups=x.size(1))

    @staticmethod
    def _debris_occlusion(
        x: torch.Tensor,
        count: int = 5,
        size_ratio: float = 0.08,
    ) -> torch.Tensor:
        """Simulate debris, mucus, or air bubbles via random patches."""
        B, C, H, W = x.shape
        out = x.clone()
        for _ in range(count):
            ph = max(1, int(H * size_ratio * random.uniform(0.5, 1.5)))
            pw = max(1, int(W * size_ratio * random.uniform(0.5, 1.5)))
            y0 = random.randint(0, H - ph)
            x0 = random.randint(0, W - pw)
            # Debris colour: whitish-grey blobs
            color = torch.rand(B, C, 1, 1, device=x.device) * 0.4 + 0.6
            out[:, :, y0:y0+ph, x0:x0+pw] = color
        return out.clamp(0, 1)

    @staticmethod
    def _gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        return (x + torch.randn_like(x) * std).clamp(0, 1)

    @staticmethod
    def _chromatic_aberration(x: torch.Tensor, shift_px: int = 3) -> torch.Tensor:
        """Simulate chromatic aberration by shifting colour channels."""
        B, C, H, W = x.shape
        out = x.clone()
        for c in range(C):
            s = random.randint(-shift_px, shift_px)
            if s > 0:
                out[:, c, :, s:] = x[:, c, :, :-s]
                out[:, c, :, :s] = 0
            elif s < 0:
                out[:, c, :, :s] = x[:, c, :, -s:]
                out[:, c, :, s:] = 0
        return out

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply random subset of artifact augmentations.

        Args:
            x         : Input images [B, 3, H, W] in [0, 1]
            intensity : Artifact strength multiplier (0–1)

        Returns:
            x_aug : Perturbed images [B, 3, H, W]
        """
        p = self.p_each * intensity
        x_aug = x.clone()

        if random.random() < p:
            x_aug = self._illumination_drift(x_aug, factor=0.3 * intensity)

        if random.random() < p:
            x_aug = self._stain_shift(x_aug, scale=0.25 * intensity)

        if random.random() < p:
            sigma = random.uniform(0.5, 3.0) * intensity
            x_aug = self._focus_blur(x_aug, sigma=sigma)

        if random.random() < p:
            x_aug = self._debris_occlusion(
                x_aug,
                count=max(1, int(self.debris_count * intensity)),
                size_ratio=self.debris_size_ratio,
            )

        if random.random() < p:
            x_aug = self._gaussian_noise(x_aug, std=0.05 * intensity)

        if random.random() < p:
            x_aug = self._chromatic_aberration(
                x_aug, shift_px=max(1, int(4 * intensity))
            )

        return x_aug

    # ──────────────────────────────────────────────────────────────────────────

    def adversarial_sample(
        self,
        x: torch.Tensor,
        model_loss_fn,
        n_trials: int = 5,
    ) -> torch.Tensor:
        """
        Adversarial worst-case sampling over perturbation family.
        Returns the x+δ that maximises the given loss function.

        Args:
            x            : Input images [B, 3, H, W]
            model_loss_fn: Callable(x_aug) → scalar loss
            n_trials     : Number of random artifact samples to evaluate

        Returns:
            x_adv : Worst-case augmented images [B, 3, H, W]
        """
        best_loss   = float("-inf")
        best_x_aug  = x.clone()

        for trial in range(n_trials):
            intensity = (trial + 1) / n_trials  # escalating intensity
            with torch.no_grad():
                x_try = self.forward(x, intensity=intensity)
                loss  = model_loss_fn(x_try)
            if loss > best_loss:
                best_loss  = loss
                best_x_aug = x_try

        return best_x_aug
