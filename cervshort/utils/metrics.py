"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  CervShort — Evaluation Metrics                              ║
║                                                                              ║
║  Implements all metrics used in the paper's Tables 1–5:                     ║
║    • AUC (↑)              — Table 1, 2, 3, 4                               ║
║    • FPR95 (↓)            — Table 1, 3                                     ║
║    • Attribution Overlap  — Table 2 (Attr. Overlap ↑)                      ║
║    • Shortcut Correlation — Table 2 (Shortcut Corr ↓)                      ║
║    • Causal Consistency   — Table 2 (Causal Cons. ↑)                       ║
║    • Sensitivity/Specificity/F1 — Table 5                                  ║
║    • ΔAUC under perturbations   — Table 4                                  ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    f1_score, precision_recall_curve,
)
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str = "ovr",
    average: str = "macro",
) -> float:
    """
    Macro-averaged one-vs-rest AUC (Table 1).

    Args:
        y_true   : Integer class labels [N]
        y_score  : Softmax probabilities [N, K]
        multi_class: 'ovr' or 'ovo'
        average  : 'macro' or 'weighted'

    Returns:
        AUC in percent (multiplied by 100)
    """
    return roc_auc_score(
        y_true, y_score, multi_class=multi_class, average=average
    ) * 100.0


def compute_fpr95(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    FPR at 95% True Positive Rate (Table 1).
    Computed per-class (OvR) then averaged.

    Returns:
        FPR95 in percent
    """
    K = y_score.shape[1]
    fpr95_list = []
    for k in range(K):
        binary_y = (y_true == k).astype(int)
        scores_k = y_score[:, k]
        # Sort descending by score; threshold at 95% TPR
        order       = np.argsort(-scores_k)
        tpr_cumsum  = np.cumsum(binary_y[order])
        total_pos   = binary_y.sum()
        total_neg   = len(binary_y) - total_pos
        if total_pos == 0 or total_neg == 0:
            continue
        tpr_arr = tpr_cumsum / total_pos
        # Find index where TPR ≥ 0.95
        idx_95  = np.searchsorted(tpr_arr, 0.95)
        # Count FPs at that threshold
        fp_cumsum = np.cumsum(1 - binary_y[order])
        fpr95     = fp_cumsum[min(idx_95, len(fp_cumsum) - 1)] / total_neg
        fpr95_list.append(fpr95 * 100.0)
    return float(np.mean(fpr95_list)) if fpr95_list else float("nan")


def compute_sensitivity_specificity_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Sensitivity (recall), specificity, and F1 (Table 5).

    Returns:
        dict with sensitivity, specificity, f1 (all in %)
    """
    cm   = confusion_matrix(y_true, y_pred)
    K    = cm.shape[0]
    sens_list = []
    spec_list = []
    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / max(tp + fn, 1))
        spec_list.append(tn / max(tn + fp, 1))

    f1 = f1_score(y_true, y_pred, average=average, zero_division=0) * 100.0
    return {
        "sensitivity": np.mean(sens_list) * 100.0,
        "specificity": np.mean(spec_list) * 100.0,
        "f1"         : f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Causal / interpretability metrics (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_attribution_overlap(
    grad_maps: torch.Tensor,
    nuc_masks: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Attribution–Mask Overlap (Table 2, Attr. Overlap ↑).

    Measures what fraction of the top-attention mass falls within the
    nucleus–cytoplasm ROI mask (morphologically meaningful).

    Args:
        grad_maps : GradCAM / attention maps [B, H, W], positive values
        nuc_masks : Binary NC ROI masks      [B, H, W]
        threshold : Binarise grad_maps above this quantile

    Returns:
        Mean overlap score ∈ [0, 1]
    """
    grad_maps = grad_maps.float()
    nuc_masks = nuc_masks.float()

    # Normalise each map
    B = grad_maps.shape[0]
    flat      = grad_maps.view(B, -1)
    mins      = flat.min(dim=1).values.view(B, 1, 1)
    maxs      = flat.max(dim=1).values.view(B, 1, 1)
    grad_norm = (grad_maps - mins) / (maxs - mins + 1e-8)  # [0, 1]

    # Binary attention mask
    thr         = torch.quantile(grad_norm.view(B, -1), 1 - threshold, dim=1)
    thr         = thr.view(B, 1, 1)
    attention_bin = (grad_norm >= thr).float()

    # Intersection over attention area
    intersection = (attention_bin * nuc_masks).sum(dim=(1, 2))
    attention_area = attention_bin.sum(dim=(1, 2)).clamp(min=1)
    overlap = (intersection / attention_area).mean().item()
    return overlap


def compute_shortcut_correlation(
    features: torch.Tensor,
    artifact_labels: torch.Tensor,
) -> float:
    """
    Shortcut Correlation (Table 2, Shortcut Corr ↓).

    Pearson correlation between feature norm and binary artifact presence.
    Low value = features not driven by shortcuts.

    Args:
        features        : Feature embeddings [B, D]
        artifact_labels : Binary [B] (1 = artifact patch, 0 = clean)

    Returns:
        Absolute Pearson correlation (lower = better)
    """
    feat_norm = features.norm(dim=-1).cpu().float().numpy()   # [B]
    art_lbl   = artifact_labels.cpu().float().numpy()         # [B]
    if feat_norm.std() < 1e-8 or art_lbl.std() < 1e-8:
        return 0.0
    corr = float(np.corrcoef(feat_norm, art_lbl)[0, 1])
    return abs(corr)


def compute_causal_consistency(
    logits_clean: torch.Tensor,
    logits_perturbed: torch.Tensor,
) -> float:
    """
    Causal Consistency (Table 2, Causal Cons. ↑).

    Measures stability of predictions between clean and morphology-preserving
    perturbed samples. High value = decisions robust to non-causal changes.

    Args:
        logits_clean     : Logits on clean inputs       [B, K]
        logits_perturbed : Logits on perturbed inputs   [B, K]

    Returns:
        Fraction of samples with matching top-1 prediction ∈ [0, 1]
    """
    pred_clean = logits_clean.argmax(dim=-1)
    pred_perturb = logits_perturbed.argmax(dim=-1)
    return (pred_clean == pred_perturb).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Robustness metrics (Table 4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_auc(
    auc_clean: float,
    auc_perturbed: float,
) -> float:
    """
    ΔAUC = AUC_perturbed − AUC_clean  (Table 4, lower magnitude = better).
    """
    return auc_perturbed - auc_clean


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated evaluation for a model checkpoint (used in evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

class CervShortEvaluator:
    """
    Evaluator that computes all metrics reported in Tables 1–5.

    Usage:
        evaluator = CervShortEvaluator(num_classes=5)
        evaluator.update(logits, labels, domain_ids, grad_maps, nuc_masks,
                         logits_perturbed, artifact_labels)
        results = evaluator.compute()
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._logits       : List[torch.Tensor] = []
        self._labels       : List[torch.Tensor] = []
        self._domain_ids   : List[torch.Tensor] = []
        self._grad_maps    : List[torch.Tensor] = []
        self._nuc_masks    : List[torch.Tensor] = []
        self._logits_aug   : List[torch.Tensor] = []
        self._art_labels   : List[torch.Tensor] = []

    def update(
        self,
        logits:           torch.Tensor,
        labels:           torch.Tensor,
        domain_ids:       Optional[torch.Tensor] = None,
        grad_maps:        Optional[torch.Tensor] = None,
        nuc_masks:        Optional[torch.Tensor] = None,
        logits_augmented: Optional[torch.Tensor] = None,
        artifact_labels:  Optional[torch.Tensor] = None,
    ):
        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())
        if domain_ids is not None:
            self._domain_ids.append(domain_ids.detach().cpu())
        if grad_maps is not None:
            self._grad_maps.append(grad_maps.detach().cpu())
        if nuc_masks is not None:
            self._nuc_masks.append(nuc_masks.detach().cpu())
        if logits_augmented is not None:
            self._logits_aug.append(logits_augmented.detach().cpu())
        if artifact_labels is not None:
            self._art_labels.append(artifact_labels.detach().cpu())

    def compute(self) -> Dict[str, float]:
        all_logits = torch.cat(self._logits, dim=0)
        all_labels = torch.cat(self._labels, dim=0).numpy()
        all_scores = torch.softmax(all_logits, dim=-1).numpy()
        all_preds  = all_logits.argmax(dim=-1).numpy()

        results: Dict[str, float] = {}

        # ── Table 1 metrics ───────────────────────────────────────────────────
        results["auc"]   = compute_auc(all_labels, all_scores)
        results["fpr95"] = compute_fpr95(all_labels, all_scores)

        # ── Table 5 metrics ───────────────────────────────────────────────────
        sens_spec = compute_sensitivity_specificity_f1(all_labels, all_preds)
        results.update(sens_spec)

        # ── Table 2 metrics (if attribution maps available) ───────────────────
        if self._grad_maps and self._nuc_masks:
            all_grad = torch.cat(self._grad_maps, dim=0)
            all_mask = torch.cat(self._nuc_masks, dim=0)
            results["attr_overlap"] = compute_attribution_overlap(all_grad, all_mask)

        # ── Table 2 metrics (shortcut correlation) ────────────────────────────
        if self._art_labels:
            all_art = torch.cat(self._art_labels, dim=0)
            results["shortcut_corr"] = compute_shortcut_correlation(
                all_logits, all_art
            )

        # ── Table 2 metrics (causal consistency) ──────────────────────────────
        if self._logits_aug:
            all_aug = torch.cat(self._logits_aug, dim=0)
            results["causal_consistency"] = compute_causal_consistency(
                all_logits, all_aug
            )

        return results
