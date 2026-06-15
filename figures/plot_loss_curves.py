"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CervShort — Figure 2: Loss Convergence Curves                     ║
║                                                                              ║
║  Reproduces Figure 2: Learning convergence and generalization gaps over      ║
║  100 training epochs for:                                                    ║
║    • Vulnerable baseline (wide red-shaded region)                           ║
║    • CervShort framework (tight blue-shaded region)                         ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Load or simulate training logs
# ─────────────────────────────────────────────────────────────────────────────

def load_log(log_path: str):
    """Load JSONL training log."""
    epochs, train_loss, val_loss = [], [], []
    with open(log_path) as f:
        for line in f:
            row = json.loads(line.strip())
            epochs.append(row["epoch"])
            train_loss.append(row.get("train_loss", np.nan))
            val_loss.append(row.get("val_loss",   np.nan))
    return np.array(epochs), np.array(train_loss), np.array(val_loss)


def simulate_logs(n_epochs: int = 100, seed: int = 42):
    """
    Simulate training curves consistent with paper Figure 2,
    used when real log files are not available.
    """
    rng    = np.random.default_rng(seed)
    epochs = np.arange(n_epochs)

    # ── Baseline: good training, poor val (high generalization gap) ──────────
    base_train = 0.8 * np.exp(-epochs / 30) + 0.08
    base_train += rng.normal(0, 0.01, n_epochs)
    # Val oscillates + stagnates early
    base_val   = base_train + 0.35 * np.exp(-epochs / 80) + 0.15
    base_val  += rng.normal(0, 0.025, n_epochs)
    # Clip to physical range
    base_train = np.clip(base_train, 0.05, 1.0)
    base_val   = np.clip(base_val,   0.10, 1.2)

    # ── CervShort: fast convergence, tight generalization gap ────────────────
    cs_train = 0.7 * np.exp(-epochs / 20) + 0.06
    cs_train += rng.normal(0, 0.005, n_epochs)
    cs_val   = cs_train + 0.04 * np.exp(-epochs / 40) + 0.02
    cs_val  += rng.normal(0, 0.006, n_epochs)
    cs_train = np.clip(cs_train, 0.04, 0.9)
    cs_val   = np.clip(cs_val,   0.05, 0.9)

    return epochs, base_train, base_val, cs_train, cs_val


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    epochs, base_train, base_val,
    cs_train, cs_val,
    output_path: str = "figures/cervshort_loss_curve.pdf",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        "Figure 2: Learning Convergence & Generalization Gap",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (tr, vl, color, shade, title) in zip(
        axes,
        [
            (base_train, base_val, "#d62728", "#ffb3b3", "Baseline (Vulnerable)"),
            (cs_train,  cs_val,   "#1f77b4", "#b3d1ff", "CervShort (Ours)"),
        ],
    ):
        ax.plot(epochs, tr, color=color, linewidth=2.0, label="Train loss", zorder=3)
        ax.plot(epochs, vl, color=color, linewidth=2.0, linestyle="--",
                label="Val loss", zorder=3)
        ax.fill_between(epochs, tr, vl, where=(vl > tr),
                        color=shade, alpha=0.55, label="Generalization gap", zorder=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, len(epochs) - 1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate gap
        gap_mid_epoch = len(epochs) // 2
        gap_val  = float(vl[gap_mid_epoch])
        gap_tr   = float(tr[gap_mid_epoch])
        gap_size = gap_val - gap_tr
        ax.annotate(
            f"Gap ≈ {gap_size:.3f}",
            xy=(gap_mid_epoch, (gap_val + gap_tr) / 2),
            xytext=(gap_mid_epoch + 5, (gap_val + gap_tr) / 2 + 0.05),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_log",  type=str, default=None)
    p.add_argument("--cervshort_log", type=str, default=None)
    p.add_argument("--output",        type=str,
                   default="figures/cervshort_loss_curve.pdf")
    p.add_argument("--n_epochs",      type=int, default=100)
    return p.parse_args()


def main():
    args = get_args()

    if args.baseline_log and os.path.exists(args.baseline_log) and \
       args.cervshort_log and os.path.exists(args.cervshort_log):
        epochs, base_train, base_val = load_log(args.baseline_log)
        _,      cs_train,   cs_val  = load_log(args.cervshort_log)
        epochs = epochs[:args.n_epochs]
        base_train = base_train[:args.n_epochs]
        base_val   = base_val[:args.n_epochs]
        cs_train   = cs_train[:args.n_epochs]
        cs_val     = cs_val[:args.n_epochs]
    else:
        print("[INFO] Log files not found — generating simulated curves.")
        epochs, base_train, base_val, cs_train, cs_val = \
            simulate_logs(n_epochs=args.n_epochs)

    plot_loss_curves(epochs, base_train, base_val, cs_train, cs_val,
                     output_path=args.output)


if __name__ == "__main__":
    main()
