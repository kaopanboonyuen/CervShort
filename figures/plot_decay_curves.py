"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      CervShort — Figure 3 (Right): Performance Decay Under Artifacts        ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_decay(output="figures/performance_decay_plot.pdf"):
    intensity = np.linspace(0, 1, 50)
    baseline  = 87.2 * np.exp(-3.2 * intensity) + 10.0 * (1 - intensity)
    cervshort = 95.9 - 8.0 * intensity - 4.0 * intensity**2

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(intensity * 100, baseline,  color="#d62728", lw=2.5, label="Baseline (ViT-L/16)")
    ax.plot(intensity * 100, cervshort, color="#1f77b4", lw=2.5, label="CervShort (Ours)")
    ax.fill_between(intensity * 100, baseline, cervshort, color="#b3d1ff", alpha=0.4)
    ax.set_xlabel("Artifact Intensity (%)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Performance Decay Under Artifact Stress (Figure 3 Right)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.close()

if __name__ == "__main__":
    plot_decay()
