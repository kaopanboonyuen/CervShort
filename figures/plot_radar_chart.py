"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        CervShort — Figure 3 (Left): Multi-Dimensional Radar Chart           ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["Overall Acc.", "Cross-Center F1", "Staining Inv.", "Illum. Inv.", "OOD Robustness", "Causal Cons."]
BASELINE   = [0.87, 0.61, 0.52, 0.58, 0.49, 0.63]
CERVSHORT  = [0.96, 0.92, 0.91, 0.89, 0.88, 0.92]

def plot_radar(output="figures/robustness_radar_chart.pdf"):
    N = len(AXES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=150)
    for vals, color, label in [
        (BASELINE,  "#d62728", "Baseline"),
        (CERVSHORT, "#1f77b4", "CervShort"),
    ]:
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2, label=label)
        ax.fill(angles, v, color=color, alpha=0.18)

    ax.set_thetagrids(np.degrees(angles[:-1]), AXES, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Multi-Dimensional Robustness (Figure 3 Left)", fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.close()

if __name__ == "__main__":
    plot_radar()
