# -*- coding: utf-8 -*-
"""Clean four-module architecture figure for the paper."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PAPER = r"D:\多模态虚假新闻检测\MMDFND\outputs\marf\paper"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
})

fig, ax = plt.subplots(figsize=(8.8, 3.3))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.0); ax.axis("off")

NAVY = "#2F4B7C"; ORANGE = "#D97757"; GREEN = "#3D8B6D"; PURPLE = "#8064A2"


def box(x, y, w, h, title, body, fc, ec, badge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=ec, zorder=4)
    ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center",
            fontsize=9.8, color="#222222", zorder=4, linespacing=1.35)
    if badge:
        ax.add_patch(plt.Circle((x + 0.22, y + h - 0.02), 0.16, color=ec, zorder=5))
        ax.text(x + 0.22, y + h - 0.02, badge, ha="center", va="center",
                color="white", fontsize=8.0, fontweight="bold", zorder=6)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#555555", zorder=2))


# Module 1: evidence
box(0.35, 1.25, 2.55, 1.55, "Frozen evidence",
    "MMDFND backbone\n4 heads  +  6 experts", "#EAF1F9", NAVY, "1")

# Module 2: reliability router
box(3.55, 2.25, 2.75, 1.10, "Reliability router",
    "21-D features  →  trigger $r$\n→  strategy selector $\\pi$", "#FBF3E4", "#A67C2E", "2")

# Module 3: calibration
box(3.55, 0.55, 2.75, 1.05, "Calibration",
    "uniform ensemble\n+ temperature $T$", "#F1EAF6", PURPLE, "3")

# Module 4: decision
box(7.05, 1.25, 3.55, 1.55, "Decision",
    "$r > 0.5$ ? selector : calibrated uniform\n$p$,  $u$,  fake / real / abstain", "#EAF4EE", GREEN, "4")

# arrows
arrow(2.92, 2.35, 3.53, 2.65)
arrow(2.92, 1.75, 3.53, 1.20)
arrow(6.32, 2.75, 7.03, 2.55)
arrow(6.32, 1.10, 7.03, 1.55)
arrow(10.62, 2.02, 11.05, 2.02)

ax.text(0.35, 0.22, "Frozen backbone: no retraining.", fontsize=8.0, color="#555555")
ax.text(3.55, 0.22, "Learned router: trigger + selector + one temperature (< 2K parameters).", fontsize=8.0, color="#555555")
ax.text(8.55, 0.22, "Abstention supports human review.", fontsize=8.0, color="#555555")

fig.tight_layout()
fig.savefig(os.path.join(PAPER, "fig_architecture.png"), dpi=320)
print("architecture saved")
