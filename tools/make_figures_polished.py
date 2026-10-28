# -*- coding: utf-8 -*-
"""Polished paper figures for the teacher-feedback revision (no table writes)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = r"D:\多模态虚假新闻检测\MMDFND"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import corruptions  # noqa: E402
from train_trigger import ece, feats, logit, sigmoid  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "marf")
PAPER = os.path.join(OUT, "paper")
CONDS = [c[0] for c in corruptions.conditions()]
GROUPS = {
    "Clean": ["clean"],
    "Img-miss": ["image_missing"],
    "Txt-miss": ["text_missing"],
    "Both-miss": ["both_missing"],
    "Img-corr": [c for c in CONDS if c.startswith(("image_noise", "image_blur", "image_occlusion"))],
    "Txt-corr": [c for c in CONDS if c.startswith(("text_truncate", "text_dropout"))],
    "Mismatch": ["image_shuffled", "text_shuffled"],
    "Held-out": [c for c in CONDS if c.startswith(("image_blur", "text_dropout"))],
    "All": CONDS,
}

NAVY = "#2F4B7C"
ORANGE = "#D97757"
GREEN = "#3D8B6D"
LIGHT = "#F7F7F5"
GRID = "#D9D9D9"


def load_cond(cond):
    with np.load(os.path.join(OUT, "benchmark_full", "test_%s.npz" % cond)) as z:
        d = {k: z[k] for k in z.files}
    with np.load(os.path.join(OUT, "preds", "selector6_%s.npz" % cond)) as z:
        p = {k: z[k] for k in z.files}
    return d, p


def f1(y, p):
    pred = (p >= 0.5).astype(int)
    vals = []
    for cls in (0, 1):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        vals.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def main():
    t = np.load(os.path.join(OUT, "trigger_model.npz"))
    coef, b, mu, sd, T = t["coef"], float(t["intercept"][0]), t["mu"], t["sd"], float(t["T"])
    marf, final, uni, ys = {}, {}, {}, {}
    for c in CONDS:
        d, p = load_cond(c)
        X = np.clip((feats(d) - mu) / sd, -5, 5)
        r = sigmoid(X @ coef.T + b).ravel()
        p_uni = p["p_uniform"]
        p_cal = sigmoid(logit(p_uni) / T)
        marf[c] = np.where(r > 0.5, p["p_selector"], p_cal)
        final[c] = p["p_final"]
        uni[c] = p_uni
        ys[c] = d["y"].astype(int)

    def g(src, cs):
        return float(np.mean([f1(ys[c], src[c]) for c in cs]))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # ---------- Figure 2: grouped bars ----------
    names = list(GROUPS.keys())
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    vals_f = [g(final, GROUPS[n]) for n in names]
    vals_u = [g(uni, GROUPS[n]) for n in names]
    vals_m = [g(marf, GROUPS[n]) for n in names]
    ax.bar(x - w, vals_f, w, label="Final head", color=NAVY, edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(x, vals_u, w, label="Uniform ensemble", color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(x + w, vals_m, w, label="MARF (ours)", color=GREEN, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, v in zip(x + w, vals_m):
        ax.text(xi, v + 0.008, "%.3f" % v, ha="center", va="bottom", fontsize=6.3, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.60, 0.99)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.38), fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_main.png"), dpi=300)
    plt.close(fig)

    # ---------- Figure 3: reliability ----------
    yy = np.concatenate([ys[c] for c in CONDS])
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    for src, lab, col in [(final, "Final head", NAVY), (uni, "Uniform", ORANGE), (marf, "MARF", GREEN)]:
        p = np.concatenate([src[c] for c in CONDS])
        conf = np.maximum(p, 1 - p)
        acc = ((p >= 0.5).astype(int) == yy).astype(float)
        xs, ys_ = [], []
        for lo in np.linspace(0, 1, 11)[:-1]:
            m = (conf >= lo) & (conf < lo + 0.1)
            if m.sum() > 0:
                xs.append(conf[m].mean()); ys_.append(acc[m].mean())
        ax.plot(xs, ys_, marker="o", ms=3.8, lw=1.4, color=col, label="%s (ECE %.3f)" % (lab, np.mean([ece(ys[c], src[c]) for c in CONDS])))
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.9, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_reliability.png"), dpi=300)
    plt.close(fig)

    # ---------- Figure 4: coverage-risk ----------
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    for src, lab, col in [(final, "Final head", NAVY), (uni, "Uniform", ORANGE), (marf, "MARF", GREEN)]:
        p = np.concatenate([src[c] for c in CONDS])
        err = ((p >= 0.5).astype(int) != yy).astype(float)
        unc = 1 - np.maximum(p, 1 - p)
        order = np.argsort(unc)
        cov = np.arange(1, len(order) + 1) / len(order)
        risk = np.cumsum(err[order]) / np.arange(1, len(order) + 1)
        ax.plot(cov, risk, lw=1.7, color=col, label=lab)
    ax.axvline(0.9, color="#777777", lw=0.8, ls=":")
    ax.text(0.905, 0.045, "90% coverage", fontsize=7, color="#555555")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Error rate")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_abstention.png"), dpi=300)
    plt.close(fig)

    # ---------- Figure 1: architecture ----------
    fig, ax = plt.subplots(figsize=(11.6, 4.7))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.7); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8.2, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=0.9, ls=ls, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=4, wrap=True)

    def arrow(x1, y1, x2, y2, color="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, lw=0.9, color=color, zorder=2))

    # groups
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 3.25, 4.15, boxstyle="round,pad=0.04",
                                fc="#F5F8FC", ec="#B8C7DA", lw=0.8, ls=(0, (4, 3)), zorder=1))
    ax.text(0.34, 4.22, "1  Frozen evidence", fontsize=8.2, color="#2F4B7C", fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.75, 0.25), 3.45, 4.15, boxstyle="round,pad=0.04",
                                fc="#FBF7EE", ec="#D9C9A3", lw=0.8, ls=(0, (4, 3)), zorder=1))
    ax.text(3.89, 4.22, "2  Reliability routing", fontsize=8.2, color="#8A6D1F", fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.45, 0.25), 3.9, 4.15, boxstyle="round,pad=0.04",
                                fc="#F3F8F3", ec="#B7D0BD", lw=0.8, ls=(0, (4, 3)), zorder=1))
    ax.text(7.59, 4.22, "3  Calibrated decision", fontsize=8.2, color="#2E6B4F", fontweight="bold")

    box(0.35, 2.55, 1.15, 0.85, "News text\n+ image", fc="#FFFFFF", fs=8.0)
    box(1.75, 2.70, 1.55, 1.25, "Frozen\nMMDFND", fc="#DCE9F5", fs=9.0, bold=True)
    box(1.75, 1.20, 1.55, 0.95, "Four heads\n$p_f,p_{fu},p_t,p_i$", fc="#FFFFFF", fs=7.6)
    box(1.75, 0.45, 1.55, 0.48, "6 experts\n+ uniform, rule", fc="#FFFFFF", fs=7.2)
    arrow(1.52, 2.97, 1.72, 2.97)
    arrow(2.52, 2.68, 2.52, 2.17)
    arrow(2.52, 1.18, 2.52, 0.95)

    box(3.95, 2.70, 1.45, 0.75, "Reliability\nfeatures $\\phi(x)$", fc="#FFFFFF", fs=7.8)
    box(3.95, 1.85, 1.45, 0.72, "Trigger\n$r=P(\\mathrm{missing}|x)$", fc="#F6E7C1", fs=7.4)
    box(5.60, 2.70, 1.35, 0.75, "Strategy\nselector $\\pi$", fc="#DDEBD8", fs=7.8)
    box(5.60, 1.85, 1.35, 0.72, "Calibrated\nuniform $p_{cal}$", fc="#EADCF3", fs=7.4)
    arrow(3.55, 1.02, 3.95, 1.02)
    arrow(3.75, 2.70, 3.95, 2.70)
    arrow(5.40, 2.70, 5.60, 2.70)
    arrow(4.68, 1.85, 4.68, 1.60)
    arrow(4.68, 1.58, 5.60, 1.58)

    box(7.70, 2.55, 1.35, 0.85, "Selector\n$p_{sel}$", fc="#DDEBD8", fs=7.8)
    box(7.70, 1.20, 1.35, 0.85, "Uniform + T\n$p_{cal}$", fc="#EADCF3", fs=7.8)
    box(9.35, 1.95, 0.95, 0.75, "Switch\n$r>0.5$?", fc="#F6E7C1", fs=7.6)
    box(9.35, 0.40, 1.35, 0.85, "$p$, $u$\nfake / real / abstain", fc="#FFFFFF", fs=7.2)
    arrow(6.95, 3.05, 7.70, 3.05)
    arrow(6.95, 2.20, 7.70, 2.20)
    arrow(9.05, 2.95, 9.35, 2.55)
    arrow(9.05, 1.65, 9.35, 2.25)
    arrow(10.00, 1.95, 10.00, 1.28)

    ax.text(5.55, 0.06, "Frozen backbone: no retraining.  Learned router: trigger + selector + one temperature (< 2K parameters).",
            ha="center", va="bottom", fontsize=7.2, color="#555555")

    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_architecture.png"), dpi=300)
    plt.close(fig)
    print("figures saved")


if __name__ == "__main__":
    main()
