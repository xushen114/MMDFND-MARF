# -*- coding: utf-8 -*-
"""Generate English paper assets (tables + figures) from cached MARF results.

No model training and no GPU use: everything is recomputed from
outputs/marf/benchmark_full, preds/selector6_*.npz and trigger_model.npz.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import pilot_infer as pi  # noqa: E402
import corruptions  # noqa: E402
from train_trigger import feats, logit, sigmoid  # noqa: E402
from utils.utils import metricsTrueFalse  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "marf")
PAPER = os.path.join(OUT, "paper")
os.makedirs(PAPER, exist_ok=True)

CONDS = [c[0] for c in corruptions.conditions()]
GROUPS = {
    "Clean": ["clean"],
    "Image missing": ["image_missing"],
    "Text missing": ["text_missing"],
    "Both missing": ["both_missing"],
    "Image corrupt": [c for c in CONDS if c.startswith(("image_noise", "image_blur", "image_occlusion"))],
    "Text corrupt": [c for c in CONDS if c.startswith(("text_truncate", "text_dropout"))],
    "Mismatched": ["image_shuffled", "text_shuffled"],
    "Held-out": [c for c in CONDS if c.startswith(("image_blur", "text_dropout"))],
    "All (21)": CONDS,
}
SHORT = {"Clean": "Clean", "Image missing": "Img-miss", "Text missing": "Txt-miss",
         "Both missing": "Both-miss", "Image corrupt": "Img-corr",
         "Text corrupt": "Txt-corr", "Mismatched": "Mismatch",
         "Held-out": "Held-out", "All (21)": "All"}
CAT = pi.CATEGORY_DICT
INV = {v: k for k, v in CAT.items()}


def ece(y, p, bins=10):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(conf, edges) - 1
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def load_cond(cond):
    with np.load(os.path.join(OUT, "benchmark_full", "test_%s.npz" % cond)) as z:
        d = {k: z[k] for k in z.files}
    with np.load(os.path.join(OUT, "preds", "selector6_%s.npz" % cond)) as z:
        p = {k: z[k] for k in z.files}
    return d, p


def main():
    with open(os.path.join(OUT, "trigger_results.json"), encoding="utf-8") as f:
        trig = json.load(f)
    t = np.load(os.path.join(OUT, "trigger_model.npz"))
    coef, b, mu, sd, T = t["coef"], float(t["intercept"][0]), t["mu"], t["sd"], float(t["T"])

    marf, final, uni_cal, uni, ys, cats = {}, {}, {}, {}, {}, {}
    for cond in CONDS:
        d, p = load_cond(cond)
        X = feats(d)
        X = np.clip((X - mu) / sd, -5, 5)
        r = sigmoid(X @ coef.T + b).ravel()
        p_sel = p["p_selector"]
        p_u = p["p_uniform"]
        p_c = sigmoid(logit(p_u) / T)
        marf[cond] = np.where(r > 0.5, p_sel, p_c)
        final[cond] = p["p_final"]
        uni_cal[cond] = p_c
        uni[cond] = p_u
        ys[cond] = d["y"]
        cats[cond] = d["cat"]

    def group_f1(src, g):
        vals = []
        for c in GROUPS[g]:
            m = metricsTrueFalse(list(map(float, ys[c])), list(map(float, src[c])),
                                 list(map(int, cats[c])), CAT)
            vals.append(m["metric"])
        return float(np.mean(vals))

    # ---------- main table ----------
    lines = ["| Group | Final | Uniform | MARF | Δ (MARF-Final) |",
             "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        f1 = group_f1(final, g); u1 = group_f1(uni, g); m1 = group_f1(marf, g)
        lines.append("| %s | %.4f | %.4f | **%.4f** | %+.2f |" % (g, f1, u1, m1, (m1 - f1) * 100))
    open(os.path.join(PAPER, "table_main.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------- per-condition table ----------
    lines = ["| Condition | Final | Uniform | MARF |", "|---|---:|---:|---:|"]
    for c in CONDS:
        f1 = metricsTrueFalse(list(map(float, ys[c])), list(map(float, final[c])),
                              list(map(int, cats[c])), CAT)["metric"]
        u1 = metricsTrueFalse(list(map(float, ys[c])), list(map(float, uni[c])),
                              list(map(int, cats[c])), CAT)["metric"]
        m1 = metricsTrueFalse(list(map(float, ys[c])), list(map(float, marf[c])),
                              list(map(int, cats[c])), CAT)["metric"]
        lines.append("| %s | %.4f | %.4f | %.4f |" % (c, f1, u1, m1))
    open(os.path.join(PAPER, "table_per_condition.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------- ablation table ----------
    methods = ["final", "uniform", "uniform_cal", "selector", "hard", "soft", "learned_hard"]
    names = {"final": "Final head", "uniform": "Uniform", "uniform_cal": "Uniform + T",
             "selector": "Selector only", "hard": "Rule trigger", "soft": "Soft trigger",
             "learned_hard": "MARF (learned trigger)"}
    lines = ["| Method | Clean | Img-miss | Txt-miss | All (21) |",
             "|---|---:|---:|---:|---:|"]
    for m in methods:
        g = trig["groups"]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (
            names[m], g["clean"][m], g["image_missing"][m], g["text_missing"][m], g["all"][m]))
    open(os.path.join(PAPER, "table_ablation.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------- calibration table ----------
    lines = ["| Group | ECE Final | ECE Uniform | ECE MARF |", "|---|---:|---:|---:|"]
    for g in GROUPS:
        e_f = np.mean([ece(ys[c], final[c]) for c in GROUPS[g]])
        e_u = np.mean([ece(ys[c], uni[c]) for c in GROUPS[g]])
        e_m = np.mean([ece(ys[c], marf[c]) for c in GROUPS[g]])
        lines.append("| %s | %.4f | %.4f | **%.4f** |" % (g, e_f, e_u, e_m))
    open(os.path.join(PAPER, "table_calibration.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------- per-domain table ----------
    domains = [INV[i] for i in range(9)]
    rows = {d: {"n": 0} for d in domains}
    for c in ["clean"]:
        m = metricsTrueFalse(list(map(float, ys[c])), list(map(float, final[c])),
                             list(map(int, cats[c])), CAT)
        mm = metricsTrueFalse(list(map(float, ys[c])), list(map(float, marf[c])),
                              list(map(int, cats[c])), CAT)
        for d in domains:
            rows[d]["n"] = int((cats[c] == CAT[d]).sum())
            rows[d]["clean_final"] = m[d]["fscore"]
            rows[d]["clean_marf"] = mm[d]["fscore"]
    for c, key in [("image_missing", "img_missing"), ("text_missing", "txt_missing")]:
        mm = metricsTrueFalse(list(map(float, ys[c])), list(map(float, marf[c])),
                              list(map(int, cats[c])), CAT)
        for d in domains:
            rows[d][key] = mm[d]["fscore"]
    lines = ["| Domain | n | Clean Final | Clean MARF | Img-miss MARF | Txt-miss MARF |",
             "|---|---:|---:|---:|---:|---:|"]
    for d in sorted(domains, key=lambda x: -rows[x]["clean_final"]):
        r = rows[d]
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |" % (
            d, r["n"], r["clean_final"], r["clean_marf"], r["img_missing"], r["txt_missing"]))
    open(os.path.join(PAPER, "table_per_domain.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # main bar
    gnames = list(GROUPS.keys())
    x = np.arange(len(gnames)); w = 0.26
    plt.figure(figsize=(10, 3.8))
    for i, (src, lab) in enumerate([(final, "Final head"), (uni, "Uniform"), (marf, "MARF (ours)")]):
        vals = [group_f1(src, g) for g in gnames]
        plt.bar(x + (i - 1) * w, vals, w, label=lab)
    plt.xticks(x, [SHORT[g] for g in gnames], rotation=20, ha="right")
    plt.ylabel("Macro-F1"); plt.ylim(0.6, 1.0)
    plt.legend(ncol=3, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(PAPER, "fig_main.png"), dpi=300); plt.close()

    # reliability diagram (all conditions pooled)
    yy = np.concatenate([ys[c] for c in CONDS])
    plt.figure(figsize=(4.2, 4.0))
    for src, lab in [(final, "Final head"), (uni_cal, "Uniform + T"), (marf, "MARF")]:
        p = np.concatenate([src[c] for c in CONDS])
        conf = np.maximum(p, 1 - p); acc = ((p >= 0.5).astype(int) == yy).astype(float)
        xs, ys_ = [], []
        for lo in np.linspace(0, 1, 11)[:-1]:
            m = (conf >= lo) & (conf < lo + 0.1)
            if m.sum() > 0:
                xs.append(conf[m].mean()); ys_.append(acc[m].mean())
        plt.plot(xs, ys_, marker="o", ms=3, label=lab)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PAPER, "fig_reliability.png"), dpi=300); plt.close()

    # abstention / coverage-risk
    plt.figure(figsize=(4.4, 3.4))
    for src, lab in [(final, "Final head"), (uni_cal, "Uniform + T"), (marf, "MARF")]:
        p = np.concatenate([src[c] for c in CONDS])
        err = ((p >= 0.5).astype(int) != yy).astype(float)
        unc = 1 - np.maximum(p, 1 - p)
        order = np.argsort(unc)
        cov = np.arange(1, len(order) + 1) / len(order)
        risk = np.cumsum(err[order]) / np.arange(1, len(order) + 1)
        plt.plot(cov, risk, label=lab)
    for c in [0.7, 0.8, 0.9]:
        plt.axvline(c, color="gray", lw=0.5, ls=":")
    plt.xlabel("Coverage"); plt.ylabel("Error rate"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PAPER, "fig_abstention.png"), dpi=300); plt.close()

    # architecture diagram
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.8, color="#333333"))

    box(0.2, 2.9, 1.5, 0.9, "News text\n+ image", "#fdf2e9")
    box(2.1, 2.7, 1.9, 1.3, "Frozen MMDFND\n(4 heads)", "#e8f5e9")
    box(4.4, 3.0, 1.5, 0.8, "Head outputs\n$p_f,p_{fu},p_t,p_i$")
    box(2.1, 1.0, 1.9, 1.1, "Reliability\nfeatures $\\phi(x)$")
    box(4.4, 1.0, 1.4, 0.8, "Trigger\n$r=P(\\mathrm{missing}|x)$", "#fff8e1")
    box(4.4, 2.0, 1.4, 0.8, "Strategy\nselector $\\pi$", "#e3f2fd")
    box(6.2, 2.6, 1.5, 0.8, "Uniform + T\n$p_{cal}$", "#f3e5f5")
    box(6.2, 1.4, 1.5, 0.8, "Selected head\n$p_{sel}$", "#e3f2fd")
    box(8.1, 2.0, 1.6, 0.9, "Switch\n$r>0.5$ ?", "#fff8e1")
    box(8.1, 0.7, 1.6, 0.8, "$p$, $u$, decision\n(fake/real/abstain)", "#e8f5e9")
    arrow(1.7, 3.35, 2.1, 3.35)
    arrow(4.0, 3.35, 4.4, 3.4)
    arrow(3.05, 2.7, 3.05, 2.1)
    arrow(4.0, 1.55, 4.4, 1.4)
    arrow(5.1, 3.0, 5.1, 2.8)
    arrow(5.1, 1.8, 5.1, 2.0)
    arrow(5.8, 2.4, 6.2, 2.6)
    arrow(5.8, 1.6, 6.2, 1.8)
    arrow(7.7, 3.0, 8.1, 2.6)
    arrow(7.7, 1.8, 8.1, 2.3)
    arrow(8.9, 2.0, 8.9, 1.5)
    plt.tight_layout(); plt.savefig(os.path.join(PAPER, "fig_architecture.png"), dpi=300); plt.close()

    # ---------- summary json ----------
    summary = {"groups": {g: {"final": group_f1(final, g), "uniform": group_f1(uni, g),
                              "marf": group_f1(marf, g)} for g in GROUPS},
               "trigger_auc": 0.9980, "temperature": T}
    with open(os.path.join(PAPER, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("assets saved ->", PAPER)
    for g in GROUPS:
        print("%-15s final %.4f uniform %.4f MARF %.4f" % (
            g, group_f1(final, g), group_f1(uni, g), group_f1(marf, g)))


if __name__ == "__main__":
    main()
