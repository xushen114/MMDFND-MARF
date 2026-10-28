# -*- coding: utf-8 -*-
"""Bootstrap CIs for MARF - Final and Rule - Final on key conditions."""
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"D:\多模态虚假新闻检测\MMDFND"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import corruptions  # noqa: E402
from train_trigger import feats, sigmoid  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "marf")
t = np.load(os.path.join(OUT, "trigger_model.npz"))
coef, b, mu, sd, T = t["coef"], float(t["intercept"][0]), t["mu"], t["sd"], float(t["T"])
CONDS = [c[0] for c in corruptions.conditions()]
GROUPS = {"clean": ["clean"], "image_missing": ["image_missing"],
          "text_missing": ["text_missing"], "both_missing": ["both_missing"],
          "all": CONDS}


def load(cond):
    with np.load(os.path.join(OUT, "benchmark_full", "test_%s.npz" % cond)) as z:
        d = {k: z[k] for k in z.files}
    with np.load(os.path.join(OUT, "preds", "selector6_%s.npz" % cond)) as z:
        p = {k: z[k] for k in z.files}
    return d, p


data = {}
for c in CONDS:
    d, p = load(c)
    X = feats(d)
    X = np.clip((X - mu) / sd, -5, 5)
    r = sigmoid(X @ coef.T + b).ravel()
    p_cal = sigmoid(np.log(np.clip(p["p_uniform"], 1e-6, 1 - 1e-6) /
                           (1 - np.clip(p["p_uniform"], 1e-6, 1 - 1e-6))) / T)
    data[c] = {"y": d["y"].astype(int), "final": p["p_final"],
               "rule": p["p_rule"], "marf": np.where(r > 0.5, p["p_selector"], p_cal)}


def f1(y, p):
    pred = (p >= 0.5).astype(int)
    out = []
    for cls in (0, 1):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        out.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def boot_cond(cond, n=2000):
    y = data[cond]["y"]; n_s = len(y)
    d_f, d_r = [], []
    rng = np.random.default_rng(0)
    for _ in range(n):
        idx = rng.integers(0, n_s, n_s)
        yy = y[idx]
        d_f.append(f1(yy, data[cond]["marf"][idx]) - f1(yy, data[cond]["final"][idx]))
        d_r.append(f1(yy, data[cond]["rule"][idx]) - f1(yy, data[cond]["final"][idx]))
    return np.array(d_f), np.array(d_r)


def boot_all(n=1000):
    rng = np.random.default_rng(1)
    d_f, d_r = [], []
    for _ in range(n):
        cs = rng.choice(CONDS, len(CONDS), replace=True)
        vf, vr = [], []
        for c in cs:
            y = data[c]["y"]; idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            vf.append(f1(yy, data[c]["marf"][idx]) - f1(yy, data[c]["final"][idx]))
            vr.append(f1(yy, data[c]["rule"][idx]) - f1(yy, data[c]["final"][idx]))
        d_f.append(np.mean(vf)); d_r.append(np.mean(vr))
    return np.array(d_f), np.array(d_r)


res = {}
for g, cs in GROUPS.items():
    if g == "all":
        df, dr = boot_all()
    else:
        df, dr = boot_cond(cs[0])
    res[g] = {"marf_minus_final": {"mean": float(df.mean()),
                                    "lo": float(np.percentile(df, 2.5)),
                                    "hi": float(np.percentile(df, 97.5))},
              "rule_minus_final": {"mean": float(dr.mean()),
                                    "lo": float(np.percentile(dr, 2.5)),
                                    "hi": float(np.percentile(dr, 97.5))}}
    print(g, "MARF-Final", res[g]["marf_minus_final"], "Rule-Final", res[g]["rule_minus_final"])
json.dump(res, open(os.path.join(OUT, "bootstrap_ci.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
