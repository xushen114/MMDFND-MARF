# -*- coding: utf-8 -*-
"""P4: learned missing-modality trigger + temperature calibration.

1. Train a small logistic detector on val features to predict "missing modality"
   (image_missing / text_missing / both_missing) vs everything else.
2. Fit a temperature T on val clean uniform predictions.
3. Evaluate the soft policy:
       p = (1 - r) * p_uniform_calibrated + r * p_selector
   and compare with the hard-rule trigger and with uniform/final.

Outputs: outputs/marf/trigger_results.json, trigger_model.npz
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import pilot_infer as pi  # noqa: E402
import corruptions  # noqa: E402

BENCH = os.path.join(ROOT, "outputs", "marf", "benchmark_full")
PRED = os.path.join(ROOT, "outputs", "marf", "preds")
CONDS = [c[0] for c in corruptions.conditions()]
MISSING = {"image_missing", "text_missing", "both_missing"}
HEADS = ["final", "fusion", "text", "image"]


def load(split, cond):
    with np.load(os.path.join(BENCH, "%s_%s.npz" % (split, cond))) as z:
        return {k: z[k] for k in z.files}


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def feats(d):
    n = len(d["y"])
    idx = np.arange(n)
    ph = np.stack([d["p_" + h][idx, d["cat"]] for h in HEADS], 1).astype(np.float32)
    ent = entropy(ph).astype(np.float32)
    pairs = np.stack([np.abs(ph[:, i] - ph[:, j])
                      for i in range(4) for j in range(i + 1, 4)], 1).astype(np.float32)
    stats = np.stack([ph.mean(1), ph.std(1), ph.min(1), ph.max(1)], 1).astype(np.float32)
    return np.concatenate([
        ph, ent, pairs, stats,
        d["img_std"][:, None].astype(np.float32),
        np.log1p(d["text_len"])[:, None].astype(np.float32),
        d["clip_sim"][:, None].astype(np.float32),
    ], 1).astype(np.float32)


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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def met(y, p, cat):
    m = pi.evaluate(y, p, cat)
    return {"f1": float(m["metric"]), "acc": float(m["acc"]), "auc": float(m.get("auc") or 0.0),
            "ece": ece(y, p)}


def main():
    # ---- train trigger on val ----
    Xs, ys = [], []
    for cond in CONDS:
        d = load("val", cond)
        Xs.append(feats(d))
        ys.append(np.full(len(d["y"]), 1 if cond in MISSING else 0))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.clip((X - mu) / sd, -5, 5)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(Xn, y)
    print("[trigger] val n=%d pos=%d" % (len(y), int(y.sum())))

    # ---- fit temperature on val clean uniform ----
    d = load("val", "clean")
    ph = np.stack([d["p_" + h][np.arange(len(d["y"])), d["cat"]] for h in HEADS], 1)
    p_uni = ph.mean(1)

    def nll(T):
        p = sigmoid(logit(p_uni) / T)
        return -np.mean(d["y"] * np.log(p + 1e-9) + (1 - d["y"]) * np.log(1 - p + 1e-9))

    res = minimize_scalar(nll, bounds=(0.2, 8.0), method="bounded")
    T = float(res.x)
    print("[trigger] temperature T=%.3f (val clean NLL %.4f)" % (T, float(res.fun)))

    def calib(p):
        return sigmoid(logit(p) / T)

    # ---- test ----
    rows = {}
    probs = {}
    for cond in CONDS:
        d = load("test", cond)
        with np.load(os.path.join(PRED, "selector6_%s.npz" % cond)) as z:
            p_sel = z["p_selector"]
        ph = np.stack([d["p_" + h][np.arange(len(d["y"])), d["cat"]] for h in HEADS], 1)
        p_uni = ph.mean(1)
        p_fin = ph[:, 0]
        Xc = np.clip((feats(d) - mu) / sd, -5, 5)
        r = clf.predict_proba(Xc)[:, 1]
        p_soft = (1 - r) * calib(p_uni) + r * p_sel
        p_lhard = np.where(r > 0.5, p_sel, calib(p_uni))
        hard = (d["img_std"] < 1e-3) | (d["text_len"] <= 2)
        p_hard = np.where(hard, p_sel, p_uni)
        p_hard_cal = np.where(hard, p_sel, calib(p_uni))
        rows[cond] = {
            "final": met(d["y"], p_fin, d["cat"]),
            "uniform": met(d["y"], p_uni, d["cat"]),
            "uniform_cal": met(d["y"], calib(p_uni), d["cat"]),
            "selector": met(d["y"], p_sel, d["cat"]),
            "hard": met(d["y"], p_hard, d["cat"]),
            "soft": met(d["y"], p_soft, d["cat"]),
            "learned_hard": met(d["y"], p_lhard, d["cat"]),
        }
        probs[cond] = {"r_mean": float(r.mean()), "r_pos_rate": float((r > 0.5).mean())}
    print("[trigger] test AUC=%.4f AP=%.4f" % (
        roc_auc_score(np.concatenate([np.full(len(load("test", c)["y"]), 1 if c in MISSING else 0)
                                      for c in CONDS]),
                      np.concatenate([clf.predict_proba(np.clip((feats(load("test", c)) - mu) / sd, -5, 5))[:, 1]
                                      for c in CONDS])),
        average_precision_score(np.concatenate([np.full(len(load("test", c)["y"]), 1 if c in MISSING else 0)
                                                for c in CONDS]),
                                np.concatenate([clf.predict_proba(np.clip((feats(load("test", c)) - mu) / sd, -5, 5))[:, 1]
                                                for c in CONDS]))))

    groups = {
        "clean": ["clean"], "image_missing": ["image_missing"],
        "text_missing": ["text_missing"], "both_missing": ["both_missing"],
        "image_corrupt": [c for c in CONDS if c.startswith(("image_noise", "image_blur", "image_occlusion"))],
        "text_corrupt": [c for c in CONDS if c.startswith(("text_truncate", "text_dropout"))],
        "shuffled": ["image_shuffled", "text_shuffled"],
        "heldout": [c for c in CONDS if c.startswith(("image_blur", "text_dropout"))],
        "all": CONDS,
    }
    methods = ["final", "uniform", "uniform_cal", "selector", "hard", "soft", "learned_hard"]
    print("\n%-16s " % "group" + " ".join("%11s" % m for m in methods))
    for g, cs in groups.items():
        vals = [np.mean([rows[c][m]["f1"] for c in cs]) for m in methods]
        print("%-16s " % g + " ".join("%11.4f" % v for v in vals))
    print("\n%-16s " % "group" + " ".join("%11s" % m for m in ["final", "uniform", "uniform_cal", "soft", "learned_hard"]))
    for g, cs in groups.items():
        vals = [np.mean([rows[c][m]["ece"] for c in cs]) for m in
                ["final", "uniform", "uniform_cal", "soft", "learned_hard"]]
        print("%-16s " % g + " ".join("%11.4f" % v for v in vals))
    print("\ntrigger mean r: clean=%.3f image_missing=%.3f text_missing=%.3f both=%.3f" % (
        probs["clean"]["r_mean"], probs["image_missing"]["r_mean"],
        probs["text_missing"]["r_mean"], probs["both_missing"]["r_mean"]))

    with open(os.path.join(ROOT, "outputs", "marf", "trigger_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"temperature": T, "test": rows, "groups": {
            g: {m: float(np.mean([rows[c][m]["f1"] for c in cs])) for m in methods}
            for g, cs in groups.items()}}, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(ROOT, "outputs", "marf", "trigger_model.npz"),
             coef=clf.coef_, intercept=clf.intercept_, mu=mu, sd=sd, T=T)
    print("\nsaved outputs/marf/trigger_results.json")


if __name__ == "__main__":
    main()
