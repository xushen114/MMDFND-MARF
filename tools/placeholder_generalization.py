# -*- coding: utf-8 -*-
"""Zero-shot generalization to unseen image placeholder types.

The trigger and selector are trained on the original 21 conditions.  Here we
only run inference on four *unseen* placeholder tensors (white, black, checker,
random) and evaluate the frozen router without any retraining.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = r"D:\多模态虚假新闻检测\MMDFND"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import pilot_infer as pi  # noqa: E402
import build_benchmark_full as bb  # noqa: E402
import corruptions  # noqa: E402
from train_trigger import feats, logit, sigmoid  # noqa: E402
import train_selector as ts  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "marf")
PLACEHOLDERS = ["image_white", "image_black", "image_checker", "image_random"]


def f1(y, p):
    pred = (p >= 0.5).astype(int)
    vals = []
    for cls in (0, 1):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        vals.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def load_router():
    trig = np.load(os.path.join(OUT, "trigger_model.npz"))
    coef, b, mu_t, sd_t, T = trig["coef"], float(trig["intercept"][0]), trig["mu"], trig["sd"], float(trig["T"])
    ck = torch.load(os.path.join(OUT, "selector6.pt"), map_location="cpu")
    sel = ts.Selector(23, 64)
    sel.load_state_dict(ck["state_dict"])
    sel.eval()
    return coef, b, mu_t, sd_t, T, sel, ck["mu"], ck["sd"]


def evaluate(d, coef, b, mu_t, sd_t, T, sel, mu_s, sd_s):
    Xt = np.clip((feats(d) - mu_t) / sd_t, -5, 5)
    r = sigmoid(Xt @ coef.T + b).ravel()

    Xs, ph4, phx = ts.build(d, "full")
    Xs = np.clip((Xs - mu_s) / sd_s, -5, 5)
    with torch.no_grad():
        w = sel(torch.from_numpy(Xs)).numpy()
    p_sel = (w * phx).sum(1)
    p_uni = ph4.mean(1)
    p_cal = sigmoid(logit(p_uni) / T)
    p_marf = np.where(r > 0.5, p_sel, p_cal)
    y = d["y"].astype(int)
    return {
        "n": int(len(y)),
        "mean_r": float(r.mean()),
        "final_f1": f1(y, ph4[:, 0]),
        "uniform_f1": f1(y, p_uni),
        "marf_f1": f1(y, p_marf),
        "r": r,
        "y": y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--batchsize", type=int, default=32)
    ap.add_argument("--tag", default="placeholders")
    args = ap.parse_args()

    out_dir = os.path.join(OUT, "benchmark_" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    model = pi.build_model(pi.CKPT)
    store, clip = bb.attach(model)
    loader, _ = pi.build_loader("test", args.batchsize)

    for kind in PLACEHOLDERS:
        print("[placeholder] %s ..." % kind, flush=True)
        data, seen = bb.run_condition(model, loader, store, clip, kind, None, 3074, args.limit_batches)
        if args.limit_batches:
            for k in data:
                data[k] = data[k][:seen]
        np.savez_compressed(os.path.join(out_dir, "test_%s.npz" % kind), **data)
        print("  n=%d mean_img_std=%.4f mean_len=%.1f" % (
            len(data["y"]), float(data["img_std"].mean()), float(data["text_len"].mean())), flush=True)
        if args.limit_batches:
            continue

        # clean reference
        with np.load(os.path.join(OUT, "benchmark_full", "test_clean.npz")) as z:
            clean = {k: z[k] for k in z.files}
        coef, b, mu_t, sd_t, T, sel, mu_s, sd_s = load_router()
        res = evaluate(data, coef, b, mu_t, sd_t, T, sel, mu_s, sd_s)
        clean_res = evaluate(clean, coef, b, mu_t, sd_t, T, sel, mu_s, sd_s)
        auc = float(roc_auc_score(np.r_[np.zeros(len(clean_res["y"])), np.ones(len(res["y"]))],
                                  np.r_[clean_res["r"], res["r"]]))
        res.pop("r"); res.pop("y")
        res["auc_vs_clean"] = auc
        with open(os.path.join(out_dir, "test_%s.json" % kind), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("  mean_r=%.3f auc=%.3f final=%.4f uniform=%.4f MARF=%.4f" % (
            res["mean_r"], auc, res["final_f1"], res["uniform_f1"], res["marf_f1"]), flush=True)

    print("[placeholder] done ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
