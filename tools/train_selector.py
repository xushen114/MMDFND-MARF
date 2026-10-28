# -*- coding: utf-8 -*-
"""Train the MARF strategy selector (P2/P3).

Five candidate strategies: final / fusion / text / image / uniform.
A small MLP maps reliability features to strategy weights; the backbone is
frozen and only cached head outputs are used.

Protocol
    train : val conditions, excluding the held-out corruption types
    test  : test conditions, including the held-out types
    seeds : 3, early stopping on a 15% per-condition val split

Usage:
    python tools/train_selector.py
    python tools/train_selector.py --heldout image_blur,text_dropout --feature_set full
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import pilot_infer as pi  # noqa: E402
import corruptions  # noqa: E402

STRATS = ["final", "fusion", "text", "image", "uniform", "rule"]
IMG_CORR = ("image_noise", "image_blur", "image_occlusion")
TXT_CORR = ("text_truncate", "text_dropout")


def condition_names():
    return [c[0] for c in corruptions.conditions()]


def load(split, cond, bench):
    with np.load(os.path.join(bench, "%s_%s.npz" % (split, cond))) as z:
        return {k: z[k] for k in z.files}


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def build(d, feature_set="full"):
    n = len(d["y"])
    idx = np.arange(n)
    ph4 = np.stack([d["p_" + h][idx, d["cat"]] for h in ["final", "fusion", "text", "image"]], 1)
    ph5 = np.concatenate([ph4, ph4.mean(1, keepdims=True)], 1).astype(np.float32)
    phx = np.concatenate([ph5, rule_p(ph4, d["img_std"], d["text_len"])[:, None]], 1).astype(np.float32)
    if feature_set == "basic":
        ent = binary_entropy(ph4).astype(np.float32)
        pairs = np.stack([np.abs(ph4[:, i] - ph4[:, j])
                          for i in range(4) for j in range(i + 1, 4)], 1).astype(np.float32)
        X = np.concatenate([ph4, ent, pairs], 1).astype(np.float32)
        return X, ph4, phx
    ent = binary_entropy(ph4).astype(np.float32)
    pairs = np.stack([np.abs(ph4[:, i] - ph4[:, j])
                      for i in range(4) for j in range(i + 1, 4)], 1).astype(np.float32)
    stats = np.stack([ph4.mean(1), ph4.std(1), ph4.min(1), ph4.max(1)], 1).astype(np.float32)
    img = d["img_std"][:, None].astype(np.float32)
    txt = np.log1p(d["text_len"])[:, None].astype(np.float32)
    sim = d["clip_sim"][:, None].astype(np.float32)
    miss = np.stack([(d["img_std"] < 1e-3).astype(np.float32),
                     (d["text_len"] <= 2).astype(np.float32)], 1)
    X = np.concatenate([ph4, ent, pairs, stats, img, txt, sim, miss], 1).astype(np.float32)
    return X, ph4, phx


class Selector(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, len(STRATS)))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


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


def met(y, p, cat):
    m = pi.evaluate(y, p, cat)
    return {"f1": float(m["metric"]), "acc": float(m["acc"]),
            "auc": float(m.get("auc") or 0.0), "ece": ece(y, p)}


def rule_p(ph4, img_std, text_len):
    p = ph4.mean(1).copy()
    img = img_std < 1e-3
    txt = text_len <= 2
    p[img & ~txt] = ph4[img & ~txt, 2]
    p[txt & ~img] = ph4[txt & ~img, 1]
    p[img & txt] = ph4[img & txt, 0]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(ROOT, "outputs", "marf", "benchmark_full"))
    ap.add_argument("--heldout", default="image_blur,text_dropout")
    ap.add_argument("--feature_set", default="full", choices=["full", "basic"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--clean_weight", type=float, default=1.0)
    ap.add_argument("--tag", default="selector")
    args = ap.parse_args()

    all_conds = condition_names()
    heldout = tuple(args.heldout.split(",")) if args.heldout else ()
    train_conds = [c for c in all_conds if not c.startswith(heldout)]
    print("[selector] train conds=%d test conds=%d heldout=%s" % (
        len(train_conds), len(all_conds), args.heldout))

    # ---- training data ----
    Xs, Ys, PHs, cond_ids = [], [], [], []
    for ci, cond in enumerate(train_conds):
        d = load("val", cond, args.bench)
        X, ph4, phx = build(d, args.feature_set)
        Xs.append(X); Ys.append(d["y"]); PHs.append(phx)
        cond_ids.append(np.full(len(d["y"]), ci))
    X = np.concatenate(Xs); Y = np.concatenate(Ys); PH = np.concatenate(PHs)
    CID = np.concatenate(cond_ids)
    mu, sd = X.mean(0), X.std(0) + 1e-6

    # per-condition 85/15 split
    rng = np.random.default_rng(0)
    tr_idx, va_idx = [], []
    for ci in range(len(train_conds)):
        idx = np.where(CID == ci)[0]
        rng.shuffle(idx)
        n_va = max(1, int(0.15 * len(idx)))
        va_idx.append(idx[:n_va]); tr_idx.append(idx[n_va:])
    tr_idx = np.concatenate(tr_idx); va_idx = np.concatenate(va_idx)

    Xn = np.clip((X - mu) / sd, -5, 5)
    Xt = torch.tensor(Xn[tr_idx], device="cuda")
    yt = torch.tensor(Y[tr_idx], dtype=torch.float32, device="cuda")
    Xv = torch.tensor(Xn[va_idx], device="cuda")
    yv = torch.tensor(Y[va_idx], dtype=torch.float32, device="cuda")
    pht = torch.tensor(PH[tr_idx], device="cuda")
    phv = torch.tensor(PH[va_idx], device="cuda")
    cond_of = np.array(train_conds)[CID]
    sw = np.ones(len(tr_idx), dtype=np.float32)
    sw[cond_of[tr_idx] == "clean"] = args.clean_weight
    wt = torch.tensor(sw, device="cuda")

    test = {}
    for cond in all_conds:
        d = load("test", cond, args.bench)
        Xc, ph4, phx = build(d, args.feature_set)
        test[cond] = {"d": d, "X": np.clip((Xc - mu) / sd, -5, 5), "ph4": ph4, "phx": phx}

    results = {c: {} for c in all_conds}
    sel_metrics = {c: [] for c in all_conds}
    last_state = None
    pred_dir = os.path.join(ROOT, "outputs", "marf", "preds")
    os.makedirs(pred_dir, exist_ok=True)

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = Selector(Xt.shape[1]).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCELoss(reduction="none")
        best = (1e9, None)
        for epoch in range(args.epochs):
            model.train()
            p = (model(Xt) * pht).sum(1)
            loss = (lossf(p.clamp(1e-6, 1 - 1e-6), yt) * wt).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    pv = (model(Xv) * phv).sum(1).clamp(1e-6, 1 - 1e-6)
                    vl = lossf(pv, yv).mean().item()
                if vl < best[0]:
                    best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[1]); model.eval()
        last_state = model.state_dict()

        for cond in all_conds:
            t = test[cond]
            with torch.no_grad():
                w = model(torch.tensor(t["X"], device="cuda")).cpu().numpy()
            p = (w * t["phx"]).sum(1)
            if args.shrink > 0:
                p = (1 - args.shrink) * p + args.shrink * t["ph4"].mean(1)
            m = met(t["d"]["y"], p, t["d"]["cat"])
            sel_metrics[cond].append(m)
            if seed == 0:
                results[cond]["selector"] = m
                np.save(os.path.join(ROOT, "outputs", "marf",
                                     "weights_%s_%s.npy" % (args.tag, cond)), w)
                np.savez_compressed(
                    os.path.join(pred_dir, "%s_%s.npz" % (args.tag, cond)),
                    y=t["d"]["y"], cat=t["d"]["cat"], p_selector=p,
                    p_final=t["ph4"][:, 0], p_uniform=t["ph4"].mean(1),
                    p_rule=rule_p(t["ph4"], t["d"]["img_std"], t["d"]["text_len"]),
                    w=w)

    # ---- baselines ----
    for cond in all_conds:
        t = test[cond]; d = t["d"]; ph4 = t["ph4"]; idx = np.arange(len(d["y"]))
        for h, col in zip(["final", "fusion", "text", "image"], range(4)):
            results[cond][h] = met(d["y"], ph4[:, col], d["cat"])
        results[cond]["uniform"] = met(d["y"], ph4.mean(1), d["cat"])
        results[cond]["rule"] = met(d["y"], rule_p(ph4, d["img_std"], d["text_len"]), d["cat"])
        best = max([(results[cond][s]["f1"], s) for s in STRATS])
        results[cond]["oracle"] = results[cond][best[1]]

    for cond in all_conds:
        for k in ["f1", "acc", "auc", "ece"]:
            results[cond]["selector"][k + "_std"] = float(np.std([m[k] for m in sel_metrics[cond]]))

    # ---- print ----
    methods = ["final", "uniform", "rule", "selector", "oracle"]
    print("\n%-22s %-7s %7s %7s %7s %7s" % ("condition", "held", "F1", "Acc", "AUC", "ECE"))
    for cond in all_conds:
        held = "HELD" if cond.startswith(heldout) else ""
        for m in methods:
            r = results[cond][m]
            std = ("+/-%.3f" % r["f1_std"]) if m == "selector" else ""
            print("%-22s %-7s %-7s %.4f%-8s %.4f %.4f %.4f" % (
                cond, held, m, r["f1"], std, r["acc"], r["auc"], r["ece"]))
        print()

    def avg(conds, method):
        vals = [results[c][method]["f1"] for c in conds]
        return float(np.mean(vals))

    groups = {
        "clean": ["clean"],
        "image_missing": ["image_missing"],
        "text_missing": ["text_missing"],
        "both_missing": ["both_missing"],
        "image_corrupt": [c for c in all_conds if c.startswith(IMG_CORR)],
        "text_corrupt": [c for c in all_conds if c.startswith(TXT_CORR)],
        "shuffled": ["image_shuffled", "text_shuffled"],
        "heldout": [c for c in all_conds if c.startswith(heldout)],
        "all": all_conds,
    }
    groups = {g: [c for c in cs if c in results] for g, cs in groups.items()}
    groups = {g: cs for g, cs in groups.items() if cs}
    print("\n%-16s %7s %7s %7s %7s" % ("group", "final", "uniform", "rule", "selector"))
    for g, cs in groups.items():
        print("%-16s %.4f  %.4f  %.4f  %.4f" % (
            g, avg(cs, "final"), avg(cs, "uniform"), avg(cs, "rule"), avg(cs, "selector")))

    out = {"config": vars(args), "groups": {g: {m: avg(cs, m) for m in methods} for g, cs in groups.items()},
           "test": results, "train_conditions": train_conds}
    with open(os.path.join(ROOT, "outputs", "marf", "%s_results.json" % args.tag), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    torch.save({"state_dict": last_state, "mu": mu, "sd": sd, "strats": STRATS},
               os.path.join("outputs", "marf", "%s.pt" % args.tag))
    print("\nsaved outputs/marf/%s_results.json and %s.pt" % (args.tag, args.tag))


if __name__ == "__main__":
    main()
