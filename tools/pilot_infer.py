# -*- coding: utf-8 -*-
"""Pilot step 2: how much does MMDFND depend on the domain label?

Runs the *existing* checkpoint with different domain-label sources and measures
the test-set macro-F1.  No model code is modified: the nine per-domain
probabilities are captured with forward hooks, and the domain label fed to the
gating network is overridden per pass.

Passes
    oracle  : ground-truth domain (sanity check: should reproduce 0.9242)
    pred    : domain predicted from content (RoBERTa+CLIP probe, ~70% acc)
    random  : uniformly random domain (worst case)

Variants per pass
    hard    : the probability of the routed domain (what the baseline does)
    soft    : sum_d pi_d * p_d, with pi from the content router
    uniform : mean of the nine domain probabilities

Usage
    python tools/pilot_infer.py --passes oracle,pred,random
    python tools/pilot_infer.py --passes oracle --limit_batches 2   # smoke
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MMDFND import MultiDomainPLEFENDModel  # noqa: E402
from utils.utils import clipdata2gpu, metricsTrueFalse  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs", "pilot")
BERT_DIR = os.path.join(ROOT, "pretrained_model", "chinese_roberta_wwm_base_ext_pytorch")
VOCAB = os.path.join(BERT_DIR, "vocab.txt")
CKPT = os.path.join(ROOT, "param_model", "MMDFND", "parameter_mmdfnd.pkl")
MAX_LEN = 197
CATEGORY_DICT = {
    "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4,
    "国际": 5, "教育": 6, "娱乐": 7, "社会": 8,
}
INV = {v: k for k, v in CATEGORY_DICT.items()}


def build_loader(split, batchsize):
    import cn_clip.clip as clip
    from utils.clip_dataloader import word2input

    df = pd.read_csv(os.path.join(DATA, split + "_origin.csv"), encoding="utf-8")
    df["content"] = df["content"].fillna("").astype(str)
    content = df["content"].astype("object").to_numpy()
    label = torch.tensor(df["label"].astype(int).to_numpy())
    category = torch.tensor(df["category"].map(CATEGORY_DICT).to_numpy())
    token_ids, masks = word2input(content, VOCAB, MAX_LEN)
    ordered_image = pickle.load(open(os.path.join(DATA, split + "_loader.pkl"), "rb"))
    clip_image = pickle.load(open(os.path.join(DATA, split + "_clip_loader.pkl"), "rb"))
    clip_text = clip.tokenize(content)
    ds = TensorDataset(token_ids, masks, label, category, ordered_image, clip_image, clip_text)
    dl = DataLoader(ds, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
    return dl, category.numpy()


def build_model(ckpt):
    print("[pilot] building model ...", flush=True)
    model = MultiDomainPLEFENDModel(768, [384], BERT_DIR, 320, 0.2)
    t0 = time.time()
    try:
        sd = torch.load(ckpt, map_location="cpu", mmap=True)
    except Exception as e:  # older torch / unusual zip layout
        print("[pilot] mmap load failed (%s), falling back" % e, flush=True)
        sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    del sd
    gc.collect()
    model = model.cuda().eval()
    print("[pilot] model ready in %.1fs" % (time.time() - t0), flush=True)
    return model


def attach_hooks(model):
    preds = [None] * 9

    def make(i):
        def hook(module, inputs, output):
            preds[i] = torch.sigmoid(output.squeeze(1)).detach().float().cpu()
        return hook

    handles = [m.register_forward_hook(make(i)) for i, m in enumerate(model.final_classifier_list)]
    return preds, handles


def run_pass(model, loader, preds, override, limit=0):
    n_total = len(loader.dataset)
    P = np.zeros((n_total, 9), dtype=np.float32)
    Y = np.zeros(n_total, dtype=np.int64)
    C = np.zeros(n_total, dtype=np.int64)
    start = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        bd = clipdata2gpu(batch)
        b = bd["label"].size(0)
        if override is not None:
            bd["category"] = torch.as_tensor(override[start:start + b], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            model(**bd)
        P[start:start + b] = torch.stack(preds, 1).float().numpy()
        Y[start:start + b] = bd["label"].detach().cpu().numpy().astype(np.int64)
        C[start:start + b] = batch[3].numpy().astype(np.int64)
        start += b
        if bi % 10 == 0:
            print("  batch %d/%d  %.0fs" % (bi, len(loader), time.time() - t0), flush=True)
    return P[:start], Y[:start], C[:start], time.time() - t0


def evaluate(y, p, c):
    return metricsTrueFalse([float(v) for v in y], [float(v) for v in p],
                            [int(v) for v in c], CATEGORY_DICT)


def brief(m):
    return {
        "metric": m.get("metric"),
        "acc": m.get("acc"),
        "auc": m.get("auc"),
        "fake_f1": m.get("fake", {}).get("F1"),
        "real_f1": m.get("real", {}).get("F1"),
        "domain_f1": {k: v.get("fscore") for k, v in m.items()
                      if isinstance(v, dict) and "fscore" in v},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--batchsize", type=int, default=32)
    ap.add_argument("--passes", default="oracle,pred,random")
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3074)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    loader, true_cat = build_loader("test", args.batchsize)
    n = len(loader.dataset)
    print("[pilot] test n=%d batches=%d" % (n, len(loader)), flush=True)

    pred_dom = np.load(os.path.join(OUT, "test_domain_pred_mm.npy"))
    proba = np.load(os.path.join(OUT, "test_domain_proba_mm.npy"))
    assert len(pred_dom) == n and proba.shape == (n, 9), (len(pred_dom), proba.shape, n)
    router_acc = float((pred_dom == true_cat).mean())
    print("[pilot] router (RoBERTa+CLIP) test domain acc = %.4f" % router_acc, flush=True)

    rng = np.random.default_rng(args.seed)
    random_dom = rng.integers(0, 9, size=n)

    model = build_model(args.ckpt)
    preds, _ = attach_hooks(model)

    results = {"config": vars(args), "n": n, "router_acc": router_acc, "passes": {}}
    arrays = {}

    for pname in [p.strip() for p in args.passes.split(",") if p.strip()]:
        if pname == "oracle":
            override = true_cat.copy()
        elif pname == "pred":
            override = pred_dom.copy()
        elif pname == "random":
            override = random_dom.copy()
        else:
            raise ValueError("unknown pass: " + pname)
        print("\n[pilot] ===== pass %s =====" % pname, flush=True)
        P, Y, C, dt = run_pass(model, loader, preds, override, args.limit_batches)
        print("[pilot] pass %s done in %.1fs" % (pname, dt), flush=True)
        arrays[pname] = P
        if args.limit_batches:
            print("[pilot] smoke shapes P=%s Y=%s C=%s" % (P.shape, Y.shape, C.shape), flush=True)
            continue

        idx = np.arange(len(Y))
        variants = {}
        if pname == "oracle":
            variants["hard_oracle"] = P[idx, C]
            variants["uniform"] = P.mean(1)
            variants["soft_router"] = (proba * P).sum(1)
        elif pname == "pred":
            variants["hard_pred"] = P[idx, C]
            variants["soft_router"] = (proba * P).sum(1)
            variants["uniform"] = P.mean(1)
        else:
            variants["hard_random"] = P[idx, C]
            variants["uniform"] = P.mean(1)

        entry = {"seconds": dt, "domain_acc": float((override == C).mean())}
        for vname, vp in variants.items():
            m = evaluate(Y, vp, C)
            entry[vname] = brief(m)
            print("  %-14s F1=%.4f acc=%.4f AUC=%.4f fakeF1=%.4f" % (
                vname, m["metric"], m["acc"], m["auc"], m["fake"]["F1"]), flush=True)
        results["passes"][pname] = entry

    if not args.limit_batches:
        np.savez_compressed(os.path.join(OUT, "pass_arrays_%s.npz" % args.tag),
                            labels=true_cat, pred_dom=pred_dom, **arrays)
        with open(os.path.join(OUT, "pilot_infer_%s.json" % args.tag), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n[pilot] saved outputs/pilot/pilot_infer_%s.json" % args.tag, flush=True)
        print("[pilot] summary")
        for pname, e in results["passes"].items():
            for vname, v in e.items():
                if isinstance(v, dict) and "metric" in v:
                    print("  %-6s %-13s F1=%.4f acc=%.4f" % (pname, vname, v["metric"], v["acc"]))
    print("[pilot] done", flush=True)


if __name__ == "__main__":
    main()
