# -*- coding: utf-8 -*-
"""Build the full modality-corruption benchmark cache (20 conditions).

For each split and condition, run the frozen backbone once and cache the four
head probabilities plus reliability features.

Usage:
    python tools/build_benchmark_full.py --limit_batches 2 --conditions image_noise_s3,text_truncate_s3 --splits val --tag smoke
    python tools/build_benchmark_full.py
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import pilot_infer as pi  # noqa: E402
import corruptions  # noqa: E402


def attach(model):
    store = {"final": [None] * 9, "fusion": [None] * 9,
             "text": [None] * 9, "image": [None] * 9}
    groups = {"final": model.final_classifier_list,
              "fusion": model.fusion_classifier_list,
              "text": model.text_classifier_list,
              "image": model.image_classifier_list}
    for name, mods in groups.items():
        for i, m in enumerate(mods):
            def hook(module, inputs, output, name=name, i=i):
                store[name][i] = torch.sigmoid(output.squeeze(1)).detach().float().cpu()
            m.register_forward_hook(hook)
    clip = {}

    def clip_hook(module, inputs):
        clip["x"] = inputs[0].detach()

    model.clip_fusion.register_forward_pre_hook(clip_hook)
    return store, clip


def run_condition(model, loader, store, clip, kind, severity, seed, limit=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(loader.dataset)
    out = {k: np.zeros((n, 9), dtype=np.float32) for k in store}
    y = np.zeros(n, dtype=np.int64)
    cat = np.zeros(n, dtype=np.int64)
    img_std = np.zeros(n, dtype=np.float32)
    text_len = np.zeros(n, dtype=np.float32)
    clip_sim = np.zeros(n, dtype=np.float32)
    s = 0
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        bd = pi.clipdata2gpu(batch)
        b = bd["label"].size(0)
        corruptions.apply(bd, kind, severity, rng)
        img_std[s:s + b] = bd["image"].flatten(1).std(dim=1).detach().cpu().numpy()
        text_len[s:s + b] = bd["content_masks"].sum(dim=1).detach().cpu().numpy()
        with torch.inference_mode():
            model(**bd)
        x = clip["x"]
        sim = (x[:, :512] * x[:, 512:]).sum(dim=1)
        clip_sim[s:s + b] = torch.nan_to_num(sim, nan=0.0).detach().cpu().numpy()
        for name in store:
            out[name][s:s + b] = torch.stack(store[name], 1).float().numpy()
        y[s:s + b] = bd["label"].detach().cpu().numpy().astype(np.int64)
        cat[s:s + b] = batch[3].numpy().astype(np.int64)
        s += b
    return dict(p_final=out["final"], p_fusion=out["fusion"], p_text=out["text"],
                p_image=out["image"], y=y, cat=cat, img_std=img_std,
                text_len=text_len, clip_sim=clip_sim), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    all_conds = corruptions.conditions()
    if args.conditions != "all":
        wanted = set(args.conditions.split(","))
        all_conds = [c for c in all_conds if c[0] in wanted]
    out_dir = os.path.join(ROOT, "outputs", "marf", "benchmark_" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print("[bench] %d conditions -> %s" % (len(all_conds), out_dir), flush=True)

    model = pi.build_model(pi.CKPT)
    store, clip = attach(model)
    meta = {}
    for split in args.splits.split(","):
        print("[bench] loading %s ..." % split, flush=True)
        loader, _ = pi.build_loader(split, 32)
        for name, kind, sev in all_conds:
            t0 = time.time()
            data, seen = run_condition(model, loader, store, clip, kind, sev, 3074,
                                       args.limit_batches)
            if args.limit_batches:
                valid = slice(0, seen)
                for k in data:
                    data[k] = data[k][valid]
            path = os.path.join(out_dir, "%s_%s.npz" % (split, name))
            np.savez_compressed(path, **data)
            n = len(data["y"])
            idx = np.arange(n)
            if args.limit_batches:
                f1 = float("nan")
            else:
                f1 = pi.evaluate(data["y"], data["p_final"][idx, data["cat"]], data["cat"])["metric"]
            print("[bench] %-5s %-20s n=%-4d F1(final)=%.4f img_std=%.3f len=%.1f sim=%.3f %.1fs" % (
                split, name, n, f1, float(np.mean(data["img_std"])),
                float(np.mean(data["text_len"])), float(np.mean(data["clip_sim"])),
                time.time() - t0), flush=True)
            meta["%s/%s" % (split, name)] = {"kind": kind, "severity": sev, "n": int(n),
                                             "f1_final": float(f1)}
        del loader
        gc.collect()
        torch.cuda.empty_cache()
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[bench] done", flush=True)


if __name__ == "__main__":
    main()
