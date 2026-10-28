# -*- coding: utf-8 -*-
"""Training/evaluation loop with run tags, per-epoch logging and predictions.

Kept separate from model/MMDFND.py so the original baseline code stays intact.
"""
import csv
import json
import os
import subprocess
import sys
import time
import unicodedata

import numpy as np
import torch
import tqdm

from utils.utils import clipdata2gpu, Averager, metricsTrueFalse
from .MMDFND import MultiDomainPLEFENDModel


def _dlen(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s, w):
    s = str(s)
    return s + " " * max(0, w - _dlen(s))


class Trainer:
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict,
                 weight_decay, early_stop=5, epoches=100, tag="run",
                 output_dir="./outputs", limit_batches=0, seed=3074,
                 batchsize=32, dataset="weibo", max_len=197,
                 eval_only=False, eval_ckpt=None, amp=False,
                 optim="adam", ada_lr=1e-3):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.inv_category = {v: k for k, v in category_dict.items()}
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.tag = tag
        self.output_dir = output_dir
        self.limit_batches = limit_batches
        self.seed = seed
        self.batchsize = batchsize
        self.dataset = dataset
        self.max_len = max_len
        self.eval_only = eval_only
        self.eval_ckpt = eval_ckpt
        self.amp = amp
        self.optim_name = optim
        self.ada_lr = ada_lr

        self.run_dir = os.path.join(output_dir, tag)
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_path = os.path.join(self.run_dir, "best.pkl")
        self.log_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.last_test = None

    # ---------------- config / io ----------------
    def _git_commit(self):
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    def save_config(self):
        cfg = {
            "tag": self.tag, "output_dir": self.output_dir, "seed": self.seed,
            "epoch": self.epoches, "batchsize": self.batchsize, "lr": self.lr,
            "weight_decay": self.weight_decay, "dataset": self.dataset,
            "max_len": self.max_len, "early_stop": self.early_stop,
            "limit_batches": self.limit_batches, "variant": "baseline",
            "optim": self.optim_name, "ada_lr": self.ada_lr,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_commit": self._git_commit(), "command": " ".join(sys.argv),
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def _log(self, obj):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---------------- test ----------------
    def _attach_hooks(self):
        self._final_preds = [None] * 9

        def make(i):
            def hook(module, inputs, output):
                self._final_preds[i] = torch.sigmoid(output.squeeze(1)).detach()
            return hook

        for i, m in enumerate(self.model.final_classifier_list):
            m.register_forward_hook(make(i))

    def test(self, dataloader, collect=False):
        self.model.eval()
        p0, ys, cats = [], [], []
        for batch in tqdm.tqdm(dataloader, disable=True):
            with torch.no_grad():
                bd = clipdata2gpu(batch)
                self.model(**bd)
                y = bd["label"].detach().cpu().numpy().astype(int).tolist()
                c = bd["category"].detach().cpu().numpy().astype(int).tolist()
                b = len(y)
                P = torch.stack(self._final_preds, 1)
                ar = torch.arange(b, device=P.device)
                prob = P[ar, bd["category"].long()]
                p0.extend(prob.detach().cpu().numpy().tolist())
                ys.extend(y)
                cats.extend(c)
        if collect:
            self.last_test = {"y": list(ys), "p": list(p0), "cat": list(cats)}
        res = metricsTrueFalse(ys, p0, cats, self.category_dict)
        return res

    # ---------------- reporting ----------------
    def _report(self, m, best_epoch):
        line = "=" * 78
        print("\n" + line)
        print(" MMDFND | tag=%s | seed=%d | batch=%d | best epoch=%d" % (
            self.tag, self.seed, self.batchsize, best_epoch))
        print(line)
        print(" Overall")
        print("   Macro-F1  %.4f    Accuracy  %.4f    AUC  %.4f" % (
            m["metric"], m["acc"], m.get("auc", 0.0)))
        print("   Precision %.4f    Recall    %.4f" % (m["precision"], m["recall"]))
        print("\n By class (threshold 0.5)")
        print("   %s %6s %8s %8s %8s" % (_pad("class", 8), "n", "P", "R", "F1"))
        for name in ("fake", "real"):
            r = m.get(name, {})
            n = sum(1 for x in (self.last_test["y"] if self.last_test else [])
                    if x == (1 if name == "fake" else 0))
            print("   %s %6d %8.4f %8.4f %8.4f" % (
                _pad(name, 8), n, r.get("precision", 0), r.get("recall", 0), r.get("F1", 0)))
        print("\n By domain (macro over classes, sorted by F1)")
        rows = []
        counts = np.bincount(self.last_test["cat"], minlength=9) if self.last_test else [0] * 9
        for c in range(9):
            r = m.get(self.inv_category[c], {})
            rows.append((r.get("fscore", 0), self.inv_category[c], int(counts[c]), r))
        rows.sort(reverse=True)
        print("   %s %5s %8s %8s %8s %8s" % (_pad("domain", 8), "n", "P", "R", "F1", "Acc"))
        for f1, name, n, r in rows:
            print("   %s %5d %8.4f %8.4f %8.4f %8.4f" % (
                _pad(name, 8), n, r.get("precision", 0), r.get("recall", 0), f1, r.get("acc", 0)))
        print("\n Files")
        for fn in ("config.json", "metrics.jsonl", "test_metrics.json",
                   "test_predictions.csv", "best.pkl"):
            print("   %s" % os.path.join(self.run_dir, fn))
        print(line + "\n")

    def _save_predictions(self):
        t = self.last_test
        if not t:
            return
        path = os.path.join(self.run_dir, "test_predictions.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "label", "prob_fake", "pred", "domain", "correct"])
            for i, (y, p, c) in enumerate(zip(t["y"], t["p"], t["cat"])):
                pred = int(p >= 0.5)
                w.writerow([i, y, "%.6f" % p, pred, self.inv_category[c],
                            int(pred == y)])

    # ---------------- train ----------------
    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.save_config()
        print("MMDFND | tag=%s | seed=%d | batch=%d | epochs=%d | lr=%g" % (
            self.tag, self.seed, self.batchsize, self.epoches, self.lr), flush=True)
        print("[mem] before model: alloc %.2f GB reserved %.2f GB" % (
            torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9),
            flush=True)

        self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims,
                                             self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        nparam = sum(p.numel() for p in self.model.parameters()) / 1e6
        ntrain = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print("[mem] after model: alloc %.2f GB reserved %.2f GB params %.1fM trainable %.1fM" % (
            torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9,
            nparam, ntrain), flush=True)
        self._attach_hooks()

        if self.eval_only:
            sd = torch.load(self.eval_ckpt, map_location="cpu")
            self.model.load_state_dict(sd)
            del sd
            self.model.eval()
            res = self.test(self.test_loader, collect=True)
            with open(os.path.join(self.run_dir, "test_metrics.json"), "w",
                      encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            self._save_predictions()
            self._report(res, 0)
            return

        loss_fn = torch.nn.BCELoss()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.optim_name == "adafactor":
            from transformers.optimization import Adafactor
            optimizer = Adafactor(trainable, lr=self.ada_lr, scale_parameter=False,
                                  relative_step=False, warmup_init=False,
                                  weight_decay=self.weight_decay)
            print("[optim] Adafactor lr=%g" % self.ada_lr, flush=True)
        else:
            optimizer = torch.optim.Adam(params=trainable, lr=self.lr,
                                         weight_decay=self.weight_decay)
            print("[optim] Adam lr=%g" % self.lr, flush=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        best_metric, best_epoch, bad = -1.0, 0, 0
        total_batches = len(self.train_loader)
        for epoch in range(self.epoches):
            t0 = time.time()
            self.model.train()
            avg = Averager()
            it = tqdm.tqdm(self.train_loader, disable=True)
            for step, batch in enumerate(it):
                if self.limit_batches and step >= self.limit_batches:
                    break
                bd = clipdata2gpu(batch)
                label = bd["label"]
                category = bd["category"]
                idxs = category.view(-1, 1)
                batch_label = torch.cat((
                    label[idxs.squeeze() == 0], label[idxs.squeeze() == 1],
                    label[idxs.squeeze() == 2], label[idxs.squeeze() == 3],
                    label[idxs.squeeze() == 4], label[idxs.squeeze() == 5],
                    label[idxs.squeeze() == 6], label[idxs.squeeze() == 7],
                    label[idxs.squeeze() == 8]))
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.amp):
                    outs = self.model(**bd)
                    loss = (0.7 * loss_fn(outs[0].float(), batch_label.float())
                            + 0.1 * loss_fn(outs[1].float(), batch_label.float())
                            + 0.1 * loss_fn(outs[2].float(), batch_label.float())
                            + 0.1 * loss_fn(outs[3].float(), batch_label.float()))
                if self.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                avg.add(loss.item())
                if (step + 1) % 20 == 0:
                    print("  batch %d/%d | loss %.4f | %.1fmin" % (
                        step + 1, total_batches, avg.item(), (time.time() - t0) / 60.0),
                        flush=True)

            if self.limit_batches:
                print("SMOKE OK | trained %d batches | loss %.4f | %.1fs" % (
                    self.limit_batches, avg.item(), time.time() - t0), flush=True)
                return

            results0 = self.test(self.val_loader)
            dt = time.time() - t0
            improved = results0["metric"] > best_metric
            if improved:
                best_metric, best_epoch, bad = results0["metric"], epoch + 1, 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                bad += 1
            self._log({
                "epoch": epoch + 1, "train_loss": avg.item(), "val": results0,
                "lr": optimizer.param_groups[0]["lr"], "seconds": dt,
                "best": bool(improved),
            })
            print("Epoch %02d | loss %.4f | val F1 %.4f | acc %.4f | AUC %.4f | %.1fmin%s" % (
                epoch + 1, avg.item(), results0["metric"], results0["acc"],
                results0.get("auc", 0.0), dt / 60.0, "  *" if improved else ""), flush=True)
            if bad >= self.early_stop:
                print("Early stop at epoch %d (best epoch %d)" % (epoch + 1, best_epoch))
                break

        self.model.load_state_dict(torch.load(self.best_path))
        results0 = self.test(self.test_loader, collect=True)
        with open(os.path.join(self.run_dir, "test_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results0, f, ensure_ascii=False, indent=2)
        self._save_predictions()
        self._report(results0, best_epoch)
