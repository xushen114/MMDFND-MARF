# -*- coding: utf-8 -*-
"""Corruption operators for the MMDFND modality-robustness benchmark.

All operators work in-place on the batch dict produced by clipdata2gpu():
    content, content_masks, image, clip_image, clip_text
and keep the tensors on the same device/dtype.

Severity semantics
    image_noise      sigma of Gaussian noise (normalized image space)
    image_blur       box-blur kernel size (3/5/7)
    image_occlusion  fraction of image area covered by a black square
    text_truncate    fraction of tokens kept (0.75/0.50/0.25)
    text_dropout     fraction of non-special tokens replaced by [UNK]
"""
import numpy as np
import torch
import torch.nn.functional as F

UNK_ID = 100
SEP_ID = 102


def _blur(x, k):
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)


def _occlude(x, frac, rng):
    b, c, h, w = x.shape
    side = int(round((h * w * frac) ** 0.5))
    side = max(1, min(side, h, w))
    for i in range(b):
        top = int(rng.integers(0, h - side + 1))
        left = int(rng.integers(0, w - side + 1))
        x[i, :, top:top + side, left:left + side] = 0.0
    return x


def _truncate_text(content, masks, clip_text, frac):
    b = content.size(0)
    lc = clip_text.size(1)
    for i in range(b):
        n = int(masks[i].sum().item())
        keep = max(2, min(n, int(round(n * frac))))
        content[i, keep - 1] = SEP_ID
        content[i, keep:] = 0
        masks[i, :keep] = 1.0
        masks[i, keep:] = 0.0
        keep_c = max(2, min(lc, int(round(lc * frac))))
        clip_text[i, keep_c:] = 0


def _dropout_text(content, masks, clip_text, frac, rng):
    b = content.size(0)
    lc = clip_text.size(1)
    for i in range(b):
        n = int(masks[i].sum().item())
        if n <= 2:
            continue
        k = max(1, int(round((n - 2) * frac)))
        pos = rng.choice(np.arange(1, n - 1), size=min(k, n - 2), replace=False)
        content[i, pos] = UNK_ID
        pos_c = rng.choice(np.arange(1, lc), size=min(k, lc - 1), replace=False)
        clip_text[i, pos_c] = UNK_ID


def _zero_image(bd):
    bd["image"] = torch.zeros_like(bd["image"])
    bd["clip_image"] = torch.zeros_like(bd["clip_image"])


# ImageNet / CLIP normalization constants, used to synthesize placeholders in
# normalized tensor space. The frozen backbone receives the same normalized
# tensors as it does for real images.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _placeholder_image(bd, kind):
    """White/black/checker/random image placeholders in normalized space."""
    for key, mean, std in (("image", _IMAGENET_MEAN, _IMAGENET_STD),
                           ("clip_image", _CLIP_MEAN, _CLIP_STD)):
        x = bd[key]
        b, c, h, w = x.shape
        if kind == "image_white":
            base = torch.ones(1, 1, h, w, device=x.device, dtype=x.dtype)
        elif kind == "image_black":
            base = torch.zeros(1, 1, h, w, device=x.device, dtype=x.dtype)
        elif kind == "image_checker":
            rows = torch.arange(h, device=x.device) // 16
            cols = torch.arange(w, device=x.device) // 16
            base = ((rows[:, None] + cols[None, :]) % 2).to(x.dtype).view(1, 1, h, w)
        elif kind == "image_random":
            base = torch.rand(b, c, h, w, device=x.device, dtype=x.dtype)
        else:
            raise ValueError("unknown placeholder: " + kind)
        if base.shape[0] == 1:
            base = base.expand(b, c, h, w)
        m = mean.to(device=x.device, dtype=x.dtype)
        s = std.to(device=x.device, dtype=x.dtype)
        bd[key] = (base - m) / s


def _missing_text(bd):
    c, m = bd["content"], bd["content_masks"]
    c.zero_()
    c[:, 0] = 101
    c[:, 1] = SEP_ID
    m.zero_()
    m[:, :2] = 1
    bd["clip_text"] = torch.zeros_like(bd["clip_text"])


def apply(bd, kind, severity, rng):
    """Apply one corruption.  kind == 'clean' is a no-op."""
    if kind == "clean":
        return
    if kind == "image_missing":
        _zero_image(bd)
    elif kind in ("image_white", "image_black", "image_checker", "image_random"):
        _placeholder_image(bd, kind)
    elif kind == "image_noise":
        for k in ("image", "clip_image"):
            bd[k] = bd[k] + torch.randn_like(bd[k]) * float(severity)
    elif kind == "image_blur":
        k = int(severity)
        for key in ("image", "clip_image"):
            bd[key] = _blur(bd[key], k)
    elif kind == "image_occlusion":
        for key in ("image", "clip_image"):
            bd[key] = _occlude(bd[key].clone(), float(severity), rng)
    elif kind == "image_shuffled":
        b = bd["image"].size(0)
        perm = torch.randperm(b, device=bd["image"].device)
        bd["image"] = bd["image"][perm]
        bd["clip_image"] = bd["clip_image"][perm]
    elif kind == "text_missing":
        _missing_text(bd)
    elif kind == "text_truncate":
        _truncate_text(bd["content"], bd["content_masks"], bd["clip_text"], float(severity))
    elif kind == "text_dropout":
        _dropout_text(bd["content"], bd["content_masks"], bd["clip_text"],
                      float(severity), rng)
    elif kind == "text_shuffled":
        b = bd["content"].size(0)
        perm = torch.randperm(b, device=bd["content"].device)
        bd["content"] = bd["content"][perm]
        bd["content_masks"] = bd["content_masks"][perm]
        bd["clip_text"] = bd["clip_text"][perm]
    elif kind == "both_missing":
        _zero_image(bd)
        _missing_text(bd)
    else:
        raise ValueError("unknown corruption: " + kind)


def conditions():
    """Ordered list of (name, kind, severity)."""
    out = [("clean", "clean", None)]
    out.append(("image_missing", "image_missing", None))
    out.append(("text_missing", "text_missing", None))
    for name, sig in [("s1", 0.05), ("s2", 0.10), ("s3", 0.20)]:
        out.append(("image_noise_" + name, "image_noise", sig))
    for name, k in [("s1", 3), ("s2", 5), ("s3", 7)]:
        out.append(("image_blur_" + name, "image_blur", k))
    for name, f in [("s1", 0.10), ("s2", 0.30), ("s3", 0.50)]:
        out.append(("image_occlusion_" + name, "image_occlusion", f))
    for name, f in [("s1", 0.75), ("s2", 0.50), ("s3", 0.25)]:
        out.append(("text_truncate_" + name, "text_truncate", f))
    for name, f in [("s1", 0.10), ("s2", 0.30), ("s3", 0.50)]:
        out.append(("text_dropout_" + name, "text_dropout", f))
    out.append(("image_shuffled", "image_shuffled", None))
    out.append(("text_shuffled", "text_shuffled", None))
    out.append(("both_missing", "both_missing", None))
    return out
