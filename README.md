# MMDFND-MARF

Code release for:

**Reliability-Aware Strategy Fusion for Multi-Domain Multimodal Fake News Detection under Missing Modalities**

This repository contains the code for the MARF router on top of a frozen
MMDFND backbone. It does **not** contain model weights, the Weibo dataset,
precomputed features, or result files. Those must be obtained separately
(see below), so that readers can reproduce the pipeline rather than download
our outputs.

## What is included

```text
main.py / run.py                 entry points for MMDFND training/evaluation
models_mae.py                    MAE definition
model/                           MMDFND backbone, BERT/layers/pivot, trainer
utils/                           data loaders and metrics
tools/                           benchmark, selector, trigger, bootstrap,
                                 placeholder generalization, figure scripts
data_pre.py, clip_data_pre.py    Weibo data/feature preparation
weibo21_*_pre.py                 optional Weibo21 preparation
requirements.txt
```

## What is not included

- `param_model/` (the 3GB MMDFND checkpoint)
- `pretrained_model/` (BERT/CLIP weights)
- `data/*.pkl`, `data/*.csv`, images (third-party data)
- `outputs/` (caches, trained router, predictions, results)
- `*.pkl`, `*.pt`, `*.npz` (any trained weights)

The `.gitignore` excludes these paths, so they cannot be committed by accident.

## Environment

Tested with Python 3.9, PyTorch 2.0.1 + CUDA 11.8, one RTX 4060 Laptop GPU
(8GB). Install PyTorch for your CUDA version first, then:

```bash
conda create -n mmdfnd python=3.9 -y
conda activate mmdfnd
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If you use CPU only, install the CPU build of PyTorch; the backbone inference
will be slow but the router code itself is small.

## Data and checkpoint

1. Obtain the Weibo data used by MMDFND from the
   [MMDFND repository](https://github.com/yutchina/MMDFND) or from the
   dataset release described there. The expected files are:

```text
data/train_origin.csv
data/val_origin.csv
data/test_origin.csv
data/train_loader.pkl
data/val_loader.pkl
data/test_loader.pkl
data/train_clip_loader.pkl
data/val_clip_loader.pkl
data/test_clip_loader.pkl
```

2. If you only have raw CSV/images, run the preprocessing scripts:

```bash
python data_pre.py
python clip_data_pre.py
# optional: python weibo21_data_pre.py / weibo21_clip_data_pre.py
```

3. Obtain the MMDFND checkpoint from its authors and place it at:

```text
param_model/MMDFND/parameter_mmdfnd.pkl
```

The repository does not redistribute weights. Exact paper numbers require the
same 3-epoch checkpoint used in the paper; a newly trained checkpoint will give
different values.

## Reproduce the pipeline

```bash
# 1) cache the four heads on 21 conditions (val + test), about 45 min on 8GB
python tools/build_benchmark_full.py

# 2) train the strategy selector on validation conditions
python tools/train_selector.py --tag selector6

# 3) train the missing-modality trigger, fit temperature, print tables
python tools/train_trigger.py

# 4) zero-shot placeholder generalization (Table 8)
python tools/placeholder_generalization.py

# 5) bootstrap confidence intervals
python tools/bootstrap_ci.py
```

`train_trigger.py` prints the grouped tables and saves
`outputs/marf/trigger_results.json`; generated outputs are ignored by Git.
The paper figures can be regenerated with:

```bash
python tools/make_figures_polished.py
python tools/make_architecture.py
```

## Hardware and runtime

| Step | Time (RTX 4060 8GB) |
|---|---:|
| `build_benchmark_full.py` | ~45 min |
| `train_selector.py` | ~2–3 min |
| `train_trigger.py` | ~1–2 min |
| `placeholder_generalization.py` | ~10 min |
| `bootstrap_ci.py` | ~5 min |

The backbone needs about 3.1GB of GPU memory for inference. Training the full
backbone with Adam does not fit in 8GB; the paper therefore uses a released
3-epoch checkpoint and freezes it.

## Notes and limitations

- **Single checkpoint.** The reported numbers use one 3-epoch MMDFND checkpoint;
  the selector is trained with three seeds, but backbone variance is not
  estimated.
- **Synthetic missing modalities.** Missing images are zero tensors and missing
  text is reduced to the CLS/SEP tokens.
- **Placeholder generalization is mixed.** White/black placeholders still trigger
  the router; checker/random placeholders do not. See Section 5.12 of the paper.
- **Do not expect exact reproduction** without the same checkpoint and data
  preprocessing. Report the checkpoint hash and data version if you need strict
  reproducibility.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The MMDFND checkpoint, Weibo data, BERT/CLIP weights, and other third-party
artifacts remain under their original licenses and are not redistributed here.
