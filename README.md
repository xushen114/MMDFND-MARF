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

## Data, pretrained models, and checkpoint

> This repository does **not** redistribute the Weibo/Weibo21 datasets,
> pretrained encoders, or the MMDFND checkpoint. The links below point to the
> official sources; please follow their licenses and terms of use.

### 1. Weibo dataset

The processed Weibo data with domain labels are distributed by the MMDFND
authors. As listed in the official MMDFND README:

- Baidu Pan: https://pan.baidu.com/s/1TGc-8RUt6BIHO1rjnzuPxQ (code: `qwer`)
- Official project: https://github.com/yutchina/MMDFND

Place the extracted files under `data/`:

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

If the Baidu link expires, use the official MMDFND repository or contact its
authors. The processed files bypass the preprocessing step.

### 2. Weibo21 (optional)

Weibo21 is **not** redistributed here. Following the MMDFND README and
Ying et al. (2023), request the complete multimodal multi-domain dataset from
Dr. Qiong Nan (`nanqiong19z@ict.ac.cn`), and see
https://github.com/yingqichao/fnd-bootstrap for the original benchmark.

Place the Weibo21 files under `Weibo_21/`:

```text
Weibo_21/train_clip_loader.pkl
Weibo_21/train_loader.pkl
Weibo_21/train_datasets.xlsx
Weibo_21/val_...
Weibo_21/test_...
```

### 3. Pretrained encoders

The backbone code expects the same pretrained encoders as the official
MMDFND implementation:

- **RoBERTa**: download from
  https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing
  and put all files under `pretrained_model/`.
- **MAE**: follow https://github.com/facebookresearch/mae and place
  `mae_pretrain_vit_base.pth` in the repository root.
- **Chinese-CLIP**: follow https://github.com/OFA-Sys/Chinese-CLIP and place
  `clip_cn_vit-b-16.pt` in the repository root.

These files are not included in this repository.

### 4. MMDFND checkpoint

The paper uses:

```text
param_model/MMDFND/parameter_mmdfnd.pkl
```

This checkpoint is **not** part of the official MMDFND repository and is not
redistributed here. Request it from the authors, or train MMDFND with
`python main.py`. Exact paper numbers require the same 3-epoch checkpoint; a
newly trained checkpoint will give different values.

### 5. Preprocessing (only if you have raw data)

If you start from raw CSV/images instead of the processed files:

```bash
python data_pre.py
python clip_data_pre.py
# optional Weibo21:
python weibo21_data_pre.py
python weibo21_clip_data_pre.py
```

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
