# Uncued MI BCI — Ubuntu GPU code-only package

This folder is the **actual upload set**. Make this folder, not its parent
research workspace, the root of a new private GitHub repository. It contains
Python/PowerShell source, tests, a minimal dependency list, and this README.
It contains **no EEG recordings, cached scores, model checkpoints, figures,
old Git history, OpenViBE artifacts, or research notebooks**.
Upload the **contents** of this folder as the repository root, so that
`src/`, `scripts/`, `tests/`, `requirements.txt`, and `.gitignore` are at top level.
Keep the code package separate from the large research workspace and raw EEG.

## Contents

| Path | Purpose |
| --- | --- |
| `src/uncued_mi/` | EEG loaders, preprocessing, decision methods and model classes |
| `scripts/` | Existing analysis dependencies plus the new server downloader, training and evaluation runners |
| `tests/` | Code checks, including causal-model tests |
| `requirements.txt`, `.gitignore` | Minimal Python dependencies and data/secret exclusions |

## Current large-model experiment: pretrained CBraMod

The intended new model is **CBraMod**, not a widened CNN-GRU. Its published
backbone is loaded strictly from an external checkpoint and kept frozen;
a fixed 800-dimensional source-only logistic probe is fitted on the other 53 Lee
subjects' original-continuous trials 0–49. This is the first representation
transfer test, **not full-backbone fine-tuning or a performance result**.

New uploadable code:

- `src/uncued_mi/cbramod_transfer.py`: 19-channel montage, samplewise CAR and
  causal 1000→200 Hz preprocessing, 2-s past-only input, strict weight loader.
- `scripts/cache_cbramod_features_original_continuous_20261014.py`: frozen
  feature cache for all 54 subjects × 2 sessions.
- `scripts/fit_cbramod_loso_original_continuous_20261014.py`: 53-donor LOSO
  linear probe and target score cache.
- `scripts/finetune_cbramod_loso_original_continuous_20261014.py`: separate
  full-backbone source-only fine-tuning arm (fixed three epochs); trained
  checkpoints are saved under ignored `data/models/`.
- `scripts/run_cbramod_original_continuous_20261014.py`: frozen I4 FP1/2/4/8
  evaluation on the original chronological Lee recording.
- `tests/test_cbramod_transfer.py`: CPU-only causality, montage, source-split
  and target-exclusion checks. A real checkpoint/GPU smoke test is still needed.

The older `large_causal_cnn_gru.py` and its runners are a separate, optional
capacity control. **Do not run them as a substitute for CBraMod.**

## On the Ubuntu server

Run from this package root. Set up a Python environment (Python 3.10+), install
the required packages, and install a CUDA-compatible PyTorch wheel for the
server's GPU. `requirements.txt` alone does not guarantee CUDA availability.
The historical scripts may have optional dependencies not listed there.

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python scripts/download_lee2019_mi_server.py --subjects 1 --sessions 1 --dry-run
python scripts/download_lee2019_mi_server.py --subjects 1 --sessions 1
python scripts/download_lee2019_mi_server.py
python scripts/download_lee2019_mi_server.py --verify-only
python -m unittest discover -s tests -p test_cbramod_transfer.py -v
python scripts/prepare_tier0_cache.py --all --sessions 1 2
python scripts/preflight_e4_full_trial_blocks_20261009.py
python scripts/preflight_e4b_task_idle_20261010.py
```

Clone the [official CBraMod source](https://github.com/wjq-learning/CBraMod)
under ignored `third_party/CBraMod/`, and obtain its published
`pretrained_weights.pth` from the [author-linked Hugging Face repository](https://huggingface.co/weighting666/CBraMod).
Record the repository commit and checkpoint SHA-256 before interpreting any
results. These external files are **not** part of this GitHub code package.

```bash
mkdir -p third_party
git clone https://github.com/wjq-learning/CBraMod.git third_party/CBraMod
mkdir -p third_party/CBraMod/pretrained_weights
curl -L --fail --retry 3 -o third_party/CBraMod/pretrained_weights/pretrained_weights.pth \
  https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
git -C third_party/CBraMod rev-parse HEAD
sha256sum third_party/CBraMod/pretrained_weights/pretrained_weights.pth
```

After the Lee download is complete, first run one feature-cache GPU smoke
case. The input unit is explicit; inspect the Lee file scale before choosing
`uV` versus `V` (older Lee audits generally use `uV`):

```bash
python scripts/cache_cbramod_features_original_continuous_20261014.py --subjects 1 --sessions 1 --input-unit uV --device cuda
```

Inspect the GPU memory, feature shapes and source-only trial IDs. Then run:

```bash
python scripts/cache_cbramod_features_original_continuous_20261014.py --input-unit uV --device cuda
python scripts/fit_cbramod_loso_original_continuous_20261014.py --subjects 1 --sessions 1
python scripts/run_cbramod_original_continuous_20261014.py --limit-cases 1
python scripts/fit_cbramod_loso_original_continuous_20261014.py
python scripts/run_cbramod_original_continuous_20261014.py
```

The LOSO probe requires all 54 frozen feature caches in a session, even for
the one-target smoke command. The feature cache also stores only source-trial
raw windows for the separate full-backbone training arm. After checking the
frozen probe, train one full model and then the cohort with the **same fixed
three-epoch protocol**:

```bash
python scripts/finetune_cbramod_loso_original_continuous_20261014.py --subjects 1 --sessions 1 --input-unit uV --device cuda
python scripts/run_cbramod_original_continuous_20261014.py --cache-root data/interim/lee2019_cbramod_finetuned_loso_scores --model-name CBraMod_pretrained_finetuned_LOSO --limit-cases 1
python scripts/finetune_cbramod_loso_original_continuous_20261014.py --input-unit uV --device cuda
python scripts/run_cbramod_original_continuous_20261014.py --cache-root data/interim/lee2019_cbramod_finetuned_loso_scores --model-name CBraMod_pretrained_finetuned_LOSO --output data/processed/cbramod_finetuned_original_continuous_20261014.json
```

Full fine-tuning repeats a backbone fit for every held-out subject and will
cost much more GPU time than the frozen probe. Neither the downloaded EEG,
the upstream repo, the published weights, trained models nor generated caches
should be committed to GitHub.

The downloader uses a URL pattern previously recorded for the public
[Lee2019 GigaDB dataset](https://gigadb.org/dataset/100542). The official
dataset lists the MI files, but server access to the direct file endpoint
must be checked with the one-file trial. Full MI data are 108 MAT files and
about 60.4 GiB locally. All downloads and outputs stay under ignored `data/`.

CBraMod instead uses original-continuous source windows and frozen published
weights. This differs from the old CNN-GRU in backbone, training domain,
channel montage and 2-s versus 1-s input support; it is **not** a controlled
parameter-count comparison. Lee is cue-paced, so even a positive result is
exploratory cue–task alignment, not genuine self-initiated MI detection.
