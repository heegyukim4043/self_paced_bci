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
  and target-exclusion checks, plus an import check for the evaluation path.
  The published checkpoint loaded in a one-subject GPU smoke test. The
  frozen-probe 54-subject x 2-session run was subsequently completed on the
  server; it showed no consistent low-FP gain over the Riemannian comparator.

The older `large_causal_cnn_gru.py` and its runners are a separate, optional
capacity control. **Do not run them as a substitute for CBraMod.**

## Second pretrained model: LaBraM base (prepared, not yet run)

LaBraM is a separate frozen-backbone experiment. It uses the official
`labram-base.pth` weights, 19 Lee electrodes with published channel-position
indices, causal 0.1-75 Hz and 50 Hz-notch filtering, 200 Hz sampling, and only
the trailing 2 s at each decision. A source-only LOSO linear probe uses the
same 256 trial-0..49 examples per donor as CBraMod. Its scores follow the same
I4 FP1/2/4/8 threshold and circular-shift evaluation. The preprocessing
differs from CBraMod, so results compare **model plus input pipeline**, not
isolated architecture effects. No LaBraM performance result exists yet.

New files to upload with this README and `requirements.txt`:

- `src/uncued_mi/labram_transfer.py`
- `scripts/cache_labram_features_original_continuous_20261014.py`
- `scripts/fit_labram_loso_original_continuous_20261014.py`
- `scripts/run_labram_original_continuous_20261014.py`
- `tests/test_labram_transfer.py`

On the server, update this code repository. Clone the [official LaBraM
repository](https://github.com/935963004/LaBraM) separately into ignored
`third_party/LaBraM`. Verify that its checkpoint is a real weight file, not a
Git LFS pointer. Do not commit upstream code, weights, EEG or generated caches.

```bash
conda activate eegfm
python -c 'import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())'
python -m pip install -r requirements.txt
# On the current server (torch 2.11.0+cu128), force the matching CUDA 12.8 build.
# The default package index can install torchvision 0.26.0 built for CUDA 13.0.
python -m pip install --force-reinstall --no-deps 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -c 'import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.is_available())'
python -m pip check
python -m unittest discover -s tests -p test_labram_transfer.py -v
git clone https://github.com/935963004/LaBraM.git third_party/LaBraM
git -C third_party/LaBraM rev-parse HEAD
ls -lh third_party/LaBraM/checkpoints/labram-base.pth
sha256sum third_party/LaBraM/checkpoints/labram-base.pth
python scripts/cache_labram_features_original_continuous_20261014.py --subjects 1 --sessions 1 --input-unit uV --device cuda
```

The one-subject extraction must load every encoder weight and return finite
800-dimensional features. A safe-load failure is a **stop condition**; do not
use random weights or disable checkpoint validation. After checking the cache,
run the full cohort. Even `--subjects 1` during fitting requires all 54
feature caches for that session (53 donors plus target).

The server-observed LaBraM base checkpoint is 96,612,769 bytes with SHA-256
`7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
Its training metadata includes NumPy scalars. The loader inspects pickle
globals, permits only the known Namespace/NumPy metadata types, and retains
`weights_only=True`; never switch to `weights_only=False` as a workaround.

```bash
python -u scripts/cache_labram_features_original_continuous_20261014.py --input-unit uV --device cuda
python scripts/fit_labram_loso_original_continuous_20261014.py --subjects 1 --sessions 1
python scripts/run_labram_original_continuous_20261014.py --limit-cases 1
python -u scripts/fit_labram_loso_original_continuous_20261014.py
python -u scripts/run_labram_original_continuous_20261014.py
```

The result is `data/processed/labram_original_continuous_20261014.json`.
Report actual FP/min, SEN@2s and circular-shift excess; compare LaBraM with
CBraMod and Riemannian at matched achieved FP as well as the frozen operating
point. Lee remains cue-paced throughout.

## On the Ubuntu server

Run from this package root. Set up a Python environment (Python 3.10+), install
the required packages, and install a CUDA-compatible PyTorch wheel for the
server's GPU. `requirements.txt` alone does not guarantee CUDA availability.
Activate the existing `eegfm` environment first; running `pip` from `(base)`
can install a different PyTorch/CUDA stack into the base environment.
The E4b/I4 evaluation path imports `numba` transitively; it is a required
dependency for this CBraMod evaluation, not an optional historical package.
The historical scripts outside this pathway may have other optional dependencies.

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python scripts/download_lee2019_mi_server.py --subjects 1 --sessions 1 --dry-run
python scripts/download_lee2019_mi_server.py --subjects 1 --sessions 1
python scripts/download_lee2019_mi_server.py
python scripts/download_lee2019_mi_server.py --verify-only
python -m unittest discover -s tests -p test_cbramod_transfer.py -v
python -m pip check
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
