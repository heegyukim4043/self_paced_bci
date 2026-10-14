# Uncued MI BCI — Ubuntu GPU code-only package

This folder is the **actual upload set**. Make this folder, not its parent
research workspace, the root of a new private GitHub repository. It contains
Python/PowerShell source, tests, a minimal dependency list, and this README.
It contains **no EEG recordings, cached scores, model checkpoints, figures,
old Git history, OpenViBE artifacts, or research notebooks**.
Upload the **contents** of this folder as the repository root, so that
`src/`, `scripts/`, `tests/`, `requirements.txt`, and `.gitignore` are at top level.
At preparation there were 419 source files plus these 3 root files (422 files,
about 2.83 MB in total).

## Contents

| Path | Purpose |
| --- | --- |
| `src/uncued_mi/` | EEG loaders, preprocessing, decision methods and model classes |
| `scripts/` | Existing analysis dependencies plus the new server downloader, training and evaluation runners |
| `tests/` | Code checks, including causal-model tests |
| `requirements.txt`, `.gitignore` | Minimal Python dependencies and data/secret exclusions |

The new exploratory model is `src/uncued_mi/large_causal_cnn_gru.py`.
Training and score generation are in
`scripts/cache_large_cnn_gru_original_continuous_20261014.py`; evaluation is
in `scripts/run_large_cnn_gru_original_continuous_20261014.py`.
The comparator is the frozen I6 CNN-GRU. The new model has **not** been trained
or shown to improve performance. Lee2019 is cue-paced; neither model's score
establishes genuine self-initiated MI detection.

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
python -m unittest discover -s tests -p test_large_causal_cnn_gru.py -v
python scripts/prepare_lee2019_mil_raw_cache.py --subjects $(seq 1 54) --sessions 1 2
python scripts/prepare_tier0_cache.py --all --sessions 1 2
python scripts/preflight_e4_full_trial_blocks_20261009.py
python scripts/preflight_e4b_task_idle_20261010.py
python scripts/cache_large_cnn_gru_original_continuous_20261014.py --subjects 1 --sessions 1 --device cuda
```

After inspecting the one-case cache and GPU memory use, run the full cohort:

```bash
python scripts/cache_large_cnn_gru_original_continuous_20261014.py --device cuda
python scripts/run_large_cnn_gru_original_continuous_20261014.py --limit-cases 1
python scripts/run_large_cnn_gru_original_continuous_20261014.py
```

The downloader uses a URL pattern previously recorded for the public
[Lee2019 GigaDB dataset](https://gigadb.org/dataset/100542). The official
dataset lists the MI files, but server access to the direct file endpoint
must be checked with the one-file trial. Full MI data are 108 MAT files and
about 60.4 GiB locally. All downloads and outputs stay under ignored `data/`.

The large-model runner reuses the old **simulated-source training** recipe and
scores the original-continuous Lee target. It does not implement original-
continuous source retraining or Dreyer/WBCIC pretraining. Comparisons must
retain subject-level LOSO isolation and report both sessions at matched
achieved false-positive rates as exploratory Lee analyses.
