"""Check A0 normalized source-window identities and target-rest reference."""

import json
from pathlib import Path

import numpy as np


root = Path("data/interim/lee2019_revalidation_norm_cov")
paths = sorted(root.glob("session*/sub-*.npz"))
if not paths:
    raise SystemExit(f"no covariance files under {root}")
max_reference_diff = 0.0
command_count_difference_files = 0
maximum_command_count_difference = 0
idle_count_difference_files = 0
maximum_idle_count_difference = 0
for path in paths:
    old_path = Path("data/interim/lee2019_four_rest") / path.parent.name / path.name
    rest_path = Path("data/interim/lee2019_four_rest_raw") / path.parent.name / path.name
    with np.load(path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
        for name in ("train_command", "train_idle_iti"):
            old_trials = old[f"{name}_trial"]
            old_trials = old_trials[old_trials < 50]
            current_trials = new[f"{name}_trial"]
            assert np.all((current_trials >= 0) & (current_trials < 50)), (path, name)
            assert np.all(np.diff(current_trials) >= 0), (path, name)
            difference = int(np.max(np.abs(
                np.bincount(current_trials, minlength=50)
                - np.bincount(old_trials, minlength=50))))
            if name == "train_command":
                assert np.array_equal(np.unique(current_trials), np.arange(50)), path
                command_count_difference_files += int(difference > 0)
                maximum_command_count_difference = max(maximum_command_count_difference, difference)
            else:
                idle_count_difference_files += int(difference > 0)
                maximum_idle_count_difference = max(maximum_idle_count_difference, difference)
            assert len(new[f"{name}_covariance"]) == len(current_trials), (path, name)
            assert np.all(np.isfinite(new[f"{name}_covariance"])), (path, name)
        reference = new["pre_train_covariance"]
    with np.load(rest_path, allow_pickle=False) as raw:
        pre = raw["pre_train"]
    ends = np.arange(750, len(pre) + 1, 25)
    assert len(reference) == len(ends), path
    direct = np.asarray([np.cov(pre[end-250:end], rowvar=False) for end in ends])
    diff = float(np.max(np.abs(reference - direct)))
    max_reference_diff = max(max_reference_diff, diff)
    assert np.allclose(reference, direct, rtol=1e-5, atol=1e-6), (path, diff)
print(json.dumps({"files": len(paths), "max_reference_cov_abs_diff": max_reference_diff,
                  "source_trial_ids": "0-49 only", "window_source": "normalized mil_raw",
                  "command_rule": "CNN-GRU source end-time rule, stop exclusive",
                  "old_command_count_difference_files": command_count_difference_files,
                  "max_old_command_count_difference_per_trial": maximum_command_count_difference,
                  "old_idle_count_difference_files": idle_count_difference_files,
                  "max_old_idle_count_difference_per_trial": maximum_idle_count_difference}, indent=2))
