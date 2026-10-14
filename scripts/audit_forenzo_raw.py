"""Audit one original Forenzo2023 MATLAB v7.3 run without MOABB.

The script intentionally reports the raw HDF5 schema before applying any
dataset-specific interpretation. This makes event-type mistakes and
trial-cell concatenation visible during the first-subject smoke audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def matlab_text(value: object, root: h5py.File) -> str:
    """Decode common MATLAB v7.3 string/cell representations."""
    if isinstance(value, h5py.Reference):
        if not value:
            return ""
        return matlab_text(root[value][()], root)
    array = np.asarray(value)
    if array.dtype == h5py.ref_dtype or array.dtype.kind == "O":
        return " ".join(matlab_text(item, root) for item in array.ravel())
    if array.dtype.kind in "ui" and array.size:
        chars = array.ravel()
        if np.all((chars == 0) | ((chars >= 9) & (chars <= 0x10FFFF))):
            return "".join(chr(int(char)) for char in chars if char)
    if array.dtype.kind in "SU":
        return " ".join(str(item) for item in array.ravel())
    return str(array.squeeze())


def describe(group: h5py.Group, root: h5py.File, prefix: str = "") -> None:
    for name, item in group.items():
        path = f"{prefix}/{name}"
        if isinstance(item, h5py.Group):
            print(f"GROUP   {path}")
            describe(item, root, path)
        else:
            print(
                f"DATASET {path} shape={item.shape} dtype={item.dtype} "
                f"matlab_class={item.attrs.get('MATLAB_class', b'')!r}"
            )
            if item.size <= 100 and item.dtype.kind not in "f":
                try:
                    print(f"        decoded={matlab_text(item[()], root)[:500]}")
                except Exception as error:  # schema audit should continue
                    print(f"        decode_error={error!r}")


def dereference_cell(dataset: h5py.Dataset, root: h5py.File) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for reference in dataset[()].ravel():
        values.append(np.asarray(root[reference][()]).squeeze())
    return values


def find_datasets(root: h5py.File, terms: tuple[str, ...]) -> list[str]:
    matches: list[str] = []

    def visitor(name: str, item: object) -> None:
        if isinstance(item, h5py.Dataset) and any(
            term in name.lower() for term in terms
        ):
            matches.append(name)

    root.visititems(visitor)
    return matches


def summarize_candidate(path: str, root: h5py.File) -> None:
    dataset = root[path]
    print(f"\nCANDIDATE {path}")
    if dataset.dtype == h5py.ref_dtype or dataset.dtype.kind == "O":
        cells = dereference_cell(dataset, root)
        print(f"  cells={len(cells)} shapes={[cell.shape for cell in cells[:10]]}")
        texts = [matlab_text(cell, root) for cell in cells]
        if all(len(text) < 100 for text in texts):
            print(f"  values={texts[:30]}")
        return
    values = np.asarray(dataset[()]).squeeze()
    print(f"  shape={values.shape} dtype={values.dtype}")
    if values.size <= 100:
        print(f"  decoded={matlab_text(values, root)[:1000]}")
    elif np.issubdtype(values.dtype, np.number):
        finite = values[np.isfinite(values)]
        if finite.size:
            print(
                "  numeric="
                f"min={finite.min():.6g} max={finite.max():.6g} "
                f"median={np.median(finite):.6g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("--schema", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.mat_file, "r") as root:
        print(f"FILE {args.mat_file.resolve()}")
        print(f"TOP_LEVEL {list(root.keys())}")
        if args.schema:
            describe(root, root)

        terms = (
            "event",
            "type",
            "latency",
            "duration",
            "trial",
            "sample",
            "srate",
            "rate",
            "chan",
            "label",
            "cursor",
            "target",
            "position",
            "time",
            "eeg",
            "data",
        )
        candidates = find_datasets(root, terms)
        print(f"\nCANDIDATE_PATHS ({len(candidates)})")
        print("\n".join(candidates))
        for path in candidates:
            summarize_candidate(path, root)

        leaf_names = Counter(path.rsplit("/", 1)[-1].lower() for path in candidates)
        duplicates = {name: count for name, count in leaf_names.items() if count > 1}
        if duplicates:
            print(f"\nDUPLICATE_CANDIDATE_LEAVES {duplicates}")


if __name__ == "__main__":
    main()
