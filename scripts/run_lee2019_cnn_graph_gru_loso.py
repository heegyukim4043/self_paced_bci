"""Exact-stream LOSO runner for parameter-near CNN-GCN/G-KAN-GRU ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.graph_kan import CausalCNNGCNGRU, CausalCNNGKANGRU
from scripts.run_lee2019_cnn_gru import CausalCNNGRU
import scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 as base


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spatial-head", choices=("gcn", "gkan"), required=True)
    selected, remaining = parser.parse_known_args()
    model_class = CausalCNNGCNGRU if selected.spatial_head == "gcn" else CausalCNNGKANGRU
    counts = {
        "cnn_gru": sum(parameter.numel() for parameter in CausalCNNGRU().parameters()),
        selected.spatial_head: sum(parameter.numel() for parameter in model_class().parameters()),
    }
    print(f"parameter counts: {counts}", flush=True)
    base.CausalCNNGRU = model_class
    base.MODEL_PROTOCOL_NAME = (
        "causal CNN-GCN-GRU spatial ablation, source-pooled LOSO"
        if selected.spatial_head == "gcn"
        else "causal CNN-Fourier-GKAN-GRU spatial ablation, source-pooled LOSO"
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
