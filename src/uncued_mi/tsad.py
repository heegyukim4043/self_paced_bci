from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from torch import nn


class RobustMahalanobis:
    def fit(self, values: np.ndarray) -> "RobustMahalanobis":
        self.center_ = np.median(values, axis=0)
        centered = values - self.center_
        scale = np.median(np.abs(centered), axis=0) * 1.4826
        self.scale_ = np.where(scale > 1e-6, scale, 1.0)
        standardized = np.clip(centered / self.scale_, -8.0, 8.0)
        self.precision_ = LedoitWolf().fit(standardized).precision_
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        centered = (values - self.center_) / self.scale_
        return np.einsum("bi,ij,bj->b", centered, self.precision_, centered)


class IsolationForestDetector:
    def __init__(self, seed: int):
        self.model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )

    def fit(self, values: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(values)
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(values)


class CausalTCNPredictor(nn.Module):
    def __init__(self, features: int, hidden: int = 32):
        super().__init__()
        self.input = nn.Conv1d(features, hidden, 1)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=0),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, 2, padding=0),
            nn.ELU(),
        )
        self.output = nn.Linear(hidden, features)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must be (batch, steps, features)")
        latent = self.temporal(self.input(history.transpose(1, 2)))
        return self.output(latent[:, :, -1])


def make_contexts(sequence: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    if len(sequence) <= context:
        return (
            np.empty((0, context, sequence.shape[1]), dtype=np.float32),
            np.empty((0, sequence.shape[1]), dtype=np.float32),
        )
    x = np.stack([sequence[index - context : index] for index in range(context, len(sequence))])
    y = sequence[context:]
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def fit_tcn_predictor(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int = 25,
) -> CausalTCNPredictor:
    torch.manual_seed(seed)
    model = CausalTCNPredictor(contexts.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, y = torch.from_numpy(contexts), torch.from_numpy(targets)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def tcn_prediction_scores(
    model: CausalTCNPredictor,
    values: np.ndarray,
    context: int,
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    x, y = make_contexts(values, context)
    if len(x):
        predictions = []
        for start in range(0, len(x), 256):
            predictions.append(model(torch.from_numpy(x[start : start + 256])).numpy())
        prediction = np.concatenate(predictions)
        output[context:] = np.mean(np.square(prediction - y), axis=1)
    return output


def zscore_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = float(np.median(reference))
    scale = float(np.median(np.abs(reference - center)) * 1.4826)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(reference))
    return (values - center) / max(scale, 1e-6)

