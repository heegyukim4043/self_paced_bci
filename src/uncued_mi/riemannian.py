from __future__ import annotations

import numpy as np


def window_covariances(
    signal: np.ndarray, end_samples: np.ndarray, window_samples: int
) -> np.ndarray:
    signal = np.asarray(signal)
    ends = np.asarray(end_samples, dtype=np.int64)
    result = np.empty(
        (len(ends), signal.shape[1], signal.shape[1]), dtype=np.float64
    )
    for index, end in enumerate(ends):
        segment = np.asarray(
            signal[end - window_samples : end], dtype=np.float64
        )
        segment -= segment.mean(axis=0, keepdims=True)
        result[index] = segment.T @ segment / (window_samples - 1)
    return result


def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, max(float(np.max(values)) * 1e-8, 1e-12))
    return (vectors * values**-0.5) @ vectors.T


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite matrix square root."""
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    values = np.maximum(values, max(float(np.max(values)) * 1e-8, 1e-12))
    return (vectors * np.sqrt(values)) @ vectors.T


def matrix_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    values = np.maximum(values, max(float(np.max(values)) * 1e-8, 1e-12))
    return (vectors * np.log(values)) @ vectors.T


def matrix_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    return (vectors * np.exp(values)) @ vectors.T


def riemannian_mean(covariances: np.ndarray, *, max_iter: int = 20,
                    tolerance: float = 1e-7) -> np.ndarray:
    """Affine-invariant (Karcher) mean of SPD covariance matrices."""
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3 or not len(covariances):
        raise ValueError("covariances must be a non-empty (windows, ch, ch) array")
    mean = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        root = matrix_sqrt(mean)
        invroot = inverse_sqrt(mean)
        whitened = invroot @ covariances @ invroot.T
        values, vectors = np.linalg.eigh(whitened)
        values = np.maximum(
            values,
            np.maximum(np.max(values, axis=1, keepdims=True) * 1e-8, 1e-12),
        )
        logs = (vectors * np.log(values)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
        update = np.mean(logs, axis=0)
        mean = root @ matrix_exp(update) @ root.T
        mean = (mean + mean.T) / 2
        if np.linalg.norm(update, ord="fro") <= tolerance:
            break
    return mean


def transport_covariance(covariance: np.ndarray, current_reference: np.ndarray,
                         anchor_reference: np.ndarray) -> np.ndarray:
    """Congruence-align a covariance from a current reference to an anchor."""
    transform = matrix_sqrt(anchor_reference) @ inverse_sqrt(current_reference)
    aligned = transform @ np.asarray(covariance, dtype=float) @ transform.T
    return (aligned + aligned.T) / 2


def tangent_features(
    covariances: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    whitening = inverse_sqrt(reference)
    matrices = whitening @ covariances @ whitening.T
    eigenvalues, vectors = np.linalg.eigh(matrices)
    eigenvalues = np.maximum(
        eigenvalues,
        np.maximum(
            np.max(eigenvalues, axis=1, keepdims=True) * 1e-8,
            1e-12,
        ),
    )
    logs = (
        vectors * np.log(eigenvalues)[:, None, :]
    ) @ np.swapaxes(vectors, 1, 2)
    upper = np.triu_indices(logs.shape[1])
    features = logs[:, upper[0], upper[1]]
    features[:, upper[0] != upper[1]] *= np.sqrt(2.0)
    return features
