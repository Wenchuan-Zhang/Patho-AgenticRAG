import numpy as np
import torch
from scipy.stats import skew, kurtosis

# Flag to enable GPU computations
USE_GPU = False


def set_device(use_gpu: bool):
    """Enable or disable GPU mode for basic functions."""
    global USE_GPU
    USE_GPU = use_gpu


class Matrix2D:
    pass


class Vector1D:
    pass


def mean2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        return torch.mean(m_t, dim=1)
    return np.mean(m, axis=1)


def std2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        return torch.std(m_t, dim=1, unbiased=False)
    return np.std(m, axis=1)


def max2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        return torch.max(m_t, dim=1).values
    return np.max(m, axis=1)


def skew2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        mean = torch.mean(m_t, dim=1, keepdim=True)
        dev = m_t - mean
        std = torch.std(m_t, dim=1, unbiased=False, keepdim=True)
        return torch.mean(dev**3, dim=1) / (std.squeeze() ** 3 + 1e-8)
    return skew(m, axis=1)


def kurtosis2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        mean = torch.mean(m_t, dim=1, keepdim=True)
        dev = m_t - mean
        var = torch.var(m_t, dim=1, unbiased=False, keepdim=True)
        return torch.mean(dev**4, dim=1) / (var.squeeze() ** 2 + 1e-8) - 3
    return kurtosis(m, axis=1)


def range2Dto1D(m: np.ndarray) -> Vector1D:
    if USE_GPU:
        m_t = (
            torch.tensor(m, device="cuda")
            if not isinstance(m, torch.Tensor)
            else m.to("cuda")
        )
        return torch.max(m_t, dim=1).values - torch.min(m_t, dim=1).values
    return np.max(m, axis=1) - np.min(m, axis=1)


def mean1Dto1D(a: np.ndarray, b: np.ndarray) -> Vector1D:
    if USE_GPU:
        a_t = (
            torch.tensor(a, device="cuda")
            if not isinstance(a, torch.Tensor)
            else a.to("cuda")
        )
        b_t = (
            torch.tensor(b, device="cuda")
            if not isinstance(b, torch.Tensor)
            else b.to("cuda")
        )
        return torch.mean(torch.stack([a_t, b_t], dim=0), dim=0)
    return np.mean([a, b], axis=0)


def max1Dto1D(a: np.ndarray, b: np.ndarray) -> Vector1D:
    if USE_GPU:
        a_t = (
            torch.tensor(a, device="cuda")
            if not isinstance(a, torch.Tensor)
            else a.to("cuda")
        )
        b_t = (
            torch.tensor(b, device="cuda")
            if not isinstance(b, torch.Tensor)
            else b.to("cuda")
        )
        return torch.maximum(a_t, b_t)
    return np.maximum(a, b)


def min1Dto1D(a: np.ndarray, b: np.ndarray) -> Vector1D:
    if USE_GPU:
        a_t = (
            torch.tensor(a, device="cuda")
            if not isinstance(a, torch.Tensor)
            else a.to("cuda")
        )
        b_t = (
            torch.tensor(b, device="cuda")
            if not isinstance(b, torch.Tensor)
            else b.to("cuda")
        )
        return torch.minimum(a_t, b_t)
    return np.minimum(a, b)


def mul1Dto1D(a: np.ndarray, b: np.ndarray) -> Vector1D:
    if USE_GPU:
        a_t = (
            torch.tensor(a, device="cuda")
            if not isinstance(a, torch.Tensor)
            else a.to("cuda")
        )
        b_t = (
            torch.tensor(b, device="cuda")
            if not isinstance(b, torch.Tensor)
            else b.to("cuda")
        )
        return torch.mul(a_t, b_t)
    return np.multiply(a, b)


def mean1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        return torch.mean(v)
    return float(np.mean(v))


def std1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        return torch.std(v, unbiased=False)
    return float(np.std(v))


def max1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        return torch.max(v)
    return float(np.max(v))


def skew1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        mean = torch.mean(v)
        dev = v - mean
        std = torch.std(v, unbiased=False)
        return torch.mean(dev**3) / (std**3 + 1e-8)
    return float(skew(v))


def kurtosis1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        mean = torch.mean(v)
        dev = v - mean
        var = torch.var(v, unbiased=False)
        return torch.mean(dev**4) / (var**2 + 1e-8) - 3
    return float(kurtosis(v))


def range1DtoScalar(v: Vector1D) -> float:
    if USE_GPU and isinstance(v, torch.Tensor):
        return torch.max(v) - torch.min(v)
    return float(np.max(v) - np.min(v))


def add(a: float, b: float) -> float:
    if USE_GPU and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a + b
    return a + b


def sub(a: float, b: float) -> float:
    if USE_GPU and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a - b
    return a - b


def mul(a: float, b: float) -> float:
    if USE_GPU and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a * b
    return a * b
