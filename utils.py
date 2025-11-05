import torch
import numpy as np
from typing import List


def trapezoid(values: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Trapezoidal rule for numerical integration over uniform grid.

    Formula: ∫f(t)dt ≈ dt/2 * [f(0) + 2f(1) + 2f(2) + ... + 2f(n-1) + f(n)]

    Args:
        values: [T] tensor of function evaluations at grid points
        dt: time step (assumed uniform)

    Returns:
        Scalar integral approximation
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 points for trapezoidal rule")

    return dt * (0.5 * values[0] + values[1:-1].sum() + 0.5 * values[-1])


def rectangle(values: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Rectangle/Euler rule for numerical integration over uniform grid.

    Formula: ∫f(t)dt ≈ dt * Σf(i)

    Args:
        values: [T] tensor of function evaluations at grid points
        dt: time step (assumed uniform)

    Returns:
        Scalar integral approximation
    """
    return values.mean() * dt * len(values)


def simpson(values: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Simpson's rule for numerical integration over uniform grid.

    Formula: ∫f(t)dt ≈ (dt/3) * [f(0) + 4f(1) + 2f(2) + 4f(3) + ... + 4f(n-1) + f(n)]

    Requires odd number of points (even number of intervals).
    If even number of points provided, uses trapezoid for last interval.

    Args:
        values: [T] tensor of function evaluations at grid points
        dt: time step (assumed uniform)

    Returns:
        Scalar integral approximation
    """
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 points for Simpson's rule")

    # If odd number of points (even intervals), use pure Simpson
    if n % 2 == 1:
        result = values[0] + values[-1]
        result += 4 * values[1:-1:2].sum()  # Odd indices (4x weight)
        result += 2 * values[2:-2:2].sum()  # Even indices (2x weight)
        return (dt / 3) * result

    # If even number of points, use Simpson for first n-1, trapezoid for last
    else:
        # Simpson on first n-1 points
        result = values[0] + values[-2]
        result += 4 * values[1:-2:2].sum()
        result += 2 * values[2:-3:2].sum()
        simpson_part = (dt / 3) * result

        # Trapezoid for last interval
        trap_part = (dt / 2) * (values[-2] + values[-1])

        return simpson_part + trap_part


def torch_to_numpy(tensor_list: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert list of tensors to list of numpy arrays"""
    return [t.cpu().detach().numpy() for t in tensor_list]
