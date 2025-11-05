from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class LyapunovFunction(nn.Module, ABC):
    """Abstract base class for Lyapunov functions"""
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def potential(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute V(x) relative to x_target.

        Args:
            x: Current state [n_x] or [B, n_x]
            x_target: Target state [n_x] or [B, n_x]

        Returns:
            V(x): Scalar or [B] potential value
        """
        pass

    @abstractmethod
    def gradient(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂V/∂x at current state.

        Args:
            x: Current state [n_x] or [B, n_x]
            x_target: Target state [n_x] or [B, n_x]

        Returns:
            grad: [n_x] or [B, n_x] gradient vector
        """
        pass


class QuadraticLyapunov(LyapunovFunction):
    """
    Quadratic Lyapunov function: V(x) = (x - x_target)ᵀ P (x - x_target)

    Args:
        P: Positive definite matrix [n_x, n_x] or scalar (for P = pI)
    """

    def __init__(self, P: torch.Tensor):
        super().__init__()
        
        if P.ndim == 0:
            # Scalar: treat as P = pI (will broadcast)
            self.register_buffer('P', P)
        elif P.ndim == 2:
            # Matrix: check symmetry and register as buffer
            if not torch.allclose(P, P.T):
                raise ValueError("P must be symmetric")
            self.register_buffer('P', P)
        else:
            raise ValueError("P must be scalar or 2D matrix")

    def potential(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """V(x) = (x - x_target)ᵀ P (x - x_target)"""
        delta = x - x_target

        if self.P.ndim == 0:
            # Scalar case: P * ||delta||²
            return self.P * (delta ** 2).sum(dim=-1)
        else:
            # Matrix case: delta^T @ P @ delta
            if delta.ndim == 1:
                return delta @ self.P @ delta
            else:
                # Batched: [B, n_x]
                return torch.einsum('bi,ij,bj->b', delta, self.P, delta)

    def gradient(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """∂V/∂x = 2P(x - x_target)"""
        delta = x - x_target

        if self.P.ndim == 0:
            # Scalar case: 2P * delta
            return 2 * self.P * delta
        else:
            # Matrix case: 2P @ delta
            if delta.ndim == 1:
                return 2 * self.P @ delta
            else:
                # Batched: [B, n_x] - more efficient with einsum
                return 2 * torch.einsum('ij,bj->bi', self.P, delta)