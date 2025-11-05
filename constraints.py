import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List


class ConstraintHandler:
    """
    Constraint penalty handler based on paper eq. 373-383.

    Implements: β · [smooth_max{0, g(x,u)}]²

    Uses smooth approximation of max{0, ·} for differentiability:
        smooth_max(0, g) = τ · softplus(g/τ)

    where τ controls sharpness (smaller τ → sharper, closer to ReLU)

    Reference: L-NODEC paper, equation 373-383
    Default β=5.0 from OLDER/double_integrator.py implementation
    """

    def __init__(
        self,
        constraint_fns: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        beta: float = 5.0,
        tau: float = 0.1,
    ):
        """
        Args:
            constraint_fns: List of constraint functions g_i(x, u)
                           Each returns scalar or tensor (≤ 0 when satisfied)
            beta: Penalty weight (β in paper eq. 373-383)
                  Default: 5.0 (from original implementation)
            tau: Temperature for smooth max approximation
                 Smaller → sharper (closer to ReLU)
                 Larger → smoother (further from ReLU)
                 Default: 0.1 (good balance)
        """
        self.constraint_fns = constraint_fns
        self.beta = beta
        self.tau = tau

    def smooth_max_zero(self, g: torch.Tensor) -> torch.Tensor:
        """
        Smooth approximation of max{0, g} using softplus.

        smooth_max(0, g) = τ · softplus(g/τ)

        Properties:
        - When g << 0: ≈ 0 (satisfied constraint)
        - When g >> 0: ≈ g (violated constraint)
        - Smooth everywhere (differentiable)
        - As τ→0: approaches max{0, g}

        Args:
            g: Constraint value

        Returns:
            Smooth approximation of max{0, g}
        """
        return self.tau * F.softplus(g / self.tau)

    def penalty(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute total constraint penalty: β · Σ [smooth_max{0, g_i(x,u)}]²

        Paper eq. 373-383:
            𝒱_c(x,t) = max{0, dV/dt + κV} + β·[max{0, g(x,u)}]²

        Implementation uses smooth approximation for differentiability.

        Args:
            x: State [n_x]
            u: Control [n_u]

        Returns:
            Scalar penalty value (always ≥ 0)
        """
        total_penalty = 0.0

        for g_fn in self.constraint_fns:
            # Evaluate constraint: g(x,u) ≤ 0
            g_val = g_fn(x, u)

            # Compute penalty: β · [smooth_max{0, g}]²
            violation = self.smooth_max_zero(g_val)
            penalty_i = self.beta * (violation ** 2)
            total_penalty = total_penalty + penalty_i

        return total_penalty

    def get_violation_stats(self, x: torch.Tensor, u: torch.Tensor) -> dict:
        """
        Get statistics about constraint violations.

        Args:
            x: State [n_x]
            u: Control [n_u]

        Returns:
            Dictionary with violation info for each constraint
        """
        stats = {}
        for i, g_fn in enumerate(self.constraint_fns):
            g_val = g_fn(x, u)
            g_item = g_val.item() if g_val.numel() == 1 else g_val.mean().item()
            violated = g_item > 0
            stats[f'constraint_{i}'] = {
                'value': g_item,
                'violated': violated,
                'margin': -g_item,  # How far from boundary (positive = safe)
            }
        return stats

    def get_penalty_params(self) -> dict:
        """Get current penalty parameter for logging"""
        return {'beta': self.beta}
