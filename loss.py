import torch
import torch.nn as nn
from typing import Callable, Optional

from lyapunov import LyapunovFunction
from utils import trapezoid, rectangle, simpson


class LyapunovLoss:
    """
    Computes Lyapunov-based losses for L-NODEC training.

    The pointwise Lyapunov loss is:
        𝒱(x, x_target) = max{0, dV/dt + κV} + α·P(x, u)

    where:
        - dV/dt = (∂V/∂x)ᵀ f(x, u, t)
        - P(x, u) is the constraint penalty (optional)
    """

    def __init__(self, lyapunov_fn: LyapunovFunction, kappa: float = 5.0,
                 constraint_handler: Optional['ConstraintHandler'] = None):
        """
        Args:
            lyapunov_fn: Lyapunov function (e.g., QuadraticLyapunov)
            kappa: Exponential stability rate (κ > 0)
            constraint_handler: Optional constraint handler for state/input constraints
        """
        self.V = lyapunov_fn
        self.kappa = kappa
        self.constraint_handler = constraint_handler

    def pointwise_loss(
        self,
        x: torch.Tensor,
        x_target: torch.Tensor,
        dynamics_fn: Callable,
        u: torch.Tensor,
        t: float,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute pointwise Lyapunov loss: max{0, dV/dt + κV} + constraint_penalty

        Args:
            x: Current state [n_x]
            x_target: Target state [n_x]
            dynamics_fn: f(x, u, t) -> xdot
            u: Control input [n_u]
            t: Current time (scalar)
            return_components: If True, return (total, lyapunov, penalty) tuple

        Returns:
            Scalar pointwise loss (or tuple if return_components=True)
        """
        # Ensure x requires grad for autograd
        x_input = x.clone().requires_grad_(True) if not x.requires_grad else x

        # Compute V(x)
        V = self.V.potential(x_input, x_target)

        # Compute ∂V/∂x using autograd (preserves computational graph)
        dVdx = torch.autograd.grad(
            outputs=V,
            inputs=x_input,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute xdot = f(x, u, t)
        xdot = dynamics_fn(x_input, u, t)

        # Compute dV/dt = (∂V/∂x)ᵀ · f(x, u, t)
        dVdt = (dVdx * xdot).sum()

        # Lyapunov condition violation: max{0, dV/dt + κV}
        lyapunov_penalty = torch.relu(dVdt + self.kappa * V)

        # Add constraint penalty if handler provided
        if self.constraint_handler is not None:
            constraint_penalty = self.constraint_handler.penalty(x_input, u)
            total_loss = lyapunov_penalty + constraint_penalty

            if return_components:
                return total_loss, lyapunov_penalty, constraint_penalty
            else:
                return total_loss
        else:
            if return_components:
                return lyapunov_penalty, lyapunov_penalty, torch.tensor(0.0)
            else:
                return lyapunov_penalty

    def integrated_loss(
        self,
        trajectory: torch.Tensor,
        x_target: torch.Tensor,
        controls: torch.Tensor,
        dynamics_fn: Callable,
        time_grid: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute integrated Lyapunov loss: ∫₀¹ max{0, dV/dt + κV} dt

        Approximated as: Σᵢ V(x(tᵢ))

        Note: For optimization, the constant Δt factor cancels out, so we just sum.
        This follows LyaNet's approach (Algorithm 2, Eq. 840).

        Args:
            trajectory: [T, n_x] state trajectory
            x_target: [n_x] target state
            controls: [T, n_u] control inputs over trajectory
            dynamics_fn: f(x, u, t) -> xdot
            time_grid: [T] time points
            return_components: If True, return (total, lyapunov, penalty) tuple

        Returns:
            Scalar sum of pointwise losses (or tuple if return_components=True)
        """
        T = len(trajectory)
        pointwise_losses = []
        lyapunov_losses = []
        penalty_losses = []

        for i in range(T):
            x = trajectory[i]
            u = controls[i]
            t = time_grid[i]

            if return_components:
                total_i, lyap_i, pen_i = self.pointwise_loss(
                    x, x_target, dynamics_fn, u, t, return_components=True
                )
                pointwise_losses.append(total_i)
                lyapunov_losses.append(lyap_i)
                penalty_losses.append(pen_i)
            else:
                loss_i = self.pointwise_loss(x, x_target, dynamics_fn, u, t)
                pointwise_losses.append(loss_i)

        losses_tensor = torch.stack(pointwise_losses)
        total_loss = losses_tensor.sum()

        if return_components:
            lyapunov_tensor = torch.stack(lyapunov_losses)
            penalty_tensor = torch.stack(penalty_losses)
            return total_loss, lyapunov_tensor.sum(), penalty_tensor.sum()
        else:
            return total_loss


    def get_constraint_stats(self, x: torch.Tensor, u: torch.Tensor) -> dict:
        """
        Get constraint violation statistics.

        Args:
            x: State
            u: Control

        Returns:
            Dictionary with violation info (empty if no constraints)
        """
        if self.constraint_handler is not None:
            return self.constraint_handler.get_violation_stats(x, u)
        return {}
