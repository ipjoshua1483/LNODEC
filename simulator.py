import torch
import torch.nn as nn
from typing import Callable, Tuple
from torchdiffeq import odeint, odeint_adjoint


class Simulator:
    """
    Handles ODE integration for closed-loop dynamics x' = f(x, π(x), t).

    Provides trajectory generation with optional adjoint method for
    memory-efficient backpropagation.
    """

    def __init__(
        self,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ):
        """
        Args:
            dynamics_fn: System dynamics f(x, u, t) -> xdot
            method: ODE solver method
                Fixed-step: 'euler' (fastest), 'midpoint', 'rk4', 'explicit_adams'
                Adaptive: 'dopri5' (recommended), 'dopri8', 'bosh3', 'adaptive_heun'
                Implicit: 'bdf' (for stiff problems)
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
        """
        self.dynamics_fn = dynamics_fn
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def rollout(
        self,
        x0: torch.Tensor,
        policy: nn.Module,
        time_grid: torch.Tensor,
        adjoint: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate closed-loop system from initial state x0.

        Args:
            x0: [n_x] initial state
            policy: Neural policy π(x) -> u
            time_grid: [T] time points for evaluation
            adjoint: If True, use adjoint method for backprop

        Returns:
            trajectory: [T, n_x] state trajectory
            controls: [T, n_u] control inputs at each state
        """

        # Wrapper class for adjoint method (needs nn.Module)
        class ClosedLoopDynamics(nn.Module):
            def __init__(self, dynamics_fn, policy):
                super().__init__()
                self.dynamics_fn = dynamics_fn
                self.policy = policy

            def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                """xdot = f(x, π(x), t)"""
                u = self.policy(x)
                return self.dynamics_fn(x, u, t)

        closed_loop_dynamics = ClosedLoopDynamics(self.dynamics_fn, policy)

        # Choose solver
        ode_solver = odeint_adjoint if adjoint else odeint

        # Integrate
        trajectory = ode_solver(
            closed_loop_dynamics,
            x0,
            time_grid,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Compute controls at each state (KEEP GRADIENTS for training!)
        controls = torch.stack([policy(x) for x in trajectory])

        return trajectory, controls

    def rollout_open_loop(
        self,
        x0: torch.Tensor,
        controls: torch.Tensor,
        time_grid: torch.Tensor,
        adjoint: bool = False,
    ) -> torch.Tensor:
        """
        Integrate with pre-specified open-loop controls (for evaluation).

        Args:
            x0: [n_x] initial state
            controls: [T, n_u] pre-specified control sequence
            time_grid: [T] time points
            adjoint: Whether to use adjoint method

        Returns:
            trajectory: [T, n_x] state trajectory
        """

        # Define open-loop dynamics
        def open_loop_dynamics(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Find nearest time index
            idx = (torch.abs(time_grid - t)).argmin()
            u = controls[idx]
            return self.dynamics_fn(x, u, t)

        ode_solver = odeint_adjoint if adjoint else odeint

        trajectory = ode_solver(
            open_loop_dynamics,
            x0,
            time_grid,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        return trajectory
