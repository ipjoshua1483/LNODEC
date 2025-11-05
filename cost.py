import torch
from abc import ABC, abstractmethod


class CostFunction(ABC):
    """Abstract base class for cost functions used in NODEC baselines"""

    @abstractmethod
    def __call__(self, trajectory: torch.Tensor, controls: torch.Tensor,
                 x_target: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute cost for a trajectory.

        Args:
            trajectory: [T, n_x] state trajectory
            controls: [T, n_u] control inputs
            x_target: [n_x] target state
            time_grid: [T] time points

        Returns:
            Scalar cost value
        """
        pass


class StageCost(CostFunction):
    """
    Stage cost (running cost): ∫₀ᵀ ℓ(x(t), u(t)) dt

    Where ℓ(x,u) = (x-z)ᵀQ(x-z) + uᵀRu

    Approximated as: Σᵢ ℓ(x(tᵢ), u(tᵢ))

    Note: Constant Δt cancels in optimization, so we just sum.
    """

    def __init__(self, Q: torch.Tensor, R: torch.Tensor = None):
        """
        Args:
            Q: State cost matrix [n_x, n_x] or scalar (for Q = qI)
            R: Control cost matrix [n_u, n_u] or scalar (optional, for control regularization)
        """
        self.Q = Q
        self.R = R

    def __call__(self, trajectory: torch.Tensor, controls: torch.Tensor,
                 x_target: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute Σ [(x-z)ᵀQ(x-z) + uᵀRu]

        Args:
            trajectory: [T, n_x] state trajectory
            controls: [T, n_u] control inputs
            x_target: [n_x] target state
            time_grid: [T] time points (unused, for API consistency)

        Returns:
            Scalar sum of stage costs
        """
        T = len(trajectory)
        stage_costs = []

        for i in range(min(T, len(controls))):
            x = trajectory[i]
            u = controls[i]

            # State cost: (x-z)ᵀQ(x-z)
            delta = x - x_target
            if self.Q.ndim == 0:
                state_cost = self.Q * (delta ** 2).sum()
            else:
                state_cost = delta @ self.Q @ delta

            # Control cost: uᵀRu (optional)
            if self.R is not None:
                if self.R.ndim == 0:
                    control_cost = self.R * (u ** 2).sum()
                else:
                    control_cost = u @ self.R @ u
                stage_costs.append(state_cost + control_cost)
            else:
                stage_costs.append(state_cost)

        costs_tensor = torch.stack(stage_costs)

        # Just sum (constant dt doesn't affect optimization)
        return costs_tensor.sum()


class TerminalCost(CostFunction):
    """
    Terminal cost: (x(T)-z)ᵀP_φ(x(T)-z)

    Only penalizes deviation at final time T.
    Common in trajectory optimization and MPC.
    """

    def __init__(self, P_phi: torch.Tensor):
        """
        Args:
            P_phi: Terminal cost matrix [n_x, n_x] or scalar
        """
        self.P_phi = P_phi

    def __call__(self, trajectory: torch.Tensor, controls: torch.Tensor,
                 x_target: torch.Tensor, time_grid: torch.Tensor = None) -> torch.Tensor:
        """
        Compute (x(T)-z)ᵀP_φ(x(T)-z)

        Args:
            trajectory: [T, n_x] state trajectory
            controls: [T, n_u] control inputs (unused)
            x_target: [n_x] target state
            time_grid: [T] time points (unused)

        Returns:
            Scalar terminal cost
        """
        x_final = trajectory[-1]
        delta = x_final - x_target

        if self.P_phi.ndim == 0:
            return self.P_phi * (delta ** 2).sum()
        else:
            return delta @ self.P_phi @ delta


class MixedCost(CostFunction):
    """
    Mixed cost: ∫₀ᵀ ℓ(x,u) dt + φ(x(T))

    Combines stage cost and terminal cost.
    Allows weighting the relative importance of trajectory cost vs final state.
    """

    def __init__(self, stage_cost: StageCost, terminal_cost: TerminalCost,
                 stage_weight: float = 1.0, terminal_weight: float = 1.0):
        """
        Args:
            stage_cost: StageCost instance
            terminal_cost: TerminalCost instance
            stage_weight: Weight for stage cost (default: 1.0)
            terminal_weight: Weight for terminal cost (default: 1.0)
        """
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.stage_weight = stage_weight
        self.terminal_weight = terminal_weight

    def __call__(self, trajectory: torch.Tensor, controls: torch.Tensor,
                 x_target: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted stage + terminal cost

        Args:
            trajectory: [T, n_x] state trajectory
            controls: [T, n_u] control inputs
            x_target: [n_x] target state
            time_grid: [T] time points

        Returns:
            Scalar total cost
        """
        stage = self.stage_cost(trajectory, controls, x_target, time_grid)
        terminal = self.terminal_cost(trajectory, controls, x_target)
        return self.stage_weight * stage + self.terminal_weight * terminal
