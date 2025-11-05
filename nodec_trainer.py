import torch
import torch.nn as nn
from typing import Callable, Dict

from cost import CostFunction
from simulator import Simulator


class NODECTrainer:
    """
    Trainer for NODEC baseline (no stability guarantees).

    Uses standard cost minimization without Lyapunov structure:
        - StageCost: ∫(x-z)ᵀQ(x-z)dt
        - TerminalCost: (x(T)-z)ᵀP(x(T)-z)
        - MixedCost: stage + terminal
    """

    def __init__(
        self,
        policy: nn.Module,
        dynamics_fn: Callable,
        cost_fn: CostFunction,
        simulator: Simulator,
        optimizer: torch.optim.Optimizer,
        dataset_sampler: Callable[[], tuple],
    ):
        """
        Args:
            policy: Neural policy π_θ(x) -> u
            dynamics_fn: System dynamics f(x, u, t) -> xdot
            cost_fn: Cost function (StageCost, TerminalCost, or MixedCost)
            simulator: Simulator instance
            optimizer: PyTorch optimizer for policy parameters
            dataset_sampler: Function () -> (x0, x_target) that samples pairs
        """
        self.policy = policy
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.simulator = simulator
        self.optimizer = optimizer
        self.sample_pair = dataset_sampler

    def train_step(self, time_grid: torch.Tensor, batch_size: int = 1) -> Dict[str, float]:
        """
        Single training iteration with batched initial conditions.

        Steps:
            1. Sample batch of B initial conditions from X_0 (with fixed target z)
            2. For each IC, generate closed-loop trajectory
            3. Compute average cost over batch
            4. Backpropagate and update policy

        Args:
            time_grid: [T] time points for trajectory
            batch_size: Number of initial conditions to sample

        Returns:
            Dictionary of metrics for logging
        """
        batch_loss = 0.0
        batch_metrics = {
            "final_distance": 0.0,
        }

        # Sample batch of (x0, x_target) pairs
        for b in range(batch_size):
            x0, x_target = self.sample_pair()
            # Generate closed-loop trajectory with current policy
            trajectory, controls = self.simulator.rollout(
                x0=x0, policy=self.policy, time_grid=time_grid, adjoint=True
            )

            # Compute cost for this trajectory
            cost_b = self.cost_fn(
                trajectory=trajectory,
                controls=controls,
                x_target=x_target,
                time_grid=time_grid,
            )

            # Accumulate batch cost
            batch_loss += cost_b / batch_size

            # Compute metrics (no grad)
            with torch.no_grad():
                x_final = trajectory[-1]
                final_dist = torch.norm(x_final - x_target).item()
                batch_metrics["final_distance"] += final_dist / batch_size

        # Optimization step on averaged batch cost
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return {
            "loss": batch_loss.item(),
            "final_distance": batch_metrics["final_distance"],
        }

    def train(
        self,
        num_epochs: int,
        time_grid: torch.Tensor,
        batch_size: int = 1,
        log_interval: int = 10,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            num_epochs: Number of training iterations
            time_grid: [T] time points for trajectories
            batch_size: Number of initial conditions per iteration
            log_interval: Print every N epochs
            verbose: Whether to print progress

        Returns:
            history: Dictionary of training metrics over epochs
        """
        history = {
            "loss": [],
            "final_distance": [],
        }

        for epoch in range(num_epochs):
            metrics = self.train_step(time_grid, batch_size=batch_size)

            # Log metrics
            for key, value in metrics.items():
                history[key].append(value)

            # Print progress
            if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch:4d}/{num_epochs}: "
                    f"Loss={metrics['loss']:.4e}, "
                    f"Final Dist={metrics['final_distance']:.4f}"
                )

        return history

    def evaluate(
        self,
        x0: torch.Tensor,
        x_target: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate policy on a single (x0, x_target) pair.

        Args:
            x0: [n_x] initial state
            x_target: [n_x] target state
            time_grid: [T] time points

        Returns:
            Dictionary with trajectory, controls, and metrics
        """
        self.policy.eval()

        with torch.no_grad():
            trajectory, controls = self.simulator.rollout(
                x0=x0, policy=self.policy, time_grid=time_grid, adjoint=False
            )

            x_final = trajectory[-1]
            final_dist = torch.norm(x_final - x_target)

            cost = self.cost_fn(trajectory, controls, x_target, time_grid)

        self.policy.train()

        return {
            "trajectory": trajectory,
            "controls": controls,
            "final_distance": final_dist,
            "cost": cost,
        }
