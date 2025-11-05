import torch
import torch.nn as nn
from typing import Callable, Dict, Optional
import copy

from loss import LyapunovLoss
from simulator import Simulator


class LNODECTrainer:
    """
    Trainer for L-NODEC policy learning.

    Implements Algorithm 1 from the paper with Option A (single pair per iteration).
    Samples (x0, x_target) pairs and trains policy to achieve exponential stability.
    """

    def __init__(
        self,
        policy: nn.Module,
        dynamics_fn: Callable,
        lyapunov_loss: LyapunovLoss,
        simulator: Simulator,
        optimizer: torch.optim.Optimizer,
        dataset_sampler: Callable[[], tuple],
    ):
        """
        Args:
            policy: Neural policy π_θ(x) -> u
            dynamics_fn: System dynamics f(x, u, t) -> xdot
            lyapunov_loss: LyapunovLoss instance
            simulator: Simulator instance
            optimizer: PyTorch optimizer for policy parameters
            dataset_sampler: Function () -> (x0, x_target) that samples pairs
        """
        self.policy = policy
        self.dynamics_fn = dynamics_fn
        self.loss_fn = lyapunov_loss
        self.simulator = simulator
        self.optimizer = optimizer
        self.sample_pair = dataset_sampler

    def train_step(self, time_grid: torch.Tensor, batch_size: int = 1) -> Dict[str, float]:
        """
        Single training iteration with batched initial conditions (Algorithm 1).

        Steps:
            1. Sample batch of B initial conditions from X_0 (with fixed target z)
            2. For each IC, generate closed-loop trajectory
            3. Compute average Lyapunov loss over batch
            4. Backpropagate and update policy

        Args:
            time_grid: [T] time points for trajectory
            batch_size: Number of initial conditions to sample (B in Algorithm 1)

        Returns:
            Dictionary of metrics for logging
        """
        batch_loss = 0.0
        batch_metrics = {
            "final_distance": 0.0,
            "V_initial": 0.0,
            "V_final": 0.0,
            "V_decay": 0.0,
        }

        # Sample batch of (x0, x_target) pairs
        # Note: For set-to-point stabilization, x_target should be FIXED
        for b in range(batch_size + 1):
            if b == 0:
                x0, x_target = self.sample_pair.nominal()
            else:
                x0, x_target = self.sample_pair()
            # Generate closed-loop trajectory with current policy
            trajectory, controls = self.simulator.rollout(
                x0=x0, policy=self.policy, time_grid=time_grid, adjoint=True
            )

            # Compute integrated Lyapunov loss for this trajectory
            # Exclude last point to match controls length
            loss_b = self.loss_fn.integrated_loss(
                trajectory=trajectory[:-1],
                x_target=x_target,
                controls=controls[:-1],
                dynamics_fn=self.dynamics_fn,
                time_grid=time_grid[:-1],
            )

            # Accumulate batch loss (will average later)
            batch_loss += loss_b / batch_size

            # Compute metrics (no grad)
            with torch.no_grad():
                x_final = trajectory[-1]
                final_dist = torch.norm(x_final - x_target).item()

                V_initial = self.loss_fn.V.potential(x0, x_target).item()
                V_final = self.loss_fn.V.potential(x_final, x_target).item()

                V_decay = V_final / (V_initial + 1e-8)  # Avoid division by zero

                # Accumulate metrics
                batch_metrics["final_distance"] += final_dist / batch_size
                batch_metrics["V_initial"] += V_initial / batch_size
                batch_metrics["V_final"] += V_final / batch_size
                batch_metrics["V_decay"] += V_decay / batch_size

        # Optimization step on averaged batch loss
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return {
            "loss": batch_loss.item(),
            **batch_metrics,
        }

    def train(
        self,
        num_epochs: int,
        time_grid: torch.Tensor,
        batch_size: int = 1,
        log_interval: int = 10,
        verbose: bool = True,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            num_epochs: Number of training iterations
            time_grid: [T] time points for trajectories
            batch_size: Number of initial conditions per iteration (B in Algorithm 1)
            log_interval: Print every N epochs
            verbose: Whether to print progress
            save_best: If True, restore policy parameters with lowest loss at end

        Returns:
            history: Dictionary of training metrics over epochs
        """
        history = {
            "loss": [],
            "final_distance": [],
            "V_initial": [],
            "V_final": [],
            "V_decay": [],
        }

        # Best model tracking
        best_loss = float('inf')
        best_state_dict = None
        best_epoch = 0

        for epoch in range(num_epochs):
            metrics = self.train_step(time_grid, batch_size=batch_size)

            # Log metrics
            for key, value in metrics.items():
                if key in history:
                    history[key].append(value)
                else:
                    history[key] = [value]

            # Save best model (based on total loss)
            if save_best and metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_epoch = epoch
                # Deep copy policy state dict
                best_state_dict = copy.deepcopy(self.policy.state_dict())

            # Print progress
            if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch:4d}/{num_epochs}: "
                    f"Loss={metrics['loss']:.4e}, "
                    f"Final Dist={metrics['final_distance']:.4f}, "
                    f"V_decay={metrics['V_decay']:.4f}"
                )

        # Restore best model
        if save_best and best_state_dict is not None:
            self.policy.load_state_dict(best_state_dict)
            if verbose:
                print(f"\n[Best Model] Restored policy from epoch {best_epoch} "
                      f"(loss={best_loss:.4e})\n")

        return history

    def evaluate(
        self,
        x0: torch.Tensor,
        x_target: torch.Tensor,
        time_grid: torch.Tensor,
        compute_stability_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate policy on a single (x0, x_target) pair.

        Args:
            x0: [n_x] initial state
            x_target: [n_x] target state
            time_grid: [T] time points
            compute_stability_metrics: If True, compute V(x(t)) and stability violations along trajectory

        Returns:
            Dictionary with trajectory, controls, and metrics
        """
        self.policy.eval()

        # Generate trajectory (no grad for efficiency)
        with torch.no_grad():
            trajectory, controls = self.simulator.rollout(
                x0=x0, policy=self.policy, time_grid=time_grid, adjoint=False
            )

            x_final = trajectory[-1]
            final_dist = torch.norm(x_final - x_target)

            V_initial = self.loss_fn.V.potential(x0, x_target)
            V_final = self.loss_fn.V.potential(x_final, x_target)

        results = {
            "trajectory": trajectory,
            "controls": controls,
            "final_distance": final_dist,
            "V_initial": V_initial,
            "V_final": V_final,
        }

        # Compute stability metrics along entire trajectory
        # (requires gradients for dV/dx computation)
        if compute_stability_metrics:
            T = len(trajectory)
            V_trajectory = torch.zeros(T)
            V_decay_trajectory = torch.zeros(T)
            stability_violation = torch.zeros(T - 1)  # Same size as pointwise_losses
            pointwise_losses = torch.zeros(T - 1)

            for i in range(T):
                with torch.no_grad():
                    V_trajectory[i] = self.loss_fn.V.potential(trajectory[i], x_target)
                    # V_decay at time t: V(x(t)) / (V(x_0) * e^(-κt))
                    # Should be ≤ 1 if exponential stability holds
                    V_decay_trajectory[i] = V_trajectory[i] / (
                        V_initial * torch.exp(-self.loss_fn.kappa * time_grid[i]) + 1e-10
                    )

                # Compute pointwise Lyapunov loss (needs gradients for dV/dx)
                if i < T - 1:
                    # Clone tensors and enable gradients (they were created in no_grad context)
                    x = trajectory[i].detach().clone().requires_grad_(True)
                    u = controls[i].detach().clone()
                    t = time_grid[i]

                    # Compute pointwise loss with gradients enabled
                    pointwise_losses[i] = self.loss_fn.pointwise_loss(
                        x, x_target, self.dynamics_fn, u, t
                    ).detach()
                    stability_violation[i] = pointwise_losses[i].item()

            # Find first time exponential stability is achieved
            # (pointwise loss = 0 means stability holds)
            with torch.no_grad():
                stable_mask = pointwise_losses < 1e-6  # Tolerance for numerical precision
                if stable_mask.any():
                    first_stable_idx = torch.where(stable_mask)[0][0].item()
                    first_stable_time = time_grid[first_stable_idx].item()
                else:
                    first_stable_idx = None
                    first_stable_time = None

            results.update(
                {
                    "V_trajectory": V_trajectory,  # [T] V(x(t)) along trajectory
                    "V_decay_trajectory": V_decay_trajectory,  # [T] V(x(t))/(V(x_0)*e^(-κt))
                    "stability_violation": stability_violation,  # [T] pointwise loss
                    "pointwise_losses": pointwise_losses,  # [T-1] max{0, dV/dt+κV}
                    "first_stable_time": first_stable_time,  # When stability first achieved
                    "first_stable_idx": first_stable_idx,
                }
            )

        self.policy.train()

        return results
