import torch
from typing import Tuple

torch.manual_seed(100)

class UniformBoxSampler:
    """
    Samples (x0, x_target) pairs from uniform distributions over boxes.

    Example:
        x0 ~ Uniform([x0_min, x0_max])
        x_target ~ Uniform([xt_min, xt_max])
    """

    def __init__(
        self,
        x0_min: torch.Tensor,
        x0_max: torch.Tensor,
        xt_min: torch.Tensor,
        xt_max: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            x0_min: [n_x] lower bounds for initial state
            x0_max: [n_x] upper bounds for initial state
            xt_min: [n_x] lower bounds for target state
            xt_max: [n_x] upper bounds for target state
            device: torch device
            dtype: torch dtype
        """
        self.x0_min = x0_min.to(device=device, dtype=dtype)
        self.x0_max = x0_max.to(device=device, dtype=dtype)
        self.xt_min = xt_min.to(device=device, dtype=dtype)
        self.xt_max = xt_max.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one (x0, x_target) pair.

        Returns:
            x0: [n_x] initial state
            x_target: [n_x] target state
        """
        # Sample x0 ~ Uniform([x0_min, x0_max])
        x0 = (
            torch.rand(self.x0_min.shape, device=self.device, dtype=self.dtype)
            * (self.x0_max - self.x0_min)
            + self.x0_min
        )

        # Sample x_target ~ Uniform([xt_min, xt_max])
        x_target = (
            torch.rand(self.xt_min.shape, device=self.device, dtype=self.dtype)
            * (self.xt_max - self.xt_min)
            + self.xt_min
        )

        return x0, x_target


class FixedPairSampler:
    """
    Always returns the same fixed (x0, x_target) pair.

    Useful for debugging or single-trajectory problems.
    """

    def __init__(self, x0: torch.Tensor, x_target: torch.Tensor):
        self.x0 = x0
        self.x_target = x_target

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x0, self.x_target


class SetToPointSampler:
    """
    Sample initial conditions from a set, with FIXED target point.

    This implements the "robust set-to-point stabilization" problem:
    - Multiple initial conditions (robustness)
    - Single fixed target (one Lyapunov function)
    """

    def __init__(
        self,
        x0_min: torch.Tensor,
        x0_max: torch.Tensor,
        x_target: torch.Tensor,  # FIXED target, not sampled!
        # seed: int = None, 
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            x0_min: [n_x] lower bounds for initial state set
            x0_max: [n_x] upper bounds for initial state set
            x_target: [n_x] FIXED target state (not sampled)
            device: torch device
            dtype: torch dtype
        """
        self.x0_min = x0_min.to(device=device, dtype=dtype)
        self.x0_max = x0_max.to(device=device, dtype=dtype)
        self.x_target = x_target.to(device=device, dtype=dtype)

        # self.seed = seed
        self.device = device
        self.dtype = dtype

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial condition from set, return with fixed target.

        Returns:
            x0: [n_x] sampled initial state from X_0
            x_target: [n_x] FIXED target state (always the same!)
        """
        # Sample x0 ~ Uniform(X_0)
        x0 = (
            torch.rand(self.x0_min.shape, device=self.device, dtype=self.dtype)
            * (self.x0_max - self.x0_min)
            + self.x0_min
        )

        # Return sampled IC with FIXED target
        return x0, self.x_target

    def nominal(self):
        """
        Yields nominal initial condition

        Returns:
            x0_mean: [n_x] as specified in __init__
            x_target: [n_x] FIXED target state (always the same!)
        """
        return (self.x0_min + self.x0_max) / 2, self.x_target
    
    def sample(self, n, seed):
        torch.manual_seed(seed)
        x0 = (
            torch.rand((n, *self.x0_min.shape), device=self.device, dtype=self.dtype)
            * (self.x0_max - self.x0_min)
            + self.x0_min
        )
        return x0, self.x_target


class NormalSetToPointSampler:
    """
    Sample initial conditions from Normal distribution, with FIXED target.

    For robust set-to-point stabilization with realistic perturbations:
    - ICs drawn from Normal(x0_mean, x0_std) elementwise
    - Single fixed target (one Lyapunov function)
    """

    def __init__(
        self,
        x0_mean: torch.Tensor,
        x0_std: torch.Tensor,
        x_target: torch.Tensor,  # FIXED target, not sampled!
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            x0_mean: [n_x] mean for each state component
            x0_std: [n_x] std dev for each state component
            x_target: [n_x] FIXED target state (not sampled)
            device: torch device
            dtype: torch dtype
        """
        self.x0_mean = x0_mean.to(device=device, dtype=dtype)
        self.x0_std = x0_std.to(device=device, dtype=dtype)
        self.x_target = x_target.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial condition from Normal distribution.

        Returns:
            x0: [n_x] sampled initial state ~ Normal(mean, std)
            x_target: [n_x] FIXED target state (always the same!)
        """
        # Sample x0 ~ Normal(mean, std) elementwise
        x0 = torch.randn(self.x0_mean.shape, device=self.device, dtype=self.dtype)
        x0 = self.x0_mean + self.x0_std * x0

        # Return sampled IC with FIXED target
        # torch.stack([self.x0_mean, x0])
        return x0, self.x_target
    
    def nominal(self):
        """
        Yields nominal initial condition

        Returns:
            x0_mean: [n_x] as specified in __init__
            x_target: [n_x] FIXED target state (always the same!)
        """
        return self.x0_mean, self.x_target



