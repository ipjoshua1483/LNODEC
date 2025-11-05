from typing import List, Sequence, Optional, Union, Callable

import torch
import torch.nn as nn


class NeuralPolicy(nn.Module):
    """
    MLP policy with elementwise bounds on output via sigmoid/tanh.
    Supports scalar or vector bounds per control dimension.
    Optionally normalizes input states before processing.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        u_min: Union[torch.Tensor, float] = -1.0,
        u_max: Union[torch.Tensor, float] = 1.0,
        bound: str = 'sigmoid',
        normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [32, 32])
        self.bound = bound
        self.normalize_fn = normalize_fn

        dims = [state_dim] + hidden_dims + [control_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        # Bounds as buffers for device dtype management
        u_min_t = u_min if isinstance(u_min, torch.Tensor) else torch.tensor(u_min, dtype=torch.float32)
        u_max_t = u_max if isinstance(u_max, torch.Tensor) else torch.tensor(u_max, dtype=torch.float32)
        self.register_buffer('u_min', u_min_t)
        self.register_buffer('u_max', u_max_t)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=5 / 3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization if provided
        if self.normalize_fn is not None:
            x = self.normalize_fn(x)

        u_raw = self.net(x)

        if self.bound == 'sigmoid':
            u01 = torch.sigmoid(u_raw)
        elif self.bound == 'tanh':
            u01 = 0.5 * (torch.tanh(u_raw) + 1.0)
        else:
            raise ValueError("bound must be 'sigmoid' or 'tanh'")

        # Broadcast bounds
        u_min = self.u_min
        u_max = self.u_max
        if u_min.ndim == 0:
            u_min = u_min.view(1)
        if u_max.ndim == 0:
            u_max = u_max.view(1)

        # Ensure shapes broadcast with [B, m_u]
        u = u_min + (u_max - u_min) * u01
        return u
