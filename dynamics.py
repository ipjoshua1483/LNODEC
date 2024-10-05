import torch
import torch.nn as nn
from typing import List, Callable
from utils import device
from torchdiffeq import odeint_adjoint as odeint

class Dynamics(nn.Module):
    def __init__(
        self, 
        dynamics: Callable[[torch.tensor], torch.tensor]
    ):
        super(Dynamics, self).__init__()
        self.dynamics = dynamics

    def forward(self, *args):
        return self.dynamics(*args)

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    if len(x.shape) != 1:
        x1, x2 = x[..., 0], x[..., 1]
    else:
        x1, x2 = x[0], x[1]
    x_dot_1 = x2
    x_dot_2 = u
    return torch.stack([x_dot_1, x_dot_2], dim = -1)

double_integrator_test = Dynamics(dynamics)
