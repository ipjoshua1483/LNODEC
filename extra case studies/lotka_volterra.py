import torch
import torch.nn as nn
#from policy import Policy

from utils import device
from case_study import Case_Study

c0, c1 = 0.4, 0.2

init_cond = torch.tensor([0.5, 0.7, 0.], dtype = torch.float32).to(device)

time_interval = torch.tensor([0., 12.], dtype = torch.float32).to(device)

time_step = 0.02

def pointwise_obj_cost(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return x[-1, 2]

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    x_dot_0 = x0 - x0 * x1 - c0 * x0 * u
    x_dot_1 = -x1 + x0 * x1 - c1 * x1 * u
    x_dot_2 = (x0 - 1) ** 2 + (x1 - 1) ** 2
    return torch.stack([x_dot_0, x_dot_1, x_dot_2], dim = -1)

# def penalty_example(x: torch.tensor, expr, penalty_weight: float, pos = False) -> torch.tensor:
#     if pos:
#         coeff = -1
#     else:
#         coeff = 1
#     return penalty_weight * nn.relu(coeff * expr)

#define penalty cost for u violation (lower and upper)

#penalty_list = [state_penalty]
penalty_list = None

fish_obj = Case_Study(init_cond,
                        time_interval,
                        time_step,
                        dynamics,
                        pointwise_obj_cost,
                        penalty_list,
                        )