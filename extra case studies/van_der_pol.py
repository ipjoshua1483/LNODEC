import torch
import torch.nn as nn
#from policy import Policy

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([5., 5.], dtype = torch.float32).to(device)

time_interval = torch.tensor([0., 20.], dtype = torch.float32).to(device)

time_step = 0.02

def pointwise_obj_cost(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return x[..., 0] ** 2 + x[..., 1] ** 2 + u ** 2

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    x1, x2 = x[..., 0], x[..., 1]
    x_dot_1 = x1 * (1 - x2 ** 2) - x2 + u
    x_dot_2 = x1
    return torch.stack([x_dot_1, x_dot_2], dim = -1)

# def penalty_example(x: torch.tensor, expr, penalty_weight: float, pos = False) -> torch.tensor:
#     if pos:
#         coeff = -1
#     else:
#         coeff = 1
#     return penalty_weight * nn.relu(coeff * expr)

#define penalty cost for x[1] violation
def state_penalty(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    """
    Constructs state penalty for the van der pol constraints
    """
    return 1000 * nn.relu(-1 * (x[..., 1] + 0.4))

#define penalty cost for u violation (lower and upper)

#penalty_list = [state_penalty]
penalty_list = None

van_der_pol_obj = Case_Study(init_cond,
                        time_interval,
                        time_step,
                        dynamics,
                        pointwise_obj_cost,
                        penalty_list,
                        )