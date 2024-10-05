import torch
import torch.nn as nn

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([1., 0., 0.], dtype = torch.float32).to(device)

time_interval = torch.linspace(0., 20., steps = 500, dtype = torch.float32).to(device)

x_0_ref_val = 0.
x_1_ref_val = 0.
x_2_ref_val = 0. #3.1762
#float('nan')
x_ref = torch.tensor([x_0_ref_val, x_1_ref_val, float('nan')], dtype = torch.float32).to(device)

x_0_ref = torch.tensor([x_0_ref_val], dtype = torch.float32).to(device)
x_1_ref = torch.tensor([x_1_ref_val], dtype = torch.float32).to(device)
x_2_ref = torch.tensor([x_2_ref_val], dtype = torch.float32).to(device)

def objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    #return (x[-1, 0] - x_0_ref) ** 2 + (x[-1, 1] - x_1_ref) ** 2 + (x[-1, 2] - x_2_ref) ** 2
    return (x[-1, 2] - x_2_ref) ** 2

def potential_objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    #return (x[-1, 0] - x_0_ref) ** 2 + (x[-1, 1] - x_1_ref) ** 2 + (x[..., 2] - x_2_ref) ** 2
    return (x[..., 2] - x_2_ref) ** 2

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    if len(x.shape) != 1:
        x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    else:
        x0, x1, x2 = x[0], x[1], x[2]
    x_dot_0 = x1
    x_dot_1 = u * (1 - x0 ** 2) * x1 - x0
    x_dot_2 = x0 ** 2 + x1 ** 2 + u ** 2
    return torch.stack([x_dot_0, x_dot_1, x_dot_2], dim = -1)

penalty_list = None

van_der_pol_2_lya_obj = Case_Study(init_cond,
                        time_interval,
                        #time_step,
                        dynamics,
                        objective,
                        potential_objective,
                        penalty_list,
                        x_ref,
                        )