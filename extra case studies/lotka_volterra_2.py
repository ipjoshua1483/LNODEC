import torch
import torch.nn as nn

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([0.5, 0.7, 0.], dtype = torch.float32).to(device)

time_interval = torch.linspace(0., 20., steps = 500, dtype = torch.float32).to(device)

x_0_ref_val = 1.0
x_1_ref_val = 1.0
x_2_ref_val = 1.34408
#float('nan')
x_ref = torch.tensor([x_0_ref_val, x_1_ref_val, x_2_ref_val], dtype = torch.float32).to(device)

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
    x_dot_0 = x0 - x0 * x1 - 0.4 * x0 * u
    x_dot_1 = -x1 + x0 * x1 - 0.2 * x1 * u
    x_dot_2 = (x0 - 1) ** 2 + (x1 - 1) ** 2
    return torch.stack([x_dot_0, x_dot_1, x_dot_2], dim = -1)

penalty_list = None

lotka_volterra_lya_obj = Case_Study(init_cond,
                        time_interval,
                        #time_step,
                        dynamics,
                        objective,
                        potential_objective,
                        penalty_list,
                        x_ref,
                        )