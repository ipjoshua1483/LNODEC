import torch
import torch.nn as nn

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([0., 0., 0., 0.], dtype = torch.float32).to(device)

time_interval = torch.linspace(0., 2., steps = 500, dtype = torch.float32).to(device)

x_2_ref_val = 4.0
x_3_ref_val = 10.0
x_4_ref_val = 0.0

x_ref = torch.tensor([float('nan'), x_2_ref_val, x_3_ref_val, x_4_ref_val], dtype = torch.float32).to(device)

x_2_ref = torch.tensor([x_2_ref_val], dtype = torch.float32).to(device)
x_3_ref = torch.tensor([x_3_ref_val], dtype = torch.float32).to(device)
x_4_ref = torch.tensor([x_4_ref_val], dtype = torch.float32).to(device)

def objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[-1, 1] - x_2_ref) ** 2 + (x[-1, 2] - x_3_ref) ** 2 + (x[-1, 3] - x_4_ref) ** 2

def potential_objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[..., 1] - x_2_ref) ** 2 + (x[..., 2] - x_3_ref) ** 2 + (x[..., 3] - x_4_ref) ** 2

def objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[-1, 1] - x_2_ref) ** 2 + (x[-1, 2] - x_3_ref) ** 2 
    return (x[-1, 1] - x_2_ref) ** 2

def potential_objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[..., 1] - x_2_ref) ** 2 + (x[..., 2] - x_3_ref) ** 2 
    return (x[..., 1] - x_2_ref) ** 2

def dynamics(x: torch.tensor, u: torch.tensor, t: float, a: float = 15.0) -> torch.tensor:
    if len(x.shape) != 1:
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    else:
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    x_dot_1 = x3
    x_dot_2 = x4
    x_dot_3 = a * torch.cos(u)
    x_dot_4 = a * torch.sin(u)
    return torch.stack([x_dot_1, x_dot_2, x_dot_3, x_dot_4], dim = -1)

penalty_list = None

particle_steering_lya_obj = Case_Study(init_cond,
                        time_interval,
                        #time_step,
                        dynamics,
                        objective,
                        potential_objective,
                        penalty_list,
                        x_ref,
                        )

particle_steering_node_obj = Case_Study(init_cond,
                        time_interval,
                        #time_step,
                        dynamics,
                        objective_node,
                        potential_objective_node,
                        penalty_list,
                        x_ref,
                        )