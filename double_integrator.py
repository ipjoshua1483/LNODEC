import torch
import torch.nn as nn

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([0., 0.], dtype = torch.float32).to(device)

time_interval = torch.linspace(0., 1.5, steps = 500, dtype = torch.float32).to(device)
time_interval_constrained = torch.linspace(0., 2., steps = 1000, dtype = torch.float32).to(device)

x_1_ref_val = 1.0
x_2_ref_val = 0.0

x_1_ref = torch.tensor([x_1_ref_val], dtype = torch.float32).to(device)
x_2_ref = torch.tensor([x_2_ref_val], dtype = torch.float32).to(device)

x_ref = torch.tensor([x_1_ref_val, 0.0], dtype = torch.float32).to(device)

def objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[-1, 0] - x_1_ref) ** 2

def potential_objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[..., 0] - x_1_ref) ** 2
    
def objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return torch.sum((x[-1] - x_ref) ** 2).unsqueeze(-1)

def potential_objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return torch.sum((x - x_ref) ** 2, dim = -1)

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    if len(x.shape) != 1:
        x1, x2 = x[..., 0], x[..., 1]
    else:
        x1, x2 = x[0], x[1]
    x_dot_1 = x2
    x_dot_2 = u
    return torch.stack([x_dot_1, x_dot_2], dim = -1)

def state_penalty(x: torch.tensor, u: torch.tensor, penalty: float = 5.0) -> torch.tensor:
    """
    Constructs state penalty for the constraints
    """
    x_2_constraint = 2.
    if len(x.shape) != 1:
        temp = penalty * torch.relu(x[..., 1] - x_2_constraint) ** 2
    else:
        temp = penalty * torch.relu(x[1] - x_2_constraint) ** 2
    return temp

penalty_list = None
penalty_list_constrained = [state_penalty]

double_integrator_lya_obj = Case_Study(init_cond,
    time_interval,
    dynamics,
    objective,
    potential_objective,
    penalty_list,
    )

double_integrator_lya_obj_constrained = Case_Study(init_cond,
    time_interval_constrained,
    dynamics,
    objective,
    potential_objective,
    penalty_list_constrained,
    )

double_integrator_node_obj = Case_Study(init_cond,
    time_interval,
    dynamics,
    objective_node,
    potential_objective_node,
    penalty_list,
    )
