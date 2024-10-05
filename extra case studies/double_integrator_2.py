import torch
import torch.nn as nn
#from policy import Policy

from utils import device
from case_study import Case_Study

init_cond = torch.tensor([0., 0., 0., 0.], dtype = torch.float32).to(device)

#time_interval = torch.tensor([0., 1.], dtype = torch.float32).to(device)

#time_step = 0.2

time_interval = torch.linspace(0., 2., steps = 500, dtype = torch.float32).to(device)

x_1_ref_val = 5.0
x_2_ref_val = -6.0

x_1_ref = torch.tensor([x_1_ref_val], dtype = torch.float32).to(device)
x_2_ref = torch.tensor([x_2_ref_val], dtype = torch.float32).to(device)

x_ref = torch.tensor([x_1_ref_val, x_2_ref_val, 0.0, 0.0], dtype = torch.float32).to(device)

def objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[-1, 0] - x_1_ref) ** 2 + (x[-1, 1] - x_2_ref) ** 2
    #return torch.sqrt((x[-1, 0] - x_1_ref) ** 2 + 10e-8*(x[-1, 1] - x_2_ref) ** 2)
    #return torch.sum((x[-1] - x_ref) ** 2)

def potential_objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[..., 0] - x_1_ref) ** 2 + (x[..., 1] - x_2_ref) ** 2
    #return torch.sqrt((x[..., 0] - x_1_ref) ** 2 + 10e-8*(x[..., 1] - x_2_ref) ** 2)
    #return torch.sum((x - x_ref) ** 2, dim = -1)

def objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    #return (x[-1, 0] - x_1_ref) ** 2
    return torch.sum((x[-1] - x_ref) ** 2).unsqueeze(-1)

def potential_objective_node(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    #return (x[..., 0] - x_1_ref) ** 2
    return torch.sum((x - x_ref) ** 2, dim = -1)

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    if len(x.shape) != 1:
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    else:
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    if len(u.shape) != 1:
        u1, u2 = u[..., 0], u[..., 1]
    else:
        u1, u2 = u[0], u[1]
    x_dot_1 = x3
    x_dot_2 = x4
    x_dot_3 = u1
    x_dot_4 = u2
    return torch.stack([x_dot_1, x_dot_2, x_dot_3, x_dot_4], dim = -1)

# def penalty_example(x: torch.tensor, expr, penalty_weight: float, pos = False) -> torch.tensor:
#     if pos:
#         coeff = -1
#     else:
#         coeff = 1
#     return penalty_weight * nn.relu(coeff * expr)

#define penalty cost for x[1] violation
#penalty = 0.0001 for log
def state_penalty(x: torch.tensor, u: torch.tensor, penalty: float = 5.0) -> torch.tensor:
    """
    Constructs state penalty for the constraints
    """
    x_2_constraint = 2.3
    if len(x.shape) != 1:
        temp = penalty * torch.relu(x[..., 1] - x_2_constraint) ** 2
        #temp = penalty * -torch.log(2.0 - x[..., 1] + 1e-4)
    else:
        temp = penalty * torch.relu(x[1] - x_2_constraint) ** 2
        #temp = penalty * -torch.log(2.0 - x[1] + 1e-4)
    #print(temp)
    return temp

#define penalty cost for u violation (lower and upper)

#penalty_list = [state_penalty]
penalty_list = None
penalty_list_constrained = [state_penalty]

double_integrator_lya_obj_2 = Case_Study(init_cond,
                        time_interval,
                        #time_step,
                        dynamics,
                        objective,
                        potential_objective,
                        penalty_list,
                        x_ref,
                        )

# double_integrator_lya_obj_constrained = Case_Study(init_cond,
#                         time_interval,
#                         #time_step,
#                         dynamics,
#                         objective,
#                         potential_objective,
#                         penalty_list_constrained,
#                         )

# double_integrator_node_obj = Case_Study(init_cond,
#                         time_interval,
#                         #time_step,
#                         dynamics,
#                         objective_node,
#                         potential_objective_node,
#                         penalty_list,
#                         )

# test_x = torch.tensor([[0.5, 0.5]], dtype = torch.float32).to(device)
# test_u = torch.tensor([0.5], dtype = torch.float32).to(device)
# print(double_integrator_obj.objective(test_x, test_u))