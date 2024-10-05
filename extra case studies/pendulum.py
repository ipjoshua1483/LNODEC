import torch
import torch.nn as nn

from utils import device
from case_study import Case_Study

#pos, v, theta, omega
init_cond = torch.tensor([2., 1., 0.5, 1.5], dtype = torch.float32).to(device)

time_interval = torch.tensor([0., 20.], dtype = torch.float32).to(device)

time_step = 0.02

def pointwise_obj_cost(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return 10 * x[..., 0] ** 2 + x[..., 1] ** 2 + 100 * x[..., 2] ** 2 + x[..., 3] ** 2 + 0.01 * u ** 2

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    pos, v, theta, omega = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    
    #x dot
    pos_dot = v
    
    #v dot
    v_dot_denom = 1.0 + 0.1 - 0.1 * torch.cos(theta) ** 2
    v_dot_num = u + 0.1 * 0.5 * omega ** 2 * torch.sin(theta) - 0.1 * 0.5 * 9.81 * torch.sin(theta) * torch.cos(theta)
    v_dot = v_dot_num / v_dot_denom
    
    #theta dot
    theta_dot = omega

    #omega dot
    omega_dot_denom = 0.5 * (1.0 + 0.1 - 0.1 * torch.cos(theta) ** 2)
    omega_dot_num = u * torch.cos(theta) + 0.1 * 0.5 * omega ** 2 * torch.sin(theta) * torch.cos(theta) - (1.0 + 0.1) * 9.81 * torch.sin(theta) 
    omega_dot = omega_dot_num / omega_dot_denom

    return torch.stack([pos_dot, v_dot, theta_dot, omega_dot], dim = -1)

def state_penalty(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    """
    Constructs state penalty for the van der pol constraints
    """
    return 1000 * nn.relu(-1 * (x[..., 1] + 0.4))

#penalty_list = [state_penalty]
penalty_list = None

pendulum_obj = Case_Study(init_cond,
                        time_interval,
                        time_step,
                        dynamics,
                        pointwise_obj_cost,
                        penalty_list,
                        )