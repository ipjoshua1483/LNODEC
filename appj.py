import torch
import torch.nn as nn
import numpy as np
from utils import device
from case_study import Case_Study
from case_study_2 import Case_Study as Case_Study_2

T_up = 318.15

k2cel = 273.15
init_cond = torch.tensor([310.15 - k2cel, 0.], dtype = torch.float32).to(device)

time_interval = torch.linspace(0., 100., steps = 500, dtype = torch.float32).to(device)
x_ref = torch.tensor([T_up - k2cel, 1.5], dtype = torch.float32).to(device)

u_lower = torch.tensor([1.], dtype = torch.float32).to(device)
u_upper = torch.tensor([5.], dtype = torch.float32).to(device)

#constants
K_cons = 0.5
T_ref = T_up - k2cel -2
T_inf = 298.15 - k2cel
T_b = 308.15 - k2cel
ρ = 2800
cp = 795
r = 1.5e-3
d = 0.2e-3
k = 1.43
β = 90.82
μ = 9.84e-4
T_0_cons = 310.15 - k2cel
T_bar = T_up - k2cel
Cp = torch.tensor(ρ * cp * torch.pi * r * r * d / μ)
φ_numerator = (2 * torch.pi * r * d * k * β / μ) / Cp * (T_bar - (T_b + T_inf) / 2) * (np.log(T_bar - T_inf) - np.log(T_bar - T_b))
print(Cp)
print(φ_numerator)
normalization_factor = 40.

def objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[-1, 1] - x_ref[1]) ** 2

def potential_objective(x: torch.tensor, u: torch.tensor) -> torch.tensor:
    return (x[..., 1] - x_ref[1]) ** 2

def dynamics(x: torch.tensor, u: torch.tensor, t: float) -> torch.tensor:
    if len(x.shape) != 1:
        T, CEM = x[..., 0], x[..., 1]
    else:
        T, CEM = x[0], x[1] 
    T_dot = (u / Cp) - (φ_numerator / (torch.log(T - T_inf) - torch.log(T - T_b)))
    CEM_dot = torch.pow(K_cons, T_ref - T)/60.
    return torch.stack([T_dot, CEM_dot], dim = -1)

def state_penalty(x: torch.tensor, u: torch.tensor, penalty: float = 50.) -> torch.tensor:
    """
    Constructs state penalty for the constraints
    """
    temp_ref = T_bar
    if len(x.shape) != 1:
        temp = penalty * torch.relu(x[..., 0] - temp_ref) ** 2
    else:
        temp = penalty * torch.relu(x[0] - temp_ref) ** 2
    return temp

def normalize(x):
    y = x.clone()
    if len(y.shape) != 1:
        y[..., 0] = y[..., 0] / normalization_factor
    else:
        y[0] = y[0] / normalization_factor
    return y

def unnormalize(x):
    y = x.clone()
    if len(y.shape) != 1:
        y[..., 0] = y[..., 0] * normalization_factor
    else:
        y[0] = y[0] * normalization_factor
    return y

penalty_list = None
penalty_list = [state_penalty]

appj_lya_obj = Case_Study(
    init_cond,
    time_interval,
    dynamics,
    objective,
    potential_objective,
    penalty_list,
    x_ref,
    normalize,
    None,
    [u_lower, u_upper],
)

appj_lya_obj_2 = Case_Study_2(
    init_cond,
    time_interval,
    dynamics,
    objective,
    potential_objective,
    penalty_list,
    x_ref,
    normalize,
    None,
    )