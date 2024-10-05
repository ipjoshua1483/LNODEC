from typing import List, Callable, Type, Optional
import torch
import torch.nn as nn
from utils import device

class Case_Study:
    """
    Creates a case study object which contains:
    init_cond: initial condition of the state x
    policy: policy to recommend u based on state x
    dynamics: continuous model to predict time evolution of states from state x and input u
    pointwise_cost: cost in the optimal control problem at a singular time point
    penalty_cost: list of constraints enforced via soft constraint penalties
    """
    def __init__(
        self, 
        init_cond: torch.tensor,
        time_interval: torch.tensor, 
        dynamics: Callable[[torch.tensor, torch.tensor], torch.tensor], 
        objective: Callable[[torch.tensor, torch.tensor], torch.tensor], 
        potential_objective: Callable[[torch.tensor, torch.tensor], torch.tensor],
        penalty_cost_list: List[Callable[[torch.tensor, torch.tensor], torch.tensor]] = None,
        x_ref: torch.tensor = None,
        normalize: Callable[[torch.tensor], torch.tensor] = None,
        unnormalize: Callable[[torch.tensor], torch.tensor] = None,
        u_bounds: List[torch.tensor] = [
            torch.tensor([-10.], 
            dtype = torch.float32).to(device), 
            torch.tensor([10.], dtype = torch.float32).to(device)
        ]
    ):
        
        self.init_cond = init_cond
        self.time_interval = time_interval
        self.dynamics = dynamics
        self.penalty_cost_list = penalty_cost_list
        self.objective = self.add_penalty(objective)
        self.potential_objective = potential_objective
        self.x_ref = x_ref
        self.normalize = normalize
        self.unnormalize = unnormalize
        self.u_bounds = u_bounds

    def add_penalty(self, function: Callable[[torch.tensor, torch.tensor], torch.tensor]) -> Callable[[torch.tensor, torch.tensor], torch.tensor]:
        """
        Adds penalties to the function passed into args if there are penalties in the case study. Otherwise returns the same function
        """
        if self.penalty_cost_list is not None:
            def function_w_penalty(*args):
                function_tensor = function(*args)
                for penalty in self.penalty_cost_list:
                    penalty_tensor = penalty(*args)
                    if len(function_tensor.shape) == 1:
                        function_tensor += torch.sum(penalty_tensor, dim = -1)
                    elif len(function_tensor.shape) > 1:
                        function_tensor += penalty_tensor
                return function_tensor
            return function_w_penalty
        else:
            return function 
    