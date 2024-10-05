import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
from torchdiffeq import odeint, odeint_adjoint
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt
import numpy as np

from utils import device, simpson_rule, torch_to_numpy
from case_study import Case_Study
from van_der_pol import van_der_pol_obj

class Policy(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        output_bounds: Tuple, 
        normalize: Callable[[torch.tensor], torch.tensor] = None,
        unnormalize: Callable[[torch.tensor], torch.tensor] = None,
    ):
        super(Policy, self).__init__()
        self.output_bounds = output_bounds
        self.normalize = normalize
        self.unnormalize = unnormalize
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.Tanh(),         
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh(),         
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),         
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),      
            nn.Linear(hidden_size, output_size)  
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x * (self.output_bounds[1] - self.output_bounds[0]) + self.output_bounds[0]  
        return x

    def objective(self, x: torch.tensor, case_study: Case_Study) -> torch.tensor:
        u = self(x).squeeze()
        objective_value = case_study.objective(x, u)
        return objective_value

    def potential(self, x: torch.tensor, case_study: Case_Study) -> torch.tensor:
        u = self(x).squeeze()
        potential_value = case_study.potential_objective(x, u)
        return potential_value

    def pointwise_lyapunov_loss(self, x: torch.tensor, case_study: Case_Study, kappa: float = 5.) -> torch.tensor:
        V = self.potential(x, case_study)
        dVdx = torch.autograd.functional.jacobian(
            lambda x: self.potential(x, case_study), 
            x, 
            create_graph = True
        ).diagonal(dim1 = 0, dim2 = 1).T
        dVdt = (dVdx @ case_study.dynamics(x, self(x).squeeze(), 0).T)
        if len(dVdt.shape) != 1:
            dVdt = dVdt.diagonal()
        return dVdt + kappa * V

    def pointwise_lyapunov_loss_penalty(self, x: torch.tensor, case_study: Case_Study):
        pointwise_loss = torch.relu(self.pointwise_lyapunov_loss(x, case_study))
        if case_study.penalty_cost_list is not None:
            for penalty in case_study.penalty_cost_list:
                pointwise_loss = pointwise_loss + penalty(x, self(x).squeeze())
        return pointwise_loss

    class Dynamics(nn.Module):    
        def __init__(self, policy, dynamics):
            super(Policy.Dynamics, self).__init__()
            self.policy = policy
            self.dynamics = dynamics
        
        def forward(self, t: torch.tensor, x: torch.tensor) -> torch.tensor:
            return self.dynamics(x, self.policy(x).squeeze(), t)

        def train_policy(
            self, 
            case_study: Case_Study, 
            node: bool = True, 
            num_epochs: int = 300, 
            lr: float = 0.025, 
            verbose = True
        ):
            optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
            total_loss_history = []
            control_loss_history = []
            states_all = []
            inputs_all = []
            objectives_all = []
            pointwise_loss_all = []
            potentials_all = []
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                trajectory = odeint_adjoint(
                    self, 
                    case_study.init_cond, 
                    case_study.time_interval, 
                    method = 'rk4'
                )
                objective = self.policy.objective(trajectory[:-1], case_study)
                loss = None
                if node:
                    loss = objective
                else:
                    pointwise_loss = self.policy.pointwise_lyapunov_loss(trajectory[:-1], case_study, kappa = 5.)
                    pointwise_loss_penalty = self.policy.pointwise_lyapunov_loss_penalty(trajectory[:-1], case_study)
                    loss = simpson_rule(pointwise_loss_penalty, case_study.time_interval[1] - case_study.time_interval[0])
                    control_loss_history.append(objective.item())
                total_loss_history.append(loss.item())
                loss.backward()
                optimizer.step()
                
                if verbose:
                    if epoch % 10 == 0:
                        print(f"Epoch: {epoch}, Loss: {loss.item()}")
                        if node == False:
                            print(f"Control Loss: {objective.item()}")
            
                [states, inputs, objectives] = torch_to_numpy([trajectory, self.policy(trajectory[:-1]), objective])
                states_all.append(states)
                inputs_all.append(inputs)
                objectives_all.append(objectives)
                if node == False:
                    [pointwise_losses] = torch_to_numpy([pointwise_loss])
                    pointwise_loss_all.append(pointwise_losses)
            print(f"Final states: {states_all[-1][-1]}")
            if node:
                times = torch_to_numpy(case_study.time_interval)
                return self, times, states_all, inputs_all, objectives_all, total_loss_history
            else:
                potential_traj = self.policy.potential(trajectory[:-1], case_study)
                potential_traj_norm = potential_traj
                times = case_study.time_interval.detach().numpy()
                if len(potential_traj.shape) == 1:
                    potential_traj_norm = potential_traj_norm / potential_traj[0]
                else:
                    potential_traj_norm = potential_traj_norm / potential_traj[..., 0]
                [potential_traj_norm] = torch_to_numpy([potential_traj_norm])
                return self, times, states_all, inputs_all, pointwise_loss_all, potential_traj_norm, total_loss_history, control_loss_history
        
    def generate_adversarial_trajectories(
        self, 
        case_study: Case_Study, 
        num_ICs: int = 10, 
        perturbation: torch.tensor = torch.tensor([2., 0.]), 
        seed: int = 42
    ):
        sobol_eng = SobolEngine(dimension = len(case_study.init_cond), seed = seed)
        init_conds = case_study.init_cond.repeat(num_ICs).reshape(-1, len(case_study.init_cond)) + sobol_eng.draw(num_ICs) * 2 * perturbation - perturbation
        [time_interval_np] = torch_to_numpy([case_study.time_interval])
        trajectory_all = []
        input_all = []
        terminal_states_np = np.zeros((num_ICs, len(case_study.init_cond)))
        constraint_violation_count = 0
        for i, init_cond in enumerate(init_conds):
            trajectory = odeint(
                lambda t, x: case_study.dynamics(x, self(x).squeeze(), t), 
                init_cond, 
                case_study.time_interval, 
                method = 'rk4'
            )
            trajectory_np, input_np = torch_to_numpy([trajectory, self(trajectory[:-1]).squeeze()])
            trajectory_all.append(trajectory_np)
            input_all.append(input_np)
            terminal_states_np[i] = trajectory_np[-1]
            if case_study.penalty_cost_list is not None:
                for penalty in case_study.penalty_cost_list:
                    if (penalty(trajectory, self(trajectory[:-1])) > 0).any():
                        constraint_violation_count += 1
        terminal_states_mean_np = np.mean(terminal_states_np, axis = 0)
        terminal_states_std_np = np.std(terminal_states_np, axis = 0)
        print("Terminal states mean: ", terminal_states_mean_np)
        print("Terminal states std dev: ", terminal_states_std_np)
        print("Constraint violations: ", constraint_violation_count)
        return time_interval_np, trajectory_all, input_all, terminal_states_mean_np, terminal_states_std_np, constraint_violation_count
    
    def generate_vector_fields(self, case_study: Case_Study):
        x1 = np.linspace(-0.5, 1.5, 10)
        x2 = np.linspace(-0.5, 3.5, 20)
        X1, X2 = np.meshgrid(x1, x2)
        X1_tensor, X2_tensor = torch.tensor(X1, dtype = torch.float32), torch.tensor(X2, dtype = torch.float32)
        X_tensor = torch.stack([X1_tensor, X2_tensor], dim = -1)
        X_dot_tensor = case_study.dynamics(X_tensor, self(X_tensor).squeeze(-1), 0)
        [X_dot] = torch_to_numpy([X_dot_tensor])
        return [X1, X2, X_dot[..., 0], X_dot[..., 1]]
        