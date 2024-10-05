import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from datetime import datetime

from policy_2 import Policy
from utils import (
    device, 
    plot_loss_epoch, 
    plot_x_t, 
    plot_u_t, 
)
from appj import appj_lya_obj as obj

hidden_size = 32

def main():
    timestamp = datetime.now()
    timestamp_name = timestamp.strftime("%Y%m%d %H%M%S")

    node_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    lya_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    print("Training NODE policy with no velocity reference")
    node_policy_dynamics = node_policy.Dynamics(node_policy, obj.dynamics)
    node_policy, node_times, node_states, node_inputs, node_objectives, node_loss_history = node_policy_dynamics.train_policy(obj)
    
    print("Training lyapunov policy")
    lya_policy_dynamics = lya_policy.Dynamics(lya_policy, obj.dynamics)
    lya_policy, lya_times, lya_states, lya_inputs, lya_pointwise_losses, lya_potential_norm, lya_total_loss_history, lya_control_loss_history = lya_policy_dynamics.train_policy(obj, node = False)

    loss_history_dict = {
        "NODE no vel ref": node_loss_history,
        "Lyapunov Control Loss": lya_control_loss_history,
    }

    plot_epoch = -1
    #view loss histories
    plot_loss_epoch(loss_history_dict, timestamp_name)

    plot_epoch_min = torch.argmin(torch.tensor(lya_total_loss_history, dtype = torch.float32).to(device))
    print(f"plot_epoch_min total loss: {plot_epoch_min}")
    plot_epoch_min_2 = torch.argmin(torch.tensor(lya_control_loss_history, dtype = torch.float32).to(device))
    print(f"plot_epoch_min control loss: {plot_epoch_min_2}")

    for i in range(len(obj.init_cond)):
        ref = obj.x_ref[i]
        plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch)
        plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch_min)
        plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch_min_2)
        
    plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch)
    plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch_min)
    plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch_min_2)

    print("Plots saved successfully")

if __name__ == "__main__":
    main()