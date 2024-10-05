import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime
import os

device = torch.device('cpu')
constraint = 2.3

def torch_to_numpy(torch_list: List[torch.tensor]) -> List[np.array]:
    np_list = []
    for item in torch_list:
        np_list.append(item.cpu().detach().numpy())
    return np_list

def simpson_rule(values, h):
    if len(values.shape) == 0:
        return values

    if len(values) % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")

    result = values[0] + values[-1] + 4 * sum(values[1:-1:2]) + 2 * sum(values[2:-2:2])
    result *= h / 3
    return result

def plot_loss_epoch(loss_dict: Dict[str, List], timestamp_name: str,) -> str:
    plot_name = "loss_epoch.png"
    plt.figure()
    plt.title("Loss over epochs")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for key, value in loss_dict.items():
        plt.plot(np.arange(len(value)), np.array(value), label = key)
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()
    
def plot_pointwise_comparison(time: np.array, pointwise_dict: Dict[str, List], timestamp_name: str,) -> str:
    plot_name = "Lyapunov_pointwise.png"
    plt.figure()
    plt.title("Lyapunov pointwise comparison")
    plt.xlabel('Time')
    plt.ylabel('Pointwise')
    for key, value in pointwise_dict.items():
        #plt.plot(np.arange(len(node_loss_history)), np.array(node_loss_history), label = "Base NODE loss")
        #plt.plot(np.arange(len(lyapunov_control_loss_history)), np.array(lyapunov_control_loss_history), label = "Lyapunov Control Loss")
        plt.plot(time[:-1], np.array(value), label = key)
    plt.legend()
    #plt.saveplot(plot_name)
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_potential_norm_shifted(
    time: np.array, 
    potential_norm_dict: Dict[str, np.array], 
    timestamp_name: str, 
    kappa: float = 5.
):
    plot_name = "Lyapunov_potential_normalized_shifted.png"
    plt.figure()
    for key, value in potential_norm_dict.items():
        plt.plot(time[-len(value):] - 0.2, value, label = key)
    filtered_time = time[time >= 0.2]
    exponential_decay = np.exp(-kappa*filtered_time)/np.exp(-kappa*0.2)
    plt.plot(filtered_time - 0.2, exponential_decay, linestyle = "--", color = "red", label = "Exponential decay")
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_potential_norm(
    time: np.array, 
    potential_norm_dict: Dict[str, np.array], 
    timestamp_name: str, 
    kappa: float = 5.
):
    plot_name = "Lyapunov_potential_normalized.png"
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Potential')
    for key, value in potential_norm_dict.items():
        if key == "Unconstrained L-NODEC":
            color = 'green'
        elif key == "Constrained L-NODEC":
            color = 'orange'
        plt.plot(time[:-1], value, label = key, color = color)
    exponential_decay = np.exp(-kappa*np.array(time))
    plt.plot(time, exponential_decay, linestyle = "--", color = "red", label = "Exponential Stability Threshold")
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_potential_norm_2(
    potential_times_dict: Dict[str, np.array],
    potential_norm_dict: Dict[str, np.array], 
    timestamp_name: str, 
    kappa: float = 5.
):
    plot_name = "Lyapunov_potential_normalized.png"
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Potential')
    for key, value in potential_norm_dict.items():
        time = potential_times_dict[key]
        if key == "Unconstrained L-NODEC":
            color = 'green'
        elif key == "Constrained L-NODEC":
            color = 'orange'
        plt.plot(time[:-1], value, label = key, color = color)
    exponential_decay = np.exp(-kappa*np.array(time))
    plt.plot(time, exponential_decay, linestyle = "--", color = "red", label = "Exponential Stability Threshold")
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_potential_adversarial(
    potential_times_dict: Dict[str, List[np.array]], 
    potential_norm_dict: Dict[str, List[np.array]], 
    timestamp_name: str, 
    kappa: float = 5.
):
    plot_name = "potential_norm_adversarial.png"
    plt.figure()
    for key, value in potential_norm_dict.items():
        time = potential_times_dict[key]
        if key == "Unconstrained L-NODEC":
            color = 'green'
        elif key == "Constrained L-NODEC":
            color = 'orange'
        for i, traj in enumerate(value):
            plt.plot(time[:-1], traj, alpha = 0.5, color = color, label = key if i == 0 else None)
    exponential_decay = np.exp(-kappa * np.array(time))
    plt.plot(time, exponential_decay, linestyle = "--", color = "red", label = "Exponential Stability Threshold")
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_potential_adversarial_truncated(
    potential_times_dict: Dict[str, List[np.array]], 
    potential_norm_dict: Dict[str, List[np.array]], 
    timestamp_name: str, kappa: float = 5.
):
    plot_name = "potential_norm_adversarial_truncated.png"
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Potential function V(x(t))")
    time_truncate = 0.5
    index = 0
    for key, value in potential_norm_dict.items():
        time = potential_times_dict[key]
        cond = time - time_truncate > 1e-3
        index = torch.where(torch.tensor(cond))[0][0]
        #print(index)
        if key == "Unconstrained L-NODEC":
            color = 'green'
        elif key == "Constrained L-NODEC":
            color = 'orange'
        for i, traj in enumerate(value):
            plt.plot(time[index:-1], traj[index:], alpha = 0.5, color = color, label = key if i == 0 else None)
    exponential_decay = np.exp(-kappa * np.array(time))
    plt.plot(time[index:], exponential_decay[index:], linestyle = "--", color = "red", label = "Exponential Stability Threshold")
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def plot_xu_t(
    t: np.array, 
    x: List[np.array], 
    u: List[np.array], 
    timestamp_name: str, 
    epoch: int, 
    prefix: str, 
    constraint: float = constraint
) -> str:
    x = x[epoch]
    u = u[epoch]

    handles, labels, handles_args = [], [], []
    plot_name = prefix + "_xu_t.png"
    fig, ax1 = plt.subplots()
    plt.title(prefix + f" States & Inputs over time @ epoch {epoch}")
    plt.xlabel('Time')
    plt.axhline(y = constraint, linestyle = "--")

    for i in range(x.shape[-1]):
        line, = ax1.plot(t, x[:, i], label = f"x{i + 1}")
        handles_args.append(line)
    ax1.set_ylabel("States (x)")
    ax1.set_ylim(-0.5, 4.0)

    ax2 = ax1.twinx()
    line2, = ax2.plot(t[:-1], u, label = f"u", color = 'r')
    handles_args.append(line2)
    ax2.set_ylabel('Inputs (u)')
    ax2.set_ylim(-10.0, 10.0)

    handles.extend(handles_args)
    labels.extend([line.get_label() for line in handles])
    plt.legend(handles, labels)
    save_plot(timestamp_name, plot_name)

def plot_u_t(
    t: np.array, 
    node_u: List[np.array], 
    lya_u: List[np.array], 
    timestamp_name: str, 
    epoch: int = -1,
    node_vel_u: List[np.array] = None,
) -> None:
    plt.figure(figsize = (8, 6))
    lya_u = lya_u[epoch]
    node_epoch = -1
    plot_name = f"u_t_{epoch}.png"
    plt.xlabel('Time')
    plt.ylabel('Power (u)')
    if node_u != []:
        node_u = node_u[node_epoch]
        plt.plot(t[1:], node_u, label = "NODEC", color = 'blue')
    if node_vel_u is not None:
        node_vel_u = node_vel_u[node_epoch]
        plt.plot(t[1:], node_vel_u, label = "NODEC", color = 'blue')
    plt.plot(t[1:], lya_u, label = "L-NODEC", color = 'green')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x_t(
    index: int, 
    ref: float, 
    node_t: np.array, 
    lyapunov_t: np.array, 
    node_x: List[np.array], 
    lyapunov_x: List[np.array], 
    timestamp_name: str, 
    epoch: int = -1,
    node_vel_t: List[np.array] = None,
    node_vel_x: List[np.array] = None,
) -> None:
    node_epoch = -1
    lyapunov_x = lyapunov_x[epoch][:, index]
    state_name = f"x{index+1}"
    plot_name = state_name + f"_t_{epoch}.png"
    plt.figure()
    plt.xlabel("Time")
    if index == 0:
        plt.ylabel(f"T ({state_name})")
    elif index == 1:
        plt.ylabel(f"CEM ({state_name})")
    if node_x != []:
        node_x = node_x[node_epoch][:, index]
        plt.plot(node_t, node_x, label = "NODEC", color = 'blue')
    
    if node_vel_x is not None:
        node_vel_x = node_vel_x[node_epoch][:, index]
        plt.plot(node_vel_t, node_vel_x, label = "NODEC", color = 'blue')
    
    plt.plot(lyapunov_t, lyapunov_x, label = "L-NODEC", color = 'green')
    if ref is not float('nan'):
        plt.plot(lyapunov_t, np.full(len(lyapunov_t), ref), linestyle = '--', color = 'red')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x_t_all(
    node_times, 
    node_states_control, 
    lya_times, 
    lya_states_lyapunov, 
    index, 
    timestamp_name
):
    plot_name = f"NODEC_LNODEC_{index}.png"
    node_states_copy = node_states_control[..., index]
    lya_states_copy = lya_states_lyapunov[..., index]
    plt.figure()
    plt.xlabel('Time (s)')
    if index == 0:
        plt.ylabel(f"T (C)")
    elif index == 1:
        plt.ylabel(f"CEM (min)")
    for i, node_traj in enumerate(node_states_copy):
        plt.plot(node_times, node_traj, label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
    for j, lya_traj in enumerate(lya_states_copy):
        plt.plot(lya_times, lya_traj, label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
    if index == 0:
        plt.axhline(y = 45, color = 'red', linestyle = '--')
    else:
        plt.axhline(y = 1.5, color = 'red', linestyle = '--')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x_t_all_truncated_subplots(
    node_times, 
    node_states_control,
    node_inputs_control, 
    lya_times, 
    lya_states_lyapunov,
    lya_inputs_lyapunov, 
    timestamp_name
):
    node_cond = None
    lya_cond = None

    plot_name = "NODEC_LNODEC_x_t_subplots.png"
    plt.figure(figsize = (24, 6))
    for index in range(lya_states_lyapunov.shape[-1] - 1, -1, -1):
        
        node_states_copy = node_states_control[..., index]
        lya_states_copy = lya_states_lyapunov[..., index]

        plt.subplot(1, 3, 2 - index)
        plt.xlabel('Time (s)')
        if index == 0:
            plt.ylabel(f"T (C)")
        elif index == 1:
            plt.ylabel(f"CEM (min)")

        for i, node_traj in enumerate(node_states_copy):
            if index == 1:
                node_cond = node_traj[node_traj < 1.5]
                plt.plot(node_times[:len(node_cond)], node_cond, label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
            else:
                plt.plot(node_times[:len(node_cond)], node_traj[:len(node_cond)], label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
        for j, lya_traj in enumerate(lya_states_copy):
            if index == 1:
                lya_cond = lya_traj[lya_traj < 1.5]
                plt.plot(lya_times[:len(lya_cond)], lya_cond, label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
            else:
                plt.plot(lya_times[:len(lya_cond)], lya_traj[:len(lya_cond)], label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
        if index == 0:
            plt.axhline(y = 45, color = 'red', linestyle = '--', label = "Temperature constraint")
        else:
            plt.axhline(y = 1.5, color = 'red', linestyle = '--', label = "Desired terminal thermal dose")
        plt.legend()
    plt.subplot(1, 3, 3)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.plot(node_times[:-1], node_inputs_control[0, :, 0], label = "NODEC", color = 'blue')
    plt.plot(lya_times[:-1], lya_inputs_lyapunov[0, :, 0], label = "L-NODEC", color = 'green')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x_t_all_truncated_individual_plots(
    node_times, 
    node_states_control, 
    lya_times, 
    lya_states_lyapunov, 
    timestamp_name
):
    node_cond = None
    lya_cond = None

    for index in range(lya_states_lyapunov.shape[-1] - 1, -1, -1):
        plot_name = f"NODEC_LNODEC_x_t_{index}.png"
        node_states_copy = node_states_control[..., index]
        lya_states_copy = lya_states_lyapunov[..., index]
        
        plt.figure()
        plt.xlabel('Time (s)')
        if index == 0:
            plt.ylabel(f"T (C)")
        elif index == 1:
            plt.ylabel(f"CEM (min)")

        for i, node_traj in enumerate(node_states_copy):
            if index == 1:
                node_cond = node_traj[node_traj < 1.5]
                plt.plot(node_times[:len(node_cond)], node_cond, label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
            else:
                plt.plot(node_times[:len(node_cond)], node_traj[:len(node_cond)], label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
        for j, lya_traj in enumerate(lya_states_copy):
            if index == 1:
                lya_cond = lya_traj[lya_traj < 1.5]
                plt.plot(lya_times[:len(lya_cond)], lya_cond, label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
            else:
                plt.plot(lya_times[:len(lya_cond)], lya_traj[:len(lya_cond)], label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
        if index == 0:
            plt.axhline(y = 45, color = 'red', linestyle = '--', label = "Temperature constraint")
        else:
            plt.axhline(y = 1.5, color = 'red', linestyle = '--', label = "Desired terminal thermal dose")
        plt.legend()
        save_plot(timestamp_name, plot_name)

def plot_x_t_all_truncated(node_times, node_states_control, lya_times, lya_states_lyapunov, timestamp_name):
    node_condition = node_states_control[..., 1] < 1.5
    node_condition_index_1, node_condition_index_2,  = np.where(node_condition)
    lya_condition = lya_states_lyapunov[..., 1] < 1.5
    lya_condition_index_1, lya_condition_index_2 = np.where(lya_condition)

    node_cond = None
    lya_cond = None

    for index in range(lya_states_lyapunov.shape[-1] - 1, -1, -1):
        plot_name = f"NODEC_LNODEC_{index}.png"
        
        node_states_copy = node_states_control[..., index]
        lya_states_copy = lya_states_lyapunov[..., index]

        plt.figure()
        plt.xlabel('Time (s)')
        if index == 0:
            plt.ylabel(f"T (C)")
        elif index == 1:
            plt.ylabel(f"CEM (min)")

        for i, node_traj in enumerate(node_states_copy):
            if index == 1:
                node_cond = node_traj[node_traj < 1.5]
                plt.plot(node_times[:len(node_cond)], node_cond, label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
            else:
                plt.plot(node_times[:len(node_cond)], node_traj[:len(node_cond)], label = 'NODEC' if i == 0 else '', color = 'blue', alpha = 0.5)
        for j, lya_traj in enumerate(lya_states_copy):
            if index == 1:
                lya_cond = lya_traj[lya_traj < 1.5]
                plt.plot(lya_times[:len(lya_cond)], lya_cond, label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
            else:
                plt.plot(lya_times[:len(lya_cond)], lya_traj[:len(lya_cond)], label = 'L-NODEC' if j == 0 else '', color = 'green', alpha = 0.5)
        if index == 0:
            plt.axhline(y = 45, color = 'red', linestyle = '--')
        else:
            plt.axhline(y = 1.5, color = 'red', linestyle = '--')
        plt.legend()
        save_plot(timestamp_name, plot_name)

def plot_x_t_mean(node_times, node_states_control, lya_times, lya_states_lyapunov, index, timestamp_name):
    plot_name = f"NODEC_LNODEC_mean_{index}.png"
    node_states_copy = node_states_control[:, index]
    lya_states_copy = lya_states_lyapunov[:, index]
    plt.figure()
    plt.xlabel('Time')
    if index == 0:
        plt.title(f"T over time")
        plt.ylabel(f"T (C)")
    elif index == 1:
        plt.title(f"CEM over time")
        plt.ylabel(f"CEM (min)")
    plt.plot(node_times, node_states_copy, label = "NODEC", color = 'cyan')
    plt.plot(lya_times, lya_states_copy, label = "L-NODEC", color = 'green')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x1_t(node_t: np.array, lyapunov_t: np.array, lyapunov_constrained_t: np.array, node_x: List[np.array], lyapunov_x: List[np.array], lyapunov_constrained_x: List[np.array], timestamp_name: str, epoch: int = -1, x_val: float = 1.0) -> None:
    node_x = node_x[epoch][:, 0]
    lyapunov_constrained_x = lyapunov_constrained_x[epoch][:, 0]
    lyapunov_x = lyapunov_x[epoch][:, 0]
    plot_name = "x1_t.png"
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.plot(node_t, node_x, label = "NODEC", color = 'blue')
    plt.plot(lyapunov_t, lyapunov_x, label = "Unconstrained L-NODEC", color = 'green')
    plt.plot(lyapunov_constrained_t, lyapunov_constrained_x, label = "Constrained L-NODEC", color = 'orange')
    plt.axhline(y = x_val, linestyle = '--', color = 'red')
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_x1_t_adversarial(
    node_t: np.array,
    lyapunov_t: np.array,
    lyapunov_constrained_t: np.array,
    node_adversartial_x: List[np.array],
    lyapunov_adversarial_x: List[np.array],
    lyapunov_constrained_adversarial_x: List[np.array],
    timestamp_name: str, 
    x_val: float = 1.0,
):
    plot_name = "x1_t_adversarial.png"
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    adversarial_t = [node_t, lyapunov_t, lyapunov_constrained_t]
    adversarial_x = [node_adversartial_x, lyapunov_adversarial_x, lyapunov_constrained_adversarial_x]
    adversarial_label = ["NODEC", "Unconstrained L-NODEC", "Constrained L-NODEC"]
    adversarial_color = ["blue", "green", "orange"]
    for i in range(len(adversarial_t)):
        time = adversarial_t[i]
        method = adversarial_x[i]
        label = adversarial_label[i]
        color = adversarial_color[i]
        for j in range(len(method)):
            plt.plot(time, method[j][:, 0], alpha = 0.5, color = color, label = label if j == 0 else None)
    plt.axhline(y = x_val, linestyle = '--', color = 'red', label = "Desired terminal position")
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_pointwise_losses_t(
    t: np.array, 
    pointwise_loss: List[np.array], 
    timestamp_name: str, 
    epoch: int, 
    node: bool = False
) -> str:
    pointwise_loss = pointwise_loss[epoch]
    prefix = ""
    if node:
        prefix += "Node"
    else:
        prefix += "Lyapunov"
    plot_name = prefix + "_pointwise_loss_t.png"
    plt.figure()
    plt.title(prefix + f" Pointwise loss over time @ epoch {epoch}")
    plt.xlabel('Time')
    plt.ylabel('Pointwise loss')
    plt.plot(t[:-1], np.array(pointwise_loss))
    save_plot(timestamp_name, plot_name)

def plot_phase_portrait(
    node_x: List[np.array], 
    lyapunov_x: List[np.array], 
    timestamp_name: str, 
    epoch: int, 
    constraint: float = constraint
) -> str:
    node_x, lyapunov_x = node_x[epoch], lyapunov_x[epoch]
    prefix = ""
    plot_name = prefix + "phase_portrait.png"

    plt.figure()
    plt.title(prefix + f" Phase portrait x2 vs x1 @ epoch {epoch}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(node_x[:, 0], node_x[:, 1], label = "NODE no vel ref")
    plt.plot(lyapunov_x[:, 0], lyapunov_x[:, 1], label = "Lyapunov")
    plt.axhline(y = constraint, linestyle = '--', label = "Constraint")
    plt.plot()
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_policy_params(policy_list, timestamp_name: str, before: bool):
    for i, policy in enumerate(policy_list):
        plt.figure()
        plt.xlabel("Policy parameters")
        plt.ylabel("Frequency")

        params_list = []
        for param in policy.named_parameters():
            params_list.extend(param[1].reshape(-1).cpu().detach().numpy().tolist())
        plt.hist(params_list)
        
        title = "Policy parameters"
        if i == 0:
            title += " NODE"
        else:
            title += " Lyapunov"
        
        if before:
            title += " before training"
        else:
            title += " after training"
        plt.title(title)
        save_plot(timestamp_name, title + ".png")
    
def plot_adversarial(
    time: np.array, 
    adversarial_trajectories: List[np.array], 
    adversarial_inputs: List[np.array], 
    prefix: str, 
    timestamp_name: str
):
    plot_name = prefix + "_x1.png"
    plt.figure()
    plt.title(prefix + " x1")
    plt.xlabel("Time")
    plt.ylabel("Position (x1)")
    plt.ylim(-0.5, 1.5)
    for adversarial_traj in adversarial_trajectories:
        plt.plot(time, adversarial_traj[:, 0])
    save_plot(timestamp_name, plot_name)

    plot_name = prefix + "_x2.png"
    plt.figure()
    plt.title(prefix + " x2")
    plt.xlabel("Time")
    plt.ylabel("Velocity (x2)")
    plt.ylim(-0.5, 4.0)
    for adversarial_traj in adversarial_trajectories:
        plt.plot(time, adversarial_traj[:, 1])
    save_plot(timestamp_name, plot_name)

    plot_name = prefix + "_u.png"
    plt.figure()
    plt.title(prefix + " u")
    plt.xlabel("Time")
    plt.ylabel("Input (u)")
    plt.ylim(-10.0, 10.0)
    for adversarial_input in adversarial_inputs:
        plt.plot(time[:-1], adversarial_input)
    save_plot(timestamp_name, plot_name)

    plot_name = prefix + "_phase_portrait.png"
    plt.figure()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 4.0)
    for adversarial_traj in adversarial_trajectories:
        plt.plot(adversarial_traj[:, 0], adversarial_traj[:, 1])
    save_plot(timestamp_name, plot_name)

def plot_adversarial_all(
    times_list: List[np.array], 
    adversarial_trajectories_list: List[List[np.array]], 
    adversarial_inputs_list: List[List[np.array]],
    names: List[str],
    colors: List[str], 
    timestamp_name: str
):
    title = "Lyapunov vs NODE with vel ref"
    plot_name = "adversarial_all_x1.png"
    plt.figure()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("x1")
    plt.ylim(-0.5, 1.5)
    for time, adversarial_trajs, name, color in zip(times_list, adversarial_trajectories_list, names, colors):
        for i, adversarial_traj in enumerate(adversarial_trajs):
            plt.plot(time, adversarial_traj[:, 0], label = name if i == 0 else None, color = color)
    plt.legend()
    save_plot(timestamp_name, plot_name)

    plot_name = "adversarial_all_x2.png"
    plt.figure()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("x2")
    plt.ylim(-0.5, 4.0)
    for time, adversarial_trajs, name, color in zip(times_list, adversarial_trajectories_list, names, colors):
        for i, adversarial_traj in enumerate(adversarial_trajs):
            plt.plot(time, adversarial_traj[:, 1], label = name if i == 0 else None, color = color)
    plt.legend()
    save_plot(timestamp_name, plot_name)

    plot_name = "adversarial_all_u.png"
    plt.figure()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.ylim(-10.0, 10.0)
    for time, adversarial_inputs, name, color in zip(times_list, adversarial_inputs_list, names, colors):
        for i, adversarial_input in enumerate(adversarial_inputs):
            plt.plot(time[:-1], adversarial_input, label = name if i == 0  else None, color = color)
    plt.legend()
    save_plot(timestamp_name, plot_name)

def plot_streamplot(
    vector_field_args, 
    adversarial_trajs, 
    nominal_traj, 
    prefix: str, 
    timestamp_name: str
):
    plot_name = prefix + "_phase_portrait_w_adversarial.png"
    plt.figure()
    plt.title(f"Phase portrait for {prefix}")
    plt.xlabel("Position (x1)")
    plt.ylabel("Velocity (x2)")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 3.5)
    plt.streamplot(*vector_field_args, zorder = 1)
    for i, adversarial_traj in enumerate(adversarial_trajs):
        plt.plot(adversarial_traj[:, 0], adversarial_traj[:, 1], color = "orange", zorder = 2, label = "Adversarial trajectory" if i == 0 else None)
    plt.plot(nominal_traj[:, 0], nominal_traj[:, 1], linestyle= "--", color = "red", linewidth = 5, label = "Nominal trajectory", zorder = 3)
    plt.scatter(nominal_traj[0, 0], nominal_traj[0, 1], marker = "*", s = 300, color = "black", label = "Initial State", zorder = 4)
    plt.scatter(nominal_traj[-1, 0], nominal_traj[-1, 1], marker = "*", s = 300, color = "blue", label = "Final State", zorder = 4)
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

    plot_name = prefix + "_phase_portrait.png"
    plt.figure()
    plt.title(f"Phase portrait for {prefix}")
    plt.xlabel("Position (x1)")
    plt.ylabel("Velocity (x2)")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 3.5)
    plt.streamplot(*vector_field_args, zorder = 1)
    plt.plot(nominal_traj[:, 0], nominal_traj[:, 1], linestyle= "--", color = "red", linewidth = 5, label = "Nominal trajectory", zorder = 2)
    plt.scatter(nominal_traj[0, 0], nominal_traj[0, 1], marker = "*", s = 300, color = "black", label = "Initial State", zorder = 4)
    plt.scatter(nominal_traj[-1, 0], nominal_traj[-1, 1], marker = "*", s = 300, color = "blue", label = "Final State", zorder = 4)
    plt.legend()
    save_plot(timestamp_name, plot_name)
    plt.close()

def subplot_streamplot_adversarial(
    node_vector_field_args,
    node_adversarial_trajs,
    node_nominal_traj,
    lya_vector_field_args,
    lya_adversarial_trajs,
    lya_nominal_traj,
    timestamp_name: str
):
    num_subplots = 2
    fig, axs = plt.subplots(1, num_subplots, figsize = (16, 6))
    subplot_name_list = ["NODEC", "L-NODEC"]
    vector_field_args_list = [node_vector_field_args, lya_vector_field_args]
    nominal_traj_list = [node_nominal_traj, lya_nominal_traj]
    adversarial_trajs_list = [node_adversarial_trajs, lya_adversarial_trajs]

    plot_name = "phase_portrait_subplots_adversarial.png"
    for i in range(num_subplots):
        axs[i].set_xlabel("Position (m)")
        axs[i].set_ylabel("Velocity (m/s)")
        axs[i].set_xlim(-0.5, 1.5)
        axs[i].set_ylim(-0.5, 3.5)
        axs[i].streamplot(*vector_field_args_list[i], zorder = 1)
        for j, adversarial_traj in enumerate(adversarial_trajs_list[i]):
            axs[i].plot(
                adversarial_traj[:, 0], 
                adversarial_traj[:, 1], 
                color = "orange", 
                zorder = 2, 
                label = "Adversarial trajectory" if j == 0 else None
            )
        axs[i].plot(
            nominal_traj_list[i][:, 0], 
            nominal_traj_list[i][:, 1], 
            linestyle= "--", 
            color = "red", 
            linewidth = 5, 
            label = "Nominal trajectory", 
            zorder = 3
        )
        axs[i].scatter(
            nominal_traj_list[i][0, 0], 
            nominal_traj_list[i][0, 1], 
            marker = "*", 
            s = 300, 
            color = "black", 
            label = "Initial State", 
            zorder = 4
        )
        axs[i].scatter(
            nominal_traj_list[i][-1, 0], 
            nominal_traj_list[i][-1, 1], 
            marker = "*", 
            s = 300, 
            color = "blue", 
            label = "Final State", 
            zorder = 4
        )
        axs[i].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    save_plot(timestamp_name, plot_name)

def plot_mean_std(time, data_mean, data_std, index, name, timestamp_name):
    data_mean_copy = data_mean[:, index]
    data_std_copy = data_std[:, index]
    plt.figure()
    plt.title(f"{name}_{index} with 3 std devs")
    plt.fill_between(time, data_mean_copy - 3 * data_std_copy, data_mean_copy + 3 * data_std_copy, alpha = 0.2)
    plt.plot(time, data_mean_copy, label = name)
    plt.xlabel("Time")
    if index == 0:
        plt.ylabel("T (x1)")
    elif index == 1:
        plt.ylabel("CEM (x2)")
    plt.legend()
    save_plot(timestamp_name, name + f"_{index}.png")

def plot_all(time, data, index, name, timestamp_name):
    data_copy = data[..., index]
    plt.figure()
    plt.title(f"{name}_{index}")
    for traj in data_copy:
        plt.plot(time, traj)
    plt.xlabel("Time (s)")
    if index == 0:
        plt.ylabel("T (C) (x1)")
    elif index == 1:
        plt.ylabel("CEM (min) (x2)")
    save_plot(timestamp_name, name + f"_{index}.png")

def save_plot(timestamp_name: str, plot_name: str):
    save_path = '.\\plots\\'
    save_path_w_timestamp = save_path + timestamp_name
    if not os.path.exists(save_path_w_timestamp):
        os.makedirs(save_path_w_timestamp)
    save_name = save_path_w_timestamp + "\\" + plot_name
    plt.savefig(save_name)
    plt.close()

def save_data(timestamp_name: str, data_dict: Dict):
    save_path = '.\\data\\'
    save_path_w_timestamp = save_path + timestamp_name
    if not os.path.exists(save_path_w_timestamp):
        os.makedirs(save_path_w_timestamp)
    for data_name, data in data_dict.items():
        save_name = save_path_w_timestamp + "\\" + data_name
        np.save(save_name, data)
    print("Data saved successfully")
    
    