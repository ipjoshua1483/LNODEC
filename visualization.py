import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict, Any
from cycler import cycler

# =========================
# ML-conference style (sans-serif + tab10)
# =========================
PLOT_RC_PARAMS = {
    # resolution
    "figure.dpi": 300,
    "savefig.dpi": 300,

    # fonts (sans-serif is typical for ML confs)
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "stixsans",
    "axes.unicode_minus": False,

    # sizes (slightly larger)
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    # lines
    "lines.linewidth": 2.2,
    "lines.markersize": 6,

    # axes + grid
    "axes.linewidth": 1.25,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.8,

    # standard ML-conf color cycle (Tableau/tab10)
    "axes.prop_cycle": cycler("color", plt.get_cmap("tab10").colors),
}

# Color scheme for method comparison (using tab10 colors)
COLORS = {
    'primary': 'C0',      # blue - for L-NODEC
    'success': 'C2',      # green - for L-NODEC-Constrained
    'secondary': 'C1',    # orange - for NODEC-Stage
    'danger': 'C3',       # red - for NODEC-Terminal
}


def set_plot_style():
    """Apply ML-conference plotting style (sans-serif + tab10)."""
    plt.rcParams.update(PLOT_RC_PARAMS)


# =========================
# Small helpers
# =========================
def _cycle_colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _compact(ax):
    """Light grid and tidy defaults per-axis."""
    ax.grid(True, alpha=0.35)


# =========================
# Plots
# =========================
def plot_phase_portrait(
    ax,
    trajectories: List[np.ndarray],
    initial_states: np.ndarray,
    final_states: np.ndarray,  # kept for API compatibility (unused)
    target_state: np.ndarray,
    vector_field_fn=None,
    title: str = "Phase Portrait",
    x1_lim: Tuple[float, float] = (-0.5, 1.5),
    x2_lim: Tuple[float, float] = (-0.5, 3.5),
):
    """
    Plot phase portrait with trajectories.
    Args:
        ax: Matplotlib axis
        trajectories: List of [T, 2] arrays
        initial_states: [N, 2]
        final_states: [N, 2] (unused here)
        target_state: [2]
        vector_field_fn: Optional (X1, X2) -> (U1, U2)
    """
    # Vector field (streamlines) colored by speed
    if vector_field_fn is not None:
        x1_grid = np.linspace(x1_lim[0], x1_lim[1], 20)
        x2_grid = np.linspace(x2_lim[0], x2_lim[1], 30)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        U1, U2 = vector_field_fn(X1, X2)
        # speed = np.sqrt(U1**2 + U2**2)
        streams = ax.streamplot(
            X1, X2, U1, U2,
            color= 'blue', # speed,
            cmap="viridis",
            linewidth=1.0,
            density=1.0,
            arrowsize=1.0,
            # alpha = 0.5,
        )
        plt.setp(streams.lines, alpha = 0.35)

    # colors = _cycle_colors()

    # Plot all trajectories (except nominal) faint
    for i, traj in enumerate(trajectories):
        # c = colors[i % len(colors)]
        if i == 0:
            continue
        ax.plot(
            traj[:, 0], traj[:, 1], 
            color="orange", 
            alpha=0.5, 
            linewidth=2.0,
            label = "Adversarial trajectories" if i == 1 else None,
        )

    # Nominal (first) highlighted
    if len(trajectories) > 0:
        # c0 = colors[0]
        ax.plot(
            trajectories[0][:, 0],
            trajectories[0][:, 1],
            color='black',# c0,
            linestyle="--",
            linewidth=2.6,
            label="Nominal trajectory",
            zorder=4,
        )

    # Initial states (neutral)
    if initial_states is not None and len(initial_states) > 0:
        ax.scatter(
            initial_states[:, 0],
            initial_states[:, 1],
            marker="o",
            s=36,
            color="black",
            alpha=0.65,
            linewidths=0,
            label="Nominal initial state",
            zorder=5,
        )

    # Target
    ax.scatter(
        target_state[0],
        target_state[1],
        marker="*",
        s=200,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        label="Target",
        zorder=6,
    )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    ax.set_title(title)
    ax.legend(loc="upper left")
    # _compact(ax)
    #ax.set_aspect("equal", adjustable="box")
    ax.set_aspect("auto")



def plot_state_trajectories(
    ax,
    time_grid: np.ndarray,
    trajectories: List[np.ndarray],
    target_position: float,
    ylabel: str = "Position (m)",
    title: str = "State Trajectories",
):
    """
    Plot x1(t) for multiple trajectories; highlight first as nominal.
    """
    colors = _cycle_colors()
    for i, traj in enumerate(trajectories):
        c = colors[i % len(colors)]
        if i == 0:
            continue
        ax.plot(time_grid, traj[:, 0], color=c, alpha=0.35, linewidth=1.9)

    if len(trajectories) > 0:
        c0 = colors[0]
        ax.plot(
            time_grid,
            trajectories[0][:, 0],
            color=c0,
            linestyle="--",
            linewidth=2.6,
            label="Nominal",
        )

    # Target line (useful for quick read)
    ax.axhline(target_position, color="black", linestyle=":", linewidth=1.8, label="Target")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _compact(ax)


def plot_controls(
    ax,
    time_grid: np.ndarray,
    controls_list: List[np.ndarray],
    u_min: float,
    u_max: float,
    title: str = "Control Inputs",
):
    """
    Plot u(t); highlight first as nominal; show bounds band.
    """
    colors = _cycle_colors()
    for i, controls in enumerate(controls_list):
        c = colors[i % len(colors)]
        if i == 0:
            continue
        ax.plot(time_grid, controls.squeeze(), color=c, alpha=0.35, linewidth=1.9)

    if len(controls_list) > 0:
        c0 = colors[0]
        ax.plot(
            time_grid,
            controls_list[0].squeeze(),
            color=c0,
            linestyle="--",
            linewidth=2.6,
            label="Nominal",
        )

    # Bounds (subtle band + guide lines)
    ax.fill_between(time_grid, u_min, u_max, alpha=0.05, label="Bounds")
    ax.axhline(u_min, color="gray", linestyle=":", linewidth=1.6)
    ax.axhline(u_max, color="gray", linestyle=":", linewidth=1.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(title)
    ax.legend()
    _compact(ax)
    ax.set_ylim([u_min - 1, u_max + 1])


def plot_lyapunov_decay(
    ax,
    time_grid: np.ndarray,
    V_trajectories: List[np.ndarray],
    kappa: float,
    title: str = "Lyapunov Function Decay",
):
    """
    Plot V(x(t))/V(x(0)) with log scale and e^{-kappa t} bound.
    """
    colors = _cycle_colors()
    eps = 1e-12

    for i, V_traj in enumerate(V_trajectories):
        c = colors[i % len(colors)]
        if i == 0:
            continue
        V_norm = np.clip(V_traj / (V_traj[0] + eps), eps, None)
        ax.plot(time_grid, V_norm, color=c, alpha=0.35, linewidth=1.9)

    if len(V_trajectories) > 0:
        c0 = colors[0]
        V_norm0 = np.clip(V_trajectories[0] / (V_trajectories[0][0] + eps), eps, None)
        ax.plot(time_grid, V_norm0, color=c0, linestyle="--", linewidth=2.6, label="Nominal")

    exp_bound = np.exp(-kappa * time_grid)
    ax.plot(time_grid, exp_bound, "k--", linewidth=2.2, label=r"$e^{-\kappa t}$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V(x(t)) / V(x(0))")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    _compact(ax)


def plot_stability_violation(
    ax,
    time_grid: np.ndarray,
    stability_violations: List[np.ndarray],
    first_stable_times: List[float],
    title: str = "Exponential Stability Violation",
):
    """
    Plot max{0, dV/dt + κV} vs time; mark first stable time if available.
    """
    colors = _cycle_colors()

    for i, violation in enumerate(stability_violations):
        c = colors[i % len(colors)]
        if i == 0:
            continue
        ax.plot(time_grid[:-1], violation, color=c, alpha=0.35, linewidth=1.9)
        if first_stable_times[i] is not None:
            ax.axvline(first_stable_times[i], color=c, linestyle=":", linewidth=1.6, alpha=0.7)

    # Nominal
    if len(stability_violations) > 0:
        c0 = colors[0]
        ax.plot(
            time_grid[:-1],
            stability_violations[0],
            color=c0,
            linestyle="--",
            linewidth=2.6,
            label="Nominal",
        )
        if first_stable_times[0] is not None:
            ax.axvline(
                first_stable_times[0],
                color=c0,
                linestyle="-",
                linewidth=2.2,
                label=f"Stable at t={first_stable_times[0]:.3f}s",
            )

    # Zero line (stability threshold)
    ax.axhline(0, color="black", linestyle="-", linewidth=2.0, label="Stability threshold")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"max{0, dV/dt + $\kappa V$}")
    ax.set_title(title)
    ax.legend()
    _compact(ax)
    ax.set_ylim(bottom=-0.1)


def plot_exponential_stability_check(
    ax,
    time_grid: np.ndarray,
    V_decay_trajectories: List[np.ndarray],
    title: str = "Exponential Stability Check",
):
    """
    Plot ratio V(x(t)) / [V(x₀)e^{-κt}] which should be ≤ 1 if exp. stability holds.
    """
    colors = _cycle_colors()

    for i, V_decay in enumerate(V_decay_trajectories):
        c = colors[i % len(colors)]
        if i == 0:
            continue
        ax.plot(time_grid, V_decay, color=c, alpha=0.35, linewidth=1.9)

    if len(V_decay_trajectories) > 0:
        c0 = colors[0]
        ax.plot(
            time_grid,
            V_decay_trajectories[0],
            color=c0,
            linestyle="--",
            linewidth=2.6,
            label="Nominal",
        )

    ax.axhline(1.0, color="black", linestyle="-", linewidth=2.0, label="Stability threshold")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"V(x(t)) / [V(x₀) e^{-\kappa t}]")
    ax.set_title(title)
    ax.legend()
    _compact(ax)
    ax.set_ylim(bottom=0)


def _compute_lyapunov_traj(traj: np.ndarray, target_state: np.ndarray, lyapunov_fn, dtype) -> np.ndarray:
    """
    Compute V(x(t)) along a trajectory. Vectorizes if lyapunov_fn supports batching.
    """
    try:
        X = torch.tensor(traj, dtype=dtype)                   # [T, n_x]
        Z = torch.tensor(target_state, dtype=dtype).expand_as(X)
        V = lyapunov_fn.potential(X, Z)                       # expect [T] if batched
        V_np = V.detach().cpu().numpy() if hasattr(V, "detach") else np.asarray(V)
        return V_np
    except Exception:
        # Fallback: per-step (works for scalar-only potential)
        vals = []
        for x in traj:
            x_t = torch.tensor(x, dtype=dtype)
            z_t = torch.tensor(target_state, dtype=dtype)
            v = lyapunov_fn.potential(x_t, z_t)
            v = v.item() if hasattr(v, "item") else float(v)
            vals.append(v)
        return np.asarray(vals)


def plot_multi_method_comparison(
    time_grid: np.ndarray,
    methods_data: Dict[str, Dict[str, Any]],
    target_state: np.ndarray,
    lyapunov_fn,
    kappa: float,
    x2_max: float = None,
    show_all_trajectories: bool = True,
    figsize: tuple = (14, 3.8),
):
    """
    Compare multiple methods with multiple trajectories each.

    methods_data:
      {
        'method_name': {
            'trajectories': List[[T, n_x]],  # All test trajectories
            # optional: 'color': str
        }, ...
      }
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Infer dtype from Lyapunov function's P matrix if available
    dtype = getattr(getattr(lyapunov_fn, "P", torch.tensor([], dtype=torch.float32)), "dtype", torch.float32)

    colors = _cycle_colors()
    method_names = list(methods_data.keys())
    fallback_colors = {m: colors[i % len(colors)] for i, m in enumerate(method_names)}

    # Panel 1: Position
    for method_name, data in methods_data.items():
        trajs = data["trajectories"]
        color = data.get("color", fallback_colors[method_name])

        if show_all_trajectories:
            for i, traj in enumerate(trajs):
                if i == 0:
                    continue
                axes[0].plot(time_grid, traj[:, 0], color=color, alpha=0.3, linewidth=1.5)
            axes[0].plot(time_grid, trajs[0][:, 0], color=color, linewidth=2.6, label=method_name, zorder=10)
        else:
            positions = np.array([t[:, 0] for t in trajs])
            mean_pos = positions.mean(axis=0)
            std_pos = positions.std(axis=0)
            axes[0].plot(time_grid, mean_pos, color=color, linewidth=2.6, label=method_name)
            axes[0].fill_between(time_grid, mean_pos - std_pos, mean_pos + std_pos, color=color, alpha=0.18)

    axes[0].axhline(target_state[0], color="black", linestyle=":", linewidth=1.8, label="Target")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Position Comparison")
    axes[0].legend(fontsize=10)
    _compact(axes[0])

    # Panel 2: Velocity
    for method_name, data in methods_data.items():
        trajs = data["trajectories"]
        color = data.get("color", fallback_colors[method_name])

        if show_all_trajectories:
            for i, traj in enumerate(trajs):
                if i == 0:
                    continue
                axes[1].plot(time_grid, traj[:, 1], color=color, alpha=0.3, linewidth=1.5)
            axes[1].plot(time_grid, trajs[0][:, 1], color=color, linewidth=2.6, label=method_name, zorder=10)
        else:
            velocities = np.array([t[:, 1] for t in trajs])
            mean_vel = velocities.mean(axis=0)
            std_vel = velocities.std(axis=0)
            axes[1].plot(time_grid, mean_vel, color=color, linewidth=2.6, label=method_name)
            axes[1].fill_between(time_grid, mean_vel - std_vel, mean_vel + std_vel, color=color, alpha=0.18)

    if x2_max is not None:
        axes[1].axhline(x2_max, color="red", linestyle="--", linewidth=2.0, label=f"Constraint: x₂≤{x2_max}", alpha=0.9)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1.8, label="Target")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity Comparison")
    axes[1].legend(fontsize=10)
    _compact(axes[1])

    # Panel 3: Lyapunov Decay
    eps = 1e-12
    for method_name, data in methods_data.items():
        trajs = data["trajectories"]
        color = data.get("color", fallback_colors[method_name])

        if show_all_trajectories:
            for i, traj in enumerate(trajs):
                if i == 0:
                    continue
                V_traj = _compute_lyapunov_traj(traj, target_state, lyapunov_fn, dtype)
                V_norm = np.clip(V_traj / (V_traj[0] + eps), eps, None)
                axes[2].plot(time_grid, V_norm, color=color, alpha=0.3, linewidth=1.5)

            V_nom = _compute_lyapunov_traj(trajs[0], target_state, lyapunov_fn, dtype)
            V_nom = np.clip(V_nom / (V_nom[0] + eps), eps, None)
            label = method_name if "L-NODEC" in method_name else f"{method_name} (not optimized)"
            axes[2].plot(time_grid, V_nom, color=color, linewidth=2.6, label=label, zorder=10)
        else:
            V_trajs = []
            for traj in trajs:
                V_traj = _compute_lyapunov_traj(traj, target_state, lyapunov_fn, dtype)
                V_norm = np.clip(V_traj / (V_traj[0] + eps), eps, None)
                V_trajs.append(V_norm)
            V_trajs = np.array(V_trajs)
            mean_V = V_trajs.mean(axis=0)
            std_V = V_trajs.std(axis=0)
            label = method_name if "L-NODEC" in method_name else f"{method_name} (not optimized)"
            axes[2].plot(time_grid, mean_V, color=color, linewidth=2.6, label=label)
            axes[2].fill_between(time_grid, mean_V - std_V, mean_V + std_V, color=color, alpha=0.18)

    exp_bound = np.exp(-kappa * time_grid)
    axes[2].plot(time_grid, exp_bound, "k--", linewidth=2.2, label=r"$e^{-\kappa t}$")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("V(x(t)) / V(x(0))")
    axes[2].set_title("Lyapunov Decay")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=9)
    _compact(axes[2])

    return fig


def plot_method_comparison(
    time_grid: np.ndarray,
    trajectories_dict: Dict[str, np.ndarray],
    target_state: np.ndarray,
    lyapunov_fn,
    kappa: float,
    figsize: tuple = (12.5, 4.2),
):
    """
    Side-by-side comparison (single trajectory per method):
      Left: Position
      Right: Lyapunov decay (computed with provided lyapunov_fn for all)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # dtype inference
    dtype = getattr(getattr(lyapunov_fn, "P", torch.tensor([], dtype=torch.float32)), "dtype", torch.float32)

    colors = _cycle_colors()
    method_list = list(trajectories_dict.keys())
    method_colors = {name: colors[i % len(colors)] for i, name in enumerate(method_list)}

    # Left: Position
    for method_name, traj in trajectories_dict.items():
        axes[0].plot(
            time_grid,
            traj[:, 0],
            color=method_colors.get(method_name, "black"),
            linewidth=2.6,
            label=method_name,
        )

    axes[0].axhline(
        target_state[0],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label="Target",
        alpha=0.9,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Position Comparison")
    axes[0].legend()
    _compact(axes[0])

    # Right: Lyapunov Decay
    eps = 1e-12
    for method_name, traj in trajectories_dict.items():
        V_traj = _compute_lyapunov_traj(traj, target_state, lyapunov_fn, dtype)
        V_norm = np.clip(V_traj / (V_traj[0] + eps), eps, None)
        label = f"{method_name} (not optimized for this)" if "NODEC" in method_name and "L-NODEC" not in method_name else method_name
        axes[1].plot(
            time_grid,
            V_norm,
            color=method_colors.get(method_name, "black"),
            linewidth=2.6,
            label=label,
        )

    # Exponential bound
    exp_bound = np.exp(-kappa * time_grid)
    axes[1].plot(
        time_grid,
        exp_bound,
        color="black",
        linestyle="--",
        linewidth=2.2,
        label=r"$e^{-\kappa t}$",
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("V(x(t)) / V(x(0))")
    axes[1].set_title("Lyapunov Decay Comparison")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    _compact(axes[1])

    return fig
