import torch
import numpy as np


def double_integrator_dynamics(
    x: torch.Tensor, u: torch.Tensor, t: float
) -> torch.Tensor:
    """
    Double integrator dynamics: ẋ₁ = x₂, ẋ₂ = u

    Args:
        x: State [2] or [B, 2] where x = [position, velocity]
        u: Control [1] or [B, 1] where u = acceleration
        t: Time (unused for time-invariant system)

    Returns:
        xdot: State derivative [2] or [B, 2]
    """
    if x.ndim == 1:
        # Single state vector
        x1, x2 = x[0], x[1]
        u_scalar = u.squeeze() if u.ndim > 0 else u

        x_dot_1 = x2
        x_dot_2 = u_scalar

        return torch.stack([x_dot_1, x_dot_2])
    else:
        # Batched states [B, 2]
        x1, x2 = x[..., 0], x[..., 1]
        u_scalar = u.squeeze(-1) if u.ndim > 1 else u

        x_dot_1 = x2
        x_dot_2 = u_scalar

        return torch.stack([x_dot_1, x_dot_2], dim=-1)


def plasma_dynamics(
    x: torch.Tensor, u: torch.Tensor, t: float
) -> torch.Tensor:
    """
    Atmospheric Pressure Plasma Jet (APPJ) dynamics for thermal dose delivery.

    State: x = [T, CEM]
      - T: Temperature (°C)
      - CEM: Cumulative Equivalent Minutes (thermal dose metric)

    Control: u = power input (W)

    Dynamics:
      - Ṫ = (u / Cp) - φ(T)
      - CEṀ = K_cons^(T_ref - T) / 60

    where:
      - Cp: Thermal capacitance
      - φ(T): Heat transfer function
      - K_cons: Thermal dose constant (0.5)
      - T_ref: Reference temperature for thermal dose

    Physical constants from biomedical plasma literature.

    Args:
        x: State [2] or [B, 2] where x = [T (°C), CEM (min)]
        u: Control [1] or [B, 1] where u = power (W)
        t: Time (unused for time-invariant system)

    Returns:
        xdot: State derivative [2] or [B, 2]
    """
    # Physical constants (same as OLDER/appj.py)
    K_cons = 0.5
    T_up = 318.15
    k2cel = 273.15
    T_ref = T_up - k2cel - 2  # 43°C
    T_inf = 298.15 - k2cel     # 25°C (ambient)
    T_b = 308.15 - k2cel       # 35°C (body)
    T_bar = T_up - k2cel       # 45°C (upper limit)

    # Material properties
    ρ = 2800      # Density (kg/m³)
    cp = 795      # Specific heat (J/kg·K)
    r = 1.5e-3    # Radius (m)
    d = 0.2e-3    # Thickness (m)
    k = 1.43      # Thermal conductivity (W/m·K)
    β = 90.82     # Heat transfer coefficient
    μ = 9.84e-4   # Viscosity (Pa·s)

    # Compute thermal capacitance
    Cp = ρ * cp * np.pi * r * r * d / μ

    # Compute heat transfer function numerator
    φ_numerator = (2 * np.pi * r * d * k * β / μ) / Cp * (T_bar - (T_b + T_inf) / 2) * (
        np.log(T_bar - T_inf) - np.log(T_bar - T_b)
    )

    if x.ndim == 1:
        # Single state vector
        T, CEM = x[0], x[1]
        u_scalar = u.squeeze() if u.ndim > 0 else u

        # Temperature dynamics
        T_dot = (u_scalar / Cp) - (φ_numerator / (torch.log(T - T_inf) - torch.log(T - T_b)))

        # CEM (thermal dose) dynamics
        CEM_dot = torch.pow(torch.tensor(K_cons), T_ref - T) / 60.0

        return torch.stack([T_dot, CEM_dot])
    else:
        # Batched states [B, 2]
        T, CEM = x[..., 0], x[..., 1]
        u_scalar = u.squeeze(-1) if u.ndim > 1 else u

        # Temperature dynamics
        T_dot = (u_scalar / Cp) - (φ_numerator / (torch.log(T - T_inf) - torch.log(T - T_b)))

        # CEM (thermal dose) dynamics
        CEM_dot = torch.pow(torch.tensor(K_cons), T_ref - T) / 60.0

        return torch.stack([T_dot, CEM_dot], dim=-1)
