"""Variance Preserving (VP) diffusion for atom types and OT flow for coordinates.

This module implements the generative-process side of pages 15-16 of the NSFC
proposal "考虑靶蛋白活性口袋电场效应的扩散分子生成模型及其应用研究":

  * 公式(6): for atom types, a VP (variance preserving) forward process. With
    t = 0 the base (noise) distribution and t = 1 the data distribution, the
    signal coefficient is

        s(t) = 1 - exp(- integral_0^t beta(u) du)     (s(0) = 0, s(1) ~ 1)

    and the corrupted one-hot type vector x_t follows

        q(x_t | x_1) = N(sqrt(s(t)) * x_1, (1 - s(t)) * I)

    i.e. mean sqrt(s(t)) * x_1 and variance (1 - s(t)) -- the variance of the
    signal-plus-noise combination is preserved (VP). beta(t) is the linear
    noise scale schedule beta_min + (beta_max - beta_min) * t.

  * 公式(7)-(8): differentiating x_t = sqrt(s(t)) * x_1 + sqrt(1 - s(t)) * eps
    and substituting the model estimate eps-hat via the predicted clean sample
    x_1_hat gives the (noise -> data) VP vector field

        u_t(x) = 0.5 * beta(t) * ( x_1_hat / sqrt(s(t)) - x_t )

    which can be integrated to sample atom types from noise to data.

  * 公式(9)-(10): for atom positions, the optimal transport (OT) flow matching
    path x_t = (1 - t) * x_0 + t * x_1 with the constant vector field
    dx/dt = x_1 - x_0. FLOWR's `Integrator.step` in flowr/models/fm_pocket.py
    already integrates exactly this velocity (coord_velocity = (x1_hat - x_t)
    / (1 - t)), and `GeometricInterpolant` produces OT-matched noise pairs;
    `ot_coord_vector_field` is provided here as an explicit reference
    implementation.

All code here is additive and only used when the caller opts in (e.g. by
setting type interpolation to "vp").
"""

import torch

# Linear noise scale schedule beta(t) = beta_min + (beta_max - beta_min) * t
VP_BETA_MIN = 0.05
VP_BETA_MAX = 12.0


def vp_beta(t, beta_min: float = VP_BETA_MIN, beta_max: float = VP_BETA_MAX) -> torch.Tensor:
    """Noise scale function beta(t) (公式(6)).

    Args:
        t: Time in [0, 1] (scalar tensor or float), t = 0 noise, t = 1 data.

    Returns:
        torch.Tensor: beta(t).
    """
    return beta_min + (beta_max - beta_min) * torch.as_tensor(t).float()


def vp_integral_beta(t, beta_min: float = VP_BETA_MIN, beta_max: float = VP_BETA_MAX) -> torch.Tensor:
    """Integral of the noise schedule, integral_0^t beta(u) du."""
    t = torch.as_tensor(t).float()
    return beta_min * t + 0.5 * (beta_max - beta_min) * t * t


def vp_signal_coeff(t, beta_min: float = VP_BETA_MIN, beta_max: float = VP_BETA_MAX) -> torch.Tensor:
    """Signal coefficient s(t) = 1 - exp(-integral_0^t beta).

    s(0) = 0 (pure noise) and s(1) ~ 1 (data), matching FLOWR's convention of
    t = 0 being the base distribution and t = 1 the data distribution.
    """
    return 1.0 - torch.exp(-vp_integral_beta(t, beta_min, beta_max))


def vp_mean_std(t, beta_min: float = VP_BETA_MIN, beta_max: float = VP_BETA_MAX):
    """Mean and std of q(x_t | x_1) for the VP process (公式(6)).

    Returns:
        (torch.Tensor, torch.Tensor): sqrt(s(t)) and sqrt(1 - s(t)).
    """
    s = vp_signal_coeff(t, beta_min, beta_max)
    return torch.sqrt(s.clamp(min=1e-8)), torch.sqrt((1.0 - s).clamp(min=1e-8))


def vp_noisy_types(data_onehot: torch.Tensor, t) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_1) for the VP process on a one-hot type vector (公式(6)).

    Args:
        data_onehot (torch.Tensor): Clean one-hot types, shape [..., n_classes].
        t: Interpolation time in [0, 1], noise at t = 0 and data at t = 1.

    Returns:
        torch.Tensor: VP-noised type vector of the same shape as data_onehot.
    """
    mean, std = vp_mean_std(t)
    eps = torch.randn_like(data_onehot.float())
    return mean * data_onehot.float() + std * eps


def vp_type_vector_field(
    x_t: torch.Tensor, x_1_hat: torch.Tensor, t
) -> torch.Tensor:
    """VP vector field for atom types, noise -> data direction (公式(7)-(8)).

    Given the model's prediction x_1_hat of the clean type vector at time t,
    the marginal velocity of the VP path is

        u_t = 0.5 * beta(t) * ( x_1_hat / sqrt(s(t)) - x_t )

    Integrating x_{t+dt} = x_t + dt * u_t samples types from noise to data.

    Args:
        x_t (torch.Tensor): Current (noised) type vector, shape [..., n_classes].
        x_1_hat (torch.Tensor): Predicted clean type vector, same shape as x_t.
        t: Current time in [0, 1).

    Returns:
        torch.Tensor: Vector field u_t, same shape as x_t.
    """
    s_sqrt, _ = vp_mean_std(t)
    return 0.5 * vp_beta(t) * (x_1_hat.float() / s_sqrt - x_t.float())


def ot_coord_path(x_0: torch.Tensor, x_1: torch.Tensor, t) -> torch.Tensor:
    """OT flow matching interpolation path for atom positions (公式(9)).

    x_t = (1 - t) * x_0 + t * x_1 with a tiny epsilon so the density at
    t = 1 concentrates near x_1.
    """
    t = torch.as_tensor(t).float()
    return (1 - t) * x_0.float() + t * x_1.float()


def ot_coord_vector_field(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """OT flow matching vector field for atom positions (公式(10)).

    The time derivative of the linear OT path is constant, u = x_1 - x_0.
    Note that FLOWR's Integrator.step already integrates this velocity in the
    form (x_1_hat - x_t) / (1 - t).
    """
    return x_1.float() - x_0.float()
