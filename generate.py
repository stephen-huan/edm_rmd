# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import dawsn
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

# combined forward pass from training/networks.py

def model_forward(self, x, sigma, class_labels=None, force_fp32=False, no_skip=False, **model_kwargs):
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
    dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

    net_name = type(next(self.modules())).__name__
    c_skip, c_out, c_in, c_noise = {
        "VPPrecond": lambda self: (
            1,
            -sigma,
            1 / (sigma ** 2 + 1).sqrt(),
            (self.M - 1) * self.sigma_inv(sigma),
        ),
        "VEPrecond": lambda self: (
            1,
            sigma,
            1,
            (0.5 * sigma).log(),
        ),
        "iDDPMPrecond": lambda self: (
            1,
            -sigma,
            1 / (sigma ** 2 + 1).sqrt(),
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32),
        ),
        "EDMPrecond": lambda self: (
            self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2),
            sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt(),
            1 / (self.sigma_data ** 2 + sigma ** 2).sqrt(),
            sigma.log() / 4,
        ),
    }[net_name](self)
    if no_skip:
        c_skip = 0

    F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
    assert F_x.dtype == dtype
    D_x = c_skip * x + c_out * F_x.to(torch.float32)
    return D_x


def quadrature(f):
    """Definite integral of f."""

    def integrate(f, t, t_prime):
        """Integral of f from t to t_prime."""
        return quad(f, t, t_prime)[0]

    def F(t, t_prime):
        t = t.item() if isinstance(t, torch.Tensor) else t
        if isinstance(t_prime, float) or t_prime.ndim == 0:
            t_prime = t_prime.item() if isinstance(t_prime, torch.Tensor) else t_prime
            return integrate(f, t, t_prime)
        else:
            t_primep = t_prime.cpu().numpy().flatten()
            # nasty broadcasting bug
            return torch.tensor(
                [integrate(f, t, s) for s in t_primep],
                dtype=torch.float64,
                device=t_prime.device,
            ).reshape(t_prime.shape)

    return F


def inverse_sample(int_int_factor):
    """Sample from the density proportional to int_factor in [t, t + h]."""

    def inverse(t, h, y):
        """Return x such that int_int_factor(t, x) = y."""
        return root_scalar(
            lambda x: int_int_factor(t, x) - y,
            bracket=[t, t + h],
            method="brentq",
        ).root

    def sample(t, h, u):
        tp = t.cpu().numpy()
        hp = h.cpu().numpy()
        up = u.cpu().numpy()
        return torch.tensor(
            [inverse(tp, hp, u * int_int_factor(tp, tp + hp)) for u in up],
            dtype=torch.float64,
            device=u.device,
        ).reshape(u.shape)

    return sample


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None,
    rand=torch.rand, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None,
    rand=torch.rand, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    importance_sample=True, handle_skip=True, check_skip=False,
):
    assert solver in ['euler', 'heun', 'midpoint']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['ou', 'vp', 've', 'linear']
    assert scaling in ['ou', 'vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'ou':
        sigma = lambda t: torch.sqrt(torch.expm1(2 * t))
        sigma_deriv = lambda t: sigma(t) + 1 / sigma(t)
        sigma_inv = lambda sigma: torch.log1p(sigma**2) / 2
    elif schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'ou':
        s = lambda t: torch.exp(-t)
        s_deriv = lambda t: -s(t)
    elif scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    weight_x = lambda t, t_prime: int_factor(t) / int_factor(t_prime)
    weight_f = lambda t, t_prime: (int_int_factor(t_prime) - int_int_factor(t)) / int_factor(t_prime)
    weight_z = lambda t, t_prime: torch.sqrt((noise_factor(t_prime) - noise_factor(t)).clamp(min=0)) / int_factor(t_prime)

    net_name = type(next(net.modules())).__name__
    networks = ["VPPrecond", "VEPrecond", "iDDPMPrecond", "EDMPrecond"]
    assert net_name in networks, f"unknown network {net_name}."
    is_edm = net_name == "EDMPrecond"
    c_skip = (
        (lambda t: net.sigma_data**2 / (sigma(t) ** 2 + net.sigma_data**2))
        if is_edm
        else (lambda t: 1)
    )

    beta_rate = S_churn / (sigma_max - sigma_min)
    noise_term = lambda t: 0
    noise_factor = lambda t: 2 * beta_rate * noise_term(torch.clamp(t, S_min, S_max))

    if schedule == 'linear' and not handle_skip:
        assert scaling == 'none', f"scaling {scaling} not supported for {schedule} schedule."
        linear_term = lambda t: 0
        # scale_t(t) = sigma_deriv(t) / sigma(t) + s_deriv(t) / s(t)
        scale_t = lambda t: 1 / t
        # int_factor(t) = exp(int -scale_t(t) dt)
        int_factor = lambda t: 1 / t
        # int_int_factor(t) = int int_factor(t) dt
        int_int_factor = lambda t: torch.log(t)
        # sample_t(t, h, u) = int_int^{-1}((1 - u) * int_int(t) + u * int_int(t + h))
        sample_t = lambda t, h, u: t * torch.pow(1 + h / t, u)
        # optional, for numerical stability
        weight_x = lambda t, t_prime: t_prime / t
        weight_f = lambda t, t_prime: t_prime * (torch.log(t_prime) - torch.log(t))
    elif schedule == 'linear' and handle_skip:
        assert scaling == 'none', f"scaling {scaling} not supported for {schedule} schedule."
        assert is_edm, "non-edm preconditioning not supported."
        # remove internal skip connection in net
        linear_term = lambda t: c_skip(t) * sigma_deriv(t) / sigma(t)
        # scale_t(t) = (1 - c_skip(t)) sigma_deriv(t) / sigma(t) + s_deriv(t) / s(t)
        scale_t = lambda t: t / (t**2 + net.sigma_data**2)
        int_factor = lambda t: 1 / torch.sqrt(t**2 + net.sigma_data**2)
        int_int_factor = lambda t: torch.atanh(t / torch.sqrt(t**2 + net.sigma_data**2))
        sample_t = (
            lambda t, h, u: net.sigma_data
            * torch.tanh(x := (1 - u) * int_int_factor(t) + u * int_int_factor(t + h))
            * torch.cosh(x)
        )
        noise_term = lambda t: t - net.sigma_data * torch.atan(t / net.sigma_data)
    elif schedule == 've' and not handle_skip:
        assert scaling == 'none', f"scaling {scaling} not supported for {schedule} schedule."
        assert is_edm, "non-edm preconditioning not supported."
        linear_term = lambda t: 0
        scale_t = lambda t: 1 / (2 * t)
        int_factor = lambda t: 1 / torch.sqrt(t)
        int_int_factor = lambda t: 2 * torch.sqrt(t)
        sample_t = lambda t, h, u: torch.square(
            (1 - u) * torch.sqrt(t) + u * torch.sqrt(t + h)
        )
    elif schedule == 've' and handle_skip:
        assert scaling == 'none', f"scaling {scaling} not supported for {schedule} schedule."
        linear_term = lambda t: c_skip(t) * sigma_deriv(t) / sigma(t)
        scale_t = lambda t: 1 / (2 * (t + net.sigma_data**2))
        int_factor = lambda t: 1 / torch.sqrt(t + net.sigma_data**2)
        int_int_factor = lambda t: 2 * torch.sqrt(t + net.sigma_data**2)
        sample_t = (
            lambda t, h, u: torch.square(
                (1 - u) * torch.sqrt(t + net.sigma_data**2)
                + u * torch.sqrt(t + h + net.sigma_data**2)
            )
            - net.sigma_data**2
        )
    elif schedule == 'vp' and not handle_skip:
        raise ValueError(f"cannot ignore skip connection for {schedule} schedule.")
    elif schedule == 'vp' and handle_skip and is_edm:
        assert scaling == 'vp', f"scaling {scaling} not supported for {schedule} schedule."
        linear_term = lambda t: c_skip(t) * sigma_deriv(t) / sigma(t)
        scale_t = (
            lambda t: (1 - net.sigma_data**2)
            * (vp_beta_d * t + vp_beta_min)
            / (2 * (sigma(t) ** 2 + net.sigma_data**2))
        )
        int_factor = lambda t: torch.sqrt(
            (sigma(t) ** 2 + 1) / (sigma(t) ** 2 + net.sigma_data**2)
        )
        int_factor_np = lambda t: np.sqrt(
            (sigma(t) ** 2 + 1) / (sigma(t) ** 2 + net.sigma_data**2)
        )
        int_int_diff = quadrature(int_factor_np)
        sample_t = inverse_sample(int_int_diff)
        weight_f = lambda t, t_prime: int_int_diff(t, t_prime) / int_factor(t_prime)
    elif schedule == 'vp' and handle_skip and not is_edm:
        assert scaling == 'vp', f"scaling {scaling} not supported for {schedule} schedule."
        linear_term = lambda t: c_skip(t) * sigma_deriv(t) / sigma(t)
        scale_t = lambda t: -(vp_beta_d * t + vp_beta_min) / 2
        int_factor = lambda t: torch.sqrt(1 + sigma(t) ** 2)
        int_int_factor_np = lambda t: (
            2
            * np.sqrt(1 + sigma(t) ** 2)
            * dawsn((vp_beta_d * t + vp_beta_min) / (2 * np.sqrt(vp_beta_d)))
            / np.sqrt(vp_beta_d)
        )
        int_int_factor = lambda t: torch.as_tensor(
            int_int_factor_np(t.cpu().numpy()),
            dtype=torch.float64,
            device=t.device,
        )
        int_int_diff = lambda t, t_prime: int_int_factor_np(t_prime) - int_int_factor_np(t)
        sample_t = inverse_sample(int_int_diff)
    else:
        # generic implementation (exponential integrator)
        # subtract 1 to remove the exponentially integrated x
        linear_term = lambda t: sigma_deriv(t) / sigma(t) + s_deriv(t) / s(t) - 1
        scale_t = lambda t: 1
        int_factor = lambda t: torch.exp(-t)
        int_int_factor = lambda t: -torch.exp(-t)
        # numerically robust implementation of log1p(u * expm1(-h))
        sample_t = lambda t, h, u: t - torch.logaddexp(torch.log1p(-u), torch.log(u) - h)
        weight_x = lambda t, t_prime: torch.exp(t_prime - t)
        weight_f = lambda t, t_prime: weight_x(t, t_prime) - 1
        noise_term = lambda t: torch.exp(-2 * t) / 2
        raise ValueError(f"unsupported combination schedule {schedule} scaling {scaling}.")

    if importance_sample:
        importance_weight = lambda t, t_mid, t_prime: weight_f(t, t_prime)
    else:
        # uniform sampling
        importance_weight = lambda t, t_mid, t_prime: (t_prime - t) * weight_x(t_mid, t_prime)
        sample_t = lambda t, h, u: t + h * u

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if solver != 'midpoint':
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        # noise in the method later
        else:
            gamma = 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction or randomized midpoint.
        if solver == 'euler' or (i == num_steps - 1 and solver == 'heun'):
            x_next = x_hat + h * d_cur
        elif solver == 'heun':
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        else:
            assert solver == 'midpoint'
            if i == num_steps - 1 and (
                (schedule == "linear" or schedule == "ve") and not handle_skip
            ):
                x_next = x_hat + h * d_cur
                continue
            # Full Shen-Lee randomized midpoint method with exponential weighting
            # From equations 7 and 8 of https://arxiv.org/pdf/2406.00924

            # Sample random value uniformly from [0, 1]
            u = rand((latents.shape[0], 1, 1, 1), dtype=torch.float64, device=latents.device)

            # Compute the randomized midpoint time
            t_mid = sample_t(t_hat, h, u)

            # Step 1: Compute the score from denoised
            g_hat = sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            f_hat = linear_term(t_hat) * x_hat - g_hat

            # Step 2: Compute midpoint using Equation 7
            exp_x_mid = weight_x(t_hat, t_mid)
            exp_f_mid = weight_f(t_hat, t_mid)
            exp_z_mid = weight_z(t_hat, t_mid)
            z_mid = s(t_mid) * S_noise * randn_like(x_hat)
            eta_mid = exp_z_mid * z_mid
            x_mid = exp_x_mid * x_hat + exp_f_mid * f_hat + eta_mid

            # Step 3: Compute denoised_mid and score_mid
            denoised_mid = net(x_mid / s(t_mid), sigma(t_mid), class_labels).to(torch.float64)
            g_mid = sigma_deriv(t_mid) * s(t_mid) / sigma(t_mid) * denoised_mid
            f_mid = linear_term(t_mid) * x_mid - g_mid

            # Step 4: Compute x_next using Equation 8
            exp_x_next = weight_x(t_hat, t_next)
            exp_f_next = importance_weight(t_hat, t_mid, t_next)
            exp_e_next = weight_x(t_mid, t_next)
            exp_z_next = weight_z(t_mid, t_next)
            z_next = s(t_next) * S_noise * randn_like(x_hat)
            eta_next = exp_e_next * eta_mid + exp_z_next * z_next
            x_next = exp_x_next * x_hat + exp_f_next * f_mid + eta_next

            # costly model evaluations, so disabled by default
            if check_skip:
                skip = model_forward(net, x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
                assert torch.allclose(skip, denoised), 'model_forward wrong.'
                no_skip = model_forward(net, x_hat / s(t_hat), sigma(t_hat), class_labels, no_skip=True).to(torch.float64)
                assert torch.allclose(no_skip + c_skip(t_hat) * x_hat / s(t_hat), denoised, atol=1e-6), 'c_skip wrong.'
            assert torch.allclose(scale_t(t_hat) * x_hat + f_hat, d_cur, atol=1e-4), 'scale_t and linear_term wrong.'
            tau = t_mid - t_hat
            assert ((h <= tau) & (tau <= 0)).all(), 't_mid out of valid range.'
            if schedule == 'linear' and not handle_skip:
                assert torch.allclose(exp_x_mid, 1 + tau / t_hat), 'exp_x_mid wrong.'
                assert torch.allclose(exp_f_mid, t_mid * torch.log(exp_x_mid)), 'exp_f_mid wrong.'
                assert torch.allclose(exp_x_next, 1 + h / t_hat), 'exp_x_next wrong.'
                target = t_next * (torch.log(exp_x_next) if importance_sample else h / t_mid)
                assert torch.allclose(exp_f_next, target), 'exp_f_next wrong.'
            elif schedule == 'linear' and handle_skip:
                ...
            elif schedule == 've' and not handle_skip:
                ...
            elif schedule == 've' and handle_skip:
                ...
            elif schedule == 'vp' and not handle_skip:
                ...
            elif schedule == 'vp' and handle_skip:
                ...
            else:
                assert torch.allclose(exp_x_mid, torch.exp(tau)), 'exp_x_mid wrong.'
                assert torch.allclose(exp_f_mid, exp_x_mid - 1), 'exp_f_mid wrong.'
                assert torch.allclose(exp_x_next, torch.exp(h)), 'exp_x_next wrong.'
                target = (exp_x_next - 1) if importance_sample else (h * torch.exp(h - tau))
                assert torch.allclose(exp_f_next, target), 'exp_f_next wrong.'

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def rand(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.rand(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun|midpoint',                 type=click.Choice(['euler', 'heun', 'midpoint']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='ou|vp|ve|linear',           type=click.Choice(['ou', 'vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='ou|vp|none',                    type=click.Choice(['ou', 'vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        images = sampler_fn(net, latents, class_labels, rand=rnd.rand, randn_like=rnd.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0]).save(image_path)
            else:
                PIL.Image.fromarray(image_np).save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
