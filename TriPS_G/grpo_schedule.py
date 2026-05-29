# grpo_schedule.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

def _design_matrix(s: torch.Tensor, degree: int) -> torch.Tensor:
    # s: [N] in [0,1]
    cols = [torch.ones_like(s)]
    for k in range(1, degree + 1):
        cols.append(s ** k)
    return torch.stack(cols, dim=-1)  # [N, degree+1]

# --------------------------
# 1) Bernstein design matrix
# --------------------------
def _bernstein_design_matrix(s: torch.Tensor, degree: int) -> torch.Tensor:
    """
    s: [N] in [0,1]
    returns: [N, degree+1] with Bernstein basis columns
    """
    n = degree
    k = torch.arange(0, n + 1, device=s.device, dtype=s.dtype)  # [n+1]

    # binom(n,k) using lgamma for stability
    lg = torch.lgamma(torch.tensor(n + 1.0, device=s.device, dtype=s.dtype)) \
        - torch.lgamma(k + 1.0) - torch.lgamma(torch.tensor(n, device=s.device, dtype=s.dtype) - k + 1.0)
    binom = torch.exp(lg)  # [n+1]

    # [N, n+1]
    S = s[:, None]  # [N,1]
    cols = binom[None, :] * (S ** k[None, :]) * ((1.0 - S) ** (n - k)[None, :])
    return cols

def _logit(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = u.clamp(eps, 1 - eps)
    return torch.log(u) - torch.log1p(-u)

def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    simple linear interpolation: xp must be increasing
    x:  [...], xp:[N], fp:[N]
    """
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x)
    idx = idx.clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)

# --------------------------
# 2) ref schedule -> Bernstein coeff (0..1)
# --------------------------
def poly_fit_ref_schedule(
    ref: torch.Tensor,
    degree: int,
    kind: str,
    cfg_min: float = 1.0,
    cfg_max: float = 8.0,
    step_min: float = 10.0,
    step_max: float = 400.0,
    max_iter: int = 200,
    reg: float = 1e-8,         
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ref -> u(s) in [0,1] -> solve min_{c in (0,1)} ||X c - u||^2
    with parametrization c = sigmoid(w).
    Returns coeff c in (0,1), shape [degree+1]
    """
    N = ref.numel()
    if degree + 1 > N:
        degree = N - 1

    device = ref.device

    # double precision for stability
    s = torch.linspace(0, 1, N, device=device, dtype=torch.float64)
    X = _bernstein_design_matrix(s, degree).to(dtype=torch.float64)  # [N, d+1]

    ref_f = ref.detach().to(dtype=torch.float64)

    if kind == "cfg":
        u = (ref_f - cfg_min) / (cfg_max - cfg_min)
    elif kind == "step":
        u = (ref_f - step_min) / (step_max - step_min)
    elif kind == "eta":
        u = ref_f
    else:
        raise ValueError(f"Unknown kind={kind}")

    u = u.clamp(0.0, 1.0)

    # --- good init: Bernstein approximation control points c0[k] ≈ u(k/degree)
    knots = torch.linspace(0, 1, degree + 1, device=device, dtype=torch.float64)
    c0 = _interp1d(knots, s, u).clamp(eps, 1 - eps)  # [d+1]
    w = nn.Parameter(_logit(c0, eps=eps))            # unconstrained

    opt = torch.optim.LBFGS([w], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        c = torch.sigmoid(w)               # (0,1)
        pred = X @ c                       # [N]
        loss = ((pred - u) ** 2).mean() + reg * (w ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)

    c = torch.sigmoid(w.detach()).to(dtype=torch.float32)  # return float32 coeff
    return c

# --------------------------
# 3) Bernstein coeff -> schedule
# --------------------------
def coeff_to_schedule(
    coeff: torch.Tensor,
    NFE: int,
    kind: str,
    cfg_min: float = 1.0,
    cfg_max: float = 8.0,
    step_min: float = 10.0,
    step_max: float = 400.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or coeff.device
    s = torch.linspace(0, 1, NFE, device=device, dtype=torch.float32)
    X = _bernstein_design_matrix(s, coeff.numel() - 1)  # [NFE, d+1]
    u = (X @ coeff.float()).clamp(0.0, 1.0)             # schedule in [0,1]

    if kind == "cfg":
        return cfg_min + (cfg_max - cfg_min) * u
    elif kind == "step":
        return step_min + (step_max - step_min) * u
    elif kind == "eta":
        return u
    else:
        raise ValueError(f"Unknown kind={kind}")

@dataclass
class ScheduleBounds:
    cfg_min: float = 1.0
    cfg_max: float = 8.0
    step_min: float = 10.0
    step_max: float = 400.0

def split_phi(phi: torch.Tensor, d_cfg: int, d_step: int, d_eta: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # degrees here are coeff counts (degree+1)
    cfg = phi[..., :d_cfg]
    step = phi[..., d_cfg:d_cfg + d_step]
    eta = phi[..., d_cfg + d_step:d_cfg + d_step + d_eta]
    return cfg, step, eta


class DiagGaussianPolicy(nn.Module):
    """
    Episode-level policy over coefficient vector phi.
    phi = [cfg_coeffs, step_coeffs, eta_coeffs]
    """
    def __init__(self, init_mu: torch.Tensor, init_log_std: torch.Tensor):
        super().__init__()
        assert init_mu.shape == init_log_std.shape
        self.mu = nn.Parameter(init_mu.clone())
        self.log_std = nn.Parameter(init_log_std.clone())

    @torch.no_grad()
    def sample(self, G: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          phi: [G, D]
          logp: [G]
        """
        std = self.log_std.exp()
        eps = torch.randn((G, self.mu.numel()), device=self.mu.device, dtype=self.mu.dtype)
        phi = self.mu[None, :] + std[None, :] * eps
        logp = self.log_prob(phi)
        return phi, logp

    def log_prob(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: [G, D]
        std = self.log_std.exp()
        var = std * std
        # log N(phi | mu, var)
        logp = -0.5 * (((phi - self.mu[None, :]) ** 2) / var[None, :] + 2 * self.log_std[None, :] + torch.log(torch.tensor(2.0 * torch.pi, device=phi.device))).sum(dim=-1)
        return logp

    def kl_to(self, ref_mu: torch.Tensor, ref_log_std: torch.Tensor) -> torch.Tensor:
        """
        KL( N(mu, std) || N(ref_mu, ref_std) ), scalar
        """
        std = self.log_std.exp()
        ref_std = ref_log_std.exp()
        var = std * std
        ref_var = ref_std * ref_std
        kl = (ref_log_std - self.log_std) + (var + (self.mu - ref_mu) ** 2) / (2.0 * ref_var) - 0.5
        return kl.sum()

# --------------------------
# 4) Beta policy (DiagGaussianPolicy replace)
# --------------------------
class DiagBetaPolicy(nn.Module):
    """
    Episode-level policy over coefficient vector phi in (0,1).
    phi = [cfg_coeffs, step_coeffs, eta_coeffs] each coeff in (0,1)
    """
    def __init__(self, init_alpha: torch.Tensor, init_beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        assert init_alpha.shape == init_beta.shape
        self.log_alpha = nn.Parameter(torch.log(init_alpha.clamp_min(eps)))
        self.log_beta  = nn.Parameter(torch.log(init_beta.clamp_min(eps)))
        self.eps = eps

    def _dist(self):
        alpha = self.log_alpha.exp().clamp_min(self.eps)
        beta  = self.log_beta.exp().clamp_min(self.eps)
        return torch.distributions.Beta(alpha, beta)

    @torch.no_grad()
    def sample(self, G: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist()
        phi = dist.sample((G,))                         # [G, D]
        phi = phi.clamp(self.eps, 1.0 - self.eps)
        logp = self.log_prob(phi)
        return phi, logp

    def log_prob(self, phi: torch.Tensor) -> torch.Tensor:
        dist = self._dist()
        phi = phi.clamp(self.eps, 1.0 - self.eps)
        return dist.log_prob(phi).sum(dim=-1)           # [G] or [B*G]

    def kl_to(self, ref_alpha: torch.Tensor, ref_beta: torch.Tensor) -> torch.Tensor:
        p = self._dist()
        q = torch.distributions.Beta(ref_alpha.to(p.concentration1.device),
                                     ref_beta.to(p.concentration1.device))
        kl = torch.distributions.kl.kl_divergence(p, q).sum()
        return kl