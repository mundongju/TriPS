"""iqa_reward.py

Modern reward helper for GRPO schedule optimization.

This module implements a *fixed-scale* reward composed from:

  - Distortion metrics (need GT): PSNR (+ optional MS-SSIM)
  - Perception metrics:
      * patch-LPIPS (need GT)
      * CLIP-IQA (NR)
      * Q-Align / OneAlign (NR)

Key design points (matching your requirements)
---------------------------------------------
1) **No per-candidate min-max normalization**.
   GRPO already normalizes within-group via advantages.

2) **Distortion vs. Perception** are combined 0.5/0.5 by default.
   (configurable via `ModernRewardConfig.distortion_weight/perception_weight`)

3) **Metric scale mismatch is handled via fixed-range scaling** to ~[0,1]
   before weighted averaging.

4) For high-res images (e.g., 768×768), NR metrics can be computed via
   multi-view crops (center / five-crop) instead of a single global downsample.

Dependencies
------------
* torch, torchvision (already in your project)
* lpips
* torchmetrics (for CLIPImageQualityAssessment)
* transformers (+ accelerate recommended) and pillow (for Q-Align)

No `pyiqa` is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ============================================================================
# Basic FR metrics
# ============================================================================


@torch.no_grad()
def psnr(x01: torch.Tensor, y01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR for tensors in [0, 1]. Returns scalar tensor."""
    mse = torch.mean((x01 - y01) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _patch_starts(L: int, patch: int, stride: int) -> list[int]:
    """Patch start indices so the last patch aligns to the border."""
    if patch > L:
        raise ValueError(f"patch({patch}) must be <= L({L})")
    starts: list[int] = []
    s = 0
    last = L - patch
    while s < last:
        starts.append(s)
        s += stride
    if len(starts) == 0 or starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def patch_lpips_vgg(
    lpips_fn,
    x01: torch.Tensor,
    y01: torch.Tensor,
    patch: int = 224,
    stride: int = 224,
) -> torch.Tensor:
    """Patch-based LPIPS over the full image.

    Args:
        lpips_fn: `lpips.LPIPS(net='vgg')` (already .to(device).eval())
        x01,y01: [1,3,H,W] in [0,1]
    Returns:
        scalar tensor
    """
    if x01.ndim != 4 or y01.ndim != 4 or x01.shape != y01.shape:
        raise ValueError(f"Expected x01,y01 [1,3,H,W] same shape; got {tuple(x01.shape)} vs {tuple(y01.shape)}")
    B, C, H, W = x01.shape
    if B != 1 or C != 3:
        raise ValueError(f"Expected batch=1, channels=3; got B={B}, C={C}")
    if H < patch or W < patch:
        raise ValueError(f"Image too small for patch: H,W=({H},{W}) patch={patch}")

    # LPIPS expects [-1, 1]
    x = x01 * 2 - 1
    y = y01 * 2 - 1

    hs = _patch_starts(H, patch, stride)
    ws = _patch_starts(W, patch, stride)

    vals = []
    for i in hs:
        for j in ws:
            xp = x[:, :, i : i + patch, j : j + patch]
            yp = y[:, :, i : i + patch, j : j + patch]
            v = lpips_fn(xp, yp)
            vals.append(v.reshape(-1)[0])
    return torch.stack(vals).mean()


# ============================================================================
# MS-SSIM (self-contained, torch-only)
# ============================================================================


_GAUSS_CACHE: dict[tuple[str, str, int, float, int], torch.Tensor] = {}


def _gaussian_window(win_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create (C,1,win,win) gaussian kernel for depthwise conv2d."""
    key = (str(device), str(dtype), int(win_size), float(sigma), int(channels))
    w = _GAUSS_CACHE.get(key)
    if w is not None:
        return w

    coords = torch.arange(win_size, device=device, dtype=torch.float32) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :]).to(dtype)
    g2d = g2d / g2d.sum()

    w = g2d.view(1, 1, win_size, win_size).repeat(channels, 1, 1, 1).contiguous()
    _GAUSS_CACHE[key] = w
    return w


def _ssim_and_cs(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ssim, cs) per batch (shape [B])."""
    B, C, H, W = x.shape
    win_size = window.shape[-1]
    pad = win_size // 2

    # NOTE:
    #   - reflect padding fails when pad >= input spatial size (can happen at small
    #     MS-SSIM pyramid scales).
    #   - replicate padding is robust for any pad size and is commonly used.
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y_pad = F.pad(y, (pad, pad, pad, pad), mode="replicate")

    mu_x = F.conv2d(x_pad, window, groups=C)
    mu_y = F.conv2d(y_pad, window, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x_pad * x_pad, window, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y_pad * y_pad, window, groups=C) - mu_y2
    sigma_xy = F.conv2d(x_pad * y_pad, window, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # contrast-structure term
    cs_map = (2.0 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)
    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    ssim = ssim_map.mean(dim=(1, 2, 3))
    cs = cs_map.mean(dim=(1, 2, 3))
    return ssim, cs


# @torch.no_grad()
# def ms_ssim(
#     x01: torch.Tensor,
#     y01: torch.Tensor,
#     data_range: float = 1.0,
#     win_size: int = 11,
#     win_sigma: float = 1.5,
#     weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
#     K1: float = 0.01,
#     K2: float = 0.03,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """Multi-Scale SSIM for tensors in [0,1]. Returns a scalar tensor.

#     Implementation follows the common MS-SSIM recipe used in `pytorch-msssim` and
#     other widely used references.
#     """
#     if x01.ndim != 4 or y01.ndim != 4 or x01.shape != y01.shape:
#         raise ValueError(f"Expected x01,y01 [B,C,H,W] same shape; got {tuple(x01.shape)} vs {tuple(y01.shape)}")

#     x = x01.to(torch.float32)
#     y = y01.to(torch.float32)

#     B, C, H, W = x.shape

#     # Ensure window size fits (for small images) while staying odd.
#     win = int(win_size)
#     min_hw = int(min(H, W))
#     if min_hw < win:
#         win = max(3, min_hw)
#         if win % 2 == 0:
#             win -= 1
#     if win < 3:
#         # Degenerate fallback
#         return torch.tensor(0.0, device=x.device, dtype=torch.float32)

#     window = _gaussian_window(win, float(win_sigma), C, x.device, x.dtype)
#     w = torch.tensor(list(weights), device=x.device, dtype=torch.float32)
#     levels = int(w.numel())

#     mcs: list[torch.Tensor] = []
#     x_i, y_i = x, y
#     ssim_last: Optional[torch.Tensor] = None

#     for i in range(levels):
#         ssim_i, cs_i = _ssim_and_cs(x_i, y_i, window, data_range=data_range, K1=K1, K2=K2)
#         if i < levels - 1:
#             mcs.append(cs_i.clamp_min(eps))
#             # downsample by 2
#             x_i = F.avg_pool2d(x_i, kernel_size=2, stride=2, padding=0)
#             y_i = F.avg_pool2d(y_i, kernel_size=2, stride=2, padding=0)
#         else:
#             ssim_last = ssim_i.clamp_min(eps)

#     if ssim_last is None:
#         return torch.tensor(0.0, device=x.device, dtype=torch.float32)

#     # Combine using log-domain for stability: exp(sum(w * log(term)))
#     log_terms = torch.zeros((B,), device=x.device, dtype=torch.float32)
#     if len(mcs) > 0:
#         mcs_stack = torch.stack(mcs, dim=0)  # [levels-1, B]
#         log_terms = log_terms + (torch.log(mcs_stack) * w[:-1].unsqueeze(1)).sum(dim=0)
#     log_terms = log_terms + torch.log(ssim_last) * w[-1]

#     ms = torch.exp(log_terms)
#     return ms.mean().reshape(())


# @torch.no_grad()
# def ms_ssim(
#     x01: torch.Tensor,
#     y01: torch.Tensor,
#     data_range: float = 1.0,
#     win_size: int = 11,
#     win_sigma: float = 1.5,
#     weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
#     K1: float = 0.01,
#     K2: float = 0.03,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """Multi-Scale SSIM for tensors in [0,1]. Returns a scalar tensor.

#     This implementation uses `pytorch_msssim.ssim` for per-scale SSIM/CS
#     computation, and combines scales with the standard MS-SSIM recipe.

#     Requires: `pip install pytorch-msssim`.
#     """
#     if x01.ndim != 4 or y01.ndim != 4 or x01.shape != y01.shape:
#         raise ValueError(f"Expected x01,y01 [B,C,H,W] same shape; got {tuple(x01.shape)} vs {tuple(y01.shape)}")

#     try:
#         from pytorch_msssim import ssim as _ssim  # type: ignore
#     except Exception as e:  # pragma: no cover
#         raise ImportError("pytorch-msssim is required for MS-SSIM. Try `pip install pytorch-msssim`.") from e

#     x = x01.to(torch.float32)
#     y = y01.to(torch.float32)

#     B, C, H, W = x.shape

#     # Ensure window size fits (for small images) while staying odd.
#     win = int(win_size)
#     min_hw = int(min(H, W))
#     if min_hw < win:
#         win = max(3, min_hw)
#         if win % 2 == 0:
#             win -= 1
#     if win < 3:
#         return torch.tensor(0.0, device=x.device, dtype=torch.float32)

#     w = torch.tensor(list(weights), device=x.device, dtype=torch.float32)
#     levels = int(w.numel())

#     mcs: list[torch.Tensor] = []
#     x_i, y_i = x, y
#     ssim_last: Optional[torch.Tensor] = None

#     for i in range(levels):
#         # `pytorch_msssim.ssim` API differs slightly across versions.
#         # We try to obtain both (ssim, cs). If CS isn't available, we fall back to ssim.
#         ssim_i: Optional[torch.Tensor] = None
#         cs_i: Optional[torch.Tensor] = None

#         # Attempt 1: full=True returns (ssim, cs) on some versions.
#         try:
#             out = _ssim(
#                 x_i,
#                 y_i,
#                 data_range=data_range,
#                 win_size=win,
#                 win_sigma=win_sigma,
#                 K=(K1, K2),
#                 size_average=False,
#                 full=True,  # type: ignore
#             )
#         except TypeError:
#             out = None

#         if out is None:
#             # Attempt 2: some versions return tuple without `full` kw.
#             try:
#                 out = _ssim(
#                     x_i,
#                     y_i,
#                     data_range=data_range,
#                     win_size=win,
#                     win_sigma=win_sigma,
#                     K=(K1, K2),
#                     size_average=False,
#                 )
#             except TypeError:
#                 # Attempt 3: older signature may not support `K`/`win_sigma` keywords.
#                 out = _ssim(x_i, y_i, data_range=data_range, win_size=win, size_average=False)

#         if isinstance(out, (tuple, list)) and len(out) >= 2:
#             ssim_i, cs_i = out[0], out[1]
#         else:
#             ssim_i = out  # type: ignore[assignment]
#             cs_i = out  # fallback when CS is not provided

#         # Convert to per-image tensor [B]
#         if not torch.is_tensor(ssim_i):
#             ssim_i = torch.as_tensor(ssim_i, device=x.device, dtype=torch.float32)
#         if not torch.is_tensor(cs_i):
#             cs_i = torch.as_tensor(cs_i, device=x.device, dtype=torch.float32)

#         # If scalar returned, expand to [B]
#         if ssim_i.ndim == 0:
#             ssim_i = ssim_i.expand(B)
#         if cs_i.ndim == 0:
#             cs_i = cs_i.expand(B)

#         if i < levels - 1:
#             mcs.append(cs_i.clamp_min(eps))
#             # downsample by 2
#             x_i = F.avg_pool2d(x_i, kernel_size=2, stride=2, padding=0)
#             y_i = F.avg_pool2d(y_i, kernel_size=2, stride=2, padding=0)
#         else:
#             ssim_last = ssim_i.clamp_min(eps)

#     if ssim_last is None:
#         return torch.tensor(0.0, device=x.device, dtype=torch.float32)

#     # Combine using log-domain for stability: exp(sum(w * log(term)))
#     log_terms = torch.zeros((B,), device=x.device, dtype=torch.float32)
#     if len(mcs) > 0:
#         mcs_stack = torch.stack(mcs, dim=0)  # [levels-1, B]
#         log_terms = log_terms + (torch.log(mcs_stack) * w[:-1].unsqueeze(1)).sum(dim=0)
#     log_terms = log_terms + torch.log(ssim_last) * w[-1]

#     ms = torch.exp(log_terms)
#     return ms.mean().reshape(())




# ============================================================================
# Helper: fixed-range scaling to [0,1]
# ============================================================================


def _to_scalar_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().to(device).reshape(())
    return torch.tensor(float(x), device=device).reshape(())


def scale_01(x: torch.Tensor, vmin: float, vmax: float, eps: float = 1e-8) -> torch.Tensor:
    """Linear map to [0,1] with clipping."""
    return ((x - vmin) / (vmax - vmin + eps)).clamp(0.0, 1.0)


def weighted_mean(vals: Sequence[torch.Tensor], weights: Sequence[float], eps: float = 1e-8) -> torch.Tensor:
    if len(vals) == 0:
        raise ValueError("weighted_mean: empty vals")
    if len(vals) != len(weights):
        raise ValueError("weighted_mean: vals and weights length mismatch")
    w = torch.tensor(weights, device=vals[0].device, dtype=torch.float32)
    wsum = w.sum().clamp_min(eps)
    s = torch.zeros((), device=vals[0].device, dtype=torch.float32)
    for vi, wi in zip(vals, w):
        s = s + vi.to(torch.float32) * wi
    return s / wsum


# ============================================================================
# CLIP-IQA (TorchMetrics)
# ============================================================================


class CLIPIQAWrapper:
    """Stateless wrapper around TorchMetrics CLIP-IQA."""

    def __init__(
        self,
        device: torch.device,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        prompts: Tuple[str, ...] | Sequence[str] | str = ("quality", "sharpness", "noisiness"),
    ):
        try:
            from torchmetrics.multimodal import CLIPImageQualityAssessment
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "torchmetrics is required for CLIP-IQA. Try `pip install torchmetrics transformers`."
            ) from e

        if isinstance(prompts, str):
            prompts_t: Tuple[str, ...] = (prompts,)
        else:
            prompts_t = tuple(prompts)
        self.device = device
        self.prompts = prompts_t
        # IMPORTANT: torchmetrics expects prompts as a tuple, not list
        self.metric = CLIPImageQualityAssessment(
            model_name_or_path=model_name_or_path,
            prompts=prompts_t,
            data_range=1.0,
        ).to(device)
        # If available, prevent double rescaling warnings for [0,1] tensors
        try:
            proc = getattr(self.metric, "processor", None)
            ip = getattr(proc, "image_processor", None)
            if ip is not None and hasattr(ip, "do_rescale"):
                ip.do_rescale = False
        except Exception:
            pass

        self.metric.eval()

    @torch.inference_mode()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        if img01.ndim != 4:
            raise ValueError(f"Expected [N,3,H,W], got {tuple(img01.shape)}")

        x = img01.to(self.device)

        # 굳이 reset이 필요 없지만, stateful metric일 수 있으니 유지
        self.metric.reset()
        out = self.metric(x)

        # TorchMetrics는 prompts가 여러 개인 경우 dict로 반환할 수 있음
        if isinstance(out, dict):
            # 각 value는 보통 shape [N] 또는 [N, K] 텐서
            vals = []
            for v in out.values():
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, device=self.device)
                # [N, K]면 K 축 평균 → [N]
                if v.ndim == 2:
                    v = v.mean(dim=1)
                # [N, 1] 같은 경우 squeeze
                if v.ndim > 1:
                    v = v.squeeze()
                vals.append(v)

            # prompts 축 평균: [num_prompts, N] -> [N]
            out_t = torch.stack(vals, dim=0).mean(dim=0)
        else:
            # Tensor로 오는 경우
            out_t = out
            if not torch.is_tensor(out_t):
                out_t = torch.as_tensor(out_t, device=self.device)
            if out_t.ndim == 2:
                out_t = out_t.mean(dim=1)
            if out_t.ndim > 1:
                out_t = out_t.squeeze()

        # 배치 평균 -> scalar
        return out_t.mean().to(torch.float32).reshape(())


# ============================================================================
# Q-Align / OneAlign (Transformers)
# ============================================================================


def _tensor01_to_pil(img01: torch.Tensor) -> "Image.Image":
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for Q-Align conversion: pip install pillow")

    x = img01.detach().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {tuple(x.shape)}")
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


class QAlignWrapper:
    """Thin wrapper for `q-future/one-align` quality score.

    The model card indicates output is in range [1, 5] for `task_='quality'`.
    """
    # iqa_reward.py (QAlignWrapper)

    @staticmethod
    def _patch_onealign_runtime(model, attn_impl: str = "eager"):
        import inspect

        use_fa2 = (attn_impl == "flash_attention_2")
        use_sdpa = (attn_impl == "sdpa")
        use_eager = (attn_impl == "eager")

        for m in model.modules():
            cls = m.__class__.__name__
            if ("LlamaModel" in cls) or ("MPLUGOwl2LlamaModel" in cls):
                if not hasattr(m, "_use_flash_attention_2"):
                    m._use_flash_attention_2 = use_fa2
                else:
                    m._use_flash_attention_2 = use_fa2

                if not hasattr(m, "_use_sdpa"):
                    m._use_sdpa = use_sdpa
                else:
                    m._use_sdpa = use_sdpa

                if not hasattr(m, "_use_eager_attention"):
                    m._use_eager_attention = use_eager
                else:
                    m._use_eager_attention = use_eager

        for m in model.modules():
            name = m.__class__.__name__
            if "RotaryEmbedding" not in name:
                continue

            orig_forward = m.forward  # bound method
            try:
                sig = inspect.signature(orig_forward)
            except Exception:
                sig = None

            def wrapped_forward(x, *args, __orig=orig_forward, __sig=sig, **kwargs):
                if "seq_len" in kwargs:
                    seq_len = kwargs.pop("seq_len")


                    try:
                        return __orig(x, *args, seq_len=seq_len, **kwargs)
                    except TypeError:
                        pass


                    if __sig is not None and "position_ids" in __sig.parameters:
                        bsz = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 1
                        position_ids = torch.arange(int(seq_len), device=x.device, dtype=torch.long).unsqueeze(0)
                        if bsz > 1:
                            position_ids = position_ids.expand(bsz, -1)

                        try:
                            return __orig(x, *args, position_ids=position_ids, **kwargs)
                        except TypeError:
                            return __orig(x, *args, position_ids, **kwargs)


                    try:
                        return __orig(x, *args, int(seq_len), **kwargs)
                    except TypeError:
                        return __orig(x, *args, **kwargs)

                return __orig(x, *args, **kwargs)


            m.forward = wrapped_forward

    @staticmethod
    def _patch_attn_flags(model, attn_impl: str):

        attn_impl = str(attn_impl)
        use_fa2 = (attn_impl == "flash_attention_2")
        use_sdpa = (attn_impl == "sdpa")
        use_eager = (attn_impl == "eager")


        for m in model.modules():
            name = m.__class__.__name__.lower()
            if "llama" in name and "model" in name:
                if not hasattr(m, "_use_flash_attention_2"):
                    m._use_flash_attention_2 = use_fa2
                else:
                    m._use_flash_attention_2 = use_fa2

                if not hasattr(m, "_use_sdpa"):
                    m._use_sdpa = use_sdpa
                else:
                    m._use_sdpa = use_sdpa

                if not hasattr(m, "_use_eager_attention"):
                    m._use_eager_attention = use_eager
                else:
                    m._use_eager_attention = use_eager
    
    def __init__(
        self,
        model_name_or_path: str = "q-future/one-align",
        device: torch.device | str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = "auto",
        attn_implementation: str = "eager",
    ):
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:  # pragma: no cover
            raise ImportError("transformers is required for Q-Align: pip install transformers") from e

        self.device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # device_map='auto' requires accelerate; if not available, fall back to manual .to(device)
        if device_map is not None:
            try:
                import accelerate  # noqa: F401
            except Exception:
                device_map = None

        # Compatibility patch: the OneAlign (MPLUGOwl2) config may miss fields expected by
        # some Transformers versions (e.g., `mlp_bias`). We pre-load the config and add
        # missing attributes before instantiating the model.
        cfg = None
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            if not hasattr(cfg, "mlp_bias"):
                cfg.mlp_bias = False
        except Exception:
            cfg = None

        kwargs = dict(
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if cfg is not None:
            kwargs["config"] = cfg

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        # self._patch_attn_flags(self.model, attn_implementation)
        self._patch_onealign_runtime(self.model, attn_implementation)


        if device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    
    @torch.inference_mode()
    def __call__(self, img01: torch.Tensor, task: str = "quality") -> torch.Tensor:
        pil = _tensor01_to_pil(img01)
        out = self.model.score([pil], task_=task, input_="image")

        # Robustly convert to scalar tensor
        if isinstance(out, (list, tuple)):
            out = out[0]
        if torch.is_tensor(out):
            return out.detach().to(torch.float32).mean().reshape(())
        return torch.tensor(float(out), device=self.device, dtype=torch.float32).reshape(())


# ============================================================================
# Reward config + computation
# ============================================================================


@dataclass
class ModernRewardConfig:
    # Group mixing
    distortion_weight: float = 0.5
    perception_weight: float = 0.5

    # --- Distortion metrics ---
    use_psnr: bool = True
    psnr_min: float = 20.0
    psnr_max: float = 40.0
    w_psnr: float = 1.0

    use_msssim: bool = False
    msssim_min: float = 0.80
    msssim_max: float = 1.00
    w_msssim: float = 1.0

    # --- Perception metrics ---
    use_lpips: bool = True
    lpips_patch: int = 224
    lpips_stride: int = 224
    w_lpips: float = 1.0

    use_clip_iqa: bool = True
    w_clip_iqa: float = 1.0

    use_qalign: bool = True
    w_qalign: float = 1.0
    qalign_task: str = "quality"  # quality | aesthetics

    # Shared encoder input size for NR metrics
    iqa_resize: Optional[int] = 224

    # NR view strategy for high-res images
    # - 'resize': resize full image -> iqa_resize
    # - 'center': center crop at native res -> resize
    # - 'five'  : five-crop (center + 4 corners) -> resize, then average
    nr_view_mode: str = "resize"  # resize|center|five
    nr_crop_native: int = 0  # 0 uses full image; otherwise native crop size (e.g., 384/512)


class ModernIQAReward:
    """Compute a fixed-scale reward from multiple IQA metrics (stateless across candidates)."""

    def __init__(
        self,
        device: torch.device,
        lpips_fn,
        cfg: Optional[ModernRewardConfig] = None,
        clip_iqa: Optional[CLIPIQAWrapper] = None,
        qalign: Optional[QAlignWrapper] = None,
    ):
        self.device = device
        self.lpips_fn = lpips_fn
        self.cfg = cfg or ModernRewardConfig()
        self.clip_iqa = clip_iqa
        self.qalign = qalign

        # Normalize group weights
        dw = float(self.cfg.distortion_weight)
        pw = float(self.cfg.perception_weight)
        s = dw + pw
        if s <= 0:
            raise ValueError("distortion_weight + perception_weight must be > 0")
        self._dw = dw / s
        self._pw = pw / s

    @torch.no_grad()
    def _resize_to_eval(self, x01: torch.Tensor, r: Optional[int]) -> torch.Tensor:
        if r is None or int(r) <= 0:
            return x01
        r = int(r)
        if x01.shape[-2] == r and x01.shape[-1] == r:
            return x01
        return F.interpolate(x01, size=(r, r), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def _make_nr_views(self, x01: torch.Tensor) -> list[torch.Tensor]:
        mode = (self.cfg.nr_view_mode or "resize").lower()
        eval_r = self.cfg.iqa_resize

        if mode == "resize":
            return [self._resize_to_eval(x01, eval_r)]

        B, C, H, W = x01.shape
        crop = int(self.cfg.nr_crop_native) if int(self.cfg.nr_crop_native) > 0 else min(H, W)
        crop = max(1, min(crop, H, W))

        def _clip(v: int, vmax: int) -> int:
            return int(max(0, min(int(v), int(vmax))))

        yc = _clip((H - crop) // 2, H - crop)
        xc = _clip((W - crop) // 2, W - crop)

        if mode == "center":
            coords = [(yc, xc)]
        elif mode == "five":
            ys = [0, H - crop]
            xs = [0, W - crop]
            coords = [(yc, xc), (ys[0], xs[0]), (ys[0], xs[1]), (ys[1], xs[0]), (ys[1], xs[1])]
        else:
            raise ValueError(f"Unknown nr_view_mode: {self.cfg.nr_view_mode}")

        views: list[torch.Tensor] = []
        for y0, x0 in coords:
            y0 = _clip(y0, H - crop)
            x0 = _clip(x0, W - crop)
            v = x01[:, :, y0 : y0 + crop, x0 : x0 + crop]
            v = self._resize_to_eval(v, eval_r)
            views.append(v)
        return views

    @torch.inference_mode()
    def compute(self, out01: torch.Tensor, gt01: Optional[torch.Tensor] = None, task = None, A_funcs = None) -> tuple[torch.Tensor, Dict[str, float]]:
        """Return (reward, metrics_dict).

        Args:
            out01: [1,3,H,W] in [0,1]
            gt01:  [1,3,H,W] in [0,1] (required for distortion metrics + LPIPS)
        """
        out01 = out01.to(self.device)
        if gt01 is not None:
            gt01 = gt01.to(self.device)
        metrics: Dict[str, float] = {}

        # ----------------
        # Distortion group
        # ----------------
        dist_vals: list[torch.Tensor] = []
        dist_w: list[float] = []
        if self.cfg.use_psnr:
            if gt01 is None:
                raise ValueError("gt01 is required for PSNR")
            ps = psnr(out01, gt01)
            metrics["psnr"] = float(ps.item())
            ps01 = scale_01(ps, self.cfg.psnr_min, self.cfg.psnr_max)
            dist_vals.append(ps01)
            dist_w.append(float(self.cfg.w_psnr))

        if self.cfg.use_msssim:
            if gt01 is None:
                raise ValueError("gt01 is required for MS-SSIM")
            # ms = ms_ssim(out01, gt01)
            ms = ssim(out01.float(), gt01.float(), data_range=1.0)
            metrics["ms_ssim"] = float(ms.item())
            ms01 = scale_01(ms, self.cfg.msssim_min, self.cfg.msssim_max)
            dist_vals.append(ms01)
            dist_w.append(float(self.cfg.w_msssim))

        dist_score = weighted_mean(dist_vals, dist_w) if len(dist_vals) > 0 else torch.zeros((), device=self.device)
        metrics["distortion_score"] = float(dist_score.item())

        # -----------------
        # Perception group
        # -----------------
        perc_vals: list[torch.Tensor] = []
        perc_w: list[float] = []

        if self.cfg.use_lpips:
            if gt01 is None:
                raise ValueError("gt01 is required for LPIPS")
            patch = int(min(self.cfg.lpips_patch, int(out01.shape[-2]), int(out01.shape[-1])))
            stride = int(min(self.cfg.lpips_stride, patch))
            lp = patch_lpips_vgg(self.lpips_fn, out01, gt01, patch=patch, stride=stride)
            metrics["lpips_patch"] = float(lp.item())
            # lower is better -> convert to higher-better
            lp01 = 1.0 - lp.clamp(0.0, 1.0)
            perc_vals.append(lp01)
            perc_w.append(float(self.cfg.w_lpips))

        # NR metrics
        nr_views = self._make_nr_views(out01)
        metrics["nr_views_count"] = float(len(nr_views))
        nr_batch = torch.cat(nr_views, dim=0)  # [V,3,h,w]

        if self.cfg.use_clip_iqa:
            if self.clip_iqa is None:
                raise ValueError("use_clip_iqa=True but clip_iqa wrapper is None")
            ci = _to_scalar_tensor(self.clip_iqa(nr_batch), self.device)
            metrics["clip_iqa"] = float(ci.item())
            ci01 = ci.clamp(0.0, 1.0)
            perc_vals.append(ci01)
            perc_w.append(float(self.cfg.w_clip_iqa))

        if self.cfg.use_qalign:
            if self.qalign is None:
                raise ValueError("use_qalign=True but qalign wrapper is None")
            qa = _to_scalar_tensor(self.qalign(nr_batch, task=self.cfg.qalign_task), self.device)
            metrics["qalign"] = float(qa.item())
            # model card range [1,5] (higher better)
            qa01 = scale_01(qa, 1.0, 5.0)
            perc_vals.append(qa01)
            perc_w.append(float(self.cfg.w_qalign))

        perc_score = weighted_mean(perc_vals, perc_w) if len(perc_vals) > 0 else torch.zeros((), device=self.device)
        metrics["perception_score"] = float(perc_score.item())

        # -----------------
        # Final reward
        # -----------------
        reward = self._dw * dist_score + self._pw * perc_score
        metrics["reward"] = float(reward.item())
        return reward.reshape(()), metrics
