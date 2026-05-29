"""
Phase Retrieval Operator for TriPS framework.

Forward model:  y = |FFT2(pad(x))|
    - x: image in [-1, 1], shape (B, C, H, W)
    - pad: zero-padding to achieve oversampling in Fourier domain
    - FFT2: 2D FFT per channel
    - |·|: element-wise magnitude (amplitude)

This is a nonlinear inverse problem. The pseudo-inverse is not well-defined,
so A_pinv_add_eta returns an identity-like operation (returns input as-is)
to make the DC loss default to ||A(x) - y||.

Reference:
    DPS (Chung et al., ICLR 2023)
    https://github.com/DPS2022/diffusion-posterior-sampling
"""

import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

from torch.fft import fft2, ifft2
from utils_treg.img_util import fft2d


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    @abstractmethod
    def noisy_forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self,
                 oversample,
                 noise,
                 noise_scale,
                 device):
        self.pad = int((oversample / 8.0) * 768)
        self.device = device
        self.noise = get_noise(name=noise, scale=noise_scale)
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2d(padded).abs()
        return amplitude
    
    def A(self, data):
        return self.forward(data)

    def noisy_forward(self, data, **kwargs):
        return self.noise.forward(self.forward(data, **kwargs))
    

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def __init__(self, **kwargs):
        pass

    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.scale


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.scale) / 255.0 / self.scale)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)