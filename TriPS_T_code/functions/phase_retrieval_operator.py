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


class PhaseRetrievalOperator:
    """
    Fourier Phase Retrieval operator compatible with the TriPS / FlowDPS codebase.

    Parameters
    ----------
    channels : int
        Number of image channels (typically 3 for RGB).
    img_size : int
        Spatial resolution of the input image (assumes square).
    oversample : float
        Oversampling ratio. Controls zero-padding extent.
        pad = int((oversample / 8.0) * img_size)  — following DPS convention.
        Default 2.0 (standard for phase retrieval benchmarks).
    device : torch.device or str
        Device for computation.
    """

    def __init__(self, channels: int, img_size: int, oversample: float, device):
        self.channels = channels
        self.img_size = img_size
        self.oversample = oversample
        self.device = device
        self.pad = int((oversample / 8.0) * img_size)
        self.padded_size = img_size + 2 * self.pad

        # Pre-compute output dimension for flat representation
        # Per-channel Fourier magnitude: (padded_size x padded_size)
        self.meas_dim_per_channel = self.padded_size * self.padded_size
        self.meas_dim = channels * self.meas_dim_per_channel

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _to_image(self, x):
        """
        Reshape flat tensor (B, C*H*W) to image (B, C, H, W) if necessary.
        If already 4D, return as-is.
        """
        if x.dim() == 2:
            B = x.shape[0]
            return x.reshape(B, self.channels, self.img_size, self.img_size)
        return x  # already (B, C, H, W)

    def _fft2_magnitude(self, x_img):
        """
        Zero-pad the image and compute per-channel 2D FFT magnitude.

        Parameters
        ----------
        x_img : (B, C, H, W) tensor

        Returns
        -------
        amplitude : (B, C, padH, padW) tensor  — real-valued magnitudes
        """
        padded = F.pad(x_img, [self.pad] * 4, mode='constant', value=0)
        # torch.fft.fft2 operates on the last two dims by default
        # freq = torch.fft.fft2(padded, norm='ortho')  ###### fail
        freq = torch.fft.fft2(padded)  ###### modify
        amplitude = freq.abs()
        return amplitude

    # ------------------------------------------------------------------
    # Public interface  (matches SVD / FFT operator API)
    # ------------------------------------------------------------------

    def A(self, x):
        """
        Forward operator: y = |FFT2(pad(x))|

        Accepts (B, C*H*W) or (B, C, H, W). Returns in the same layout.
        """
        flat_input = (x.dim() == 2)
        x_img = self._to_image(x)
        amplitude = self._fft2_magnitude(x_img)  # (B, C, padH, padW)
        if flat_input:
            return amplitude.reshape(x.shape[0], -1)
        return amplitude

    def At(self, y):
        """
        Adjoint-like operation (for visualization / back-projection display only).

        For nonlinear phase retrieval there is no true adjoint.
        We return the center-cropped inverse FFT of the amplitude
        (treating amplitude as if it were a real-valued signal)
        so that the saved 'input1' images are at least visually meaningful.

        Accepts (B, meas_dim) or (B, C, padH, padW). Returns (B, C*H*W) or (B, C, H, W).
        """
        flat_input = (y.dim() == 2)
        if flat_input:
            B = y.shape[0]
            y_img = y.reshape(B, self.channels, self.padded_size, self.padded_size)
        else:
            y_img = y

        # Inverse FFT of real magnitude → rough spatial estimate
        # spatial = torch.fft.ifft2(y_img, norm='ortho').real

        # IFFT of magnitude -> rough spatial estimate (same norm convention)
        spatial = torch.fft.ifft2(y_img).real

        # Center-crop back to original image size
        p = self.pad
        cropped = spatial[:, :, p:p + self.img_size, p:p + self.img_size]

        if flat_input:
            return cropped.reshape(y.shape[0], -1)
        return cropped

    def A_pinv_add_eta(self, y, eta_reg=0.0):
        """
        Regularised pseudo-inverse. For phase retrieval (nonlinear), this
        does NOT exist in closed form.

        We return the input unchanged (identity) so that the BP loss term
        in data_consistency_DDPG_v2 gracefully degrades to:
            loss_BP = ||A(x) - y||   (same as loss_LS).

        This means the total DC loss is simply  (wBP + wLS) * ||A(x) - y||.
        """
        return y


# ---------------------------------------------------------------------------
# Convenience constructor (mirrors the calling convention in solve_ours_*.py)
# ---------------------------------------------------------------------------

def get_phase_retrieval_operator(channels, img_size, oversample, device):
    return PhaseRetrievalOperator(channels, img_size, oversample, device)
