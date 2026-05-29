from typing import List, Tuple, Optional
import math
import torch

from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
import numpy as np
from cores.mcmc import MCMCSampler
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F

# from custom_util import *
from munch import munchify
# from functions.degradation import get_degradation
from diffusers import AutoencoderTiny
import matplotlib.pyplot as plt

################### add #########################################
from pathlib import Path
from torchvision.utils import save_image
#################################################################

# =======================================================================
# Factory
# =======================================================================

__SOLVER__ = {}

def register_solver(name:str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name:str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

import time
def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _tic():
    _sync(); return time.perf_counter()

def _toc(t0):
    _sync(); return time.perf_counter() - t0

class StableDiffusion3Base():
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        keep_pipe_for_encoding: bool = True,
    ):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
        )

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3

        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd3", torch_dtype=torch.float16
        ).to(self.device).eval()

        self.transformer = pipe.transformer.to(self.device).eval()
        self.transformer.requires_grad_(False)

        if keep_pipe_for_encoding:
            pipe.transformer = None
            pipe.vae = None
            self._encode_prompt = pipe.encode_prompt
            self._pipe_for_encoding = pipe
        else:
            self._encode_prompt = None
            self._pipe_for_encoding = None
            del pipe

        self.seed = None

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None else 8
        )
    
    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        prompt_3: Optional[List[str]] = None,
        negative_prompt_3: Optional[List[str]] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        prompt_embeds, neg_embeds, pooled, neg_pooled = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=torch.device(self.device),
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=clip_skip,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )

        return prompt_embeds.to(self.dtype), pooled.to(self.dtype)

    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)
        if self.seed is not None:
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype, generator=torch.Generator(self.device).manual_seed(42))
        else:
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        img_latent = self.vae.encode(image, return_dict=False)[0]
        if hasattr(img_latent, "sample"):
            img_latent = img_latent.sample()
        img_latent = (img_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return img_latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(z / self.vae.config.scaling_factor + self.vae.config.shift_factor, return_dict=False)[0]
        return img

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v

class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3.5-medium', device='cuda'):
        super().__init__(model_key=model_key, device=device)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1,
                  prompt_emb:Optional[List[torch.Tensor]]=None,
                  null_emb:Optional[List[torch.Tensor]]=None):

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""])
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img).to(self.transformer.device)

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        y_1 = torch.randn_like(z)
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            pred_v_fin = pred_null_v + cfg_scale * (pred_v - pred_null_v)

            z = z + (sigma_next - sigma) * pred_v_fin

        return z


    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("TriPS_G_test")
class SD3FlowDPS(SD3Euler):

    def data_consistency(
        self, z0t, A, y, sigma,
        step_scale=1.0, sigma_y=0.03, eta_tilde=0.8, eta_min=1e-4, phi=1.0, inner_steps=3
    ):
        lambda_t = ((1 - sigma).clamp(0, 1))**phi
        wBP, wLS = (1 - lambda_t), lambda_t

        eta_reg = max(eta_min, (sigma_y**2) * eta_tilde)
        
        z = z0t.detach().to(self.vae.device).requires_grad_(True)
        decay = float(sigma)**2
        step  = (step_scale * (0.25 + 0.75 * decay)) / inner_steps

        for _ in range(inner_steps):
            x = self.decode(z).float()  # FP32

            loss_BP = torch.linalg.norm(A.A_pinv_add_eta(A.A(x.reshape(x.size(0), -1).contiguous()), eta_reg) - A.A_pinv_add_eta(y.reshape(y.size(0), -1).contiguous(), eta_reg))
            loss_LS = torch.linalg.norm(A.A(x.reshape(x.size(0), -1).contiguous()) - y.reshape(y.size(0), -1).contiguous())

            loss = wBP * loss_BP + wLS * loss_LS
            
            z_grad = torch.autograd.grad(loss, z)[0].half()

            z = (z - step * z_grad).detach().requires_grad_(True)
        z = z.detach().to(device=self.transformer.device, dtype=z0t.dtype)
        return z


    def sample_final(self, measurement, operator, task,
                     prompts: List[str], NFE:int,
                     img_shape: Optional[Tuple[int]]=None,
                     cfg_scale: float=1.0, batch_size: int = 1,
                     latent:Optional[List[torch.Tensor]]=None,
                     prompt_embs:Optional[List[torch.Tensor]]=None,
                     null_embs:Optional[List[torch.Tensor]]=None,
                     inner_steps=None,
                     sigma_y=None,
                     cfg_schedule: Optional[torch.Tensor] = None,   # shape [NFE]
                     step_schedule: Optional[torch.Tensor] = None,  # shape [NFE]
                     eta_schedule: Optional[torch.Tensor] = None,   # shape [NFE], in [0,1]
        ):
        A_funcs = operator
        y = measurement
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_embs is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_embs[0], prompt_embs[1]
            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)
            if null_embs is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_embs[0], null_embs[1]
            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        images_x0t = []
        images_x0y = []

        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-TriPS_G_test')
        for i, t in enumerate(pbar):

            cfg_scale_i = float(cfg_schedule[i].detach().to("cuda").item())
            step_scale_i = float(step_schedule[i].detach().to("cuda").item())
            eta_i = eta_schedule[i].to(device=z.device, dtype=z.dtype).clamp(0.0, 1.0)
 
            timestep = t.expand(z.shape[0]).to(self.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale_i != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
            
            sigma_max = sigmas[1]
            sigma_min = sigmas[-1]
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            delta = sigma - sigma_next
            pred_v_fin = pred_null_v + cfg_scale_i * (pred_v-pred_null_v)
            
            # denoising
            # 1. reverse process
            z_curr = z
            z_next = z - delta * pred_v_fin
            z0t = z - sigma * pred_v_fin
            z1t = z + (1-sigma) * pred_v_fin

            if i < NFE:
                if "inpainting" in task:
                    z0y = self.data_consistency(z0t, A_funcs, y, sigma,
                                                step_scale=step_scale_i, 
                                                sigma_y=sigma_y, 
                                                eta_tilde=0.8, 
                                                phi=1.0,
                                                inner_steps=12)
                else:
                    if task == "sr_bicubic":
                        eta_tilde = 0
                    else:
                        eta_tilde = 0.8
                    z0y = self.data_consistency(z0t, A_funcs, y, sigma,
                                                        step_scale=step_scale_i, 
                                                        sigma_y=sigma_y, 
                                                        eta_tilde=eta_tilde, 
                                                        phi=1.0,
                                                        inner_steps=inner_steps)
                
            else:
                z0y = z0t

            eps = torch.randn_like(z1t)
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            
            z1y = torch.sqrt(torch.clamp(1.0 - eta_i**2, min=0.0)) * z1t + eta_i * eps
            z = (1 - sigma_next) * z0y + sigma_next * z1y 
            
        # decode
        with torch.no_grad():
            img = self.decode(z)
            img = (img / 2 + 0.5).clamp(0, 1).detach().cpu()
        return img, images_x0t, images_x0y
        
