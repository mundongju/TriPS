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

from custom_util import *
from munch import munchify
from diffusers import AutoencoderTiny
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.utils import save_image
import os
import glob


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
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype, generator=torch.Generator(self.device).manual_seed(self.seed))
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

@register_solver("TriPS_T")
class SD3FlowDPS(SD3Euler):

    ######### Data consistency loss #############################
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
    ############################################################

    ######### Template functions def ###########################
    # 1) linear
    def lerp(self, a, b, t):
        return a + (b - a) * t

    # 2) exponential (normalized)
    def phi_exp(self, u, alpha=5.0):
        return (np.exp(alpha * u) - 1.0) / (np.exp(alpha) - 1.0)
    
    def phi_exp_latedrop(self, u, alpha=8.0):
        # stable version of (exp(alpha*u)-1)/(exp(alpha)-1)
        # uses expm1 to avoid cancellation and overflow issues
        u = np.asarray(u, dtype=np.float64)
        return np.expm1(alpha * u) / np.expm1(alpha)

    # 3) logarithmic (normalized)
    def phi_log(self, u, alpha=9.0):
        return np.log1p(alpha * u) / np.log1p(alpha)
    ############################################################

    ######### Main sampler loop ################################
    def sample(self, measurement, operator, task,
                     prompts: List[str], NFE:int,
                     img_shape: Optional[Tuple[int]]=None,
                     cfg_scale: float=1.0, batch_size: int = 1,
                     latent:Optional[List[torch.Tensor]]=None,
                     prompt_embs:Optional[List[torch.Tensor]]=None,
                     null_embs:Optional[List[torch.Tensor]]=None,
                     step_scale=None,
                     step_scale_2=None,
                     inner_steps=None,
                     sigma_y=None,
                     stochasticity_weight=None,
                     workdir=None,
                     function_dc=None,
                     function_cfg=None,
                     function_sto=None,
                     img_num=None,): # new
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
        images = []
        images_y = []
        u_hist = []
        step_scale_hist = []
        eta_n_hist = []
        cfg_scale_hist = []

        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-TriPS_T')
        
        step_scale_init_1 = step_scale
        step_scale_init_2 = step_scale_2

        # sample_num = 0
        
        for i, t in enumerate(pbar):
            inner_steps=inner_steps

            # endpoints
            u = i / (NFE - 1)
            half = NFE // 2
            
            if function_dc == 'linear':
                step_scale = self.lerp(step_scale_init_1,step_scale_init_2,u)
            elif function_dc == 'logarithm':
                alpha_exp = 6.0
                t_exp = self.phi_exp(u, alpha_exp)
                step_scale = self.lerp(step_scale_init_1,step_scale_init_2,t_exp)
            elif function_dc == 'exponential':
                alpha_log = 6.0
                t_log = self.phi_log(u, alpha_log)
                step_scale = self.lerp(step_scale_init_1,step_scale_init_2,t_log)


            if function_cfg == 'linear':
                cfg_scale = self.lerp(1.0,6.0,u)
            elif function_cfg == 'exponential':
                alpha_exp = 6.0
                t_exp = self.phi_exp(u, alpha_exp)
                cfg_scale = self.lerp(1.0,6.0,t_exp)
            elif function_cfg == 'logarithm':
                alpha_log = 6.0
                t_log = self.phi_log(u, alpha_log)
                cfg_scale = self.lerp(1.0,6.0,t_log)


            # # step_scale 
            u_hist.append(float(u))
            if torch.is_tensor(step_scale):
                step_scale_hist.append(float(step_scale.detach().cpu().item()))
            else:
                step_scale_hist.append(float(step_scale))

            if torch.is_tensor(cfg_scale):
                cfg_scale_hist.append(float(cfg_scale.detach().cpu().item()))
            else:
                cfg_scale_hist.append(float(cfg_scale))

            timestep = t.expand(z.shape[0]).to(self.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
            
            sigma_max = sigmas[1]
            sigma_min = sigmas[-1]
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            delta = sigma - sigma_next

            pred_v_fin = pred_null_v + cfg_scale * (pred_v-pred_null_v)

            
            # denoising
            # 1. reverse process
            z_curr = z
            z_next = z - delta * pred_v_fin
            z0t = z - sigma * pred_v_fin
            z1t = z + (1-sigma) * pred_v_fin

            if i < NFE:
                z0y = self.data_consistency(z0t, A_funcs, y, sigma,
                                                    step_scale=step_scale, 
                                                    sigma_y=sigma_y, 
                                                    eta_tilde=0.8, 
                                                    phi=1.0,
                                                    inner_steps=inner_steps)
                
            else:
                z0y = z0t

            eps = torch.randn_like(z1t)
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            
            if function_sto == 'linear':
                alpha = self.lerp(1.0,0.0,u)
            elif function_sto == 'logarithm':
                alpha_exp = 15.0
                t_exp = self.phi_exp_latedrop(u, alpha_exp)
                alpha = self.lerp(1.0,0.0,t_exp) * stochasticity_weight
            elif function_sto == 'exponential':
                alpha_log = 50.0
                t_log = self.phi_log(u, alpha_log)
                alpha = self.lerp(1.0,0.0,t_log)

            if alpha > 1.0:
                alpha = 1.0
            z1y = ((1-alpha**2)**0.5) * z1t + alpha * eps

            z = (1 - sigma_next) * z0y + sigma_next * z1y


            # next state
            z = (1 - sigma_next) * z0y + sigma_next * z1y

            # eta_schedule 
            eta_n = alpha * sigma_next
            if torch.is_tensor(eta_n):
                eta_n_hist.append(float(eta_n.detach().cpu().item()))
            else:
                eta_n_hist.append(float(eta_n))


            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)

        
        ### demo plot
        if img_num < 2:
            save_dir = workdir if workdir is not None else "."
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "step_scale_vs_u.png")

            plt.figure()
            plt.plot(u_hist, step_scale_hist, marker="o")
            plt.xlabel("u")
            plt.ylabel("step_scale")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()

            save_path = os.path.join(save_dir, "cfg_scale_vs_u.png")

            plt.figure()
            plt.plot(u_hist, cfg_scale_hist, marker="o")
            plt.xlabel("u")
            plt.ylabel("cfg_scale")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()

            save_path = os.path.join(save_dir, "eta_n_scale_vs_u.png")

            plt.figure()
            plt.plot(u_hist, eta_n_hist, marker="o")
            plt.xlabel("u")
            plt.ylabel("eta_n_scale")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()

        
        # decode
        with torch.no_grad():
            img = self.decode(z)
            img = (img / 2 + 0.5).clamp(0, 1).detach().cpu()
        return img, images
    ############################################################


########################################################################################
######################baseline #########################################################
########################################################################################

###### FlowDPS ###############
@register_solver("flowdps")
class SD3FlowDPS(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task, step_scale:int=30.0, inner_steps:int=3.0):
        z0t = z0t.requires_grad_(True)
        # num_iters = 3
        for _ in range(inner_steps): # modify
            x0t = self.decode(z0t).float()
            if "sr" in task:
                loss = torch.linalg.norm((operator.A_pinv_add_eta(measurement, 0) - operator.A_pinv_add_eta(operator.A(x0t), 0)).view(1, -1))
            else:
                loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - step_scale*grad
 
        return z0t.detach(), grad.detach()
 
 
    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_scale: float=30.0,
               inner_steps: float=3.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               stochasticity_scale=None,
               workdir=None, # new
               return_step: int=0): # new
 
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
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
 
        # Solve ODE
        images = []   
        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-FlowDPS')
        for i, t in enumerate(pbar):
            if i < return_step:
                continue
            timestep = t.expand(z.shape[0]).to(self.device)
 
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
 
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
 
            # denoising
            z_curr = z
            z0t = z - sigma * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            z1t = z + (1-sigma) * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            delta = sigma - sigma_next
            
            if i < NFE:
                z0y, dc_grad = self.data_consistency(z0t, operator, measurement, task=task, step_scale=step_scale, inner_steps=inner_steps)
                z0y = (1-sigma) * z0t + sigma * z0y
            else:
                z0y = z0t
 
            # renoising (FlowDPS style)
            eps = torch.randn_like(z1t)
            alpha = math.sqrt(1-sigma_next) * stochasticity_scale
            if alpha >= 1:
                alpha = 1.0
            noise = ((1-alpha**2)**0.5) * z1t + alpha * eps
 
            z = (1-sigma_next) * z0y + sigma_next * noise
 
 
 
            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)
 
        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img, images
 
 
 
###### Flowchef ###############
@register_solver("flowchef")
class SD3FlowChef(SD3Euler):
    def data_consistency(self, z, z0t, operator, measurement, task):
        x0t = self.decode(z0t).float()
        if "sr_bicubic" in task:
            loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
        else:
            loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
        grad = torch.autograd.grad(loss, z0t)[0].half()
        return grad.detach()
    
    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_scale: float=50.0,
               inner_steps: float=3.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               stochasticity_scale=None,
               workdir=None): # new
 
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
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        
        # Solve ODE
        images = []
        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-FlowChef')
        for i, t in enumerate(pbar):
            z = z.clone().detach().requires_grad_(True)
            timestep = t.expand(z.shape[0]).to(self.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
 
            z_curr = z
 
            # denoising
            pred_v_fin = pred_null_v + cfg_scale * (pred_v - pred_null_v)
            delta = sigma - sigma_next
            for j in range(inner_steps): # num_iters
                z0t = z - sigma * pred_v_fin
                grad = self.data_consistency(z, z0t, operator, measurement, task=task)
                z = z - step_scale * grad
 
            z0t_dc = z - sigma * pred_v_fin  # z0t after DC correction for visualization
            z0t_inp = z_curr - sigma * pred_v_fin
            
            # update
            z = z - delta * pred_v_fin
 
            img_out = self.decode(z0t_inp).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)
 
        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img, images
 
 
###### ReSample ###############
@register_solver("resample")
class SD3ReSample(SD3Euler):
    def __init__(self, model_key = 'stabilityai/stable-diffusion-3.5-medium', device='cuda'):
        super().__init__(model_key, device)
        self.gamma = 40
        
    def data_consistency(self, z0t, operator, measurement, task, step_scale:int=30.0, inner_steps:int=3.0):
        z0t = z0t.requires_grad_(True)
        for _ in range(inner_steps):
            x0t = self.decode(z0t).float()
            if "sr_bicubic" in task:
                loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
            else:
                loss = torch.linalg.norm((measurement - operator.A(x0t)).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - step_scale*grad
 
        return z0t.detach()
 
    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_scale: float=30.0,
               inner_steps: float=3.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               stochasticity_scale=None,
               workdir=None,
               return_step: int=0):
 
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
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
 
        # Solve ODE
        images = []
        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-ReSample')
        for i, t in enumerate(pbar):
            if i < return_step:
                continue
            timestep = t.expand(z.shape[0]).to(self.device)
 
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
 
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            alpha_bar = (1 - sigma) ** 2 / (sigma ** 2 + (1 - sigma) ** 2)
            alpha_bar_next = (1 - sigma_next) ** 2 / (sigma_next ** 2 + (1 - sigma_next) ** 2)
            noise = torch.randn_like(pred_v)
            
            v_final = (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            
            z0t = z - sigma * v_final
            z1t = z + (1-sigma) * v_final
            
            z1t = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * noise
            z_prime_t = sigma_next * z1t + (1 - sigma_next) * z0t
            
            z0y = z0t  # default: no DC correction
            if i < NFE - 1:
                noise = torch.randn_like(pred_v)
                resample_sigma = self.gamma * (1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next)
                z0y = self.data_consistency(z0t, operator, measurement, task=task, step_scale=step_scale, inner_steps=inner_steps)
                z = (resample_sigma * math.sqrt(alpha_bar_next) * z0y + (1 - alpha_bar_next) * z_prime_t) / (resample_sigma + 1 - alpha_bar_next) + noise * math.sqrt(1/(1/resample_sigma + 1/(1 - alpha_bar_next)))
            else:
                z = z_prime_t
 
            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)
        
        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img, images
 
###### Flower ###############
@register_solver("flower")
class SD3Flower(SD3Euler):
    def proximal_dc(self, z0t, operator, measurement, task,
                    step_scale: float = 15.0,
                    inner_steps: int = 3,
                    nu_t_sq: float = 1.0):
        z = z0t.clone().detach().requires_grad_(True)
        for _ in range(inner_steps):
            x0t = self.decode(z).float()
            if "sr" in task:
                residual = (operator.A_pinv_add_eta(measurement, 0)
                            - operator.A_pinv_add_eta(operator.A(x0t), 0))
                loss = torch.linalg.norm(residual.view(1, -1))
            else:
                residual = operator.At(measurement) - operator.At(operator.A(x0t))
                loss = torch.linalg.norm(residual.view(1, -1))
            grad = torch.autograd.grad(loss, z)[0].half()
            z = z - step_scale * nu_t_sq * grad
        return z.detach()
 
    def sample(self, measurement, operator, task,
               prompts: List[str], NFE: int,
               img_shape: Optional[Tuple[int]] = None,
               cfg_scale: float = 1.0, batch_size: int = 1,
               step_scale: float = 15.0,
               inner_steps: int = 3,
               latent: Optional[List[torch.Tensor]] = None,
               prompt_emb: Optional[List[torch.Tensor]] = None,
               null_emb: Optional[List[torch.Tensor]] = None,
               stochasticity_scale: float = 1.0,   # not used by Flower (pure noise)
               workdir=None,
               sigma_y: float = 0.03,               # measurement noise σ_n
               gamma_flower: int = 0,                # uncertainty flag γ ∈ {0,1}
               **kwargs):
 
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
 
        # ---- encode text prompts (identical to FlowDPS) ----
        with torch.no_grad():
            if prompt_emb is None or prompt_emb[0] is None:
                prompt_emb_t, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb_t, pooled_emb = prompt_emb[0], prompt_emb[1]
            prompt_emb_t = prompt_emb_t.to(self.transformer.device)
            pooled_emb   = pooled_emb.to(self.transformer.device)
 
            if null_emb is None or null_emb[0] is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]
            null_prompt_emb  = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb  = null_pooled_emb.to(self.transformer.device)
 
        # ---- initialise latent z₀ ~ N(0, I) ----
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent
 
        # ---- time schedule ----
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
 
        # ---- iterative Flower loop ----
        images = []
        pbar = tqdm(timesteps, total=NFE, desc='SD3.5-FLOWER')
 
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
 
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb_t, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(
                        z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
 
            sigma      = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0
            sigma_val  = sigma.item() if isinstance(sigma, torch.Tensor) else float(sigma)
            sigma_next_val = sigma_next.item() if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
 
            v_cfg = pred_null_v + cfg_scale * (pred_v - pred_null_v)
            z0t = z - sigma * v_cfg                      # estimated clean latent
 
            t_flower = 1.0 - sigma_val                    
            denom = math.sqrt(t_flower ** 2 + sigma_val ** 2) + 1e-8
            nu_t = sigma_val / denom
            nu_t_sq = nu_t ** 2
 
            if i < NFE:
                z0y = self.proximal_dc(
                    z0t, operator, measurement, task,
                    step_scale=step_scale,
                    inner_steps=inner_steps,
                    nu_t_sq=nu_t_sq,
                )
            else:
                z0y = z0t
 

            if gamma_flower == 1 and sigma_val > 1e-6:
                kappa = nu_t * torch.randn_like(z0y)
                z0y = z0y + kappa
 
            if sigma_next_val > 1e-8:
                eps_fresh = torch.randn_like(z)
                z = sigma_next * eps_fresh + (1.0 - sigma_next) * z0y
            else:
                z = z0y                                  # last step — no noise
 
            # ---- save intermediate visualisation ----
            with torch.no_grad():
                img_out = self.decode(z0t).float()
                img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)
 
        # ---- decode final output ----
        with torch.no_grad():
            img = self.decode(z)
        return img, images
